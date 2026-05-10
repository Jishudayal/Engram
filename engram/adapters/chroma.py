"""Chroma adapter — sync client wrapped in asyncio.to_thread.

Requires the optional 'chroma' extra::

    pip install "engram[chroma]"

Chroma's Python client is synchronous. Every collection call is wrapped in
``asyncio.to_thread`` to avoid blocking the event loop. Each adapter call
dispatches one thread; for high-throughput workloads consider configuring
a dedicated ThreadPoolExecutor on the event loop.

Call ``open()`` (or use ``async with``) before any other method.

Tenant isolation: every record's Chroma ID is derived via UUID5 from
``(agent_id, memory_id)``, preventing cross-agent ID collisions within the
shared collection. The ``agent_id`` is also stored as flat metadata for
server-side ``where`` filtering.

Null embeddings: Chroma requires a vector for every record. Memories without
an embedding are stored with a zero-vector plus ``_no_embedding: True`` in
the metadata. They are excluded from search results and the embedding is
returned as None on fetch.

Score conversion: collections are created with ``hnsw:space=cosine``. Chroma
returns distances (``1 - cosine_similarity``); this adapter converts to scores
via ``score = max(0.0, 1.0 - distance)`` so scores are always in [0, 1].

Metadata serialization: Chroma metadata values must be scalar (str, int,
float, bool). The full Memory payload is JSON-serialized under the ``_payload``
metadata key. Fields needed for server-side filtering (``agent_id``, ``status``,
``_no_embedding``) are also stored as flat metadata fields.

Metadata filters (search): user-supplied ``filters`` are applied client-side
after fetching top_k results from Chroma, by deserializing ``_payload``.
Chroma's ``where`` clause cannot filter on nested JSON. Fewer than top_k
results may be returned if many results are filtered out.

list_all note: Chroma does not support server-side ordering by payload field.
``list_all`` fetches all matching records and sorts client-side — O(n) in
the agent's record count.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from typing import Any

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.errors import ChromaError

from engram.adapters._utils import (
    POINT_NAMESPACE,
    map_adapter_errors,
    memory_to_payload,
    payload_to_memory,
)
from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus
from engram.core.exceptions import AdapterError, NotFoundError
from engram.core.models import Memory, SearchResult

__all__ = ["ChromaAdapter"]

_NO_EMBEDDING_KEY = "_no_embedding"
_RAW_EMBEDDING_KEY = "_raw_embedding"
_PAYLOAD_KEY = "_payload"
_AGENT_ID_KEY = "agent_id"
_STATUS_KEY = "status"

_CHROMA_ERRORS = (ChromaError,)

_POINT_NAMESPACE = POINT_NAMESPACE  # shared Engram-specific namespace from _utils

# When user filters are present, fetch more results than top_k from Chroma to
# compensate for client-side elimination. The final slice is still capped at top_k.
_FILTER_OVERFETCH = 5


def _chroma_id(agent_id: str, memory_id: str) -> str:
    """Deterministic Chroma record ID from (agent_id, memory_id).

    Chroma uses a flat ID space per collection. Without this derivation two
    agents sharing the same memory_id would overwrite each other's records.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{agent_id}\x00{memory_id}"))


def _to_chroma(memory: Memory, vector_size: int) -> tuple[str, list[float], dict[str, Any]]:
    """Convert a Memory to (chroma_id, vector, chroma_metadata)."""
    payload = memory_to_payload(memory)
    if memory.embedding is not None:
        # Store the raw embedding in the payload JSON so that Chroma's float32
        # truncation does not break round-trip fidelity on fetch.
        # Key matches Qdrant's convention (_raw_embedding) for cross-adapter consistency.
        payload[_RAW_EMBEDDING_KEY] = memory.embedding
    metadata: dict[str, Any] = {
        _PAYLOAD_KEY: json.dumps(payload),
        _AGENT_ID_KEY: memory.agent_id,
        _STATUS_KEY: memory.status.value,
        _NO_EMBEDDING_KEY: memory.embedding is None,
    }
    vector = memory.embedding if memory.embedding is not None else [0.0] * vector_size
    return _chroma_id(memory.agent_id, memory.memory_id), vector, metadata


def _from_chroma(raw_embedding: Any, metadata: Mapping[str, Any]) -> Memory:
    """Reconstruct a Memory from Chroma record fields.

    Prefers the raw embedding stored in the payload (exact float64 precision)
    over Chroma's stored vector, which may be float32-truncated. Falls back to
    Chroma's stored vector for records written without the payload trick (e.g.,
    by an older adapter version or an external tool).
    """
    payload = json.loads(metadata[_PAYLOAD_KEY])
    no_embedding = metadata.get(_NO_EMBEDDING_KEY, False)
    if no_embedding:
        embedding = None
    else:
        embedding = payload.get(_RAW_EMBEDDING_KEY)
        if embedding is None and raw_embedding is not None:
            embedding = list(raw_embedding)
    return payload_to_memory(payload, embedding=embedding)


def _agent_where(
    agent_id: str,
    *,
    status: MemoryStatus | None = None,
    exclude_no_embedding: bool = False,
) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = [{_AGENT_ID_KEY: agent_id}]
    if status is not None:
        conditions.append({_STATUS_KEY: status.value})
    if exclude_no_embedding:
        conditions.append({_NO_EMBEDDING_KEY: False})
    return {"$and": conditions} if len(conditions) > 1 else conditions[0]


class ChromaAdapter(AbstractAdapter):
    """Async Chroma adapter wrapping a synchronous Chroma client.

    Parameters
    ----------
    client:
        A configured Chroma client — ``chromadb.EphemeralClient()`` for
        in-memory testing, ``chromadb.PersistentClient(path=...)`` for
        local disk persistence, or ``chromadb.HttpClient(...)`` for a
        remote server. The adapter does not manage the client's lifecycle;
        callers are responsible for closing it after use.
    collection:
        Name of the Chroma collection to use. Created on ``open()`` if it
        does not already exist (with cosine distance configured automatically).
    vector_size:
        Dimensionality of the embedding vectors. Required to construct the
        zero-vector stored in place of null embeddings.
    """

    def __init__(
        self,
        client: ClientAPI,
        collection: str,
        *,
        vector_size: int,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(f"vector_size must be > 0, got {vector_size}")
        self._client = client
        self._collection_name = collection
        self._vector_size = vector_size
        self._col: ChromaCollection | None = None

    @property
    def backend_name(self) -> str:
        return "chroma"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        if self._col is not None:
            return  # idempotent
        col: ChromaCollection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        space = (col.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            raise AdapterError(
                f"collection {self._collection_name!r} uses hnsw:space={space!r}, "
                f"ChromaAdapter requires 'cosine'"
            )
        self._col = col

    async def close(self) -> None:
        self._col = None  # drop Collection reference; client lifecycle is caller's responsibility

    async def __aenter__(self) -> ChromaAdapter:
        await self.open()
        return self

    @property
    def _c(self) -> ChromaCollection:
        if self._col is None:
            raise RuntimeError(
                "ChromaAdapter is not open — use 'async with' or call open() first"
            )
        return self._col

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def store(self, memory: Memory) -> None:
        cid, vector, metadata = _to_chroma(memory, self._vector_size)
        await asyncio.to_thread(
            self._c.upsert,
            ids=[cid],
            embeddings=[vector],  # type: ignore[arg-type]
            metadatas=[metadata],
        )

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def update(self, memory: Memory) -> None:
        cid = _chroma_id(memory.agent_id, memory.memory_id)
        existing = await asyncio.to_thread(self._c.get, ids=[cid], include=["metadatas"])
        if not existing["ids"]:
            raise NotFoundError(
                f"memory {memory.memory_id!r} not found for agent {memory.agent_id!r}"
            )
        stored_agent = (existing["metadatas"] or [{}])[0].get(_AGENT_ID_KEY)
        if stored_agent != memory.agent_id:
            raise NotFoundError(
                f"memory {memory.memory_id!r} not found for agent {memory.agent_id!r}"
            )
        _, vector, metadata = _to_chroma(memory, self._vector_size)
        await asyncio.to_thread(
            self._c.upsert,
            ids=[cid],
            embeddings=[vector],  # type: ignore[arg-type]
            metadatas=[metadata],
        )

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def delete(self, agent_id: str, memory_id: str) -> bool:
        cid = _chroma_id(agent_id, memory_id)
        existing = await asyncio.to_thread(self._c.get, ids=[cid], include=["metadatas"])
        if not existing["ids"]:
            return False
        stored_agent = (existing["metadatas"] or [{}])[0].get(_AGENT_ID_KEY)
        if stored_agent != agent_id:
            return False
        await asyncio.to_thread(self._c.delete, ids=[cid])
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        cid = _chroma_id(agent_id, memory_id)
        result = await asyncio.to_thread(
            self._c.get, ids=[cid], include=["embeddings", "metadatas"]
        )
        if not result["ids"]:
            return None
        meta = (result["metadatas"] or [{}])[0]
        if meta.get(_AGENT_ID_KEY) != agent_id:
            return None
        emb = result["embeddings"][0] if result["embeddings"] is not None else None
        return _from_chroma(emb, meta)

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if top_k == 0:
            return []

        where = _agent_where(agent_id, exclude_no_embedding=True)
        # Overfetch when user filters are present: client-side post-filtering on
        # nested JSON can eliminate results, leaving fewer than top_k. Fetching
        # more upfront compensates for most practical filter selectivities.
        fetch_k = top_k * _FILTER_OVERFETCH if filters else top_k
        result = await asyncio.to_thread(
            self._c.query,
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=fetch_k,
            where=where,
            include=["embeddings", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        if not ids:
            return []

        # Chroma stubs type distances/embeddings as list[...] | None; we know
        # they are present because we requested them and have non-empty ids.
        distances = result["distances"][0]  # type: ignore[index]
        raw_embeddings = result["embeddings"][0] if result["embeddings"] is not None else [None] * len(ids)
        metadatas = result["metadatas"][0]  # type: ignore[index]

        output: list[SearchResult] = []
        rank = 1
        for emb, distance, meta in zip(raw_embeddings, distances, metadatas, strict=True):
            score = max(0.0, 1.0 - distance)
            if score_threshold is not None and score < score_threshold:
                continue
            if filters:
                # User metadata is nested in the JSON payload; Chroma's where
                # clause cannot filter on nested JSON so we post-filter here.
                payload = json.loads(meta[_PAYLOAD_KEY])
                user_meta = payload.get("metadata", {})
                if not all(user_meta.get(k) == v for k, v in filters.items()):
                    continue
            output.append(SearchResult(memory=_from_chroma(emb, meta), score=score, rank=rank))
            rank += 1
            if len(output) >= top_k:
                break

        return output

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        where = _agent_where(agent_id, status=status)
        result = await asyncio.to_thread(
            self._c.get,
            where=where,
            include=["embeddings", "metadatas"],
        )

        ids = result["ids"]
        if not ids:
            return []

        raw_embeddings = result["embeddings"]
        metadatas = result["metadatas"] or []

        memories = [
            _from_chroma(
                raw_embeddings[i] if raw_embeddings is not None else None,
                meta,
            )
            for i, meta in enumerate(metadatas)
        ]
        memories.sort(key=lambda m: m.created_at)
        sliced = memories[offset:] if offset else memories
        return sliced[:limit] if limit is not None else sliced

    # ------------------------------------------------------------------
    # Bulk overrides
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def store_batch(self, memories: list[Memory]) -> None:
        if not memories:
            return
        records = [_to_chroma(m, self._vector_size) for m in memories]
        await asyncio.to_thread(
            self._c.upsert,
            ids=[r[0] for r in records],
            embeddings=[r[1] for r in records],  # type: ignore[arg-type]
            metadatas=[r[2] for r in records],
        )

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        cids = [_chroma_id(agent_id, mid) for mid in memory_ids]
        result = await asyncio.to_thread(
            self._c.get, ids=cids, include=["embeddings", "metadatas"]
        )
        if not result["ids"]:
            return {}
        raw_embeddings = result["embeddings"]
        metadatas = result["metadatas"] or []
        no_emb_sentinel = [None] * len(metadatas)
        emb_list = raw_embeddings if raw_embeddings is not None else no_emb_sentinel
        output: dict[str, Memory] = {}
        for emb, meta in zip(emb_list, metadatas, strict=True):
            if meta.get(_AGENT_ID_KEY) != agent_id:
                continue
            memory = _from_chroma(emb, meta)
            output[memory.memory_id] = memory
        return output

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        cids = [_chroma_id(agent_id, mid) for mid in memory_ids]
        result = await asyncio.to_thread(self._c.get, ids=cids, include=["metadatas"])
        owned = [
            cid
            for cid, meta in zip(result["ids"], result["metadatas"] or [], strict=True)
            if meta.get(_AGENT_ID_KEY) == agent_id
        ]
        if not owned:
            return 0
        await asyncio.to_thread(self._c.delete, ids=owned)
        return len(owned)

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        where = _agent_where(agent_id, status=status)
        result = await asyncio.to_thread(self._c.get, where=where, include=[])
        return len(result["ids"])

    @map_adapter_errors(error=_CHROMA_ERRORS)
    async def exists(self, agent_id: str, memory_id: str) -> bool:
        result = await asyncio.to_thread(
            self._c.get, ids=[_chroma_id(agent_id, memory_id)], include=[]
        )
        return len(result["ids"]) > 0
