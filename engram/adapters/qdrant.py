"""Qdrant adapter — async, tenant-isolated via payload filters.

Requires the optional 'qdrant' extra::

    pip install "engram[qdrant]"

Call ``open()`` (or use ``async with``) before any other method.

Vector size is fixed at construction time; all embeddings must match it.
If the collection already exists, ``open()`` validates that the stored
vector_size and distance match the adapter's configuration, raising
``AdapterError`` on mismatch to catch config drift at startup.

Null embeddings: Qdrant requires a vector for every point. Memories stored
without an embedding receive a zero-vector plus a ``_no_embedding: True``
sentinel in the payload. They are excluded from search results and the
embedding is returned as None on fetch.

Tenant isolation: every point ID is derived from ``(agent_id, memory_id)``
via UUID5, so two agents that happen to share a ``memory_id`` do not
overwrite each other. Reads/writes additionally filter on ``agent_id`` in
the payload for defense-in-depth.

Metadata filters (search only): pass a flat ``{key: value}`` dict. Keys are
automatically prefixed with ``metadata.`` because user metadata is stored as
a nested payload field by ``memory_to_payload``.

list_all note: Qdrant does not support server-side ordering by payload field,
so ``list_all`` fetches all matching records, sorts client-side, then slices.
This is O(n) in the number of agent records — fine for development and
moderate workloads, but not suitable for agents with millions of memories.
A future sub-step will introduce a stored ``created_at_ts`` integer index to
enable server-side ordered scrolling.

Score range: scores are clamped to [0.0, 1.0] regardless of distance metric.
For Cosine (default), scores are semantically meaningful in that range.
For Dot/Euclid/Manhattan, raw Qdrant scores are clamped and may not be
comparable across queries.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointIdsList,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)

from engram.adapters._utils import (
    POINT_NAMESPACE,
    map_adapter_errors,
    memory_to_payload,
    payload_to_memory,
)
from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus, ResolutionStatus
from engram.core.exceptions import AdapterError, NotFoundError
from engram.core.models import ConflictRecord, Memory, SearchResult

__all__ = ["QdrantAdapter"]

_NO_EMBEDDING_KEY = "_no_embedding"
_RAW_EMBEDDING_KEY = "_raw_embedding"
_AGENT_ID_KEY = "agent_id"

# Sidecar collection for ConflictRecords. Uses a 1-dim Cosine collection with
# a zero-vector per point — Qdrant requires a vector for every point, but
# conflicts are retrieved purely by payload filters (agent_id, resolution_status).
_CONFLICT_COLLECTION_SUFFIX = "_conflicts"
_CONFLICT_VECTOR_SIZE = 1
_CONFLICT_DUMMY_VECTOR = [0.0]

_QDRANT_ERRORS = (UnexpectedResponse, ResponseHandlingException)

_DISTANCE_MAP: dict[str, Distance] = {
    "Cosine": Distance.COSINE,
    "Euclid": Distance.EUCLID,
    "Dot": Distance.DOT,
    "Manhattan": Distance.MANHATTAN,
}

_POINT_NAMESPACE = POINT_NAMESPACE  # shared Engram-specific namespace from _utils


def _point_id(agent_id: str, memory_id: str) -> str:
    """Deterministic Qdrant point ID from (agent_id, memory_id).

    Qdrant uses a flat global ID space per collection. Without this derivation,
    two agents sharing the same memory_id would silently overwrite each other's
    records, violating the multi-tenant contract.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{agent_id}\x00{memory_id}"))


def _payload(point: Record | ScoredPoint) -> dict[str, Any]:
    """Return the point's payload dict, safe against None."""
    return point.payload or {}


def _to_point(memory: Memory, vector_size: int) -> PointStruct:
    payload = memory_to_payload(memory)
    if memory.embedding is not None:
        # Store the raw embedding in the payload so Cosine-normalization by
        # Qdrant does not break round-trip fidelity on fetch.
        vector = memory.embedding
        payload[_RAW_EMBEDDING_KEY] = memory.embedding
    else:
        vector = [0.0] * vector_size
        payload[_NO_EMBEDDING_KEY] = True
    return PointStruct(
        id=_point_id(memory.agent_id, memory.memory_id),
        vector=vector,
        payload=payload,
    )


def _from_point(point: Record | ScoredPoint) -> Memory:
    p = dict(_payload(point))
    has_no_embedding = p.pop(_NO_EMBEDDING_KEY, False)
    raw_embedding: list[float] | None = p.pop(_RAW_EMBEDDING_KEY, None)
    embedding = None if has_no_embedding else raw_embedding
    return payload_to_memory(p, embedding=embedding)


def _agent_filter(
    agent_id: str,
    *,
    status: MemoryStatus | None = None,
    exclude_no_embedding: bool = False,
) -> Filter:
    must: list[Any] = [
        FieldCondition(key=_AGENT_ID_KEY, match=MatchValue(value=agent_id)),
    ]
    if status is not None:
        must.append(FieldCondition(key="status", match=MatchValue(value=status.value)))

    must_not: list[Any] = []
    if exclude_no_embedding:
        must_not.append(FieldCondition(key=_NO_EMBEDDING_KEY, match=MatchValue(value=True)))

    return Filter(must=must, must_not=must_not or None)


class QdrantAdapter(AbstractAdapter):
    """Async Qdrant adapter backed by AsyncQdrantClient.

    Parameters
    ----------
    location:
        Qdrant URL (``"http://localhost:6333"``) or ``":memory:"`` for an
        in-process instance suitable for tests.
    collection:
        Name of the Qdrant collection to use. Created on ``open()`` if it
        does not already exist.
    vector_size:
        Dimensionality of the embedding vectors. Every embedding stored via
        this adapter must have exactly this many dimensions.
    distance:
        Distance metric — one of ``"Cosine"`` (default), ``"Euclid"``,
        ``"Dot"``, ``"Manhattan"``.
    api_key:
        Optional API key for Qdrant Cloud.
    """

    def __init__(
        self,
        location: str,
        collection: str,
        *,
        vector_size: int,
        distance: str = "Cosine",
        api_key: str | None = None,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(f"vector_size must be > 0, got {vector_size}")
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"distance must be one of {list(_DISTANCE_MAP)!r}, got {distance!r}")
        self._location = location
        self._collection = collection
        self._vector_size = vector_size
        self._distance = _DISTANCE_MAP[distance]
        self._api_key = api_key
        self._client: AsyncQdrantClient | None = None

    @property
    def backend_name(self) -> str:
        return "qdrant"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        if self._client is not None:
            return  # idempotent
        kwargs: dict[str, Any] = {"location": self._location}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        self._client = AsyncQdrantClient(**kwargs)
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, VectorParams):
                if vectors_config.size != self._vector_size:
                    raise AdapterError(
                        f"collection {self._collection!r} has vector_size={vectors_config.size}, "
                        f"adapter configured with vector_size={self._vector_size}"
                    )
                if vectors_config.distance != self._distance:
                    raise AdapterError(
                        f"collection {self._collection!r} has distance={vectors_config.distance}, "
                        f"adapter configured with distance={self._distance}"
                    )
        else:
            await self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=self._distance),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> QdrantAdapter:
        await self.open()
        return self

    @property
    def _c(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantAdapter is not open — use 'async with' or call open() first")
        return self._client

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def store(self, memory: Memory) -> None:
        await self._c.upsert(
            self._collection,
            points=[_to_point(memory, self._vector_size)],
            wait=True,
        )

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def update(self, memory: Memory) -> None:
        pid = _point_id(memory.agent_id, memory.memory_id)
        existing = await self._c.retrieve(
            self._collection,
            ids=[pid],
            with_vectors=False,
            with_payload=[_AGENT_ID_KEY],
        )
        if not existing or _payload(existing[0]).get(_AGENT_ID_KEY) != memory.agent_id:
            raise NotFoundError(
                f"memory {memory.memory_id!r} not found for agent {memory.agent_id!r}"
            )
        await self._c.upsert(
            self._collection,
            points=[_to_point(memory, self._vector_size)],
            wait=True,
        )

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def delete(self, agent_id: str, memory_id: str) -> bool:
        pid = _point_id(agent_id, memory_id)
        existing = await self._c.retrieve(
            self._collection,
            ids=[pid],
            with_vectors=False,
            with_payload=[_AGENT_ID_KEY],
        )
        if not existing or _payload(existing[0]).get(_AGENT_ID_KEY) != agent_id:
            return False
        await self._c.delete(
            self._collection,
            points_selector=PointIdsList(points=[pid]),
            wait=True,
        )
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        results = await self._c.retrieve(
            self._collection,
            ids=[_point_id(agent_id, memory_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        if _payload(results[0]).get(_AGENT_ID_KEY) != agent_id:
            return None
        return _from_point(results[0])

    @map_adapter_errors(error=_QDRANT_ERRORS)
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

        query_filter = _agent_filter(agent_id, exclude_no_embedding=True)
        if filters:
            # User provides flat {key: value} metadata filters. In the Qdrant
            # payload, user metadata is stored under the nested "metadata" key,
            # so filter paths must be prefixed with "metadata.".
            extra = [
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(
                must=list(query_filter.must or []) + extra,  # type: ignore[arg-type]
                must_not=query_filter.must_not,
            )

        results = await self._c.query_points(
            self._collection,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
        )
        return [
            SearchResult(memory=_from_point(p), score=min(1.0, max(0.0, p.score)), rank=rank)
            for rank, p in enumerate(results.points, start=1)
        ]

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        scroll_filter = _agent_filter(agent_id, status=status)
        all_records: list[Record] = []
        next_offset: Any = None

        while True:
            batch, next_offset = await self._c.scroll(
                self._collection,
                scroll_filter=scroll_filter,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            all_records.extend(batch)
            if next_offset is None:
                break

        memories = [_from_point(r) for r in all_records]
        memories.sort(key=lambda m: m.created_at)
        sliced = memories[offset:] if offset else memories
        return sliced[:limit] if limit is not None else sliced

    # ------------------------------------------------------------------
    # Bulk overrides
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def store_batch(self, memories: list[Memory]) -> None:
        if not memories:
            return
        await self._c.upsert(
            self._collection,
            points=[_to_point(m, self._vector_size) for m in memories],
            wait=True,
        )

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        point_ids = [_point_id(agent_id, mid) for mid in memory_ids]
        results, _ = await self._c.scroll(
            self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key=_AGENT_ID_KEY, match=MatchValue(value=agent_id)),
                    HasIdCondition(has_id=cast(list[str | int | uuid.UUID], point_ids)),
                ]
            ),
            limit=len(memory_ids),
            with_payload=True,
            with_vectors=True,
        )
        return {_payload(r)["memory_id"]: _from_point(r) for r in results}

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        point_ids = [_point_id(agent_id, mid) for mid in memory_ids]
        results, _ = await self._c.scroll(
            self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key=_AGENT_ID_KEY, match=MatchValue(value=agent_id)),
                    HasIdCondition(has_id=cast(list[str | int | uuid.UUID], point_ids)),
                ]
            ),
            limit=len(memory_ids),
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            return 0
        existing_pids = [r.id for r in results]
        await self._c.delete(
            self._collection,
            points_selector=PointIdsList(points=existing_pids),
            wait=True,
        )
        return len(existing_pids)

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        result = await self._c.count(
            self._collection,
            count_filter=_agent_filter(agent_id, status=status),
            exact=True,
        )
        return int(result.count)

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def exists(self, agent_id: str, memory_id: str) -> bool:
        results = await self._c.retrieve(
            self._collection,
            ids=[_point_id(agent_id, memory_id)],
            with_payload=False,
            with_vectors=False,
        )
        return bool(results)

    # ------------------------------------------------------------------
    # Conflict storage — sidecar collection "{collection}_conflicts"
    # ------------------------------------------------------------------

    @property
    def _conflicts_collection(self) -> str:
        return f"{self._collection}{_CONFLICT_COLLECTION_SUFFIX}"

    async def _ensure_conflicts_collection(self) -> None:
        """Lazily create the conflicts sidecar collection if it does not exist."""
        name = self._conflicts_collection
        if not await self._c.collection_exists(name):
            await self._c.create_collection(
                name,
                vectors_config=VectorParams(
                    size=_CONFLICT_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def _conflict_point_id(self, agent_id: str, conflict_id: str) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"{agent_id}\x00{conflict_id}"))

    def _conflict_filter(
        self,
        agent_id: str,
        *,
        status: ResolutionStatus | None = None,
    ) -> Filter:
        must: list[Any] = [
            FieldCondition(key=_AGENT_ID_KEY, match=MatchValue(value=agent_id)),
        ]
        if status is not None:
            must.append(
                FieldCondition(
                    key="resolution_status",
                    match=MatchValue(value=status.value),
                )
            )
        return Filter(must=must)

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def store_conflict(self, conflict: ConflictRecord) -> None:
        await self._ensure_conflicts_collection()
        payload = conflict.model_dump(mode="json")
        pid = self._conflict_point_id(conflict.agent_id, conflict.conflict_id)
        await self._c.upsert(
            self._conflicts_collection,
            points=[
                PointStruct(
                    id=pid,
                    vector=_CONFLICT_DUMMY_VECTOR,
                    payload=payload,
                )
            ],
            wait=True,
        )

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def fetch_conflict(self, agent_id: str, conflict_id: str) -> ConflictRecord | None:
        name = self._conflicts_collection
        if not await self._c.collection_exists(name):
            return None
        pid = self._conflict_point_id(agent_id, conflict_id)
        results = await self._c.retrieve(
            name,
            ids=[pid],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        p = _payload(results[0])
        if p.get(_AGENT_ID_KEY) != agent_id:
            return None
        return ConflictRecord.model_validate(p)

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def list_conflicts(
        self,
        agent_id: str,
        *,
        status: ResolutionStatus | None = None,
    ) -> list[ConflictRecord]:
        name = self._conflicts_collection
        if not await self._c.collection_exists(name):
            return []

        scroll_filter = self._conflict_filter(agent_id, status=status)
        all_records: list[Record] = []
        next_offset: Any = None

        while True:
            batch, next_offset = await self._c.scroll(
                name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_records.extend(batch)
            if next_offset is None:
                break

        records = [ConflictRecord.model_validate(_payload(r)) for r in all_records]
        records.sort(key=lambda r: r.detected_at)
        return records

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def update_conflict(self, conflict: ConflictRecord) -> None:
        name = self._conflicts_collection
        pid = self._conflict_point_id(conflict.agent_id, conflict.conflict_id)
        if await self._c.collection_exists(name):
            existing = await self._c.retrieve(
                name,
                ids=[pid],
                with_payload=[_AGENT_ID_KEY],
                with_vectors=False,
            )
        else:
            existing = []
        if not existing or _payload(existing[0]).get(_AGENT_ID_KEY) != conflict.agent_id:
            raise NotFoundError(
                f"conflict {conflict.conflict_id!r} not found for agent {conflict.agent_id!r}"
            )
        payload = conflict.model_dump(mode="json")
        await self._c.upsert(
            name,
            points=[
                PointStruct(
                    id=pid,
                    vector=_CONFLICT_DUMMY_VECTOR,
                    payload=payload,
                )
            ],
            wait=True,
        )

    @map_adapter_errors(error=_QDRANT_ERRORS)
    async def delete_conflict(self, agent_id: str, conflict_id: str) -> bool:
        name = self._conflicts_collection
        if not await self._c.collection_exists(name):
            return False
        pid = self._conflict_point_id(agent_id, conflict_id)
        existing = await self._c.retrieve(
            name,
            ids=[pid],
            with_payload=[_AGENT_ID_KEY],
            with_vectors=False,
        )
        if not existing or _payload(existing[0]).get(_AGENT_ID_KEY) != agent_id:
            return False
        await self._c.delete(
            name,
            points_selector=PointIdsList(points=[pid]),
            wait=True,
        )
        return True
