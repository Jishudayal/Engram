"""NaiveQdrantAdapter — raw Qdrant wrapper without Memnotary's data model.

Represents a realistic "before Memnotary" baseline: a developer who reached
for Qdrant directly and stored text + embedding, but none of the reliability
machinery that Memnotary's AbstractAdapter preserves.

What it stores (Qdrant payload):
    memory_id, agent_id, text

What it silently drops on every write:
    status, supersedes, superseded_by
    importance, access_count, last_accessed
    expires_at, provenance, updated_at, metadata, memory_type

Behavioural gaps vs compliant adapters:
    update()     — no-op; field mutations are never persisted
    list_all()   — returns ALL records; status filter is silently ignored
    fetch()      — reconstructed Memory has only the three stored fields;
                   all other fields revert to model defaults
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from memnotary.adapters.base import AbstractAdapter
from memnotary.core.constants import MemoryStatus
from memnotary.core.exceptions import NotFoundError
from memnotary.core.models import Memory, SearchResult

_COLLECTION = "naive_memoryeval"
_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _pid(agent_id: str, memory_id: str) -> str:
    return str(uuid.uuid5(_NS, f"{agent_id}\x00{memory_id}"))


class NaiveQdrantAdapter(AbstractAdapter):
    """Minimal Qdrant wrapper: stores only text + embedding, no data model."""

    def __init__(self, vector_size: int = 16) -> None:
        self._vector_size = vector_size
        self._client: AsyncQdrantClient | None = None

    @property
    def backend_name(self) -> str:
        return "naive-qdrant"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncQdrantClient(location=":memory:")
        await self._client.create_collection(
            _COLLECTION,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def _c(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("call open() first")
        return self._client

    # ------------------------------------------------------------------
    # Write — only the three core fields reach Qdrant
    # ------------------------------------------------------------------

    async def store(self, memory: Memory) -> None:
        vector = memory.embedding if memory.embedding is not None else [0.0] * self._vector_size
        await self._c.upsert(
            _COLLECTION,
            points=[
                PointStruct(
                    id=_pid(memory.agent_id, memory.memory_id),
                    vector=vector,
                    payload={
                        "memory_id": memory.memory_id,
                        "agent_id": memory.agent_id,
                        "text": memory.text,
                    },
                )
            ],
        )

    async def update(self, memory: Memory) -> None:
        # Honour the contract: raise if absent. Re-upsert only the three core
        # fields — all reliability fields (status, importance, supersedes, …)
        # are dropped, which is the intended naive-adapter gap.
        pid = _pid(memory.agent_id, memory.memory_id)
        existing = await self._c.retrieve(_COLLECTION, ids=[pid], with_payload=False)
        if not existing:
            raise NotFoundError(memory.agent_id, memory.memory_id)
        await self.store(memory)

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        pid = _pid(agent_id, memory_id)
        existing = await self._c.retrieve(_COLLECTION, ids=[pid], with_payload=False)
        if not existing:
            return False
        await self._c.delete(_COLLECTION, points_selector=[pid])
        return True

    # ------------------------------------------------------------------
    # Read — reconstructed Memory carries only the three stored fields
    # ------------------------------------------------------------------

    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        results = await self._c.retrieve(
            _COLLECTION,
            ids=[_pid(agent_id, memory_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        p = results[0]
        return Memory(
            memory_id=p.payload["memory_id"],
            agent_id=p.payload["agent_id"],
            text=p.payload["text"],
            embedding=list(p.vector) if p.vector else None,
        )

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        results = await self._c.query_points(
            _COLLECTION,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
        )
        return [
            SearchResult(
                memory=Memory(
                    memory_id=p.payload["memory_id"],
                    agent_id=p.payload["agent_id"],
                    text=p.payload["text"],
                    embedding=list(p.vector) if p.vector else None,
                ),
                score=min(1.0, max(0.0, p.score)),
                rank=rank,
            )
            for rank, p in enumerate(results.points, start=1)
        ]

    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,  # silently ignored
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        records, _ = await self._c.scroll(
            _COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
            ),
            limit=limit or 1000,
            with_payload=True,
            with_vectors=True,
        )
        memories = [
            Memory(
                memory_id=r.payload["memory_id"],
                agent_id=r.payload["agent_id"],
                text=r.payload["text"],
                embedding=list(r.vector) if r.vector else None,
            )
            for r in records
        ]
        return memories[offset:]
