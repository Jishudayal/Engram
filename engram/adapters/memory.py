"""InMemory adapter — pure-Python, no external dependencies.

Stores Memory objects in nested dicts keyed by (agent_id, memory_id).
Search runs cosine similarity over stored embeddings using numpy (a core
dependency). Use this adapter for unit tests and local development; switch
to a production adapter for staging and production workloads.

Isolation semantics: every store writes a deep copy; every read returns a
deep copy. Mutations to objects before or after calling store/fetch/list_all
do not affect the stored state. This matches the behaviour of serialization-
based adapters (Qdrant, Chroma, pgvector) and makes InMemoryAdapter a valid
reference implementation for the contract test suite.

Concurrency: not safe for concurrent mutation across coroutines sharing the
same instance. Suitable for single-event-loop unit test workloads.

Metadata filters (search only): pass a flat {key: value} dict; a memory is
included only if every key in the filter dict is present in memory.metadata
with the matching value. Pass None for no filtering.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus
from engram.core.exceptions import NotFoundError
from engram.core.models import Memory, SearchResult

__all__ = ["InMemoryAdapter"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity clamped to [0.0, 1.0]; returns 0.0 for zero vectors.

    Negative cosine values (opposite-direction vectors) are clamped to 0.0
    rather than returned raw, satisfying the SearchResult.score [0, 1] contract.
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, float(np.dot(va, vb) / (norm_a * norm_b)))


def _matches_filters(memory: Memory, filters: dict[str, Any]) -> bool:
    return all(memory.metadata.get(k) == v for k, v in filters.items())


class InMemoryAdapter(AbstractAdapter):
    """In-memory adapter backed by Python dicts.

    Suitable for unit tests and local prototyping. Not suitable for
    concurrent workloads or persistence across restarts.
    """

    def __init__(self) -> None:
        # _store[agent_id][memory_id] = Memory (deep-copied on write)
        self._store: dict[str, dict[str, Memory]] = {}

    @property
    def backend_name(self) -> str:
        return "memory"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store(self, memory: Memory) -> None:
        self._store.setdefault(memory.agent_id, {})[memory.memory_id] = memory.model_copy(
            deep=True
        )

    async def update(self, memory: Memory) -> None:
        bucket = self._store.get(memory.agent_id, {})
        if memory.memory_id not in bucket:
            raise NotFoundError(
                f"memory {memory.memory_id!r} not found for agent {memory.agent_id!r}"
            )
        bucket[memory.memory_id] = memory.model_copy(deep=True)

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        bucket = self._store.get(agent_id, {})
        if memory_id not in bucket:
            return False
        del bucket[memory_id]
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        memory = self._store.get(agent_id, {}).get(memory_id)
        return memory.model_copy(deep=True) if memory is not None else None

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        scored: list[tuple[float, Memory]] = []
        for memory in self._store.get(agent_id, {}).values():
            if memory.embedding is None:
                continue
            if filters is not None and not _matches_filters(memory, filters):
                continue
            score = _cosine(query_embedding, memory.embedding)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append((score, memory))

        scored.sort(key=lambda t: t[0], reverse=True)
        # max(0, top_k) guards against negative top_k silently dropping results
        return [
            SearchResult(memory=mem.model_copy(deep=True), score=min(1.0, score), rank=rank)
            for rank, (score, mem) in enumerate(scored[: max(0, top_k)], start=1)
        ]

    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        memories = [m.model_copy(deep=True) for m in self._store.get(agent_id, {}).values()]
        if status is not None:
            memories = [m for m in memories if m.status == status]
        memories.sort(key=lambda m: m.created_at)
        sliced = memories[offset:] if offset else memories
        return sliced[:limit] if limit is not None else sliced

    async def close(self) -> None:
        pass  # nothing to release

    # ------------------------------------------------------------------
    # Bulk overrides — single dict lookup per agent instead of N lookups
    # ------------------------------------------------------------------

    async def store_batch(self, memories: list[Memory]) -> None:
        for memory in memories:
            self._store.setdefault(memory.agent_id, {})[memory.memory_id] = memory.model_copy(
                deep=True
            )

    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        bucket = self._store.get(agent_id, {})
        return {mid: bucket[mid].model_copy(deep=True) for mid in memory_ids if mid in bucket}

    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        bucket = self._store.get(agent_id, {})
        count = 0
        for mid in memory_ids:
            if mid in bucket:
                del bucket[mid]
                count += 1
        return count

    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        bucket = self._store.get(agent_id, {})
        if status is None:
            return len(bucket)
        return sum(1 for m in bucket.values() if m.status == status)

    async def exists(self, agent_id: str, memory_id: str) -> bool:
        return memory_id in self._store.get(agent_id, {})
