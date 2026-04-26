"""Engram — the main entry point.

Wrap any AbstractAdapter with Engram to gain contradiction detection,
health scoring, and consolidation on top of your vector backend:

    adapter = QdrantAdapter(url="localhost:6333", collection="memories")
    async with Engram(adapter) as eng:
        await eng.store(memory)
        results = await eng.search("agent-1", query_embedding)

What's implemented in each step:
  1.8 — constructor, adapter delegation, NotImplementedError stubs
  4   — health() scoring
  5   — contradiction detection wired into store() and search()
  6   — consolidate() planning and execution
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus
from engram.core.models import Memory, SearchResult

if TYPE_CHECKING:
    from engram.core.models import ConsolidationPlan, HealthScore

__all__ = ["Engram"]

logger = logging.getLogger(__name__)


class Engram:
    """Memory layer facade — wraps an adapter and coordinates reliability features.

    Instantiate with any AbstractAdapter subclass. Use as an async context
    manager to ensure the adapter's connections are closed on exit.
    """

    def __init__(self, adapter: AbstractAdapter) -> None:
        if not isinstance(adapter, AbstractAdapter):
            raise TypeError(
                f"adapter must be an AbstractAdapter subclass, got {type(adapter).__name__!r}"
            )
        self._adapter = adapter

    @property
    def adapter(self) -> AbstractAdapter:
        """The underlying storage adapter."""
        return self._adapter

    @property
    def backend_name(self) -> str:
        """Short identifier for the underlying backend (e.g. 'qdrant', 'chroma')."""
        return self._adapter.backend_name

    def __repr__(self) -> str:
        return f"Engram(adapter={self._adapter.backend_name!r})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release backend connections and free resources.

        Idempotent — delegates directly to the adapter's close().
        """
        await self._adapter.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def store(self, memory: Memory) -> None:
        """Store a memory.

        Step 1.8: delegates to adapter.
        Step 5:   also runs contradiction detection against existing memories.
        """
        await self._adapter.store(memory)

    async def update(self, memory: Memory) -> None:
        """Fully replace an existing memory record.

        Raises NotFoundError if the record does not exist.
        """
        await self._adapter.update(memory)

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted, False if not found."""
        return await self._adapter.delete(agent_id, memory_id)

    async def store_batch(self, memories: list[Memory]) -> None:
        """Upsert multiple memories."""
        await self._adapter.store_batch(memories)

    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        """Delete multiple memories. Returns the count actually deleted."""
        return await self._adapter.delete_batch(agent_id, memory_ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        """Return the memory, or None if it does not exist."""
        return await self._adapter.fetch(agent_id, memory_id)

    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        """Fetch multiple memories by ID. Absent IDs are omitted from the result."""
        return await self._adapter.fetch_batch(agent_id, memory_ids)

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Step 1.8: delegates to adapter (conflict fields remain at defaults).
        Step 5:   enriches results with contradiction detection output.
        """
        return await self._adapter.search(
            agent_id,
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Return memories for an agent, optionally filtered and paginated."""
        return await self._adapter.list_all(
            agent_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        """Return the count of memories for an agent, optionally filtered by status."""
        return await self._adapter.count(agent_id, status=status)

    async def exists(self, agent_id: str, memory_id: str) -> bool:
        """Return True iff a memory with this ID exists for the agent."""
        return await self._adapter.exists(agent_id, memory_id)

    # ------------------------------------------------------------------
    # Future steps (stubs)
    # ------------------------------------------------------------------

    async def health(self, agent_id: str) -> HealthScore:
        """Compute a health snapshot for an agent's memory collection.

        Not yet implemented — arrives in Step 4.
        """
        raise NotImplementedError("health() is not yet implemented (Step 4)")

    async def consolidate(self, agent_id: str) -> ConsolidationPlan:
        """Plan and execute memory consolidation for an agent.

        Not yet implemented — arrives in Step 6.
        """
        raise NotImplementedError("consolidate() is not yet implemented (Step 6)")

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Engram:
        await self._adapter.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self._adapter.__aexit__(exc_type, exc_val, exc_tb)
