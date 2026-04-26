"""AbstractAdapter — the contract all storage backends must satisfy.

Every Engram backend (Qdrant, Chroma, Weaviate, pgvector, in-memory) is
a concrete subclass of AbstractAdapter. The interface is intentionally async:
all production backends have async clients, and synchronous adapters can
trivially wrap their operations in async methods.

Minimum to implement (8 abstract methods + 1 property):
  backend_name  — stable lowercase identifier ('qdrant', 'chroma', …)
  store()       — upsert; overwrites if memory_id already exists
  update()      — full replacement; raises NotFoundError if absent
  delete()      — returns True if deleted, False if not found
  fetch()       — returns Memory or None; never raises on absent
  search()      — vector similarity search, results in rank order
  list_all()    — paginated full scan ordered by created_at asc
  close()       — idempotent teardown

Concrete helpers (override for better performance, but not required):
  store_batch()  — upserts a list; default loops over store()
  fetch_batch()  — returns {memory_id: Memory}; default loops over fetch()
  delete_batch() — deletes a list; returns count deleted; default loops
  count()        — returns record count; default len(list_all(...))
  exists()       — returns bool; default fetch(...) is not None

Semantic notes:
  - update() is scoped to memory.agent_id: the same memory_id under a
    different agent_id is treated as absent and raises NotFoundError.
  - search() applies score_threshold first, then returns up to top_k results.
    If fewer matches meet the threshold, fewer than top_k results are returned.
  - adapters return SearchResult with conflict fields at their defaults;
    the Engram wrapper populates conflict information after retrieval.
  - list_all() results are ordered by created_at ascending. This stable
    ordering makes offset-based pagination consistent across calls.
  - store_batch() default is NOT all-or-nothing; override for atomic bulk
    semantics where the backend supports it (e.g. Qdrant upsert, pgvector
    multi-row INSERT).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from engram.core.constants import MemoryStatus
from engram.core.models import Memory, SearchResult

__all__ = ["AbstractAdapter"]


class AbstractAdapter(ABC):
    """Contract all storage backends must satisfy.

    Implement the 8 abstract methods listed above. Concrete helpers
    (store_batch, fetch_batch, delete_batch, count, exists) work out of the
    box via default implementations and can be overridden for native bulk
    performance.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short, lowercase identifier for this backend (e.g. 'qdrant', 'chroma').

        Used in log messages, error strings, and health reports. Must be
        stable across restarts — do not return a runtime-generated value.
        """

    # ------------------------------------------------------------------
    # Write operations (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    async def store(self, memory: Memory) -> None:
        """Persist a memory (upsert semantics).

        If a memory with the same memory_id already exists it is overwritten.
        Raises AdapterError on backend failure.
        """

    @abstractmethod
    async def update(self, memory: Memory) -> None:
        """Fully replace an existing memory record.

        Raises NotFoundError if no record exists for memory.agent_id /
        memory.memory_id. Scoped to memory.agent_id: the same memory_id
        under a different agent_id is treated as absent.
        Raises AdapterError on other backend failures.
        """

    @abstractmethod
    async def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory.

        Returns True if the record was found and deleted, False if it was
        not found. Raises AdapterError on backend failure.
        """

    # ------------------------------------------------------------------
    # Read operations (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        """Return the memory, or None if it does not exist.

        The agent_id parameter enforces tenant isolation: a memory that
        exists for a different agent is treated as absent.
        Raises AdapterError on backend failure.
        """

    @abstractmethod
    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Vector similarity search over an agent's memories.

        Returns up to top_k results ordered by descending similarity (rank 1
        is the best match). Each SearchResult.score is a cosine similarity
        in [0, 1].

        score_threshold: exclude results below this score before ranking.
            After filtering, up to top_k results are returned. If fewer than
            top_k matches meet the threshold, fewer results are returned.
        filters: backend-specific metadata filter; intentionally untyped
            because each backend has its own filter DSL (see adapter docs).
            Pass None for no metadata filtering.

        Conflict fields on SearchResult (conflict_flag, conflicts_with, etc.)
        are left at their defaults. The Engram wrapper populates them.
        Raises AdapterError on backend failure.
        """

    @abstractmethod
    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Return memories for an agent, optionally filtered and paginated.

        status: if provided, return only memories in this lifecycle state.
        limit:  maximum records to return; None means no cap (use with care
                on large collections — prefer a reasonable bound in production).
        offset: number of records to skip (for pagination).

        Results are ordered by created_at ascending (oldest first). This
        stable ordering makes offset-based pagination consistent: the same
        offset always refers to the same position in the full sorted set,
        even across separate calls.
        Raises AdapterError on backend failure.
        """

    # ------------------------------------------------------------------
    # Lifecycle (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    async def close(self) -> None:
        """Release backend connections and free resources.

        Must be idempotent — calling close() multiple times must not raise.
        Called automatically by __aexit__ when used as an async context manager.
        """

    # ------------------------------------------------------------------
    # Bulk helpers (concrete defaults — override for native batch performance)
    # ------------------------------------------------------------------

    async def store_batch(self, memories: list[Memory]) -> None:
        """Upsert multiple memories.

        Default: loops over store(). Override with the backend's native bulk
        write (e.g. Qdrant upsert, pgvector multi-row INSERT) for significant
        throughput gains. The default is NOT all-or-nothing; override if you
        need atomic bulk semantics.
        Raises AdapterError on backend failure.
        """
        for memory in memories:
            await self.store(memory)

    async def fetch_batch(
        self, agent_id: str, memory_ids: list[str]
    ) -> dict[str, Memory]:
        """Fetch multiple memories by ID.

        Returns a dict keyed by memory_id. Absent records are simply omitted
        from the result — no exception for missing IDs.
        Default: loops over fetch(). Override with the backend's native
        multi-get for better performance.
        Raises AdapterError on backend failure.
        """
        result: dict[str, Memory] = {}
        for mid in memory_ids:
            memory = await self.fetch(agent_id, mid)
            if memory is not None:
                result[mid] = memory
        return result

    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        """Delete multiple memories.

        Returns the count of records actually deleted (absent records
        are not counted).
        Default: loops over delete(). Override with the backend's native
        bulk delete.
        Raises AdapterError on backend failure.
        """
        deleted = 0
        for mid in memory_ids:
            if await self.delete(agent_id, mid):
                deleted += 1
        return deleted

    async def count(
        self, agent_id: str, *, status: MemoryStatus | None = None
    ) -> int:
        """Return the count of memories for an agent, optionally filtered by status.

        Default: calls list_all() and returns len(). Override with a COUNT
        query for O(1) performance on large collections.
        Raises AdapterError on backend failure.
        """
        return len(await self.list_all(agent_id, status=status))

    async def exists(self, agent_id: str, memory_id: str) -> bool:
        """Return True iff a memory with this ID exists for the agent.

        Default: calls fetch() and checks for None. Override with a cheaper
        existence check if the backend supports one.
        Raises AdapterError on backend failure.
        """
        return await self.fetch(agent_id, memory_id) is not None

    # ------------------------------------------------------------------
    # Async context manager (concrete default)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AbstractAdapter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()
