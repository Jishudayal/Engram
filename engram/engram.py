"""Engram — the main entry point.

Wrap any AbstractAdapter with Engram to gain contradiction detection,
health scoring, and consolidation on top of your vector backend:

    adapter = QdrantAdapter(url="localhost:6333", collection="memories")
    async with Engram(adapter) as eng:
        await eng.store(memory)
        results = await eng.search("agent-1", query_embedding)

Pass a ContradictionDetector to enable conflict detection on every store
and conflict-flag enrichment on every search:

    from engram import Engram, ContradictionDetector
    eng = Engram(adapter, detector=ContradictionDetector())

After bulk ingestion, run a batch scan to detect conflicts across the batch:

    await eng.store_batch(memories)
    new_conflicts = await eng.scan_contradictions("agent-1")

To plan and execute consolidation of detected conflicts:

    from engram import Engram, ContradictionDetector, Consolidator
    eng = Engram(adapter, detector=ContradictionDetector(), consolidator=Consolidator())
    plan = await eng.consolidate("agent-1")  # None if no PENDING conflicts

What's implemented in each step:
  1.8 — constructor, adapter delegation, NotImplementedError stubs
  4   — health() scoring
  5.1 — ContradictionDetector core + conflict storage contract
  5.2 — store-time detection, search-time enrichment, scan_contradictions()
  6.1 — Consolidator.plan() — LLM-driven action planning
  6.2 — update_conflict() adapter contract
  6.3 — Consolidator.execute() + Engram.consolidate() wiring
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus, ResolutionStatus
from engram.core.health import HealthScorer
from engram.core.models import ConflictRecord, Memory, SearchResult

if TYPE_CHECKING:
    from engram.core.contradiction import ContradictionDetector
    from engram.core.consolidator import Consolidator
    from engram.core.models import ConsolidationPlan, HealthScore

__all__ = ["Engram"]

logger = logging.getLogger(__name__)

# Number of candidate memories to fetch during store-time detection.
# +1 to account for the self-match: the just-stored memory will appear as the top
# result (cosine=1.0) and is filtered out below, so 21 yields up to 20 true candidates.
_STORE_TIME_CANDIDATE_LIMIT = 21


class Engram:
    """Memory layer facade — wraps an adapter and coordinates reliability features.

    Instantiate with any AbstractAdapter subclass. Pass a ContradictionDetector
    to enable automatic conflict detection on store() and conflict-flag enrichment
    on search(). Use as an async context manager to ensure connections close on exit.

    Args:
        adapter:       Any AbstractAdapter subclass (Qdrant, Chroma, InMemory, …).
        detector:      Optional ContradictionDetector. When provided:
                         - store() runs conflict detection against similar existing memories.
                         - search() enriches results with conflict_flag, conflicts_with,
                           conflict_summary, and recommended fields.
                         - scan_contradictions() becomes available.
                       When None, store/search behave identically to a plain adapter call.
        health_scorer:  Optional HealthScorer instance. Supply a pre-configured scorer to
                        override freshness half-life, weights, or risk thresholds. Defaults
                        to a HealthScorer with library defaults.
        consolidator:   Optional Consolidator instance. When provided, consolidate() plans
                        and executes memory consolidation actions for PENDING conflicts.
                        When None, consolidate() raises ValueError.
    """

    def __init__(
        self,
        adapter: AbstractAdapter,
        *,
        detector: ContradictionDetector | None = None,
        health_scorer: HealthScorer | None = None,
        consolidator: Consolidator | None = None,
    ) -> None:
        if not isinstance(adapter, AbstractAdapter):
            raise TypeError(
                f"adapter must be an AbstractAdapter subclass, got {type(adapter).__name__!r}"
            )
        self._adapter = adapter
        self._detector = detector
        self._health_scorer = health_scorer or HealthScorer()
        self._consolidator = consolidator

    @property
    def adapter(self) -> AbstractAdapter:
        """The underlying storage adapter."""
        return self._adapter

    @property
    def backend_name(self) -> str:
        """Short identifier for the underlying backend (e.g. 'qdrant', 'chroma')."""
        return self._adapter.backend_name

    def __repr__(self) -> str:
        if self._detector is not None:
            return (
                f"Engram(adapter={self._adapter.backend_name!r}, "
                f"detector={type(self._detector).__name__})"
            )
        return f"Engram(adapter={self._adapter.backend_name!r})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release backend connections. Idempotent."""
        await self._adapter.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def store(self, memory: Memory) -> None:
        """Store a memory, then run contradiction detection if a detector is configured.

        Detection flow (when detector is set and memory has an embedding):
          1. Persist the memory via the adapter.
          2. Vector-search for similar existing memories above the detector's
             cluster_threshold (up to _STORE_TIME_CANDIDATE_LIMIT candidates).
          3. LLM-classify each candidate pair via detector.detect().
          4. Store each confirmed ConflictRecord via the adapter.

        If the memory has no embedding, step 2–4 are skipped (no vector to compare).
        store_batch() does NOT trigger per-memory detection — use scan_contradictions()
        after bulk ingestion instead.
        """
        await self._adapter.store(memory)
        if self._detector is not None and memory.embedding is not None:
            await self._detect_and_store_conflicts(memory)

    async def update(self, memory: Memory) -> None:
        """Fully replace an existing memory record. Raises NotFoundError if absent."""
        await self._adapter.update(memory)

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted, False if not found."""
        return await self._adapter.delete(agent_id, memory_id)

    async def store_batch(self, memories: list[Memory]) -> None:
        """Upsert multiple memories without per-memory detection.

        Contradiction detection is intentionally skipped for bulk writes.
        Call scan_contradictions(agent_id) after ingestion to detect conflicts
        across the full batch in one pass.
        """
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
        """Search for relevant memories, enriching results with conflict metadata.

        When a detector is configured, each result is annotated with:
          conflict_flag     — True if a PENDING ConflictRecord involves this memory.
          conflicts_with    — memory IDs of conflicting memories.
          conflict_summary  — LLM-generated one-sentence explanation (from description).
          recommended       — False when this memory conflicts with a higher-ranked
                              result in the same response; True otherwise.

        When no detector is configured, conflict fields remain at their defaults
        (conflict_flag=False, conflicts_with=(), recommended=True).
        """
        results = await self._adapter.search(
            agent_id,
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )
        if self._detector is not None and results:
            results = await self._enrich_with_conflicts(agent_id, results)
        return results

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
    # Reliability features
    # ------------------------------------------------------------------

    async def health(self, agent_id: str) -> HealthScore:
        """Compute a health snapshot for an agent's memory collection."""
        return await self._health_scorer.compute(agent_id, self._adapter)

    async def scan_contradictions(self, agent_id: str) -> list[ConflictRecord]:
        """Run a full batch contradiction scan across all ACTIVE memories.

        Fetches all active memories, runs embedding clustering to find candidate
        pairs, LLM-classifies each pair, and stores new ConflictRecords. Skips
        pairs that already have a PENDING ConflictRecord.

        Requires a ContradictionDetector to be configured at construction time.
        Use this after store_batch() or to bootstrap contradiction detection on
        an existing collection.

        Raises ValueError if no detector was provided.
        Raises NotImplementedError if the adapter does not support conflict storage.
        """
        if self._detector is None:
            raise ValueError(
                "scan_contradictions() requires a ContradictionDetector. "
                "Pass detector=ContradictionDetector() to Engram()."
            )
        return await self._detector.scan(agent_id, self._adapter)

    async def pending_review(self, agent_id: str) -> list[ConflictRecord]:
        """Return ConflictRecords that require human review.

        Returns all PENDING ConflictRecords for the agent. After a consolidation
        run, this is the human review queue: conflicts that were flagged (HUMAN_REVIEW
        tier) rather than auto-resolved appear here so a human can decide what to do.

        Can be called without a ContradictionDetector or Consolidator configured.
        """
        return await self._adapter.list_conflicts(agent_id, status=ResolutionStatus.PENDING)

    async def consolidate(self, agent_id: str) -> ConsolidationPlan | None:
        """Plan and execute memory consolidation for an agent's PENDING conflicts.

        Calls Consolidator.plan() to build a ConsolidationPlan from PENDING
        ConflictRecords, then Consolidator.execute() to apply each action.

        Returns the executed ConsolidationPlan, or None if there are no PENDING
        conflicts to act on.

        Raises ValueError if no Consolidator was provided at construction time.
        Raises NotImplementedError if the adapter does not support conflict storage.
        """
        if self._consolidator is None:
            raise ValueError(
                "consolidate() requires a Consolidator. "
                "Pass consolidator=Consolidator() to Engram()."
            )
        plan = await self._consolidator.plan(agent_id, self._adapter)
        if plan is None:
            return None
        return await self._consolidator.execute(plan, self._adapter)

    # ------------------------------------------------------------------
    # Internal — store-time detection
    # ------------------------------------------------------------------

    async def _detect_and_store_conflicts(self, memory: Memory) -> None:
        """Run contradiction detection for a newly stored memory.

        Uses vector search to find similar existing memories above the detector's
        cluster_threshold, then calls detector.detect() to LLM-classify each
        candidate pair. Confirmed ConflictRecords are stored via the adapter.
        """
        assert self._detector is not None  # caller guarantees this
        assert memory.embedding is not None  # caller guarantees this

        candidate_results = await self._adapter.search(
            memory.agent_id,
            memory.embedding,
            top_k=_STORE_TIME_CANDIDATE_LIMIT,
            score_threshold=self._detector.cluster_threshold,
        )
        candidates = [
            r.memory
            for r in candidate_results
            if r.memory.memory_id != memory.memory_id
        ]
        if not candidates:
            logger.debug(
                "_detect_and_store_conflicts: no candidates above threshold "
                "for memory %s (agent=%s)",
                memory.memory_id,
                memory.agent_id,
            )
            return

        conflicts = await self._detector.detect(memory, candidates)
        for conflict in conflicts:
            await self._adapter.store_conflict(conflict)

        if conflicts:
            logger.info(
                "_detect_and_store_conflicts: stored %d conflict(s) for memory %s (agent=%s)",
                len(conflicts),
                memory.memory_id,
                memory.agent_id,
            )

    # ------------------------------------------------------------------
    # Internal — search-time enrichment
    # ------------------------------------------------------------------

    async def _enrich_with_conflicts(
        self,
        agent_id: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Annotate search results with conflict metadata from stored ConflictRecords.

        Fetches all PENDING conflicts for the agent, builds a memory-id index,
        and for each result finds relevant conflicts. When two results in the same
        response conflict with each other, the lower-ranked result is marked
        recommended=False.
        """
        all_conflicts = await self._adapter.list_conflicts(
            agent_id, status=ResolutionStatus.PENDING
        )
        if not all_conflicts:
            return results

        # Build lookup: memory_id → [ConflictRecord, ...]
        conflict_index: dict[str, list[ConflictRecord]] = {}
        for conflict in all_conflicts:
            conflict_index.setdefault(conflict.memory_a_id, []).append(conflict)
            conflict_index.setdefault(conflict.memory_b_id, []).append(conflict)

        result_ids = {r.memory.memory_id for r in results}

        # First pass — collect conflict data per result
        conflict_data: list[dict[str, Any]] = []
        for result in results:
            mid = result.memory.memory_id
            relevant = conflict_index.get(mid)
            if not relevant:
                conflict_data.append({})
                continue

            other_ids = tuple(
                c.memory_b_id if c.memory_a_id == mid else c.memory_a_id
                for c in relevant
            )
            # Surface summary from highest-confidence conflict
            best = max(relevant, key=lambda c: c.confidence)
            conflict_data.append({
                "conflict_flag": True,
                "conflicts_with": other_ids,
                "conflict_summary": best.description,
            })

        # Second pass — determine recommended flag
        # If two results conflict with each other, mark the lower-ranked one
        # (higher rank number = worse score) as not recommended.
        not_recommended: set[str] = set()
        rank_map = {r.memory.memory_id: r.rank for r in results}
        for result, data in zip(results, conflict_data, strict=True):
            if not data:
                continue
            for other_id in data["conflicts_with"]:
                if other_id not in result_ids:
                    continue
                other_rank = rank_map.get(other_id)
                if other_rank is not None and other_rank > result.rank:
                    not_recommended.add(other_id)

        # Third pass — build enriched SearchResult instances
        enriched: list[SearchResult] = []
        for result, data in zip(results, conflict_data, strict=True):
            if not data:
                enriched.append(result)
            else:
                mid = result.memory.memory_id
                enriched.append(
                    result.model_copy(
                        update={
                            **data,
                            "recommended": mid not in not_recommended,
                        }
                    )
                )

        flagged = sum(1 for d in conflict_data if d)
        logger.debug(
            "_enrich_with_conflicts: agent=%s results=%d flagged=%d not_recommended=%d",
            agent_id,
            len(results),
            flagged,
            len(not_recommended),
        )
        return enriched

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
