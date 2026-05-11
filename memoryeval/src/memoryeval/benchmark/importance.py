"""Importance field test cases for MemoryEval.

Ten cases split across two tiers:

**Tier 1 — Adapter fidelity (5 cases):** verify that the importance field
is correctly stored, updated, and returned by any compliant adapter. These
should score 1.0 on InMemoryAdapter, raw Qdrant, and Engram alike.

**Tier 2 — Reliability gap (5 cases):** verify that retrieval is *weighted*
by importance. A naive cosine-only adapter scores 0.0; a reliability layer
(like Engram's importance-weighted ranker) scores 1.0. When comparing raw
Qdrant vs. Engram, the gap on these cases is the core signal.

Implementation note on gap-case determinism
--------------------------------------------
Gap cases (I6-I10) give the "correct" memory (high importance / frequently
accessed / recently accessed) a slightly **lower** cosine score than the
"wrong" memory. A cosine-only adapter ranks by similarity and always returns
the wrong memory first, giving a deterministic score of 0.0 regardless of
insertion order or backend tie-breaking behaviour.

Scoring convention
------------------
- 1.0  — property correctly handled
- 0.0  — property missing or incorrect
- 0.x  — partial correctness (fractional measurement)
"""

from __future__ import annotations

from memnotary.core.models import Memory, SearchResult
from memoryeval.benchmark._embeddings import vec
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory

# ===========================================================================
# TIER 1 — Adapter fidelity (score 1.0 on any compliant adapter)
# ===========================================================================


# ---------------------------------------------------------------------------
# I1 — Importance value survives write-read roundtrip
# ---------------------------------------------------------------------------


class ImportanceRoundtrip(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "importance_roundtrip"
    description = "A custom importance score must survive write-read roundtrip unchanged"

    _memory_id: str
    _expected: float = 0.92

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Critical security policy: all passwords expire every 90 days.",
            embedding=vec("password", "security", "expire"),
            importance=self._expected,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if abs(result.importance - self._expected) < 1e-6 else 0.0


# ---------------------------------------------------------------------------
# I2 — Default importance is in the valid range [0.0, 1.0]
# ---------------------------------------------------------------------------


class ImportanceDefaultInRange(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "importance_default_in_range"
    description = "A memory stored without explicit importance must have importance in [0.0, 1.0]"

    _memory_id: str

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Rate limit is 100 requests per minute.",
            embedding=vec("api", "rate", "limit"),
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if 0.0 <= result.importance <= 1.0 else 0.0


# ---------------------------------------------------------------------------
# I3 — Updated importance persists after adapter.update()
# ---------------------------------------------------------------------------


class ImportanceUpdatePersists(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "importance_update_persists"
    description = "Updating importance via update_fields must persist after adapter.update()"

    _memory_id: str
    _new_importance: float = 0.95

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Security policy under active review.",
            embedding=vec("security", "policy"),
            importance=0.3,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        m = await adapter.fetch(self.agent_id, self._memory_id)
        if m is None:
            return None
        m.update_fields(importance=self._new_importance)
        await adapter.update(m)
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if abs(result.importance - self._new_importance) < 1e-6 else 0.0


# ---------------------------------------------------------------------------
# I4 — Search does not implicitly filter by importance
# ---------------------------------------------------------------------------


class ImportanceNotUsedInSearchFilter(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "importance_not_used_in_search_filter"
    description = (
        "Search must return both high and low importance memories without implicit filtering"
    )

    _high_id: str
    _low_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy")
        high = Memory(
            agent_id=self.agent_id,
            text="High-priority refund policy update.",
            embedding=embedding,
            importance=0.9,
        )
        low = Memory(
            agent_id=self.agent_id,
            text="Low-priority refund policy archive.",
            embedding=embedding,
            importance=0.1,
        )
        await adapter.store(high)
        await adapter.store(low)
        self._high_id = high.memory_id
        self._low_id = low.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        found = {r.memory.memory_id for r in result}
        high_found = self._high_id in found
        low_found = self._low_id in found
        if high_found and low_found:
            return 1.0
        return 0.5 if high_found or low_found else 0.0


# ---------------------------------------------------------------------------
# I5 — All importance levels in a mixed store are preserved independently
# ---------------------------------------------------------------------------


class MultipleImportanceLevelsAllPreserved(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "multiple_importance_levels_all_preserved"
    description = "Memories at five different importance levels must each retain their exact value"

    _importance_map: dict[str, float]

    async def setup(self, adapter) -> None:
        self._importance_map = {}
        for level in (0.1, 0.3, 0.5, 0.7, 0.9):
            m = Memory(
                agent_id=self.agent_id,
                text=f"Policy with importance {level}.",
                embedding=vec("policy", "security"),
                importance=level,
            )
            await adapter.store(m)
            self._importance_map[m.memory_id] = level

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id)

    def score(self, result: list[Memory]) -> float:
        if not result:
            return 0.0
        correct = sum(
            1
            for m in result
            if (expected := self._importance_map.get(m.memory_id)) is not None
            and abs(m.importance - expected) < 1e-6
        )
        return correct / len(self._importance_map)


# ===========================================================================
# TIER 2 — Reliability gap (score 0.0 on naive adapters, 1.0 on Engram)
#
# InMemoryAdapter is cosine-only: importance, access_count, and last_accessed
# are stored but ignored for ranking. Engram's importance-weighted retrieval
# layer corrects this. These cases expose that gap.
# ===========================================================================


# ---------------------------------------------------------------------------
# I6 — High-importance memory must rank above low-importance even at slightly
#      lower cosine similarity (importance signal overrides raw vector ranking)
# ---------------------------------------------------------------------------


class HighImportanceRanksAboveLow(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "high_importance_ranks_above_low"
    description = (
        "The high-importance memory must rank first even when it has a slightly lower cosine score"
    )

    # INTENTIONAL: scores 0.0 on cosine-only adapters — importance is ignored.
    # Low-importance memory is an exact-match to the query (cosine 1.0); high-importance
    # memory is slightly off (cosine ~0.82). A raw adapter ranks low first; an
    # importance-aware ranker promotes high.

    _high_id: str
    _low_id: str

    async def setup(self, adapter) -> None:
        low = Memory(
            agent_id=self.agent_id,
            text="Low-priority archived refund note.",
            embedding=vec("refund", "policy", "days"),  # exact query match → cosine 1.0
            importance=0.1,
        )
        high = Memory(
            agent_id=self.agent_id,
            text="Critical refund policy — active enforcement.",
            embedding=vec("refund", "policy"),  # one keyword short → cosine ~0.82
            importance=0.9,
        )
        await adapter.store(low)
        await adapter.store(high)
        self._low_id = low.memory_id
        self._high_id = high.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy", "days"), top_k=2)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        return 1.0 if result[0].memory.memory_id == self._high_id else 0.0


# ---------------------------------------------------------------------------
# I7 — top_k results should be the highest-importance subset, not an
#      cosine-ranked subset, when high-importance memories score lower by cosine
# ---------------------------------------------------------------------------


class TopKSortedByImportance(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "top_k_sorted_by_importance"
    description = "top_k=2 must return the two highest-importance memories even when they score lower by cosine"

    # INTENTIONAL: scores 0.0 on cosine-only adapters.
    # Low-importance memories are exact-match (cosine 1.0); high-importance memories are
    # slightly off (cosine ~0.82). A raw adapter fills top_k with the 2 low-importance
    # exact-matches; an importance-aware ranker surfaces the 2 high-importance ones.

    _high_ids: set[str]

    async def setup(self, adapter) -> None:
        low_embedding = vec("api", "rate", "limit")  # exact query match → cosine 1.0
        high_embedding = vec("api", "rate")  # one keyword short → cosine ~0.82
        high_ids: list[str] = []
        for level, text in (
            (0.1, "Minor API note A."),
            (0.2, "Minor API note B."),
            (0.15, "Minor API note C."),
        ):
            await adapter.store(
                Memory(agent_id=self.agent_id, text=text, embedding=low_embedding, importance=level)
            )

        for level, text in (
            (0.9, "Critical API rate limit policy."),
            (0.85, "Important API rate guidance."),
        ):
            m = Memory(
                agent_id=self.agent_id, text=text, embedding=high_embedding, importance=level
            )
            await adapter.store(m)
            high_ids.append(m.memory_id)

        self._high_ids = set(high_ids)

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("api", "rate", "limit"), top_k=2)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        return len(self._high_ids & found) / len(self._high_ids)


# ---------------------------------------------------------------------------
# I8 — Frequently accessed memory must rank above an untouched one
# ---------------------------------------------------------------------------


class AccessCountBoostedRetrieval(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "access_count_boosted_retrieval"
    description = "A memory touched 5 times must rank above an untouched one even with a slightly lower cosine"

    # INTENTIONAL: scores 0.0 on cosine-only adapters — access_count is ignored.
    # Untouched memory is exact-match (cosine 1.0); frequently-accessed has one extra
    # keyword (cosine ~0.82). A raw adapter ranks untouched first; an access-aware
    # ranker promotes the frequently-accessed memory.

    _frequent_id: str
    _untouched_id: str

    async def setup(self, adapter) -> None:
        untouched = Memory(
            agent_id=self.agent_id,
            text="Untouched refund policy.",
            embedding=vec("refund", "policy"),  # exact query match → cosine 1.0
        )
        await adapter.store(untouched)
        self._untouched_id = untouched.memory_id

        frequent = Memory(
            agent_id=self.agent_id,
            text="Frequently accessed refund policy.",
            embedding=vec("refund", "policy", "window"),  # one extra keyword → cosine ~0.82
        )
        await adapter.store(frequent)
        for _ in range(5):
            frequent.touch()
        await adapter.update(frequent)
        self._frequent_id = frequent.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy"), top_k=2)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        return 1.0 if result[0].memory.memory_id == self._frequent_id else 0.0


# ---------------------------------------------------------------------------
# I9 — Recently accessed memory must rank above a never-accessed one
# ---------------------------------------------------------------------------


class RecentlyAccessedSurfaces(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "recently_accessed_surfaces"
    description = (
        "A recently touched memory must rank above a stale one even with a slightly lower cosine"
    )

    # INTENTIONAL: scores 0.0 on cosine-only adapters — last_accessed is ignored.
    # Stale memory is exact-match (cosine 1.0); recently-accessed has one extra keyword
    # (cosine ~0.82). A raw adapter ranks stale first; a recency-aware ranker promotes
    # the recently-accessed memory.

    _recent_id: str
    _stale_id: str

    async def setup(self, adapter) -> None:
        stale = Memory(
            agent_id=self.agent_id,
            text="Old security policy (never accessed).",
            embedding=vec("password", "security"),  # exact query match → cosine 1.0
        )
        await adapter.store(stale)
        self._stale_id = stale.memory_id

        recent = Memory(
            agent_id=self.agent_id,
            text="Security policy (recently reviewed).",
            embedding=vec("password", "security", "requirement"),  # one extra → cosine ~0.82
        )
        await adapter.store(recent)
        recent.touch()
        await adapter.update(recent)
        self._recent_id = recent.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("password", "security"), top_k=2)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        return 1.0 if result[0].memory.memory_id == self._recent_id else 0.0


# ---------------------------------------------------------------------------
# I10 — When importance is tied, recency (last_accessed) breaks the tie
# ---------------------------------------------------------------------------


class RecencyBreaksImportanceTie(TestCase):
    category = BenchmarkCategory.IMPORTANCE
    name = "recency_breaks_importance_tie"
    description = (
        "With equal importance, the recently-accessed memory must rank first even at lower cosine"
    )

    # INTENTIONAL: scores 0.0 on cosine-only adapters — last_accessed is ignored.
    # Old memory is exact-match (cosine 1.0); recently-accessed is one keyword short
    # (cosine ~0.82). Both have identical importance. A raw adapter ranks old first;
    # a recency-aware ranker promotes the recently-accessed memory.

    _recent_id: str
    _old_id: str

    async def setup(self, adapter) -> None:
        importance = 0.75
        old = Memory(
            agent_id=self.agent_id,
            text="Pricing plan — last reviewed 6 months ago.",
            embedding=vec("price", "plan", "subscription"),  # exact query match → cosine 1.0
            importance=importance,
        )
        await adapter.store(old)
        self._old_id = old.memory_id

        recent = Memory(
            agent_id=self.agent_id,
            text="Pricing plan — reviewed this week.",
            embedding=vec("price", "plan"),  # one keyword short → cosine ~0.82
            importance=importance,
        )
        await adapter.store(recent)
        recent.touch()
        await adapter.update(recent)
        self._recent_id = recent.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("price", "plan", "subscription"), top_k=2)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        return 1.0 if result[0].memory.memory_id == self._recent_id else 0.0


# ---------------------------------------------------------------------------
# Public list — used by the benchmark runner and __init__.py
# ---------------------------------------------------------------------------

IMPORTANCE_CASES: list[type[TestCase]] = [
    ImportanceRoundtrip,
    ImportanceDefaultInRange,
    ImportanceUpdatePersists,
    ImportanceNotUsedInSearchFilter,
    MultipleImportanceLevelsAllPreserved,
    HighImportanceRanksAboveLow,
    TopKSortedByImportance,
    AccessCountBoostedRetrieval,
    RecentlyAccessedSurfaces,
    RecencyBreaksImportanceTie,
]
