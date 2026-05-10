"""Tests for memoryeval.benchmark.importance — 10 importance field test cases.

Tier 1 (I1-I5): adapter fidelity — score 1.0 on InMemoryAdapter.
Tier 2 (I6-I10): reliability gap — score 0.0 on InMemoryAdapter (cosine-only),
  score 1.0 on Engram (importance-weighted retrieval).
"""

import pytest
from memoryeval.benchmark.importance import (
    IMPORTANCE_CASES,
    AccessCountBoostedRetrieval,
    HighImportanceRanksAboveLow,
    ImportanceDefaultInRange,
    ImportanceNotUsedInSearchFilter,
    ImportanceRoundtrip,
    ImportanceUpdatePersists,
    MultipleImportanceLevelsAllPreserved,
    RecencyBreaksImportanceTie,
    RecentlyAccessedSurfaces,
    TopKSortedByImportance,
)
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Class-attribute contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", IMPORTANCE_CASES)
def test_category_is_importance(case_class) -> None:
    assert case_class.category == BenchmarkCategory.IMPORTANCE


@pytest.mark.parametrize("case_class", IMPORTANCE_CASES)
def test_name_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.name, str) and case_class.name


@pytest.mark.parametrize("case_class", IMPORTANCE_CASES)
def test_description_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.description, str) and case_class.description


def test_ten_importance_cases_registered() -> None:
    assert len(IMPORTANCE_CASES) == 10


def test_all_case_names_unique() -> None:
    names = [c.name for c in IMPORTANCE_CASES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# End-to-end lifecycle: setup → run → score → teardown (InMemoryAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", IMPORTANCE_CASES)
async def test_score_is_in_valid_range(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    result = await case.run(adapter)
    score = case.score(result)
    await case.teardown(adapter)

    assert 0.0 <= score <= 1.0, f"{case_class.name} returned score={score}"


@pytest.mark.parametrize("case_class", IMPORTANCE_CASES)
async def test_teardown_clears_all_fixtures(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    await case.teardown(adapter)
    assert await adapter.count(case.agent_id) == 0


# ---------------------------------------------------------------------------
# Tier 1 — Adapter fidelity: InMemoryAdapter must score 1.0
# ---------------------------------------------------------------------------


async def test_importance_roundtrip_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ImportanceRoundtrip()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_importance_default_in_range_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ImportanceDefaultInRange()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_importance_update_persists_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ImportanceUpdatePersists()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_importance_not_used_in_search_filter_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ImportanceNotUsedInSearchFilter()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_multiple_importance_levels_all_preserved_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = MultipleImportanceLevelsAllPreserved()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


# ---------------------------------------------------------------------------
# Tier 2 — Reliability gap: InMemoryAdapter scores 0.0 (cosine-only, no
# importance/recency weighting). Engram's ranker scores 1.0 here.
# ---------------------------------------------------------------------------


async def test_high_importance_ranks_above_low_scores_0_on_inmemory() -> None:
    """InMemoryAdapter is cosine-only; equal cosine → insertion order → low-importance wins."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = HighImportanceRanksAboveLow()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0


async def test_top_k_sorted_by_importance_scores_0_on_inmemory() -> None:
    """InMemoryAdapter returns first 2 by insertion order (all low-importance); gap case."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = TopKSortedByImportance()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0


async def test_access_count_boosted_retrieval_scores_0_on_inmemory() -> None:
    """InMemoryAdapter ignores access_count; untouched memory inserted first wins the tie."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AccessCountBoostedRetrieval()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0


async def test_recently_accessed_surfaces_scores_0_on_inmemory() -> None:
    """InMemoryAdapter ignores last_accessed; stale memory inserted first wins the tie."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = RecentlyAccessedSurfaces()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0


async def test_recency_breaks_importance_tie_scores_0_on_inmemory() -> None:
    """InMemoryAdapter ignores last_accessed; old memory inserted first wins the tie."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = RecencyBreaksImportanceTie()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0
