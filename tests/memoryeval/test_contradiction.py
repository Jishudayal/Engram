"""Tests for memoryeval.benchmark.contradiction — 10 contradiction test cases."""

import pytest
from memoryeval.benchmark.contradiction import (
    CONTRADICTION_CASES,
    AllConflictingClaimsSearchable,
    BothClaimsActiveIsContradiction,
    ContradictionDetectableViaSearch,
    EmbeddingFidelityDistinct,
    EmbeddingFidelityHighPrecision,
    FlaggedStatusPersistable,
    ResolutionReducesActiveCount,
    ResolvedContradictionHasOneActive,
    SupersessionLinksIntact,
    ThreeClaimsOneActive,
)
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Class-attribute contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", CONTRADICTION_CASES)
def test_category_is_contradiction(case_class) -> None:
    assert case_class.category == BenchmarkCategory.CONTRADICTION


@pytest.mark.parametrize("case_class", CONTRADICTION_CASES)
def test_name_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.name, str) and case_class.name


@pytest.mark.parametrize("case_class", CONTRADICTION_CASES)
def test_description_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.description, str) and case_class.description


def test_ten_contradiction_cases_registered() -> None:
    assert len(CONTRADICTION_CASES) == 10


def test_all_case_names_unique() -> None:
    names = [c.name for c in CONTRADICTION_CASES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# End-to-end lifecycle: setup → run → score → teardown (InMemoryAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", CONTRADICTION_CASES)
async def test_score_is_in_valid_range(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    result = await case.run(adapter)
    score = case.score(result)
    await case.teardown(adapter)

    assert 0.0 <= score <= 1.0, f"{case_class.name} returned score={score}"


@pytest.mark.parametrize("case_class", CONTRADICTION_CASES)
async def test_teardown_clears_all_fixtures(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    await case.teardown(adapter)
    assert await adapter.count(case.agent_id) == 0


# ---------------------------------------------------------------------------
# Per-case correctness
# ---------------------------------------------------------------------------


async def test_both_claims_active_scores_0() -> None:
    """C1 is the bad baseline — naive adapters score 0.0 (contradiction unresolved)."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = BothClaimsActiveIsContradiction()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 0.0


async def test_resolved_contradiction_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ResolvedContradictionHasOneActive()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_embedding_fidelity_high_precision_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = EmbeddingFidelityHighPrecision()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_three_claims_one_active_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ThreeClaimsOneActive()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_flagged_status_persistable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = FlaggedStatusPersistable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_contradiction_detectable_via_search_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ContradictionDetectableViaSearch()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_supersession_links_intact_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SupersessionLinksIntact()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_embedding_fidelity_distinct_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = EmbeddingFidelityDistinct()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_resolution_reduces_active_count_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ResolutionReducesActiveCount()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_all_conflicting_claims_searchable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AllConflictingClaimsSearchable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


# ---------------------------------------------------------------------------
# C1 special: pass_threshold is lower (0.5) — still a valid threshold
# ---------------------------------------------------------------------------


def test_c1_has_lower_pass_threshold() -> None:
    assert BothClaimsActiveIsContradiction.pass_threshold == 0.5
