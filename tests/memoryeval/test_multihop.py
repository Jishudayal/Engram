"""Tests for memoryeval.benchmark.multihop — 10 multi-hop retrieval test cases."""

import pytest
from memoryeval.benchmark.multihop import (
    MULTIHOP_CASES,
    AllVersionsSurfacedBySearch,
    ChainedFactsThreeHopSurface,
    CrossClusterBridgedRetrieval,
    HighThresholdCapturesExactMatches,
    OrthogonalQueryReturnsEmpty,
    PartialQuerySurfacesRelated,
    RelatedFactsAllRetrievable,
    RelevanceRankingCorrect,
    TopicIsolationByThreshold,
    TwoHopBridgeReachable,
)
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Class-attribute contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", MULTIHOP_CASES)
def test_category_is_multihop(case_class) -> None:
    assert case_class.category == BenchmarkCategory.MULTIHOP


@pytest.mark.parametrize("case_class", MULTIHOP_CASES)
def test_name_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.name, str) and case_class.name


@pytest.mark.parametrize("case_class", MULTIHOP_CASES)
def test_description_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.description, str) and case_class.description


def test_ten_multihop_cases_registered() -> None:
    assert len(MULTIHOP_CASES) == 10


def test_all_case_names_unique() -> None:
    names = [c.name for c in MULTIHOP_CASES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# End-to-end lifecycle: setup → run → score → teardown (InMemoryAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", MULTIHOP_CASES)
async def test_score_is_in_valid_range(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    result = await case.run(adapter)
    score = case.score(result)
    await case.teardown(adapter)

    assert 0.0 <= score <= 1.0, f"{case_class.name} returned score={score}"


@pytest.mark.parametrize("case_class", MULTIHOP_CASES)
async def test_teardown_clears_all_fixtures(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    await case.teardown(adapter)
    assert await adapter.count(case.agent_id) == 0


# ---------------------------------------------------------------------------
# Per-case correctness: InMemoryAdapter must score 1.0 on all cases
# ---------------------------------------------------------------------------


async def test_related_facts_all_retrievable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = RelatedFactsAllRetrievable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_partial_query_surfaces_related_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = PartialQuerySurfacesRelated()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_relevance_ranking_correct_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = RelevanceRankingCorrect()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_topic_isolation_by_threshold_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = TopicIsolationByThreshold()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_two_hop_bridge_reachable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = TwoHopBridgeReachable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_all_versions_surfaced_by_search_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AllVersionsSurfacedBySearch()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_chained_facts_three_hop_surface_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ChainedFactsThreeHopSurface()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_high_threshold_captures_exact_matches_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = HighThresholdCapturesExactMatches()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_orthogonal_query_returns_empty_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = OrthogonalQueryReturnsEmpty()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_cross_cluster_bridged_retrieval_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = CrossClusterBridgedRetrieval()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0
