"""Tests for memoryeval.benchmark.temporal — 10 temporal test cases."""

import pytest
from memoryeval.benchmark.temporal import (
    TEMPORAL_CASES,
    AccessTrackingPersists,
    ExpirationDetectable,
    ImportanceFieldRoundtrips,
    NewerVersionActive,
    ProvenanceRoundtrips,
    SupersededByLinkRecorded,
    SupersededExcludedFromActive,
    SupersedesLinkRecorded,
    ThreeVersionsOnlyLatestActive,
    UpdatedAtPreservedOnRoundtrip,
)
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Class-attribute contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", TEMPORAL_CASES)
def test_category_is_temporal(case_class) -> None:
    assert case_class.category == BenchmarkCategory.TEMPORAL


@pytest.mark.parametrize("case_class", TEMPORAL_CASES)
def test_name_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.name, str) and case_class.name


@pytest.mark.parametrize("case_class", TEMPORAL_CASES)
def test_description_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.description, str) and case_class.description


def test_ten_temporal_cases_registered() -> None:
    assert len(TEMPORAL_CASES) == 10


def test_all_case_names_unique() -> None:
    names = [c.name for c in TEMPORAL_CASES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# End-to-end lifecycle: setup → run → score → teardown (InMemoryAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", TEMPORAL_CASES)
async def test_score_is_in_valid_range(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    result = await case.run(adapter)
    score = case.score(result)
    await case.teardown(adapter)

    assert 0.0 <= score <= 1.0, f"{case_class.name} returned score={score}"


@pytest.mark.parametrize("case_class", TEMPORAL_CASES)
async def test_teardown_clears_all_fixtures(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    await case.teardown(adapter)
    assert await adapter.count(case.agent_id) == 0


# ---------------------------------------------------------------------------
# Per-case correctness: InMemoryAdapter should score 1.0 for most temporal cases
# ---------------------------------------------------------------------------


async def test_newer_version_active_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = NewerVersionActive()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_superseded_excluded_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SupersededExcludedFromActive()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_three_versions_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ThreeVersionsOnlyLatestActive()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_supersedes_link_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SupersedesLinkRecorded()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_superseded_by_link_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SupersededByLinkRecorded()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_importance_roundtrip_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ImportanceFieldRoundtrips()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_expiration_detectable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ExpirationDetectable()
    await case.setup(adapter)
    past, future = await case.run(adapter)
    assert case.score((past, future)) == 1.0


async def test_access_tracking_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AccessTrackingPersists()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_provenance_roundtrip_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ProvenanceRoundtrips()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_updated_at_roundtrip_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = UpdatedAtPreservedOnRoundtrip()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0
