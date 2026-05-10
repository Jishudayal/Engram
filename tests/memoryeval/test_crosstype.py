"""Tests for memoryeval.benchmark.crosstype — 10 cross-type interaction test cases."""

import pytest
from memoryeval.benchmark.crosstype import (
    CROSSTYPE_CASES,
    ActiveAndSupersededCountCorrect,
    AllFourStatusesStorable,
    AllOptionalFieldsRoundtrip,
    ExpiredMemoryStillSearchable,
    ExpiryAndImportanceBothPreserved,
    MemoryTypeCoexistAllSearchable,
    ProvenanceAndSupersessionCombined,
    SearchReturnsAllStatusesByDefault,
    SupersessionChainWithProvenance,
    TouchOnlyChangesAccessFields,
)
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Class-attribute contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", CROSSTYPE_CASES)
def test_category_is_cross_type(case_class) -> None:
    assert case_class.category == BenchmarkCategory.CROSS_TYPE


@pytest.mark.parametrize("case_class", CROSSTYPE_CASES)
def test_name_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.name, str) and case_class.name


@pytest.mark.parametrize("case_class", CROSSTYPE_CASES)
def test_description_is_nonempty_string(case_class) -> None:
    assert isinstance(case_class.description, str) and case_class.description


def test_ten_crosstype_cases_registered() -> None:
    assert len(CROSSTYPE_CASES) == 10


def test_all_case_names_unique() -> None:
    names = [c.name for c in CROSSTYPE_CASES]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# End-to-end lifecycle: setup → run → score → teardown (InMemoryAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_class", CROSSTYPE_CASES)
async def test_score_is_in_valid_range(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    result = await case.run(adapter)
    score = case.score(result)
    await case.teardown(adapter)

    assert 0.0 <= score <= 1.0, f"{case_class.name} returned score={score}"


@pytest.mark.parametrize("case_class", CROSSTYPE_CASES)
async def test_teardown_clears_all_fixtures(case_class) -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = case_class()
    await case.setup(adapter)
    await case.teardown(adapter)
    assert await adapter.count(case.agent_id) == 0


# ---------------------------------------------------------------------------
# Per-case correctness: InMemoryAdapter should score 1.0 on all cases
# ---------------------------------------------------------------------------


async def test_active_and_superseded_count_correct_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ActiveAndSupersededCountCorrect()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_all_four_statuses_storable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AllFourStatusesStorable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_provenance_and_supersession_combined_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ProvenanceAndSupersessionCombined()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_expiry_and_importance_both_preserved_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ExpiryAndImportanceBothPreserved()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_supersession_chain_with_provenance_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SupersessionChainWithProvenance()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_search_returns_all_statuses_by_default_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = SearchReturnsAllStatusesByDefault()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_expired_memory_still_searchable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = ExpiredMemoryStillSearchable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_touch_only_changes_access_fields_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = TouchOnlyChangesAccessFields()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_memory_type_coexist_all_searchable_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = MemoryTypeCoexistAllSearchable()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0


async def test_all_optional_fields_roundtrip_scores_1() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    case = AllOptionalFieldsRoundtrip()
    await case.setup(adapter)
    result = await case.run(adapter)
    assert case.score(result) == 1.0
