"""Tests for memoryeval.scorer.BenchmarkScorer."""

from __future__ import annotations

import asyncio

import pytest

from memoryeval.benchmark import ALL_CASES
from memoryeval.benchmark.importance import ImportanceRoundtrip
from memoryeval.benchmark.temporal import NewerVersionActive
from memoryeval.case import TestCase
from memoryeval.scorer import BenchmarkScorer
from memoryeval.types import (
    BenchmarkCategory,
    BenchmarkReport,
    CaseResult,
    CategoryScore,
)


# ---------------------------------------------------------------------------
# Inline broken-case helpers (only needed in this module)
# ---------------------------------------------------------------------------


class _RunRaises(TestCase):
    """run() raises; score() should never be called."""

    category    = BenchmarkCategory.TEMPORAL
    name        = "_run_raises"
    description = "test helper — run() always raises"

    async def setup(self, adapter) -> None:
        pass

    async def run(self, adapter) -> None:
        raise RuntimeError("intentional test error")

    def score(self, result) -> float:
        return 1.0


class _SetupRaises(TestCase):
    """setup() raises; teardown() should NOT be called (nothing was stored)."""

    category    = BenchmarkCategory.TEMPORAL
    name        = "_setup_raises"
    description = "test helper — setup() always raises"

    _teardown_called: bool = False

    async def setup(self, adapter) -> None:
        raise ValueError("setup boom")

    async def run(self, adapter) -> None:
        pass

    def score(self, result) -> float:
        return 1.0

    async def teardown(self, adapter) -> None:
        _SetupRaises._teardown_called = True
        await super().teardown(adapter)


class _SlowScoreZero(TestCase):
    """Always scores 0.0; used to verify pass_rate calculation."""

    category    = BenchmarkCategory.MULTIHOP
    name        = "_score_zero"
    description = "test helper — always scores 0.0"

    async def setup(self, adapter) -> None:
        pass

    async def run(self, adapter) -> None:
        return None

    def score(self, result) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# run_case() — happy path
# ---------------------------------------------------------------------------


async def test_run_case_passing_case_has_score_and_passed() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(ImportanceRoundtrip())

    assert isinstance(result, CaseResult)
    assert result.error is None
    assert result.score == 1.0
    assert result.passed is True


async def test_run_case_zero_score_not_passed() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(_SlowScoreZero())

    assert result.error is None
    assert result.score == 0.0
    assert result.passed is False  # 0.0 < pass_threshold (0.7)


async def test_run_case_result_captures_case_metadata() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(ImportanceRoundtrip())

    assert result.case_name == ImportanceRoundtrip.name
    assert result.category  == BenchmarkCategory.IMPORTANCE


# ---------------------------------------------------------------------------
# run_case() — error handling
# ---------------------------------------------------------------------------


async def test_run_case_run_raises_returns_error_result() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(_RunRaises())

    assert result.error is not None
    assert "intentional test error" in result.error
    assert result.score   == 0.0
    assert result.passed  is False


async def test_run_case_setup_raises_returns_error_result() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(_SetupRaises())

    assert result.error is not None
    assert "setup" in result.error
    assert result.score  == 0.0
    assert result.passed is False


async def test_run_case_setup_raises_does_not_call_teardown() -> None:
    from engram.adapters.memory import InMemoryAdapter

    _SetupRaises._teardown_called = False
    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    await scorer.run_case(_SetupRaises())

    assert _SetupRaises._teardown_called is False


async def test_run_case_run_raises_still_calls_teardown() -> None:
    """Teardown must run after a successful setup even when run() fails."""
    from engram.adapters.memory import InMemoryAdapter
    from engram.core.models import Memory
    from memoryeval.benchmark._embeddings import vec

    teardown_ran = False

    class _RunRaisesWithSetup(TestCase):
        category    = BenchmarkCategory.TEMPORAL
        name        = "_run_raises_with_setup"
        description = "test helper — setup succeeds, run() raises"

        async def setup(self, adapter) -> None:
            await adapter.store(Memory(
                agent_id=self.agent_id,
                text="fixture",
                embedding=vec("refund", "policy"),
            ))

        async def run(self, adapter) -> None:
            raise RuntimeError("run failure")

        def score(self, result) -> float:
            return 1.0

        async def teardown(self, adapter) -> None:
            nonlocal teardown_ran
            teardown_ran = True
            await super().teardown(adapter)

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(_RunRaisesWithSetup())

    assert result.error is not None
    assert teardown_ran is True


# ---------------------------------------------------------------------------
# run() — full suite
# ---------------------------------------------------------------------------


async def test_run_returns_benchmark_report() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="my-backend")
    report  = await scorer.run()

    assert isinstance(report, BenchmarkReport)
    assert report.backend_name == "my-backend"


async def test_run_has_all_five_categories() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    assert len(report.categories) == 5
    found = {cs.category for cs in report.categories}
    assert found == set(BenchmarkCategory)


async def test_run_each_category_has_ten_results() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    for cat_score in report.categories:
        assert len(cat_score.results) == 10, (
            f"{cat_score.category} has {len(cat_score.results)} results, expected 10"
        )


async def test_run_total_results_is_fifty() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    total = sum(len(cs.results) for cs in report.categories)
    assert total == 50


async def test_run_overall_score_in_valid_range() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    assert 0.0 <= report.overall_score <= 1.0


async def test_run_category_scores_in_valid_range() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    for cs in report.categories:
        assert 0.0 <= cs.score    <= 1.0, f"{cs.category} score out of range"
        assert 0.0 <= cs.pass_rate <= 1.0, f"{cs.category} pass_rate out of range"


async def test_run_category_order_matches_enum_declaration() -> None:
    """Categories must appear in BenchmarkCategory enum order."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    expected_order = list(BenchmarkCategory)
    actual_order   = [cs.category for cs in report.categories]
    assert actual_order == expected_order


async def test_run_hallucination_risk_set() -> None:
    from engram.adapters.memory import InMemoryAdapter
    from memoryeval.types import HallucinationRisk

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    assert isinstance(report.hallucination_risk, HallucinationRisk)


# ---------------------------------------------------------------------------
# run() — category filter
# ---------------------------------------------------------------------------


async def test_run_single_category_filter() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run(categories=[BenchmarkCategory.TEMPORAL])

    assert len(report.categories) == 1
    assert report.categories[0].category == BenchmarkCategory.TEMPORAL
    assert len(report.categories[0].results) == 10


async def test_run_two_category_filter() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run(
        categories=[BenchmarkCategory.TEMPORAL, BenchmarkCategory.MULTIHOP]
    )

    assert len(report.categories) == 2
    found = {cs.category for cs in report.categories}
    assert found == {BenchmarkCategory.TEMPORAL, BenchmarkCategory.MULTIHOP}


async def test_run_empty_category_filter_raises_value_error() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    with pytest.raises(ValueError, match="No cases match"):
        await scorer.run(categories=[])


# ---------------------------------------------------------------------------
# run() — custom case list
# ---------------------------------------------------------------------------


async def test_custom_case_list() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(
        adapter,
        backend_name="test",
        cases=[ImportanceRoundtrip, NewerVersionActive],
    )
    report  = await scorer.run()

    total = sum(len(cs.results) for cs in report.categories)
    assert total == 2


async def test_custom_case_list_with_category_filter() -> None:
    """Category filter applies on top of the custom case list."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(
        adapter,
        backend_name="test",
        cases=[ImportanceRoundtrip, NewerVersionActive],
    )
    report  = await scorer.run(categories=[BenchmarkCategory.IMPORTANCE])

    total = sum(len(cs.results) for cs in report.categories)
    assert total == 1  # only ImportanceRoundtrip matches


# ---------------------------------------------------------------------------
# scores_by_category property
# ---------------------------------------------------------------------------


async def test_scores_by_category_has_five_entries() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    sbc = report.scores_by_category
    assert len(sbc) == 5
    for cat in BenchmarkCategory:
        assert cat in sbc
        assert 0.0 <= sbc[cat] <= 1.0


# ---------------------------------------------------------------------------
# Known score signatures on InMemoryAdapter
# ---------------------------------------------------------------------------


async def test_importance_gap_cases_lower_category_score() -> None:
    """Importance gap cases (I6-I10) score 0.0 on InMemoryAdapter; the
    category average must be < 1.0 as a result."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run(categories=[BenchmarkCategory.IMPORTANCE])

    imp_score = report.scores_by_category[BenchmarkCategory.IMPORTANCE]
    # 5 of 10 cases score 0.0 → average ≤ 0.5
    assert imp_score <= 0.5


async def test_temporal_fidelity_cases_score_full() -> None:
    """All temporal cases are adapter-fidelity cases → category score = 1.0."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run(categories=[BenchmarkCategory.TEMPORAL])

    assert report.scores_by_category[BenchmarkCategory.TEMPORAL] == 1.0


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class _SlowSetup(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "_slow_setup"
    description = "test helper — setup sleeps indefinitely"

    async def setup(self, adapter) -> None:
        await asyncio.sleep(100)

    async def run(self, adapter) -> None:
        return None

    def score(self, result) -> float:
        return 1.0


class _SlowRun(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "_slow_run"
    description = "test helper — run() sleeps indefinitely"

    async def setup(self, adapter) -> None:
        pass

    async def run(self, adapter) -> None:
        await asyncio.sleep(100)

    def score(self, result) -> float:
        return 1.0


async def test_setup_timeout_produces_error_result() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test", case_timeout=0.05)
    result  = await scorer.run_case(_SlowSetup())

    assert result.error is not None
    assert "setup" in result.error and "timed out" in result.error
    assert result.score  == 0.0
    assert result.passed is False


async def test_run_timeout_produces_error_result() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test", case_timeout=0.05)
    result  = await scorer.run_case(_SlowRun())

    assert result.error is not None
    assert "run" in result.error and "timed out" in result.error
    assert result.score  == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Teardown failures are logged, not raised
# ---------------------------------------------------------------------------


class _TeardownRaises(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "_teardown_raises"
    description = "test helper — teardown always raises"

    async def setup(self, adapter) -> None:
        pass

    async def run(self, adapter) -> None:
        return None

    def score(self, result) -> float:
        return 1.0

    async def teardown(self, adapter) -> None:
        raise RuntimeError("teardown explosion")


async def test_teardown_failure_does_not_raise_from_run_case() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    result  = await scorer.run_case(_TeardownRaises())  # must not raise

    # run/score succeeded despite teardown failure
    assert result.error is None
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Progress hook
# ---------------------------------------------------------------------------


async def test_progress_hook_called_once_per_case() -> None:
    import asyncio

    from engram.adapters.memory import InMemoryAdapter

    calls: list[tuple[str, int, int]] = []

    async def hook(result: CaseResult, i: int, total: int) -> None:
        calls.append((result.case_name, i, total))

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(
        adapter,
        backend_name="test",
        cases=[ImportanceRoundtrip, NewerVersionActive],
    )
    await scorer.run(on_case_complete=hook)

    assert len(calls) == 2
    assert calls[0] == (ImportanceRoundtrip.name, 1, 2)
    assert calls[1] == (NewerVersionActive.name,  2, 2)


async def test_progress_hook_none_is_fine() -> None:
    """run() with no hook must behave identically to run() without the arg."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(
        adapter,
        backend_name="test",
        cases=[ImportanceRoundtrip],
    )
    report = await scorer.run(on_case_complete=None)
    assert len(report.all_results) == 1


# ---------------------------------------------------------------------------
# Duplicate case-name guard
# ---------------------------------------------------------------------------


def test_duplicate_case_names_raises_at_init() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    with pytest.raises(ValueError, match="Duplicate case names"):
        BenchmarkScorer(
            adapter,
            backend_name="test",
            cases=[ImportanceRoundtrip, ImportanceRoundtrip],
        )


# ---------------------------------------------------------------------------
# all_results property
# ---------------------------------------------------------------------------


async def test_all_results_flat_tuple_length() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run()

    assert len(report.all_results) == 50


async def test_all_results_are_case_result_instances() -> None:
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(
        adapter,
        backend_name="test",
        cases=[ImportanceRoundtrip, NewerVersionActive],
    )
    report = await scorer.run()

    assert all(isinstance(r, CaseResult) for r in report.all_results)


async def test_all_results_subset_run() -> None:
    """all_results count matches total cases when categories are filtered."""
    from engram.adapters.memory import InMemoryAdapter

    adapter = InMemoryAdapter()
    scorer  = BenchmarkScorer(adapter, backend_name="test")
    report  = await scorer.run(categories=[BenchmarkCategory.TEMPORAL])

    assert len(report.all_results) == 10
