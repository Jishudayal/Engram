"""Tests for memoryeval.types."""

import pytest
from pydantic import ValidationError

from memoryeval.case import TestCase
from memoryeval.types import (
    BenchmarkCategory,
    BenchmarkReport,
    CaseResult,
    CategoryScore,
    HallucinationRisk,
)


# ---------------------------------------------------------------------------
# Concrete TestCase used for from_case() tests
# ---------------------------------------------------------------------------


class _MinimalCase(TestCase):
    category = BenchmarkCategory.TEMPORAL
    name = "minimal_case"
    description = "Minimal concrete case for testing CaseResult.from_case"

    async def setup(self, adapter):
        pass

    async def run(self, adapter):
        return None

    def score(self, result):
        return 0.0


class _HighThresholdCase(_MinimalCase):
    name = "high_threshold_case"
    description = "Overrides pass_threshold for testing"
    pass_threshold = 0.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    *,
    case_name: str = "test_case",
    category: BenchmarkCategory = BenchmarkCategory.TEMPORAL,
    score: float = 0.9,
    pass_threshold: float = 0.7,
    passed: bool = True,
    error: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_name=case_name,
        category=category,
        score=score,
        pass_threshold=pass_threshold,
        passed=passed,
        error=error,
    )


def make_category_score(
    category: BenchmarkCategory = BenchmarkCategory.TEMPORAL,
    score: float = 0.8,
    pass_rate: float = 0.9,
) -> CategoryScore:
    return CategoryScore(category=category, score=score, pass_rate=pass_rate)


# ---------------------------------------------------------------------------
# BenchmarkCategory
# ---------------------------------------------------------------------------


class TestBenchmarkCategory:
    def test_all_five_categories_exist(self) -> None:
        assert set(BenchmarkCategory) == {
            BenchmarkCategory.TEMPORAL,
            BenchmarkCategory.CONTRADICTION,
            BenchmarkCategory.MULTIHOP,
            BenchmarkCategory.IMPORTANCE,
            BenchmarkCategory.CROSS_TYPE,
        }

    def test_string_values(self) -> None:
        assert BenchmarkCategory.TEMPORAL == "temporal"
        assert BenchmarkCategory.CROSS_TYPE == "cross_type"
        assert BenchmarkCategory.MULTIHOP == "multihop"

    def test_is_str_subclass(self) -> None:
        assert isinstance(BenchmarkCategory.TEMPORAL, str)


# ---------------------------------------------------------------------------
# HallucinationRisk
# ---------------------------------------------------------------------------


class TestHallucinationRisk:
    def test_low_at_exactly_0_85(self) -> None:
        assert HallucinationRisk.from_score(0.85) == HallucinationRisk.LOW

    def test_low_at_1_0(self) -> None:
        assert HallucinationRisk.from_score(1.0) == HallucinationRisk.LOW

    def test_medium_just_below_low(self) -> None:
        assert HallucinationRisk.from_score(0.84) == HallucinationRisk.MEDIUM

    def test_medium_at_exactly_0_70(self) -> None:
        assert HallucinationRisk.from_score(0.70) == HallucinationRisk.MEDIUM

    def test_high_just_below_medium(self) -> None:
        assert HallucinationRisk.from_score(0.69) == HallucinationRisk.HIGH

    def test_high_at_exactly_0_50(self) -> None:
        assert HallucinationRisk.from_score(0.50) == HallucinationRisk.HIGH

    def test_critical_just_below_high(self) -> None:
        assert HallucinationRisk.from_score(0.49) == HallucinationRisk.CRITICAL

    def test_critical_at_zero(self) -> None:
        assert HallucinationRisk.from_score(0.0) == HallucinationRisk.CRITICAL

    def test_is_str_subclass(self) -> None:
        assert isinstance(HallucinationRisk.LOW, str)


# ---------------------------------------------------------------------------
# CaseResult — basic construction
# ---------------------------------------------------------------------------


class TestCaseResult:
    def test_basic_instantiation(self) -> None:
        r = make_result()
        assert r.case_name == "test_case"
        assert r.category == BenchmarkCategory.TEMPORAL
        assert r.score == 0.9
        assert r.pass_threshold == 0.7
        assert r.passed is True
        assert r.error is None

    def test_default_pass_threshold(self) -> None:
        r = CaseResult(
            case_name="x", category=BenchmarkCategory.TEMPORAL,
            score=0.9, passed=True,
        )
        assert r.pass_threshold == 0.7

    def test_default_details_empty(self) -> None:
        assert make_result().details == {}

    def test_details_preserved(self) -> None:
        r = CaseResult(
            case_name="x",
            category=BenchmarkCategory.CONTRADICTION,
            score=0.5,
            pass_threshold=0.7,
            passed=False,
            details={"retrieved_rank": 3, "expected_rank": 1},
        )
        assert r.details["retrieved_rank"] == 3

    def test_score_at_zero_and_not_passed(self) -> None:
        r = make_result(score=0.0, pass_threshold=0.7, passed=False)
        assert r.score == 0.0
        assert r.passed is False

    def test_score_at_one_and_passed(self) -> None:
        r = make_result(score=1.0, pass_threshold=0.7, passed=True)
        assert r.score == 1.0

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_result(score=-0.01)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_result(score=1.01)

    def test_frozen(self) -> None:
        r = make_result()
        with pytest.raises(Exception):
            r.score = 0.5  # type: ignore[misc]

    # --- Error invariant ---

    def test_error_forces_score_zero_and_not_passed(self) -> None:
        r = make_result(score=0.0, passed=False, error="TimeoutError: timed out")
        assert r.error == "TimeoutError: timed out"
        assert r.score == 0.0
        assert r.passed is False

    def test_error_with_nonzero_score_rejected(self) -> None:
        with pytest.raises(ValidationError, match="score must be 0.0"):
            CaseResult(
                case_name="x", category=BenchmarkCategory.TEMPORAL,
                score=0.5, pass_threshold=0.7, passed=False,
                error="boom",
            )

    def test_error_with_passed_true_rejected(self) -> None:
        with pytest.raises(ValidationError, match="passed must be False"):
            CaseResult(
                case_name="x", category=BenchmarkCategory.TEMPORAL,
                score=0.0, pass_threshold=0.7, passed=True,
                error="boom",
            )

    # --- Passed consistency invariant ---

    def test_inconsistent_passed_false_when_above_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent"):
            CaseResult(
                case_name="x", category=BenchmarkCategory.TEMPORAL,
                score=0.9, pass_threshold=0.7, passed=False,  # should be True
            )

    def test_inconsistent_passed_true_when_below_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent"):
            CaseResult(
                case_name="x", category=BenchmarkCategory.TEMPORAL,
                score=0.5, pass_threshold=0.7, passed=True,  # should be False
            )

    def test_score_exactly_at_threshold_is_passing(self) -> None:
        r = CaseResult(
            case_name="x", category=BenchmarkCategory.TEMPORAL,
            score=0.7, pass_threshold=0.7, passed=True,
        )
        assert r.passed is True

    # --- from_case factory ---

    def test_from_case_normal_passing(self) -> None:
        case = _MinimalCase()
        r = CaseResult.from_case(case, score=0.8)
        assert r.case_name == case.name
        assert r.category == case.category
        assert r.score == 0.8
        assert r.pass_threshold == 0.7
        assert r.passed is True
        assert r.error is None

    def test_from_case_below_threshold_not_passed(self) -> None:
        case = _MinimalCase()
        r = CaseResult.from_case(case, score=0.5)
        assert r.passed is False

    def test_from_case_at_threshold_boundary_passes(self) -> None:
        case = _MinimalCase()
        r = CaseResult.from_case(case, score=0.7)
        assert r.passed is True

    def test_from_case_error_forces_zero_score(self) -> None:
        case = _MinimalCase()
        r = CaseResult.from_case(case, score=0.9, error="adapter crashed")
        assert r.score == 0.0
        assert r.passed is False
        assert r.error == "adapter crashed"

    def test_from_case_captures_custom_pass_threshold(self) -> None:
        case = _HighThresholdCase()
        r = CaseResult.from_case(case, score=0.8)
        assert r.pass_threshold == 0.9
        assert r.passed is False  # 0.8 < 0.9

    def test_from_case_with_details(self) -> None:
        case = _MinimalCase()
        r = CaseResult.from_case(case, score=0.9, details={"retrieved": ["id_1"]})
        assert r.details == {"retrieved": ["id_1"]}


# ---------------------------------------------------------------------------
# CategoryScore
# ---------------------------------------------------------------------------


class TestCategoryScore:
    def test_from_results_empty(self) -> None:
        cs = CategoryScore.from_results(BenchmarkCategory.TEMPORAL, [])
        assert cs.score == 0.0
        assert cs.pass_rate == 0.0
        assert cs.results == ()

    def test_from_results_single(self) -> None:
        cs = CategoryScore.from_results(
            BenchmarkCategory.TEMPORAL,
            [make_result(score=0.8, passed=True)],
        )
        assert cs.score == pytest.approx(0.8)
        assert cs.pass_rate == 1.0

    def test_from_results_averaging(self) -> None:
        results = [
            make_result(score=0.6, passed=False),
            make_result(score=1.0, passed=True),
        ]
        cs = CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)
        assert cs.score == pytest.approx(0.8)
        assert cs.pass_rate == pytest.approx(0.5)

    def test_from_results_all_failing(self) -> None:
        results = [make_result(score=0.3, passed=False) for _ in range(3)]
        cs = CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)
        assert cs.pass_rate == 0.0

    def test_from_results_preserves_results_tuple(self) -> None:
        results = [make_result(score=0.9)]
        cs = CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)
        assert len(cs.results) == 1
        assert cs.results[0].score == 0.9

    def test_from_results_rejects_wrong_category(self) -> None:
        results = [
            make_result(score=0.9, category=BenchmarkCategory.TEMPORAL),
            make_result(score=0.8, category=BenchmarkCategory.CONTRADICTION),
        ]
        with pytest.raises(ValueError, match="wrong category"):
            CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)

    def test_from_results_rejects_multiple_wrong_categories(self) -> None:
        results = [
            make_result(score=0.5, passed=False, category=BenchmarkCategory.MULTIHOP),
            make_result(score=0.5, passed=False, category=BenchmarkCategory.IMPORTANCE),
        ]
        with pytest.raises(ValueError, match="wrong category"):
            CategoryScore.from_results(BenchmarkCategory.TEMPORAL, results)

    def test_score_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            CategoryScore(category=BenchmarkCategory.TEMPORAL, score=1.5, pass_rate=0.5)

    def test_pass_rate_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            CategoryScore(category=BenchmarkCategory.TEMPORAL, score=0.5, pass_rate=-0.1)

    def test_frozen(self) -> None:
        cs = make_category_score()
        with pytest.raises(Exception):
            cs.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def _five_category_scores(self) -> list[CategoryScore]:
        return [
            make_category_score(BenchmarkCategory.TEMPORAL, score=0.9),
            make_category_score(BenchmarkCategory.CONTRADICTION, score=0.8),
            make_category_score(BenchmarkCategory.MULTIHOP, score=0.7),
            make_category_score(BenchmarkCategory.IMPORTANCE, score=0.6),
            make_category_score(BenchmarkCategory.CROSS_TYPE, score=0.5),
        ]

    def test_from_category_scores_empty(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", [])
        assert report.overall_score == 0.0
        assert report.hallucination_risk == HallucinationRisk.CRITICAL
        assert report.categories == ()

    def test_from_category_scores_equal_weights(self) -> None:
        # mean of 0.9, 0.8, 0.7, 0.6, 0.5 = 0.7
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        assert report.overall_score == pytest.approx(0.7, abs=0.001)
        assert report.hallucination_risk == HallucinationRisk.MEDIUM

    def test_from_category_scores_custom_weights(self) -> None:
        scores = [
            make_category_score(BenchmarkCategory.TEMPORAL, score=1.0),
            make_category_score(BenchmarkCategory.CONTRADICTION, score=0.0),
        ]
        # temporal=9, contradiction=1 → overall = (1.0×9 + 0.0×1) / 10 = 0.9
        report = BenchmarkReport.from_category_scores(
            "memory",
            scores,
            weights={BenchmarkCategory.TEMPORAL: 9.0, BenchmarkCategory.CONTRADICTION: 1.0},
        )
        assert report.overall_score == pytest.approx(0.9, abs=0.001)
        assert report.hallucination_risk == HallucinationRisk.LOW

    def test_custom_weights_missing_category_defaults_to_1(self) -> None:
        scores = [
            make_category_score(BenchmarkCategory.TEMPORAL, score=1.0),
            make_category_score(BenchmarkCategory.CONTRADICTION, score=0.0),
        ]
        # Only temporal in weights; contradiction defaults to 1.0
        # → (1.0×1 + 0.0×1) / 2 = 0.5
        report = BenchmarkReport.from_category_scores(
            "memory",
            scores,
            weights={BenchmarkCategory.TEMPORAL: 1.0},
        )
        assert report.overall_score == pytest.approx(0.5, abs=0.001)

    def test_zero_total_weight_raises(self) -> None:
        scores = [make_category_score(BenchmarkCategory.TEMPORAL, score=0.9)]
        with pytest.raises(ValueError, match="total weight must be"):
            BenchmarkReport.from_category_scores(
                "memory",
                scores,
                weights={BenchmarkCategory.TEMPORAL: 0.0},
            )

    def test_negative_weight_causing_zero_total_raises(self) -> None:
        scores = [
            make_category_score(BenchmarkCategory.TEMPORAL, score=0.9),
            make_category_score(BenchmarkCategory.CONTRADICTION, score=0.5),
        ]
        with pytest.raises(ValueError, match="total weight must be"):
            BenchmarkReport.from_category_scores(
                "memory",
                scores,
                weights={
                    BenchmarkCategory.TEMPORAL: -1.0,
                    BenchmarkCategory.CONTRADICTION: -1.0,
                },
            )

    def test_scores_by_category(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        by_cat = report.scores_by_category
        assert by_cat[BenchmarkCategory.TEMPORAL] == pytest.approx(0.9)
        assert by_cat[BenchmarkCategory.CROSS_TYPE] == pytest.approx(0.5)

    def test_scores_by_category_empty(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", [])
        assert report.scores_by_category == {}

    def test_backend_name_stored(self) -> None:
        report = BenchmarkReport.from_category_scores("qdrant", self._five_category_scores())
        assert report.backend_name == "qdrant"

    def test_report_id_auto_generated(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        assert len(report.report_id) == 36

    def test_two_reports_have_different_ids(self) -> None:
        a = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        b = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        assert a.report_id != b.report_id

    def test_run_at_is_utc_aware(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        assert report.run_at.tzinfo is not None

    def test_categories_tuple_length(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        assert len(report.categories) == 5

    def test_frozen(self) -> None:
        report = BenchmarkReport.from_category_scores("memory", self._five_category_scores())
        with pytest.raises(Exception):
            report.backend_name = "other"  # type: ignore[misc]

    def test_risk_low(self) -> None:
        scores = [make_category_score(BenchmarkCategory.TEMPORAL, score=0.9)]
        assert (
            BenchmarkReport.from_category_scores("m", scores).hallucination_risk
            == HallucinationRisk.LOW
        )

    def test_risk_critical(self) -> None:
        scores = [make_category_score(BenchmarkCategory.TEMPORAL, score=0.3)]
        assert (
            BenchmarkReport.from_category_scores("m", scores).hallucination_risk
            == HallucinationRisk.CRITICAL
        )
