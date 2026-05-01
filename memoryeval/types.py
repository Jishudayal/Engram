"""Output types for MemoryEval benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    # Only needed for the CaseResult.from_case() type signature.
    # Importing at runtime would create a circular dependency:
    #   types.py → case.py → types.py
    from memoryeval.case import TestCase


__all__ = [
    "BenchmarkCategory",
    "HallucinationRisk",
    "CaseResult",
    "CategoryScore",
    "BenchmarkReport",
]


class BenchmarkCategory(StrEnum):
    """The five evaluation dimensions MemoryEval measures."""

    TEMPORAL = "temporal"
    CONTRADICTION = "contradiction"
    MULTIHOP = "multihop"
    IMPORTANCE = "importance"
    CROSS_TYPE = "cross_type"


class HallucinationRisk(StrEnum):
    """Overall reliability risk derived from the benchmark score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> HallucinationRisk:
        """Map a 0–1 composite score to a risk tier.

        Thresholds are calibrated against the static comparison table in
        memoryeval/README.md. Do not change without updating that table.
        """
        if score >= 0.85:
            return cls.LOW
        if score >= 0.70:
            return cls.MEDIUM
        if score >= 0.50:
            return cls.HIGH
        return cls.CRITICAL


class CaseResult(BaseModel):
    """Outcome of a single test case execution.

    Immutable. Three invariants are enforced by the model validator:

    - When ``error`` is set: ``score`` must be 0.0 and ``passed`` must be False.
    - When ``error`` is None: ``passed`` must equal ``score >= pass_threshold``.

    Prefer ``CaseResult.from_case()`` over direct construction — it computes
    ``score``, ``passed``, and ``pass_threshold`` correctly in one call.
    """

    model_config = ConfigDict(frozen=True)

    case_name: str
    category: BenchmarkCategory
    score: float = Field(ge=0.0, le=1.0)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def _enforce_invariants(self) -> CaseResult:
        if self.error is not None:
            if self.score != 0.0:
                raise ValueError("score must be 0.0 when error is set")
            if self.passed:
                raise ValueError("passed must be False when error is set")
        else:
            expected_passed = self.score >= self.pass_threshold
            if self.passed != expected_passed:
                raise ValueError(
                    f"passed={self.passed!r} is inconsistent with "
                    f"score={self.score} >= pass_threshold={self.pass_threshold} "
                    f"(expected passed={expected_passed!r})"
                )
        return self

    @classmethod
    def from_case(
        cls,
        case: TestCase,
        *,
        score: float = 0.0,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> CaseResult:
        """Build a CaseResult from a TestCase and its raw score.

        Enforces all three invariants automatically:
        - If ``error`` is given, ``score`` is forced to 0.0 and ``passed`` to False.
        - ``passed`` is always ``score >= case.pass_threshold``.
        - ``pass_threshold`` is captured from the case for auditability.

        Typical scorer usage::

            try:
                raw = await case.run(adapter)
                s   = case.score(raw)
                return CaseResult.from_case(case, score=s, details={"raw": raw})
            except Exception as exc:
                return CaseResult.from_case(case, error=str(exc))
        """
        effective_score = 0.0 if error else score
        passed = error is None and effective_score >= case.pass_threshold
        return cls(
            case_name=case.name,
            category=case.category,
            pass_threshold=case.pass_threshold,
            score=effective_score,
            passed=passed,
            details=details or {},
            error=error,
        )


class CategoryScore(BaseModel):
    """Aggregated reliability score for one benchmark category.

    Build via ``CategoryScore.from_results()`` rather than constructing directly.

    ``score``     — arithmetic mean of all CaseResult scores in this category.
    ``pass_rate`` — fraction of tests that individually passed their threshold.
    """

    model_config = ConfigDict(frozen=True)

    category: BenchmarkCategory
    score: float = Field(ge=0.0, le=1.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    results: tuple[CaseResult, ...] = Field(default_factory=tuple)

    @classmethod
    def from_results(
        cls,
        category: BenchmarkCategory,
        results: list[CaseResult],
    ) -> CategoryScore:
        """Build a CategoryScore from a list of test results for one category.

        Raises ``ValueError`` if any result belongs to a different category —
        mixed-category inputs produce plausible but incorrect scores.
        """
        mismatched = [r for r in results if r.category != category]
        if mismatched:
            bad = sorted({r.category.value for r in mismatched})
            raise ValueError(
                f"from_results received {len(mismatched)} result(s) with wrong category; "
                f"expected {category.value!r}, got {bad!r}"
            )
        if not results:
            return cls(category=category, score=0.0, pass_rate=0.0)
        n = len(results)
        score = sum(r.score for r in results) / n
        pass_rate = sum(1 for r in results if r.passed) / n
        return cls(
            category=category,
            score=round(score, 4),
            pass_rate=round(pass_rate, 4),
            results=tuple(results),
        )


class BenchmarkReport(BaseModel):
    """Complete output of a MemoryEval benchmark run.

    Immutable. Build via ``BenchmarkReport.from_category_scores()``.

    Quick access to per-category scores::

        report.scores_by_category[BenchmarkCategory.TEMPORAL]  # → 0.91
    """

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    run_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    backend_name: str
    categories: tuple[CategoryScore, ...]
    overall_score: float = Field(ge=0.0, le=1.0)
    hallucination_risk: HallucinationRisk

    @property
    def scores_by_category(self) -> dict[BenchmarkCategory, float]:
        """Dict of category → score for quick programmatic access."""
        return {cs.category: cs.score for cs in self.categories}

    @property
    def all_results(self) -> tuple[CaseResult, ...]:
        """Flat tuple of every CaseResult, grouped by report category order."""
        return tuple(r for cs in self.categories for r in cs.results)

    @classmethod
    def from_category_scores(
        cls,
        backend_name: str,
        category_scores: list[CategoryScore],
        *,
        weights: dict[BenchmarkCategory, float] | None = None,
    ) -> BenchmarkReport:
        """Build a BenchmarkReport from per-category scores.

        All categories are weighted equally by default. Pass ``weights`` to
        override — values are normalized so they don't need to sum to 1.0.
        Categories absent from ``weights`` receive a weight of 1.0.

        Raises ``ValueError`` if the effective total weight is zero or negative,
        which would produce a nonsensical (or divide-by-zero) overall score.
        """
        if not category_scores:
            return cls(
                backend_name=backend_name,
                categories=(),
                overall_score=0.0,
                hallucination_risk=HallucinationRisk.CRITICAL,
            )

        if weights:
            total_weight = sum(weights.get(cs.category, 1.0) for cs in category_scores)
            if total_weight <= 0:
                raise ValueError(
                    f"total weight must be > 0; got {total_weight}. "
                    "Check that at least one category has a positive weight."
                )
            overall = sum(
                cs.score * weights.get(cs.category, 1.0) / total_weight
                for cs in category_scores
            )
        else:
            overall = sum(cs.score for cs in category_scores) / len(category_scores)

        return cls(
            backend_name=backend_name,
            categories=tuple(category_scores),
            overall_score=round(overall, 4),
            hallucination_risk=HallucinationRisk.from_score(overall),
        )
