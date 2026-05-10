"""MemoryEval benchmark scorer — orchestrates case execution and metric rollup.

Does not parallelize, retry, or persist results across runs. One waist of the
hourglass: everything between the adapter and the report flows through here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from memoryeval.benchmark import ALL_CASES
from memoryeval.case import TestCase
from memoryeval.types import (
    BenchmarkCategory,
    BenchmarkReport,
    CaseResult,
    CategoryScore,
)

__all__ = ["BenchmarkScorer"]

logger = logging.getLogger(__name__)

# Callback called after each case completes: (result, index, total)
ProgressHook = Callable[[CaseResult, int, int], Awaitable[None]]

_DEFAULT_CASE_TIMEOUT: float = 30.0


class BenchmarkScorer:
    """Runs MemoryEval benchmark test cases and produces a BenchmarkReport.

    Each case is executed in isolation: setup → run → score → teardown.
    Teardown is guaranteed even when run() or score() raise. A failed setup
    is also caught and reported as an error result (the case is still counted).

    Usage::

        from engram.adapters.memory import InMemoryAdapter
        from memoryeval.scorer import BenchmarkScorer

        adapter = InMemoryAdapter()
        scorer  = BenchmarkScorer(adapter, backend_name="in-memory")
        report  = await scorer.run()

        print(report.overall_score)          # e.g. 0.82
        print(report.hallucination_risk)     # e.g. HallucinationRisk.MEDIUM
    """

    def __init__(
        self,
        adapter: Any,
        *,
        backend_name: str,
        cases: list[type[TestCase]] | None = None,
        case_timeout: float = _DEFAULT_CASE_TIMEOUT,
    ) -> None:
        """
        Args:
            adapter:      Any object implementing the AbstractAdapter protocol.
            backend_name: Human-readable label stored in the BenchmarkReport
                          (e.g. "qdrant", "chroma", "in-memory").
            cases:        Override the full case list. Defaults to ALL_CASES
                          (all 50 cases). Pass a subset for faster targeted runs.
                          Passing an empty list is accepted here but will raise
                          ValueError at run() time.
            case_timeout: Per-case wall-clock timeout in seconds applied to
                          setup() and run() individually. Defaults to 30 s.
                          Tear down also receives this timeout but its failure
                          is logged rather than propagated.
        """
        active = cases if cases is not None else ALL_CASES
        names = [cls.name for cls in active]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate case names in scorer: {duplicates}")

        self._adapter = adapter
        self._backend_name = backend_name
        self._cases = active
        self._case_timeout = case_timeout

    # ------------------------------------------------------------------
    # Single-case execution
    # ------------------------------------------------------------------

    async def run_case(self, case: TestCase) -> CaseResult:
        """Execute one test case and return a CaseResult.

        Lifecycle:
        1. ``setup()`` — store fixtures; if this raises or times out, return an
           error result immediately (teardown is not called — nothing was stored).
        2. ``run()`` and ``score()`` — execute the scenario; any exception or
           timeout → error result.
        3. ``teardown()`` — always called after a successful setup, regardless
           of what happened in run()/score(). Teardown failures are logged as
           warnings and do not affect the returned CaseResult.
        """
        try:
            await asyncio.wait_for(
                case.setup(self._adapter), timeout=self._case_timeout
            )
        except TimeoutError:
            return CaseResult.from_case(
                case, error=f"setup: timed out after {self._case_timeout}s"
            )
        except Exception as exc:
            return CaseResult.from_case(case, error=f"setup: {exc}")

        try:
            raw = await asyncio.wait_for(
                case.run(self._adapter), timeout=self._case_timeout
            )
            score = case.score(raw)
            return CaseResult.from_case(case, score=score)
        except TimeoutError:
            return CaseResult.from_case(
                case, error=f"run: timed out after {self._case_timeout}s"
            )
        except Exception as exc:
            return CaseResult.from_case(case, error=str(exc))
        finally:
            try:
                await asyncio.wait_for(
                    case.teardown(self._adapter), timeout=self._case_timeout
                )
            except TimeoutError:
                logger.warning(
                    "teardown timed out for case %s after %.1fs",
                    case.name, self._case_timeout,
                )
            except Exception as teardown_exc:
                logger.warning(
                    "teardown failed for case %s: %s",
                    case.name, teardown_exc,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Full-suite execution
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        categories: list[BenchmarkCategory] | None = None,
        on_case_complete: ProgressHook | None = None,
    ) -> BenchmarkReport:
        """Run all cases (or a filtered subset) and return a BenchmarkReport.

        Cases are executed sequentially — adapters are not assumed to be
        thread-safe or safe for concurrent access. Each case gets its own
        agent_id namespace so fixtures never bleed between cases.

        Args:
            categories:       Limit the run to these categories. Pass ``None``
                              (the default) to run all cases in ``self._cases``.
            on_case_complete: Optional async callback invoked after each case
                              completes. Signature: ``async def hook(result,
                              i, total) -> None`` where ``i`` is 1-based.

        Returns:
            A frozen BenchmarkReport. Category order follows BenchmarkCategory
            enum declaration order (temporal → contradiction → multihop →
            importance → cross_type).

        Raises:
            ValueError: if the combined filter (cases + categories) produces
                        no active cases — prevents a silent CRITICAL report from
                        a typo'd category name or an empty custom case list.
        """
        active = [
            cls for cls in self._cases
            if categories is None or cls.category in categories
        ]

        if not active:
            raise ValueError(
                f"No cases match the filter: "
                f"categories={[c.value for c in (categories or [])]!r}, "
                f"cases in scorer={len(self._cases)}"
            )

        results: list[CaseResult] = []
        total = len(active)
        for i, cls in enumerate(active, start=1):
            result = await self.run_case(cls())
            results.append(result)
            if on_case_complete is not None:
                await on_case_complete(result, i, total)

        by_category: dict[BenchmarkCategory, list[CaseResult]] = {}
        for result in results:
            by_category.setdefault(result.category, []).append(result)

        # Preserve canonical enum order regardless of run order
        category_scores = [
            CategoryScore.from_results(cat, by_category[cat])
            for cat in BenchmarkCategory
            if cat in by_category
        ]

        return BenchmarkReport.from_category_scores(self._backend_name, category_scores)
