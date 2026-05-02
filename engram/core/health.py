"""Health scoring engine for Engram.

Step 4 sub-steps:
  4.1 — HealthScorer skeleton; freshness_score, provenance_completeness
  4.2 — contradiction_score (embedding similarity clustering)
  4.3 — confidence_accuracy_gap (probe-based measurement)
  4.4 — compose compute(), wire Engram.health()
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.adapters.base import AbstractAdapter
    from engram.core.models import HealthScore, Memory

__all__ = ["HealthScorer"]

logger = logging.getLogger(__name__)

# Used in the half-life decay formula: freshness = exp(-age * LN2 / half_life)
# gives exactly 0.5 at age == half_life.
_LN2 = math.log(2.0)

_SECONDS_PER_DAY: float = 86_400.0


class HealthScorer:
    """Computes health metrics for an agent's memory collection.

    Each sub-metric is a standalone method so individual signals can be
    computed without running the full pipeline. Call compute() for the
    combined HealthScore snapshot.

    Class-level constants control scoring behaviour and can be overridden
    per-instance or via subclassing:

        scorer = HealthScorer()
        scorer.FRESHNESS_HALF_LIFE_DAYS = 14.0  # tighter freshness window
    """

    # Age in days at which a memory's freshness value decays to exactly 0.5.
    # 30 days balances fast-changing domains (product info, policies) against
    # slow-changing ones (company culture, stable facts). Override for your domain.
    FRESHNESS_HALF_LIFE_DAYS: float = 30.0

    # Freshness values strictly below this threshold count as "stale".
    # Matches the half-life by construction: at age == half_life, freshness == 0.5.
    # Change only if "stale" should mean something other than "past the half-life".
    STALE_FRESHNESS_THRESHOLD: float = 0.5

    # ------------------------------------------------------------------
    # Step 4.1 — Freshness and provenance scorers
    # ------------------------------------------------------------------

    def freshness_score(self, memories: list[Memory]) -> tuple[float, list[str]]:
        """Compute (freshness_score, stale_memory_ids) for a memory collection.

        Only ACTIVE, non-expired memories are considered (vacuously fresh
        when none exist). The score is the importance-weighted mean of
        per-memory freshness values via exponential half-life decay on
        updated_at age:

            freshness(m) = exp(-age_days * ln(2) / FRESHNESS_HALF_LIFE_DAYS)

        This gives 1.0 at age=0 and 0.5 at age=FRESHNESS_HALF_LIFE_DAYS.
        stale_memory_ids contains the memory_ids of ACTIVE memories whose
        freshness is strictly below STALE_FRESHNESS_THRESHOLD.

        Falls back to unweighted mean when all importances are zero.
        Returns (1.0, []) for an empty collection.

        Raises ValueError if FRESHNESS_HALF_LIFE_DAYS <= 0 or
        STALE_FRESHNESS_THRESHOLD is outside [0, 1].
        """
        half_life = self.FRESHNESS_HALF_LIFE_DAYS
        threshold = self.STALE_FRESHNESS_THRESHOLD

        if half_life <= 0:
            raise ValueError(f"FRESHNESS_HALF_LIFE_DAYS must be > 0, got {half_life!r}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"STALE_FRESHNESS_THRESHOLD must be in [0, 1], got {threshold!r}"
            )

        active = [m for m in memories if m.is_usable()]
        if not active:
            logger.debug("freshness_score: no active memories — returning (1.0, [])")
            return 1.0, []

        now = datetime.now(UTC)

        freshness_values: list[float] = []
        for m in active:
            age_days = (now - m.updated_at).total_seconds() / _SECONDS_PER_DAY
            # Clamp to [0, 1]: a future updated_at (negative age) would otherwise
            # produce a value > 1.0.
            freshness_values.append(min(1.0, math.exp(-age_days * _LN2 / half_life)))

        stale_ids = [
            m.memory_id for m, f in zip(active, freshness_values) if f < threshold
        ]

        total_importance = sum(m.importance for m in active)
        if total_importance > 0.0:
            weighted = sum(f * m.importance for f, m in zip(freshness_values, active))
            score = weighted / total_importance
        else:
            score = sum(freshness_values) / len(freshness_values)

        score = max(0.0, min(1.0, score))

        oldest_age = max(
            (now - m.updated_at).total_seconds() / _SECONDS_PER_DAY for m in active
        )
        logger.debug(
            "freshness_score: total=%d active=%d stale=%d oldest_age=%.1fd score=%.3f",
            len(memories),
            len(active),
            len(stale_ids),
            oldest_age,
            score,
        )

        return score, stale_ids

    def provenance_completeness(self, memories: list[Memory]) -> float:
        """Fraction of ACTIVE, non-expired memories that have a provenance record.

        This is an *operational* completeness signal — it measures provenance
        coverage of the currently-usable knowledge base. Superseded, flagged,
        archived, and expired memories are excluded from both numerator and
        denominator. For a full audit-trail check across all records (including
        historical ones), iterate over all statuses explicitly.

        Returns 1.0 for an empty collection (vacuously complete).
        """
        active = [m for m in memories if m.is_usable()]
        if not active:
            logger.debug("provenance_completeness: no active memories — returning 1.0")
            return 1.0

        with_provenance = sum(1 for m in active if m.provenance is not None)
        score = with_provenance / len(active)

        logger.debug(
            "provenance_completeness: active=%d with_provenance=%d score=%.3f",
            len(active),
            with_provenance,
            score,
        )

        return score

    # ------------------------------------------------------------------
    # Steps 4.2 and 4.3 arrive here (contradiction_score,
    # confidence_accuracy_gap) — stubs until those sub-steps land.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Step 4.4 — Full health snapshot (not yet implemented)
    # ------------------------------------------------------------------

    async def compute(self, agent_id: str, adapter: AbstractAdapter) -> HealthScore:
        """Compute a full health snapshot for an agent's memory collection.

        Not yet implemented — arrives in Step 4.4.
        """
        raise NotImplementedError("compute() is not yet implemented (Step 4.4)")
