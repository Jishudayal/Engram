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
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from engram.core.constants import CLUSTER_SIMILARITY_THRESHOLD, ResolutionStatus

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

    # Cosine similarity above which two ACTIVE memories are flagged as a
    # candidate conflict pair. Sourced from constants.py so a single change
    # propagates to both the health scorer and the Step 5 detector.
    CONTRADICTION_CLUSTER_THRESHOLD: float = CLUSTER_SIMILARITY_THRESHOLD  # 0.82

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
            raise ValueError(f"STALE_FRESHNESS_THRESHOLD must be in [0, 1], got {threshold!r}")

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
            m.memory_id for m, f in zip(active, freshness_values, strict=True) if f < threshold
        ]

        total_importance = sum(m.importance for m in active)
        if total_importance > 0.0:
            weighted = sum(f * m.importance for f, m in zip(freshness_values, active, strict=True))
            score = weighted / total_importance
        else:
            score = sum(freshness_values) / len(freshness_values)

        score = max(0.0, min(1.0, score))

        oldest_age = max((now - m.updated_at).total_seconds() / _SECONDS_PER_DAY for m in active)
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
    # Step 4.2 — Contradiction scorer (embedding similarity clustering)
    # ------------------------------------------------------------------

    def contradiction_score(
        self, memories: list[Memory]
    ) -> tuple[float, list[tuple[str, str]], float]:
        """Compute contradiction score, candidate conflict pairs, and embedding coverage.

        Scans ACTIVE memories that have embeddings for near-duplicate pairs
        using cosine similarity. Two memories form a "candidate pair" when
        their cosine similarity strictly exceeds CONTRADICTION_CLUSTER_THRESHOLD.

        Returns:
          - score: fraction of ACTIVE embedded memories involved in ≥1 candidate
            pair. 0.0 = no near-duplicates; 1.0 = every embedded memory has a
            close neighbour. High values signal unresolved near-duplicates that
            may be contradicting each other. The actual contradiction verdict is
            deferred to the LLM classifier in Step 5.

            Polarity: unlike freshness_score and provenance_completeness (where
            higher = better), higher contradiction_score means MORE near-duplicate
            conflicts, i.e., WORSE health. In the composite score (Step 4.4),
            use (1.0 - contradiction_score) when combining with other metrics.

          - candidate_pairs: list of (memory_id_a, memory_id_b) for each pair
            above threshold, in upper-triangle row-major order.

          - embedding_coverage: fraction of ACTIVE memories that have a valid
            (non-None, non-zero-norm) embedding. Coverage < 1.0 means some active
            memories could not be checked for contradictions; low coverage makes
            the score optimistic. A WARNING is logged when coverage < 0.5.

        ACTIVE memories without embeddings, and embeddings whose L2 norm is
        zero, are excluded from both numerator and denominator of score, but ARE
        counted in the denominator of embedding_coverage.

        Memory complexity is O(n²) in the number of active embedded memories —
        the full K×K similarity matrix is materialised in RAM. For collections
        exceeding ~5 000 memories with embeddings, peak memory can reach several
        hundred MB. Chunked or ANN-based computation is planned as a follow-on.

        Returns (0.0, [], coverage) when fewer than 2 ACTIVE memories have valid
        embeddings.

        Raises ValueError if CONTRADICTION_CLUSTER_THRESHOLD is outside [-1, 1],
        or if active memories carry embeddings of inconsistent dimension.
        """
        cluster_threshold = self.CONTRADICTION_CLUSTER_THRESHOLD
        if not -1.0 <= cluster_threshold <= 1.0:
            raise ValueError(
                f"CONTRADICTION_CLUSTER_THRESHOLD must be in [-1, 1], got {cluster_threshold!r}"
            )

        active_all = [m for m in memories if m.is_usable()]

        # Candidates have a non-None embedding.
        candidates = [m for m in active_all if m.embedding is not None]

        if len(candidates) < 2:
            coverage = len(candidates) / len(active_all) if active_all else 1.0
            logger.debug(
                "contradiction_score: fewer than 2 embedded active memories (%d/%d)"
                " — returning (0.0, [], %.2f)",
                len(candidates),
                len(active_all),
                coverage,
            )
            return 0.0, [], coverage

        # Guard against accidental mixing of different embedding models.
        dims = {len(m.embedding) for m in candidates}  # type: ignore[arg-type]
        if len(dims) > 1:
            raise ValueError(
                f"all embeddings must have the same dimension; got mixed dimensions: {sorted(dims)}"
            )

        # Build float matrix and drop zero-norm rows (all-zeros embeddings).
        matrix = np.array([m.embedding for m in candidates], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        valid_mask = norms > 0.0
        active_embedded = [m for m, v in zip(candidates, valid_mask, strict=True) if v]

        # Embedding coverage over all active memories (not just those with embeddings).
        embedding_coverage = len(active_embedded) / len(active_all) if active_all else 1.0
        if active_all and embedding_coverage < 0.5:
            logger.warning(
                "contradiction_score: only %d/%d active memories have valid embeddings"
                " (%.0f%%) — score may understate true contradictions",
                len(active_embedded),
                len(active_all),
                embedding_coverage * 100,
            )

        if len(active_embedded) < 2:
            logger.debug(
                "contradiction_score: fewer than 2 non-zero embeddings after filtering"
                " — returning (0.0, [], %.2f)",
                embedding_coverage,
            )
            return 0.0, [], embedding_coverage

        # Normalize each row to unit length; dot-product of unit vectors = cosine similarity.
        valid_matrix = matrix[valid_mask]
        valid_norms = np.linalg.norm(valid_matrix, axis=1, keepdims=True)
        normalized = valid_matrix / valid_norms
        # Full K×K similarity matrix — O(n²) memory, see docstring for scale warning.
        sim_matrix = normalized @ normalized.T  # values in [-1, 1]

        # Upper triangle only (k=1 excludes the diagonal) avoids self-pairs and duplicates.
        n = len(active_embedded)
        rows, cols = np.triu_indices(n, k=1)
        sims = sim_matrix[rows, cols]

        above = sims > cluster_threshold
        candidate_rows = rows[above].tolist()
        candidate_cols = cols[above].tolist()

        candidate_pairs = [
            (active_embedded[r].memory_id, active_embedded[c].memory_id)
            for r, c in zip(candidate_rows, candidate_cols, strict=True)
        ]

        involved: set[int] = set(candidate_rows) | set(candidate_cols)
        score = len(involved) / n

        logger.debug(
            "contradiction_score: active_embedded=%d candidate_pairs=%d"
            " involved=%d coverage=%.2f score=%.3f",
            n,
            len(candidate_pairs),
            len(involved),
            embedding_coverage,
            score,
        )

        return score, candidate_pairs, embedding_coverage

    # ------------------------------------------------------------------
    # Step 4.3 — Confidence-accuracy gap scorer
    # ------------------------------------------------------------------

    async def confidence_accuracy_gap(
        self,
        agent_id: str,
        adapter: AbstractAdapter,
        *,
        probe_count: int = 50,
        top_k: int = 5,
    ) -> tuple[float, int]:
        """Measure the gap between retrieval confidence and measured precision@k.

        Unlike the other HealthScorer methods this method is async and performs
        adapter I/O: it fetches all memories for the agent, samples up to
        ``probe_count`` ACTIVE embedded memories as probes, and issues one
        ``adapter.search()`` call per probe (without a status filter, simulating
        naive retrieval).

        For each probe two values are extracted from the top-k results after
        excluding the probe itself:

            retrieval_confidence = mean cosine score of the top-k results
            measured_precision   = fraction of top-k results that are currently
                                   ACTIVE (i.e., would be surfaced as truth)
            per-probe gap        = |retrieval_confidence − measured_precision|

        The score is the mean per-probe gap across all probes that yield at least
        one result after self-exclusion.  Lower is better:

            gap ≈ 0.00  — well-calibrated; high-confidence results are also
                          high-precision
            gap ≥ 0.20  — significant overconfidence; stale/superseded memories
                          are retrieved at high similarity scores
            gap ≥ 0.50  — severe miscalibration; the "silent killer" failure mode

        The "silent killer" this detects: cosine similarity keeps producing
        high-confidence matches, but a growing fraction of those matches are
        SUPERSEDED or FLAGGED — the system looks confident while returning
        stale answers.

        Returns:
          - gap_score in [0, 1]; lower is better.
          - num_probed: probes that yielded ≥ 1 result after self-exclusion.
            Surface alongside gap_score; reliability increases with num_probed.

        Returns (0.0, 0) when fewer than 2 ACTIVE memories have a valid
        (non-None, non-zero-norm) embedding.

        Raises ValueError if probe_count < 1 or top_k < 1.
        """
        if probe_count < 1:
            raise ValueError(f"probe_count must be >= 1, got {probe_count!r}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")

        all_memories = await adapter.list_all(agent_id)
        return await self._confidence_accuracy_gap_from_memories(
            agent_id,
            adapter,
            all_memories,
            probe_count=probe_count,
            top_k=top_k,
        )

    async def _confidence_accuracy_gap_from_memories(
        self,
        agent_id: str,
        adapter: AbstractAdapter,
        all_memories: list[Memory],
        *,
        probe_count: int,
        top_k: int,
    ) -> tuple[float, int]:
        """Compute confidence gap from an already-fetched memory collection."""
        active = [m for m in all_memories if m.is_usable()]
        embedded = [
            m
            for m in active
            if m.embedding is not None
            and float(np.linalg.norm(np.asarray(m.embedding, dtype=np.float64))) > 0.0
        ]

        if len(embedded) < 2:
            logger.debug(
                "confidence_accuracy_gap: fewer than 2 active embedded memories (%d)"
                " for agent %s — returning (0.0, 0)",
                len(embedded),
                agent_id,
            )
            return 0.0, 0

        probes = (
            random.sample(embedded, probe_count) if len(embedded) > probe_count else list(embedded)
        )

        gaps: list[float] = []
        for probe in probes:
            # Request one extra to account for a potential self-hit in results.
            results = await adapter.search(
                agent_id,
                probe.embedding,  # type: ignore[arg-type]
                top_k=top_k + 1,
            )
            results = [r for r in results if r.memory.memory_id != probe.memory_id][:top_k]
            if not results:
                continue

            retrieval_confidence = sum(r.score for r in results) / len(results)
            active_count = sum(1 for r in results if r.memory.is_usable())
            measured_precision = active_count / len(results)
            gaps.append(abs(retrieval_confidence - measured_precision))

        num_probed = len(gaps)
        if num_probed == 0:
            logger.debug(
                "confidence_accuracy_gap: no valid probe results for agent %s — returning (0.0, 0)",
                agent_id,
            )
            return 0.0, 0

        gap_score = max(0.0, min(1.0, sum(gaps) / num_probed))

        logger.debug(
            "confidence_accuracy_gap: agent=%s probes_attempted=%d probes_used=%d"
            " top_k=%d gap=%.3f",
            agent_id,
            len(probes),
            num_probed,
            top_k,
            gap_score,
        )

        return gap_score, num_probed

    # ------------------------------------------------------------------
    # Step 4.4 — Full health snapshot
    # ------------------------------------------------------------------

    # Weights for the composite score produced by compute().
    # contradiction_score and confidence_accuracy_gap are higher-is-worse and
    # are inverted to (1 − metric) before weighting. The composite is clamped
    # to [0, 1] regardless of whether the weights sum to 1.0.
    # Override per-instance to tune scoring for your domain.
    SCORE_WEIGHT_FRESHNESS: float = 0.25
    SCORE_WEIGHT_PROVENANCE: float = 0.25
    SCORE_WEIGHT_CONTRADICTION: float = 0.25
    SCORE_WEIGHT_CONFIDENCE_GAP: float = 0.25

    # Composite score thresholds for risk tier assignment in compute().
    RISK_THRESHOLD_LOW: float = 0.80  # score >= 0.80 → LOW
    RISK_THRESHOLD_MEDIUM: float = 0.60  # score >= 0.60 → MEDIUM
    RISK_THRESHOLD_HIGH: float = 0.40  # score >= 0.40 → HIGH; below → CRITICAL

    def _validate_composite_config(self) -> None:
        """Validate overridable weights and risk thresholds before scoring."""
        weights = {
            "SCORE_WEIGHT_FRESHNESS": self.SCORE_WEIGHT_FRESHNESS,
            "SCORE_WEIGHT_PROVENANCE": self.SCORE_WEIGHT_PROVENANCE,
            "SCORE_WEIGHT_CONTRADICTION": self.SCORE_WEIGHT_CONTRADICTION,
            "SCORE_WEIGHT_CONFIDENCE_GAP": self.SCORE_WEIGHT_CONFIDENCE_GAP,
        }
        for name, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        if sum(weights.values()) <= 0.0:
            raise ValueError("at least one SCORE_WEIGHT_* value must be > 0")

        thresholds = {
            "RISK_THRESHOLD_LOW": self.RISK_THRESHOLD_LOW,
            "RISK_THRESHOLD_MEDIUM": self.RISK_THRESHOLD_MEDIUM,
            "RISK_THRESHOLD_HIGH": self.RISK_THRESHOLD_HIGH,
        }
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if not (self.RISK_THRESHOLD_LOW >= self.RISK_THRESHOLD_MEDIUM >= self.RISK_THRESHOLD_HIGH):
            raise ValueError(
                "risk thresholds must be ordered as "
                "RISK_THRESHOLD_LOW >= RISK_THRESHOLD_MEDIUM >= RISK_THRESHOLD_HIGH"
            )

    async def compute(self, agent_id: str, adapter: AbstractAdapter) -> HealthScore:
        """Compute a full health snapshot for an agent's memory collection.

        Fetches all memories for the agent, computes all four sub-metrics, and
        assembles them into a HealthScore snapshot.

        Sub-metrics and their polarities:
          freshness_score         — higher is healthier
          provenance_completeness — higher is healthier
          contradiction_score     — lower is healthier; inverted to
                                    (1 − contradiction_score) before weighting
          confidence_accuracy_gap — lower is healthier; inverted to
                                    (1 − cag) before weighting

        Composite score formula:
          score = SCORE_WEIGHT_FRESHNESS      × freshness_score
                + SCORE_WEIGHT_PROVENANCE     × provenance_completeness
                + SCORE_WEIGHT_CONTRADICTION  × (1 − contradiction_score)
                + SCORE_WEIGHT_CONFIDENCE_GAP × (1 − confidence_accuracy_gap)
          Clamped to [0, 1].

        Risk tier (RiskLevel):
          score ≥ RISK_THRESHOLD_LOW    (0.80) → LOW
          score ≥ RISK_THRESHOLD_MEDIUM (0.60) → MEDIUM
          score ≥ RISK_THRESHOLD_HIGH   (0.40) → HIGH
          score <  RISK_THRESHOLD_HIGH         → CRITICAL

        HealthScore.total_memories counts only ACTIVE (usable) memories.
        HealthScore.pending_review is empty until Step 5 wires in the LLM
        contradiction classifier.
        """
        from engram.core.constants import RiskLevel
        from engram.core.models import HealthScore as _HealthScore

        self._validate_composite_config()

        all_memories = await adapter.list_all(agent_id)
        active = [m for m in all_memories if m.is_usable()]

        freshness, stale_ids = self.freshness_score(active)
        prov = self.provenance_completeness(active)
        contradiction, pairs, _coverage = self.contradiction_score(active)
        cag, _probed = await self._confidence_accuracy_gap_from_memories(
            agent_id,
            adapter,
            all_memories,
            probe_count=50,
            top_k=5,
        )

        # Fetch confirmed PENDING conflicts for pending_review.
        # Falls back gracefully when the adapter does not implement conflict storage.
        try:
            pending_conflicts = await adapter.list_conflicts(
                agent_id, status=ResolutionStatus.PENDING
            )
        except NotImplementedError:
            logger.debug(
                "compute: adapter %r does not support conflict storage "
                "— pending_review will be empty",
                type(adapter).__name__,
            )
            pending_conflicts = []

        # Use confirmed conflict count when available; embedding-pair count otherwise.
        conflict_count = len(pending_conflicts) if pending_conflicts else len(pairs)

        score = (
            self.SCORE_WEIGHT_FRESHNESS * freshness
            + self.SCORE_WEIGHT_PROVENANCE * prov
            + self.SCORE_WEIGHT_CONTRADICTION * (1.0 - contradiction)
            + self.SCORE_WEIGHT_CONFIDENCE_GAP * (1.0 - cag)
        )
        score = max(0.0, min(1.0, score))

        if score >= self.RISK_THRESHOLD_LOW:
            risk = RiskLevel.LOW
        elif score >= self.RISK_THRESHOLD_MEDIUM:
            risk = RiskLevel.MEDIUM
        elif score >= self.RISK_THRESHOLD_HIGH:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL

        now = datetime.now(UTC)
        avg_importance = sum(m.importance for m in active) / len(active) if active else 0.0
        oldest_age: float | None = (
            max((now - m.updated_at).total_seconds() / _SECONDS_PER_DAY for m in active)
            if active
            else None
        )

        logger.debug(
            "compute: agent=%s total=%d active=%d stale=%d conflicts=%d"
            " pending_review=%d freshness=%.3f provenance=%.3f"
            " contradiction=%.3f cag=%.3f score=%.3f risk=%s",
            agent_id,
            len(all_memories),
            len(active),
            len(stale_ids),
            conflict_count,
            len(pending_conflicts),
            freshness,
            prov,
            contradiction,
            cag,
            score,
            risk,
        )

        return _HealthScore(
            agent_id=agent_id,
            total_memories=len(active),
            stale_count=len(stale_ids),
            conflict_count=conflict_count,
            avg_importance=avg_importance,
            oldest_memory_age_days=oldest_age,
            freshness_score=freshness,
            provenance_completeness=prov,
            contradiction_score=contradiction,
            confidence_accuracy_gap=cag,
            score=score,
            risk_level=risk,
            pending_review=tuple(pending_conflicts),
        )
