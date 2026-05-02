"""Tests for HealthScorer — freshness_score, provenance_completeness, contradiction_score."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from engram.core.constants import MemoryStatus, SourceType
from engram.core.health import HealthScorer
from engram.core.models import Memory, ProvenanceRecord

# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------


def make_memory(
    *,
    age_days: float = 0.0,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    with_provenance: bool = False,
    expires_at: datetime | None = None,
    agent_id: str = "agent-1",
) -> Memory:
    updated_at = datetime.now(UTC) - timedelta(days=age_days)
    m = Memory(
        agent_id=agent_id,
        text="test memory",
        importance=importance,
        status=status,
        updated_at=updated_at,
        expires_at=expires_at,
    )
    if with_provenance:
        pr = ProvenanceRecord(memory_id=m.memory_id, source_type=SourceType.DOCUMENT)
        m.attach_provenance(pr)
    return m


# ---------------------------------------------------------------------------
# freshness_score — validation guards
# ---------------------------------------------------------------------------


class TestFreshnessScoreValidation:
    def test_zero_half_life_raises(self) -> None:
        scorer = HealthScorer()
        scorer.FRESHNESS_HALF_LIFE_DAYS = 0.0
        with pytest.raises(ValueError, match="FRESHNESS_HALF_LIFE_DAYS must be > 0"):
            scorer.freshness_score([make_memory()])

    def test_negative_half_life_raises(self) -> None:
        scorer = HealthScorer()
        scorer.FRESHNESS_HALF_LIFE_DAYS = -5.0
        with pytest.raises(ValueError, match="FRESHNESS_HALF_LIFE_DAYS must be > 0"):
            scorer.freshness_score([make_memory()])

    def test_threshold_above_one_raises(self) -> None:
        scorer = HealthScorer()
        scorer.STALE_FRESHNESS_THRESHOLD = 1.1
        with pytest.raises(ValueError, match="STALE_FRESHNESS_THRESHOLD must be in"):
            scorer.freshness_score([make_memory()])

    def test_threshold_below_zero_raises(self) -> None:
        scorer = HealthScorer()
        scorer.STALE_FRESHNESS_THRESHOLD = -0.1
        with pytest.raises(ValueError, match="STALE_FRESHNESS_THRESHOLD must be in"):
            scorer.freshness_score([make_memory()])

    def test_threshold_at_boundaries_accepted(self) -> None:
        scorer_zero = HealthScorer()
        scorer_zero.STALE_FRESHNESS_THRESHOLD = 0.0
        score, stale_ids = scorer_zero.freshness_score([make_memory(age_days=90.0)])
        assert len(stale_ids) == 0  # nothing is < 0.0

        scorer_one = HealthScorer()
        scorer_one.STALE_FRESHNESS_THRESHOLD = 1.0
        _, stale_ids = scorer_one.freshness_score([make_memory(age_days=0.0)])
        assert len(stale_ids) == 1  # 1.0 is not < 1.0... wait, fresh memory has freshness ≈ 1.0
        # freshness of a brand-new memory ≈ 1.0, threshold = 1.0 → f < 1.0 is True only if <1
        # A brand-new memory has freshness slightly < 1.0 due to execution time,
        # so stale_ids will have 1 entry.


# ---------------------------------------------------------------------------
# freshness_score — edge cases
# ---------------------------------------------------------------------------


class TestFreshnessScoreEdgeCases:
    def test_empty_collection_returns_full_freshness_and_empty_stale_ids(self) -> None:
        score, stale_ids = HealthScorer().freshness_score([])
        assert score == 1.0
        assert stale_ids == []

    def test_only_superseded_memories_treated_as_empty(self) -> None:
        memories = [make_memory(status=MemoryStatus.SUPERSEDED) for _ in range(3)]
        score, stale_ids = HealthScorer().freshness_score(memories)
        assert score == 1.0
        assert stale_ids == []

    def test_only_archived_memories_treated_as_empty(self) -> None:
        memories = [make_memory(status=MemoryStatus.ARCHIVED)]
        score, stale_ids = HealthScorer().freshness_score(memories)
        assert score == 1.0
        assert stale_ids == []

    def test_expired_memory_excluded(self) -> None:
        expired = make_memory(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        score, stale_ids = HealthScorer().freshness_score([expired])
        assert score == 1.0
        assert stale_ids == []

    def test_score_always_in_unit_interval(self) -> None:
        memories = [make_memory(age_days=0.0), make_memory(age_days=180.0)]
        score, _ = HealthScorer().freshness_score(memories)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# freshness_score — decay formula
# ---------------------------------------------------------------------------


class TestFreshnessScoreDecayFormula:
    def test_brand_new_memory_scores_near_one(self) -> None:
        score, stale_ids = HealthScorer().freshness_score([make_memory(age_days=0.0)])
        assert score == pytest.approx(1.0, abs=0.01)
        assert len(stale_ids) == 0

    def test_memory_at_half_life_scores_near_0_5(self) -> None:
        # Use 29.9 days — just under the half-life so freshness is just above 0.5.
        # Exact boundary behaviour at 30.0 is ambiguous due to execution timing.
        score, stale_ids = HealthScorer().freshness_score([make_memory(age_days=29.9)])
        assert score == pytest.approx(0.5, abs=0.02)
        assert len(stale_ids) == 0

    def test_memory_one_day_past_half_life_is_stale(self) -> None:
        _, stale_ids = HealthScorer().freshness_score([make_memory(age_days=31.0)])
        assert len(stale_ids) == 1

    def test_memory_at_two_half_lives_scores_0_25(self) -> None:
        # exp(-2 * ln2) == 0.25
        score, stale_ids = HealthScorer().freshness_score([make_memory(age_days=60.0)])
        assert score == pytest.approx(0.25, abs=0.01)
        assert len(stale_ids) == 1

    def test_memory_at_three_half_lives_scores_0_125(self) -> None:
        score, stale_ids = HealthScorer().freshness_score([make_memory(age_days=90.0)])
        assert score == pytest.approx(0.125, abs=0.01)
        assert len(stale_ids) == 1

    def test_future_updated_at_clamped_to_one(self) -> None:
        # updated_at in the future → negative age → raw value > 1 → clamped to 1.
        future_memory = make_memory(age_days=-10.0)
        score, stale_ids = HealthScorer().freshness_score([future_memory])
        assert score == pytest.approx(1.0, abs=0.01)
        assert len(stale_ids) == 0


# ---------------------------------------------------------------------------
# freshness_score — importance weighting
# ---------------------------------------------------------------------------


class TestFreshnessScoreWeighting:
    def test_high_importance_fresh_memory_pulls_score_up(self) -> None:
        fresh = make_memory(age_days=0.0, importance=0.9)
        stale = make_memory(age_days=90.0, importance=0.1)
        score, _ = HealthScorer().freshness_score([fresh, stale])
        # Weighted toward fresh (0.9 weight on ~1.0, 0.1 weight on ~0.125)
        # Expected ≈ (1.0 * 0.9 + 0.125 * 0.1) / 1.0 ≈ 0.913
        assert score > 0.85

    def test_high_importance_stale_memory_drags_score_down(self) -> None:
        stale = make_memory(age_days=90.0, importance=0.9)
        fresh = make_memory(age_days=0.0, importance=0.1)
        score, _ = HealthScorer().freshness_score([stale, fresh])
        # Weighted toward stale: expected ≈ (0.125 * 0.9 + 1.0 * 0.1) / 1.0 ≈ 0.213
        assert score < 0.3

    def test_equal_importance_gives_unweighted_mean(self) -> None:
        fresh = make_memory(age_days=0.0, importance=0.5)
        two_halflives = make_memory(age_days=60.0, importance=0.5)
        score, _ = HealthScorer().freshness_score([fresh, two_halflives])
        # Unweighted mean of ~1.0 and ~0.25 = ~0.625
        assert score == pytest.approx(0.625, abs=0.02)

    def test_all_zero_importance_uses_unweighted_mean(self) -> None:
        m1 = make_memory(age_days=0.0, importance=0.0)
        m2 = make_memory(age_days=60.0, importance=0.0)
        score, _ = HealthScorer().freshness_score([m1, m2])
        # Unweighted mean of ~1.0 and ~0.25 = ~0.625
        assert score == pytest.approx(0.625, abs=0.02)


# ---------------------------------------------------------------------------
# freshness_score — stale IDs
# ---------------------------------------------------------------------------


class TestFreshnessScoreStaleIds:
    def test_stale_ids_count_matches_expectation(self) -> None:
        fresh = make_memory(age_days=5.0)
        stale1 = make_memory(age_days=45.0)
        stale2 = make_memory(age_days=90.0)
        _, stale_ids = HealthScorer().freshness_score([fresh, stale1, stale2])
        assert len(stale_ids) == 2

    def test_stale_ids_contain_correct_memory_ids(self) -> None:
        fresh = make_memory(age_days=5.0)
        stale = make_memory(age_days=60.0)
        _, stale_ids = HealthScorer().freshness_score([fresh, stale])
        assert stale.memory_id in stale_ids
        assert fresh.memory_id not in stale_ids

    def test_superseded_memories_not_in_stale_ids(self) -> None:
        active = make_memory(age_days=5.0)
        superseded = make_memory(age_days=90.0, status=MemoryStatus.SUPERSEDED)
        _, stale_ids = HealthScorer().freshness_score([active, superseded])
        assert len(stale_ids) == 0
        assert superseded.memory_id not in stale_ids

    def test_all_fresh_gives_empty_stale_ids(self) -> None:
        memories = [make_memory(age_days=1.0) for _ in range(5)]
        _, stale_ids = HealthScorer().freshness_score(memories)
        assert stale_ids == []

    def test_all_stale_gives_full_id_list(self) -> None:
        memories = [make_memory(age_days=60.0) for _ in range(4)]
        _, stale_ids = HealthScorer().freshness_score(memories)
        assert len(stale_ids) == 4
        for m in memories:
            assert m.memory_id in stale_ids

    def test_stale_ids_returns_list_not_tuple(self) -> None:
        _, stale_ids = HealthScorer().freshness_score([make_memory(age_days=60.0)])
        assert isinstance(stale_ids, list)


# ---------------------------------------------------------------------------
# freshness_score — custom configuration
# ---------------------------------------------------------------------------


class TestFreshnessScoreCustomConfig:
    def test_custom_half_life_changes_decay_rate(self) -> None:
        scorer = HealthScorer()
        scorer.FRESHNESS_HALF_LIFE_DAYS = 7.0

        # Use 6.9 days — just under the half-life so freshness is just above 0.5.
        score, stale_ids = scorer.freshness_score([make_memory(age_days=6.9)])
        assert score == pytest.approx(0.5, abs=0.02)
        assert len(stale_ids) == 0

    def test_custom_threshold_changes_stale_ids(self) -> None:
        scorer = HealthScorer()
        scorer.STALE_FRESHNESS_THRESHOLD = 0.8

        # At age=10 days: freshness = exp(-10 * ln2 / 30) ≈ 0.794 < 0.8 → stale
        m = make_memory(age_days=10.0)
        _, stale_ids = scorer.freshness_score([m])
        assert m.memory_id in stale_ids

    def test_default_instance_has_expected_constants(self) -> None:
        scorer = HealthScorer()
        assert scorer.FRESHNESS_HALF_LIFE_DAYS == 30.0
        assert scorer.STALE_FRESHNESS_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# provenance_completeness — edge cases
# ---------------------------------------------------------------------------


class TestProvenanceCompletenessEdgeCases:
    def test_empty_collection_returns_one(self) -> None:
        assert HealthScorer().provenance_completeness([]) == 1.0

    def test_only_superseded_memories_treated_as_empty(self) -> None:
        memories = [make_memory(status=MemoryStatus.SUPERSEDED, with_provenance=False)]
        assert HealthScorer().provenance_completeness(memories) == 1.0

    def test_expired_memory_excluded(self) -> None:
        expired = make_memory(
            with_provenance=False,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        assert HealthScorer().provenance_completeness([expired]) == 1.0


# ---------------------------------------------------------------------------
# provenance_completeness — scoring
# ---------------------------------------------------------------------------


class TestProvenanceCompletenessScoring:
    def test_all_with_provenance_returns_one(self) -> None:
        memories = [make_memory(with_provenance=True) for _ in range(3)]
        assert HealthScorer().provenance_completeness(memories) == pytest.approx(1.0)

    def test_none_with_provenance_returns_zero(self) -> None:
        memories = [make_memory(with_provenance=False) for _ in range(3)]
        assert HealthScorer().provenance_completeness(memories) == pytest.approx(0.0)

    def test_half_with_provenance_returns_0_5(self) -> None:
        with_prov = [make_memory(with_provenance=True) for _ in range(2)]
        without_prov = [make_memory(with_provenance=False) for _ in range(2)]
        score = HealthScorer().provenance_completeness(with_prov + without_prov)
        assert score == pytest.approx(0.5)

    def test_one_in_three_with_provenance(self) -> None:
        memories = [
            make_memory(with_provenance=True),
            make_memory(with_provenance=False),
            make_memory(with_provenance=False),
        ]
        assert HealthScorer().provenance_completeness(memories) == pytest.approx(1 / 3, abs=0.01)

    def test_superseded_not_counted_in_denominator(self) -> None:
        active_with = make_memory(with_provenance=True)
        superseded_without = make_memory(
            with_provenance=False, status=MemoryStatus.SUPERSEDED
        )
        score = HealthScorer().provenance_completeness([active_with, superseded_without])
        # Only active counted: 1 with provenance / 1 active = 1.0
        assert score == pytest.approx(1.0)

    def test_flagged_memory_excluded_from_denominator(self) -> None:
        # FLAGGED is not ACTIVE → is_usable() returns False → excluded
        active_with = make_memory(with_provenance=True)
        flagged_without = make_memory(
            with_provenance=False, status=MemoryStatus.FLAGGED
        )
        score = HealthScorer().provenance_completeness([active_with, flagged_without])
        assert score == pytest.approx(1.0)

    def test_single_active_without_provenance(self) -> None:
        score = HealthScorer().provenance_completeness([make_memory(with_provenance=False)])
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# contradiction_score — shared test vectors
#
# All vectors are 3-dimensional. Norms are exactly 1.0, so cosine similarity
# equals the dot product. Verified values:
#
#   cosine(V_X, V_X_NEAR)  = 0.9  — above default threshold (0.82) → pair
#   cosine(V_X, V_Y)       = 0.0  — below threshold → no pair
#   cosine(V_X, V_Y_NEAR)  = 0.0  — below threshold → no pair
#   cosine(V_X_NEAR, V_Y)  = sqrt(0.19) ≈ 0.436  — below threshold → no pair
#   cosine(V_X_NEAR, V_Y_NEAR) = sqrt(0.19)*0.9 ≈ 0.392 — below threshold
#   cosine(V_Y, V_Y_NEAR)  = 0.9  — above threshold → pair
# ---------------------------------------------------------------------------

_SQ19 = math.sqrt(0.19)  # sqrt(1 - 0.9²)

V_X: list[float] = [1.0, 0.0, 0.0]
V_X_NEAR: list[float] = [0.9, _SQ19, 0.0]  # cosine with V_X = 0.9
V_Y: list[float] = [0.0, 1.0, 0.0]
V_Y_NEAR: list[float] = [0.0, 0.9, _SQ19]  # cosine with V_Y = 0.9
V_ZERO: list[float] = [0.0, 0.0, 0.0]


def make_embedded_memory(
    embedding: list[float],
    *,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    agent_id: str = "agent-1",
) -> Memory:
    return Memory(agent_id=agent_id, text="test memory", embedding=embedding, status=status)


# ---------------------------------------------------------------------------
# contradiction_score — validation
# ---------------------------------------------------------------------------


class TestContradictionScoreValidation:
    def test_threshold_above_one_raises(self) -> None:
        scorer = HealthScorer()
        scorer.CONTRADICTION_CLUSTER_THRESHOLD = 1.1
        with pytest.raises(ValueError, match="CONTRADICTION_CLUSTER_THRESHOLD"):
            scorer.contradiction_score([make_embedded_memory(V_X)])

    def test_threshold_below_minus_one_raises(self) -> None:
        scorer = HealthScorer()
        scorer.CONTRADICTION_CLUSTER_THRESHOLD = -1.1
        with pytest.raises(ValueError, match="CONTRADICTION_CLUSTER_THRESHOLD"):
            scorer.contradiction_score([make_embedded_memory(V_X)])

    def test_threshold_at_boundaries_accepted(self) -> None:
        for boundary in (-1.0, 1.0):
            scorer = HealthScorer()
            scorer.CONTRADICTION_CLUSTER_THRESHOLD = boundary
            score, pairs, coverage = scorer.contradiction_score(
                [make_embedded_memory(V_X), make_embedded_memory(V_X_NEAR)]
            )
            assert isinstance(score, float)

    def test_mixed_dimensions_raise(self) -> None:
        m1 = make_embedded_memory([1.0, 0.0])       # 2D
        m2 = make_embedded_memory([1.0, 0.0, 0.0])  # 3D
        with pytest.raises(ValueError, match="same dimension"):
            HealthScorer().contradiction_score([m1, m2])

    def test_uniform_dimension_accepted(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        score, pairs, coverage = HealthScorer().contradiction_score([m1, m2])
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# contradiction_score — edge cases
# ---------------------------------------------------------------------------


class TestContradictionScoreEdgeCases:
    def test_empty_collection_returns_zero(self) -> None:
        score, pairs, coverage = HealthScorer().contradiction_score([])
        assert score == 0.0
        assert pairs == []
        assert coverage == 1.0  # vacuously covered

    def test_no_embeddings_returns_zero(self) -> None:
        memories = [make_memory() for _ in range(3)]  # no embeddings
        score, pairs, coverage = HealthScorer().contradiction_score(memories)
        assert score == 0.0
        assert pairs == []
        assert coverage == 0.0  # 0 embedded / 3 active

    def test_single_embedded_memory_returns_zero(self) -> None:
        score, pairs, coverage = HealthScorer().contradiction_score(
            [make_embedded_memory(V_X)]
        )
        assert score == 0.0
        assert pairs == []

    def test_superseded_memories_excluded(self) -> None:
        active = make_embedded_memory(V_X)
        superseded = make_embedded_memory(V_X_NEAR, status=MemoryStatus.SUPERSEDED)
        score, pairs, coverage = HealthScorer().contradiction_score([active, superseded])
        assert score == 0.0
        assert pairs == []

    def test_zero_vector_embedding_excluded(self) -> None:
        valid = make_embedded_memory(V_X)
        zero_vec = make_embedded_memory(V_ZERO)
        score, pairs, coverage = HealthScorer().contradiction_score([valid, zero_vec])
        assert score == 0.0
        assert pairs == []

    def test_mixed_embedded_and_no_embedding(self) -> None:
        with_emb = make_embedded_memory(V_X)
        without_emb = make_memory()  # embedding=None
        score, pairs, coverage = HealthScorer().contradiction_score(
            [with_emb, without_emb]
        )
        assert score == 0.0
        assert pairs == []

    def test_score_in_unit_interval(self) -> None:
        memories = [make_embedded_memory(V_X), make_embedded_memory(V_X_NEAR)]
        score, _, _coverage = HealthScorer().contradiction_score(memories)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# contradiction_score — embedding coverage
# ---------------------------------------------------------------------------


class TestContradictionScoreEmbeddingCoverage:
    def test_all_have_embeddings_coverage_is_one(self) -> None:
        memories = [make_embedded_memory(V_X), make_embedded_memory(V_Y)]
        _, _, coverage = HealthScorer().contradiction_score(memories)
        assert coverage == pytest.approx(1.0)

    def test_none_have_embeddings_coverage_is_zero(self) -> None:
        memories = [make_memory() for _ in range(4)]
        _, _, coverage = HealthScorer().contradiction_score(memories)
        assert coverage == pytest.approx(0.0)

    def test_half_have_embeddings_coverage_is_half(self) -> None:
        memories = [
            make_embedded_memory(V_X),
            make_embedded_memory(V_X_NEAR),
            make_memory(),  # no embedding
            make_memory(),  # no embedding
        ]
        _, _, coverage = HealthScorer().contradiction_score(memories)
        assert coverage == pytest.approx(0.5)

    def test_coverage_excludes_zero_norm_embeddings(self) -> None:
        # 3 active: 1 valid, 1 zero-norm (excluded), 1 no embedding (excluded)
        valid = make_embedded_memory(V_X)
        zero = make_embedded_memory(V_ZERO)
        no_emb = make_memory()
        _, _, coverage = HealthScorer().contradiction_score([valid, zero, no_emb])
        assert coverage == pytest.approx(1 / 3, abs=0.01)

    def test_coverage_excludes_superseded_from_denominator(self) -> None:
        # Superseded memories are not ACTIVE, so they don't appear in active_all.
        active_with = make_embedded_memory(V_X)
        superseded_with = make_embedded_memory(V_X_NEAR, status=MemoryStatus.SUPERSEDED)
        _, _, coverage = HealthScorer().contradiction_score(
            [active_with, superseded_with]
        )
        # Only 1 active memory (with valid embedding) → coverage = 1/1 = 1.0
        assert coverage == pytest.approx(1.0)

    def test_empty_collection_coverage_is_one(self) -> None:
        _, _, coverage = HealthScorer().contradiction_score([])
        assert coverage == 1.0

    def test_coverage_is_float(self) -> None:
        _, _, coverage = HealthScorer().contradiction_score(
            [make_embedded_memory(V_X), make_embedded_memory(V_Y)]
        )
        assert isinstance(coverage, float)


# ---------------------------------------------------------------------------
# contradiction_score — scoring formula
# ---------------------------------------------------------------------------


class TestContradictionScoreFormula:
    def test_two_similar_memories_score_one(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert score == pytest.approx(1.0)
        assert len(pairs) == 1

    def test_two_dissimilar_memories_score_zero(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_Y)
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert score == 0.0
        assert pairs == []

    def test_one_pair_among_three_scores_two_thirds(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        m3 = make_embedded_memory(V_Y)
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2, m3])
        assert score == pytest.approx(2 / 3, abs=0.01)
        assert len(pairs) == 1

    def test_two_independent_pairs_all_involved(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        m3 = make_embedded_memory(V_Y)
        m4 = make_embedded_memory(V_Y_NEAR)
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2, m3, m4])
        assert score == pytest.approx(1.0)
        assert len(pairs) == 2

    def test_two_pairs_plus_isolated_scores_four_fifths(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        m3 = make_embedded_memory(V_Y)
        m4 = make_embedded_memory(V_Y_NEAR)
        m5 = make_embedded_memory([0.0, 0.0, 1.0])  # orthogonal to all above
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2, m3, m4, m5])
        assert score == pytest.approx(0.8, abs=0.01)
        assert len(pairs) == 2

    def test_identical_embeddings_form_pair(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X)
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert score == pytest.approx(1.0)
        assert len(pairs) == 1

    def test_opposite_embeddings_do_not_form_pair(self) -> None:
        # cosine(V_X, -V_X) = -1.0 — not a near-duplicate
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory([-1.0, 0.0, 0.0])
        score, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert score == 0.0
        assert pairs == []


# ---------------------------------------------------------------------------
# contradiction_score — pair contents
# ---------------------------------------------------------------------------


class TestContradictionScorePairs:
    def test_pairs_contain_correct_memory_ids(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        _, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert len(pairs) == 1
        pair_ids = set(pairs[0])
        assert m1.memory_id in pair_ids
        assert m2.memory_id in pair_ids

    def test_pairs_are_list_of_two_tuples(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        _, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert isinstance(pairs, list)
        assert isinstance(pairs[0], tuple)
        assert len(pairs[0]) == 2

    def test_no_duplicate_pairs(self) -> None:
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        _, pairs, _ = HealthScorer().contradiction_score([m1, m2])
        assert len(pairs) == len(set(pairs))

    def test_memory_not_paired_with_itself(self) -> None:
        memories = [make_embedded_memory(V_X), make_embedded_memory(V_X_NEAR)]
        _, pairs, _ = HealthScorer().contradiction_score(memories)
        for a, b in pairs:
            assert a != b

    def test_inactive_memory_id_not_in_pairs(self) -> None:
        active = make_embedded_memory(V_X)
        superseded = make_embedded_memory(V_X_NEAR, status=MemoryStatus.SUPERSEDED)
        _, pairs, _ = HealthScorer().contradiction_score([active, superseded])
        assert all(superseded.memory_id not in (a, b) for a, b in pairs)


# ---------------------------------------------------------------------------
# contradiction_score — custom threshold
# ---------------------------------------------------------------------------


class TestContradictionScoreCustomThreshold:
    def test_stricter_threshold_produces_fewer_pairs(self) -> None:
        scorer = HealthScorer()
        scorer.CONTRADICTION_CLUSTER_THRESHOLD = 0.95
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)  # cosine ≈ 0.9 < 0.95
        score, pairs, _ = scorer.contradiction_score([m1, m2])
        assert score == 0.0
        assert pairs == []

    def test_looser_threshold_produces_more_pairs(self) -> None:
        scorer = HealthScorer()
        scorer.CONTRADICTION_CLUSTER_THRESHOLD = 0.3
        m1 = make_embedded_memory(V_X)
        m2 = make_embedded_memory(V_X_NEAR)
        m3 = make_embedded_memory(V_Y)
        _, default_pairs, _ = HealthScorer().contradiction_score([m1, m2, m3])
        _, loose_pairs, _ = scorer.contradiction_score([m1, m2, m3])
        assert len(loose_pairs) >= len(default_pairs)

    def test_default_threshold_matches_constant(self) -> None:
        from engram.core.constants import CLUSTER_SIMILARITY_THRESHOLD
        assert HealthScorer().CONTRADICTION_CLUSTER_THRESHOLD == CLUSTER_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# confidence_accuracy_gap — helpers
# ---------------------------------------------------------------------------


async def _store_memories(adapter: object, memories: list) -> None:
    """Store a list of Memory objects in the adapter."""
    for m in memories:
        await adapter.store(m)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# confidence_accuracy_gap — validation
# ---------------------------------------------------------------------------


class TestConfidenceAccuracyGapValidation:
    async def test_probe_count_zero_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        with pytest.raises(ValueError, match="probe_count must be >= 1"):
            await HealthScorer().confidence_accuracy_gap(
                "agent-1", adapter, probe_count=0
            )

    async def test_probe_count_negative_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        with pytest.raises(ValueError, match="probe_count must be >= 1"):
            await HealthScorer().confidence_accuracy_gap(
                "agent-1", adapter, probe_count=-1
            )

    async def test_top_k_zero_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            await HealthScorer().confidence_accuracy_gap("agent-1", adapter, top_k=0)

    async def test_top_k_negative_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            await HealthScorer().confidence_accuracy_gap("agent-1", adapter, top_k=-5)


# ---------------------------------------------------------------------------
# confidence_accuracy_gap — edge cases
# ---------------------------------------------------------------------------


class TestConfidenceAccuracyGapEdgeCases:
    async def test_empty_agent_returns_zero(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        gap, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert gap == 0.0
        assert n == 0

    async def test_no_embedded_memories_returns_zero(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(adapter, [make_memory(), make_memory(), make_memory()])
        gap, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert gap == 0.0
        assert n == 0

    async def test_single_embedded_active_returns_zero(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(adapter, [make_embedded_memory(V_X)])
        gap, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert gap == 0.0
        assert n == 0

    async def test_superseded_embedded_not_counted_as_probe(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [
                make_embedded_memory(V_X, status=MemoryStatus.SUPERSEDED),
                make_embedded_memory(V_X, status=MemoryStatus.SUPERSEDED),
            ],
        )
        gap, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert gap == 0.0
        assert n == 0

    async def test_zero_norm_embedding_not_counted(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        # One valid, one zero-norm — only 1 valid embedded → below threshold
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_ZERO)],
        )
        gap, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert gap == 0.0
        assert n == 0


# ---------------------------------------------------------------------------
# confidence_accuracy_gap — num_probed
# ---------------------------------------------------------------------------


class TestConfidenceAccuracyGapNumProbed:
    async def test_num_probed_equals_active_embedded_count(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X) for _ in range(3)],
        )
        _, n = await HealthScorer().confidence_accuracy_gap(
            "agent-1", adapter, probe_count=10, top_k=2
        )
        assert n == 3

    async def test_sampling_caps_probe_count(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X) for _ in range(10)],
        )
        _, n = await HealthScorer().confidence_accuracy_gap(
            "agent-1", adapter, probe_count=3, top_k=2
        )
        assert n == 3

    async def test_returns_int_num_probed(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_Y)],
        )
        _, n = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert isinstance(n, int)


# ---------------------------------------------------------------------------
# confidence_accuracy_gap — score formula
# ---------------------------------------------------------------------------


class TestConfidenceAccuracyGapScoreFormula:
    async def test_all_active_identical_embeddings_gives_zero_gap(self) -> None:
        """Healthy system: all retrieved memories are ACTIVE → precision=1.0.

        With identical unit-norm embeddings, cosine similarity = 1.0 for all
        pairs, so retrieval_confidence = 1.0 and measured_precision = 1.0,
        giving gap = 0.0 per probe.
        """
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X) for _ in range(4)],
        )
        gap, n = await HealthScorer().confidence_accuracy_gap(
            "agent-1", adapter, probe_count=10, top_k=2
        )
        assert gap == pytest.approx(0.0, abs=1e-9)
        assert n == 4

    async def test_superseded_in_results_gives_large_gap(self) -> None:
        """Silent killer: retrieval is confident but results are superseded.

        SUPERSEDED memories are stored first so they fill the top-k results
        before any ACTIVE memory (all embeddings identical, scores all = 1.0,
        stable sort preserves insertion order).
        """
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        # Store SUPERSEDED first — they appear first in equal-score results.
        superseded = [
            make_embedded_memory(V_X, status=MemoryStatus.SUPERSEDED)
            for _ in range(5)
        ]
        active = [make_embedded_memory(V_X) for _ in range(2)]
        await _store_memories(adapter, superseded + active)

        # top_k=3 < 5 superseded memories → all top-k results are SUPERSEDED
        gap, n = await HealthScorer().confidence_accuracy_gap(
            "agent-1", adapter, probe_count=10, top_k=3
        )
        assert gap == pytest.approx(1.0, abs=1e-9)
        assert n == 2

    async def test_gap_score_in_unit_interval(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_Y)],
        )
        gap, _ = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert 0.0 <= gap <= 1.0

    async def test_gap_is_float(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_X)],
        )
        gap, _ = await HealthScorer().confidence_accuracy_gap("agent-1", adapter)
        assert isinstance(gap, float)

    async def test_isolated_memories_still_probed(self) -> None:
        """Orthogonal memories have zero cosine, but are still valid probes."""
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        # m1 and m2 are orthogonal — cosine = 0.0, but the probe still runs.
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_Y)],
        )
        gap, n = await HealthScorer().confidence_accuracy_gap(
            "agent-1", adapter, top_k=1
        )
        # Both ACTIVE, score = 0.0 → retrieval_confidence = 0.0, precision = 1.0
        # gap = |0.0 - 1.0| = 1.0. This shows low score ≠ small gap.
        assert gap == pytest.approx(1.0, abs=1e-9)
        assert n == 2


# ---------------------------------------------------------------------------
# compute() — basic
# ---------------------------------------------------------------------------


class TestComputeBasic:
    async def test_compute_returns_health_score(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.models import HealthScore

        adapter = InMemoryAdapter()
        await _store_memories(adapter, [make_embedded_memory(V_X), make_embedded_memory(V_Y)])
        result = await HealthScorer().compute("agent-1", adapter)
        assert isinstance(result, HealthScore)

    async def test_compute_empty_collection_is_healthy(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import RiskLevel

        adapter = InMemoryAdapter()
        result = await HealthScorer().compute("agent-1", adapter)
        # All empty-collection defaults are "perfect" → score = 1.0
        assert result.score == pytest.approx(1.0, abs=1e-9)
        assert result.risk_level == RiskLevel.LOW

    async def test_compute_agent_id_set_correctly(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        result = await HealthScorer().compute("my-agent", adapter)
        assert result.agent_id == "my-agent"

    async def test_compute_pending_review_is_empty_tuple(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.pending_review == ()


# ---------------------------------------------------------------------------
# compute() — counts and metrics
# ---------------------------------------------------------------------------


class TestComputeCounts:
    async def test_total_memories_counts_active_only(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [
                make_memory(),
                make_memory(),
                make_memory(status=MemoryStatus.SUPERSEDED),
            ],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.total_memories == 2

    async def test_stale_count_from_freshness(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_memory(age_days=1.0), make_memory(age_days=60.0)],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.stale_count == 1

    async def test_conflict_count_from_similar_pairs(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        # Identical embeddings → cosine = 1.0 > 0.82 threshold → 1 pair
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_X)],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.conflict_count == 1

    async def test_avg_importance_correct(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_memory(importance=0.2), make_memory(importance=0.8)],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.avg_importance == pytest.approx(0.5, abs=1e-9)

    async def test_oldest_age_days_set(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_memory(age_days=5.0), make_memory(age_days=10.0)],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.oldest_memory_age_days == pytest.approx(10.0, abs=0.01)

    async def test_oldest_age_days_none_for_empty(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.oldest_memory_age_days is None

    async def test_empty_collection_zero_avg_importance(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.avg_importance == 0.0


# ---------------------------------------------------------------------------
# compute() — composite score and risk level
# ---------------------------------------------------------------------------


class TestComputeScore:
    async def test_score_in_unit_interval(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(adapter, [make_embedded_memory(V_X), make_embedded_memory(V_Y)])
        result = await HealthScorer().compute("agent-1", adapter)
        assert 0.0 <= result.score <= 1.0

    async def test_score_uses_inverted_contradiction(self) -> None:
        """Four identical-embedding memories → contradiction_score=1.0 pulls composite down.

        Setup:
          freshness  = 1.0 (age=0)    provenance = 0.0 (no records)
          contradiction = 1.0 (all 4 form pairs)   cag = 0.0 (all ACTIVE, score=1.0)
          score = 0.25*1.0 + 0.25*0.0 + 0.25*(1-1.0) + 0.25*(1-0.0) = 0.50
        """
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X) for _ in range(4)],
        )
        result = await HealthScorer().compute("agent-1", adapter)
        assert result.score == pytest.approx(0.50, abs=0.01)

    async def test_risk_level_consistent_with_score(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import RiskLevel

        adapter = InMemoryAdapter()
        await _store_memories(adapter, [make_embedded_memory(V_X), make_embedded_memory(V_Y)])
        scorer = HealthScorer()
        result = await scorer.compute("agent-1", adapter)

        if result.score >= scorer.RISK_THRESHOLD_LOW:
            assert result.risk_level == RiskLevel.LOW
        elif result.score >= scorer.RISK_THRESHOLD_MEDIUM:
            assert result.risk_level == RiskLevel.MEDIUM
        elif result.score >= scorer.RISK_THRESHOLD_HIGH:
            assert result.risk_level == RiskLevel.HIGH
        else:
            assert result.risk_level == RiskLevel.CRITICAL

    async def test_custom_weights_change_score(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        # Two identical embeddings → contradiction_score = 1.0
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_X)],
        )
        scorer_default = HealthScorer()
        scorer_heavy = HealthScorer()
        # Heavily penalise contradiction — contradiction component gets more weight
        scorer_heavy.SCORE_WEIGHT_CONTRADICTION = 0.70
        scorer_heavy.SCORE_WEIGHT_FRESHNESS = 0.10
        scorer_heavy.SCORE_WEIGHT_PROVENANCE = 0.10
        scorer_heavy.SCORE_WEIGHT_CONFIDENCE_GAP = 0.10

        default_result = await scorer_default.compute("agent-1", adapter)
        heavy_result = await scorer_heavy.compute("agent-1", adapter)
        # Higher contradiction weight → lower composite when contradiction_score is high
        assert heavy_result.score < default_result.score

    async def test_custom_risk_thresholds_applied(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import RiskLevel

        adapter = InMemoryAdapter()
        await _store_memories(
            adapter,
            [make_embedded_memory(V_X), make_embedded_memory(V_X)],
        )
        scorer = HealthScorer()
        scorer.RISK_THRESHOLD_LOW = 0.99
        scorer.RISK_THRESHOLD_MEDIUM = 0.90
        scorer.RISK_THRESHOLD_HIGH = 0.80
        result = await scorer.compute("agent-1", adapter)
        assert result.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# compute() — composite config validation
# ---------------------------------------------------------------------------


class TestComputeConfigValidation:
    async def test_negative_weight_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        scorer = HealthScorer()
        scorer.SCORE_WEIGHT_FRESHNESS = -0.1
        with pytest.raises(ValueError, match="SCORE_WEIGHT_FRESHNESS"):
            await scorer.compute("agent-1", InMemoryAdapter())

    async def test_all_zero_weights_raise(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        scorer = HealthScorer()
        scorer.SCORE_WEIGHT_FRESHNESS = 0.0
        scorer.SCORE_WEIGHT_PROVENANCE = 0.0
        scorer.SCORE_WEIGHT_CONTRADICTION = 0.0
        scorer.SCORE_WEIGHT_CONFIDENCE_GAP = 0.0
        with pytest.raises(ValueError, match="at least one SCORE_WEIGHT"):
            await scorer.compute("agent-1", InMemoryAdapter())

    async def test_risk_threshold_out_of_range_raises(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        scorer = HealthScorer()
        scorer.RISK_THRESHOLD_LOW = 1.1
        with pytest.raises(ValueError, match="RISK_THRESHOLD_LOW"):
            await scorer.compute("agent-1", InMemoryAdapter())

    async def test_risk_thresholds_must_be_ordered(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        scorer = HealthScorer()
        scorer.RISK_THRESHOLD_LOW = 0.60
        scorer.RISK_THRESHOLD_MEDIUM = 0.80
        with pytest.raises(ValueError, match="risk thresholds must be ordered"):
            await scorer.compute("agent-1", InMemoryAdapter())


# ---------------------------------------------------------------------------
# Engram.health() — delegation
# ---------------------------------------------------------------------------


class TestEngramHealth:
    async def test_health_returns_health_score(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.models import HealthScore
        from engram.engram import Engram

        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        result = await eng.health("agent-1")
        assert isinstance(result, HealthScore)

    async def test_health_uses_correct_agent_id(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.engram import Engram

        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        result = await eng.health("my-agent")
        assert result.agent_id == "my-agent"

    async def test_health_sees_stored_memories(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.engram import Engram

        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        await eng.store(make_memory(agent_id="agent-1"))
        await eng.store(make_memory(agent_id="agent-1"))
        result = await eng.health("agent-1")
        assert result.total_memories == 2


# ---------------------------------------------------------------------------
# compute() — pending_review population (Step 5.3)
# ---------------------------------------------------------------------------


class TestComputePendingReview:
    async def test_pending_review_empty_when_no_conflicts(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        await adapter.store(make_memory(agent_id="agent-1"))
        scorer = HealthScorer()
        result = await scorer.compute("agent-1", adapter)
        assert result.pending_review == ()

    async def test_pending_review_populated_from_adapter(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import ConflictType, ResolutionStatus
        from engram.core.models import ConflictRecord

        adapter = InMemoryAdapter()
        await adapter.store(make_memory(agent_id="agent-1"))
        conflict = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.92,
        )
        await adapter.store_conflict(conflict)

        scorer = HealthScorer()
        result = await scorer.compute("agent-1", adapter)
        assert len(result.pending_review) == 1
        assert result.pending_review[0].conflict_id == conflict.conflict_id

    async def test_pending_review_excludes_resolved_conflicts(self) -> None:
        from datetime import UTC, datetime

        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import ConflictType, ResolutionStatus
        from engram.core.models import ConflictRecord

        adapter = InMemoryAdapter()
        await adapter.store(make_memory(agent_id="agent-1"))

        pending = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.92,
        )
        resolved = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-c",
            memory_b_id="mem-d",
            conflict_type=ConflictType.PARTIAL_CONFLICT,
            confidence=0.85,
        )
        resolved.resolve(ResolutionStatus.AUTO_RESOLVED)

        await adapter.store_conflict(pending)
        await adapter.store_conflict(resolved)

        scorer = HealthScorer()
        result = await scorer.compute("agent-1", adapter)
        assert len(result.pending_review) == 1
        assert result.pending_review[0].conflict_id == pending.conflict_id

    async def test_conflict_count_uses_confirmed_conflicts_when_available(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.constants import ConflictType
        from engram.core.models import ConflictRecord

        adapter = InMemoryAdapter()
        await adapter.store(make_memory(agent_id="agent-1"))
        conflict = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.92,
        )
        await adapter.store_conflict(conflict)

        scorer = HealthScorer()
        result = await scorer.compute("agent-1", adapter)
        assert result.conflict_count == 1

    async def test_pending_review_empty_when_adapter_not_implemented(self) -> None:
        from typing import Any

        from engram.adapters.base import AbstractAdapter
        from engram.core.constants import MemoryStatus, ResolutionStatus
        from engram.core.models import Memory, SearchResult

        class _NoConflictAdapter(AbstractAdapter):
            @property
            def backend_name(self) -> str:
                return "no-conflict"

            async def store(self, memory: Memory) -> None:
                pass

            async def update(self, memory: Memory) -> None:
                pass

            async def delete(self, agent_id: str, memory_id: str) -> bool:
                return False

            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return None

            async def search(
                self,
                agent_id: str,
                query_embedding: list[float],
                *,
                top_k: int = 10,
                score_threshold: float | None = None,
                filters: dict[str, Any] | None = None,
            ) -> list[SearchResult]:
                return []

            async def list_all(
                self,
                agent_id: str,
                *,
                status: MemoryStatus | None = None,
                limit: int | None = None,
                offset: int = 0,
            ) -> list[Memory]:
                return []

            async def close(self) -> None:
                pass

        scorer = HealthScorer()
        result = await scorer.compute("agent-1", _NoConflictAdapter())
        assert result.pending_review == ()
