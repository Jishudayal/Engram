"""Tests for HealthScorer — freshness_score and provenance_completeness (Step 4.1)."""

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
# compute() — not yet implemented
# ---------------------------------------------------------------------------


class TestComputeNotImplemented:
    @pytest.mark.asyncio
    async def test_compute_raises_not_implemented(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        scorer = HealthScorer()
        adapter = InMemoryAdapter()
        with pytest.raises(NotImplementedError, match="Step 4.4"):
            await scorer.compute("agent-1", adapter)
