"""Tests for HealthScore, ConsolidationAction, ConsolidationPlan, SearchResult (sub-step 1.6)."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engram.core.constants import ActionType, ConflictType, ConsolidationTier, RiskLevel, SourceType
from engram.core.models import (
    ConsolidationAction,
    ConsolidationPlan,
    HealthScore,
    Memory,
    ProvenanceRecord,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def make_health_score(**overrides: object) -> HealthScore:
    defaults: dict = {
        "agent_id": "agent-1",
        "total_memories": 100,
        "stale_count": 10,
        "conflict_count": 2,
        "avg_importance": 0.6,
        "contradiction_score": 0.9,
        "freshness_score": 0.85,
        "confidence_accuracy_gap": 0.05,
        "provenance_completeness": 0.95,
        "score": 0.85,
        "risk_level": RiskLevel.LOW,
    }
    return HealthScore(**{**defaults, **overrides})


def make_action(**overrides: object) -> ConsolidationAction:
    defaults: dict = {
        "agent_id": "agent-1",
        "action_type": ActionType.MERGE,
        "tier": ConsolidationTier.AUTO_MERGE,
        "confidence": 0.95,
        "source_memory_ids": ["mem-a", "mem-b"],
    }
    return ConsolidationAction(**{**defaults, **overrides})


def make_plan(**overrides: object) -> ConsolidationPlan:
    defaults: dict = {
        "agent_id": "agent-1",
        "actions": [make_action()],
    }
    return ConsolidationPlan(**{**defaults, **overrides})


def make_memory(**overrides: object) -> Memory:
    defaults: dict = {"agent_id": "agent-1", "text": "The refund policy is 30 days."}
    return Memory(**{**defaults, **overrides})


def make_search_result(**overrides: object) -> SearchResult:
    defaults: dict = {"memory": make_memory(), "score": 0.92, "rank": 1}
    return SearchResult(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# HealthScore
# ---------------------------------------------------------------------------


class TestHealthScoreInstantiation:
    def test_minimal_required_fields(self) -> None:
        hs = make_health_score()
        assert hs.agent_id == "agent-1"
        assert hs.total_memories == 100
        assert hs.risk_level == RiskLevel.LOW

    def test_score_id_auto_generated(self) -> None:
        hs = make_health_score()
        assert hs.score_id
        assert len(hs.score_id) == 36

    def test_two_scores_get_different_ids(self) -> None:
        assert make_health_score().score_id != make_health_score().score_id

    def test_computed_at_defaults_to_utc_aware(self) -> None:
        hs = make_health_score()
        assert hs.computed_at.tzinfo is not None

    def test_oldest_memory_age_days_defaults_to_none(self) -> None:
        assert make_health_score().oldest_memory_age_days is None

    def test_oldest_memory_age_days_accepted(self) -> None:
        hs = make_health_score(oldest_memory_age_days=365.5)
        assert hs.oldest_memory_age_days == 365.5

    def test_all_risk_levels_accepted(self) -> None:
        for level in RiskLevel:
            hs = make_health_score(risk_level=level)
            assert hs.risk_level == level


class TestHealthScoreValidators:
    def test_blank_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty or whitespace"):
            make_health_score(agent_id="")

    def test_agent_id_is_stripped(self) -> None:
        hs = make_health_score(agent_id="  agent-1  ")
        assert hs.agent_id == "agent-1"

    def test_negative_total_memories_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(total_memories=-1)

    def test_negative_stale_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(stale_count=-1)

    def test_negative_conflict_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(conflict_count=-1)

    def test_stale_count_exceeds_total_raises(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed total_memories"):
            make_health_score(total_memories=5, stale_count=6)

    def test_stale_count_equals_total_accepted(self) -> None:
        hs = make_health_score(total_memories=10, stale_count=10)
        assert hs.stale_count == 10

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(score=-0.1)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(score=1.1)

    def test_score_at_boundaries_accepted(self) -> None:
        assert make_health_score(score=0.0).score == 0.0
        assert make_health_score(score=1.0).score == 1.0

    def test_avg_importance_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(avg_importance=-0.1)
        with pytest.raises(ValidationError):
            make_health_score(avg_importance=1.1)

    def test_negative_oldest_memory_age_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(oldest_memory_age_days=-1.0)

    def test_naive_computed_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_health_score(computed_at=datetime.utcnow())  # noqa: DTZ003

    def test_zero_total_zero_stale_accepted(self) -> None:
        hs = make_health_score(total_memories=0, stale_count=0)
        assert hs.total_memories == 0


class TestHealthScoreFrozen:
    def test_assignment_after_creation_raises(self) -> None:
        hs = make_health_score()
        with pytest.raises(ValidationError):
            hs.score = 0.5  # type: ignore[misc]

    def test_risk_level_immutable(self) -> None:
        hs = make_health_score()
        with pytest.raises(ValidationError):
            hs.risk_level = RiskLevel.CRITICAL  # type: ignore[misc]


class TestHealthScoreSerialization:
    def test_model_dump_round_trip(self) -> None:
        hs = make_health_score(risk_level=RiskLevel.HIGH, score=0.4)
        data = hs.model_dump()
        restored = HealthScore(**data)
        assert restored.score_id == hs.score_id
        assert restored.risk_level == RiskLevel.HIGH
        assert restored.score == 0.4

    def test_risk_level_serializes_as_string(self) -> None:
        data = make_health_score(risk_level=RiskLevel.MEDIUM).model_dump()
        assert data["risk_level"] == "medium"


# ---------------------------------------------------------------------------
# ConsolidationAction
# ---------------------------------------------------------------------------


class TestConsolidationActionInstantiation:
    def test_minimal_required_fields(self) -> None:
        a = make_action()
        assert a.agent_id == "agent-1"
        assert a.action_type == ActionType.MERGE
        assert a.source_memory_ids == ("mem-a", "mem-b")

    def test_action_id_auto_generated(self) -> None:
        a = make_action()
        assert a.action_id
        assert len(a.action_id) == 36

    def test_two_actions_get_different_ids(self) -> None:
        assert make_action().action_id != make_action().action_id

    def test_created_at_defaults_to_utc_aware(self) -> None:
        assert make_action().created_at.tzinfo is not None

    def test_executed_at_defaults_to_none(self) -> None:
        assert make_action().executed_at is None

    def test_result_memory_id_defaults_to_none(self) -> None:
        assert make_action().result_memory_id is None

    def test_reasoning_defaults_to_none(self) -> None:
        assert make_action().reasoning is None


class TestConsolidationActionValidators:
    def test_blank_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty or whitespace"):
            make_action(agent_id="")

    def test_agent_id_is_stripped(self) -> None:
        a = make_action(agent_id="  agent-1  ")
        assert a.agent_id == "agent-1"

    def test_empty_source_memory_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            make_action(source_memory_ids=[])

    def test_blank_string_in_source_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not contain blank"):
            make_action(source_memory_ids=["mem-a", ""])

    def test_single_source_id_accepted(self) -> None:
        a = make_action(source_memory_ids=["mem-a"])
        assert a.source_memory_ids == ("mem-a",)

    def test_confidence_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            make_action(confidence=-0.1)
        with pytest.raises(ValidationError):
            make_action(confidence=1.1)

    def test_blank_result_memory_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_action(result_memory_id="")

    def test_blank_reasoning_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_action(reasoning="   ")

    def test_naive_executed_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_action(executed_at=datetime.utcnow())  # noqa: DTZ003

    def test_executed_at_before_created_at_raises(self) -> None:
        base = datetime.now(UTC)
        with pytest.raises(ValidationError, match="must not be before created_at"):
            make_action(
                created_at=base,
                executed_at=base - timedelta(seconds=1),
            )


class TestConsolidationActionMutation:
    def test_can_set_executed_at(self) -> None:
        a = make_action()
        a.result_memory_id = "mem-merged"  # execution invariant: result must precede executed_at
        now = datetime.now(UTC)
        a.executed_at = now
        assert a.executed_at == now

    def test_can_set_result_memory_id(self) -> None:
        a = make_action()
        a.result_memory_id = "mem-merged"
        assert a.result_memory_id == "mem-merged"

    def test_can_set_reasoning(self) -> None:
        a = make_action()
        a.reasoning = "Memories are near-identical."
        assert a.reasoning == "Memories are near-identical."

    def test_created_at_frozen(self) -> None:
        a = make_action()
        with pytest.raises(ValidationError):
            a.created_at = datetime.now(UTC)  # type: ignore[misc]

    def test_all_action_types_accepted(self) -> None:
        for at in ActionType:
            a = make_action(action_type=at)
            assert a.action_type == at

    def test_all_consolidation_tiers_accepted(self) -> None:
        for tier in ConsolidationTier:
            a = make_action(tier=tier)
            assert a.tier == tier


class TestConsolidationActionSerialization:
    def test_model_dump_round_trip(self) -> None:
        a = make_action(action_type=ActionType.SUPERSEDE, reasoning="Newer fact.")
        data = a.model_dump()
        restored = ConsolidationAction(**data)
        assert restored.action_id == a.action_id
        assert restored.action_type == ActionType.SUPERSEDE
        assert restored.reasoning == "Newer fact."

    def test_action_type_serializes_as_string(self) -> None:
        data = make_action(action_type=ActionType.FLAG).model_dump()
        assert data["action_type"] == "flag"

    def test_tier_serializes_as_string(self) -> None:
        data = make_action(tier=ConsolidationTier.HUMAN_REVIEW).model_dump()
        assert data["tier"] == "human_review"


# ---------------------------------------------------------------------------
# ConsolidationPlan
# ---------------------------------------------------------------------------


class TestConsolidationPlanInstantiation:
    def test_minimal_required_fields(self) -> None:
        p = make_plan()
        assert p.agent_id == "agent-1"
        assert len(p.actions) == 1

    def test_plan_id_auto_generated(self) -> None:
        p = make_plan()
        assert p.plan_id
        assert len(p.plan_id) == 36

    def test_two_plans_get_different_ids(self) -> None:
        assert make_plan().plan_id != make_plan().plan_id

    def test_created_at_defaults_to_utc_aware(self) -> None:
        assert make_plan().created_at.tzinfo is not None

    def test_executed_at_defaults_to_none(self) -> None:
        assert make_plan().executed_at is None

    def test_multiple_actions_accepted(self) -> None:
        actions = [
            make_action(source_memory_ids=["mem-a", "mem-b"]),
            make_action(action_type=ActionType.FLAG, source_memory_ids=["mem-c"]),
        ]
        p = make_plan(actions=actions)
        assert len(p.actions) == 2


class TestConsolidationPlanValidators:
    def test_blank_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty or whitespace"):
            make_plan(agent_id="")

    def test_agent_id_is_stripped(self) -> None:
        p = make_plan(agent_id="  agent-1  ")
        assert p.agent_id == "agent-1"

    def test_empty_actions_raises(self) -> None:
        with pytest.raises(ValidationError, match="actions must not be empty"):
            make_plan(actions=[])

    def test_mismatched_action_agent_id_raises(self) -> None:
        wrong_action = make_action(agent_id="agent-other")
        with pytest.raises(ValidationError, match="agent_id"):
            make_plan(agent_id="agent-1", actions=[wrong_action])

    def test_naive_executed_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_plan(executed_at=datetime.utcnow())  # noqa: DTZ003

    def test_executed_at_before_created_at_raises(self) -> None:
        base = datetime.now(UTC)
        with pytest.raises(ValidationError, match="must not be before created_at"):
            make_plan(
                created_at=base,
                executed_at=base - timedelta(seconds=1),
            )


class TestConsolidationPlanMutation:
    def test_can_set_executed_at(self) -> None:
        p = make_plan()
        now = datetime.now(UTC)
        p.executed_at = now
        assert p.executed_at == now

    def test_plan_id_frozen(self) -> None:
        p = make_plan()
        with pytest.raises(ValidationError):
            p.plan_id = "new-id"  # type: ignore[misc]

    def test_created_at_frozen(self) -> None:
        p = make_plan()
        with pytest.raises(ValidationError):
            p.created_at = datetime.now(UTC)  # type: ignore[misc]


class TestConsolidationPlanSerialization:
    def test_model_dump_round_trip(self) -> None:
        actions = [make_action(reasoning="Similar content.")]
        p = make_plan(actions=actions)
        data = p.model_dump()
        restored = ConsolidationPlan(**data)
        assert restored.plan_id == p.plan_id
        assert len(restored.actions) == 1
        assert restored.actions[0].action_id == p.actions[0].action_id

    def test_json_round_trip(self) -> None:
        p = make_plan()
        restored = ConsolidationPlan.model_validate_json(p.model_dump_json())
        assert restored.plan_id == p.plan_id
        assert restored.actions[0].action_id == p.actions[0].action_id


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResultInstantiation:
    def test_minimal_required_fields(self) -> None:
        sr = make_search_result()
        assert sr.score == 0.92
        assert sr.rank == 1
        assert sr.memory is not None

    def test_query_id_defaults_to_none(self) -> None:
        assert make_search_result().query_id is None

    def test_query_id_accepted(self) -> None:
        sr = make_search_result(query_id="q-42")
        assert sr.query_id == "q-42"

    def test_memory_fields_accessible(self) -> None:
        m = make_memory(text="Return policy.")
        sr = make_search_result(memory=m)
        assert sr.memory.text == "Return policy."


class TestSearchResultValidators:
    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_search_result(score=-0.1)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_search_result(score=1.1)

    def test_score_at_boundaries_accepted(self) -> None:
        assert make_search_result(score=0.0).score == 0.0
        assert make_search_result(score=1.0).score == 1.0

    def test_rank_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_search_result(rank=0)

    def test_rank_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_search_result(rank=-1)

    def test_rank_one_accepted(self) -> None:
        assert make_search_result(rank=1).rank == 1

    def test_blank_query_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_search_result(query_id="")

    def test_whitespace_query_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_search_result(query_id="   ")


class TestSearchResultFrozen:
    def test_assignment_after_creation_raises(self) -> None:
        sr = make_search_result()
        with pytest.raises(ValidationError):
            sr.score = 0.5  # type: ignore[misc]

    def test_rank_immutable(self) -> None:
        sr = make_search_result()
        with pytest.raises(ValidationError):
            sr.rank = 2  # type: ignore[misc]


class TestSearchResultSerialization:
    def test_model_dump_round_trip(self) -> None:
        m = make_memory(text="Shipping takes 3–5 days.")
        sr = make_search_result(memory=m, score=0.88, rank=2, query_id="q-1")
        data = sr.model_dump()
        restored = SearchResult(**data)
        assert restored.score == 0.88
        assert restored.rank == 2
        assert restored.memory.text == "Shipping takes 3–5 days."
        assert restored.query_id == "q-1"

    def test_json_round_trip(self) -> None:
        m = make_memory()
        sr = make_search_result(memory=m)
        restored = SearchResult.model_validate_json(sr.model_dump_json())
        assert restored.memory.memory_id == m.memory_id
        assert restored.score == sr.score

    def test_nested_memory_with_provenance_round_trips(self) -> None:
        m = make_memory(memory_id="mem-x")
        pr = ProvenanceRecord(
            memory_id="mem-x",
            source_type=SourceType.DOCUMENT,
            source_id="doc-99",
        )
        m.attach_provenance(pr)
        sr = make_search_result(memory=m)
        restored = SearchResult.model_validate_json(sr.model_dump_json())
        assert restored.memory.provenance is not None
        assert restored.memory.provenance.source_id == "doc-99"


# ---------------------------------------------------------------------------
# HealthScore — signature metrics and pending_review
# ---------------------------------------------------------------------------


class TestHealthScoreSignatures:
    def test_signature_metrics_accepted(self) -> None:
        hs = make_health_score(
            contradiction_score=0.7,
            freshness_score=0.8,
            confidence_accuracy_gap=0.1,
            provenance_completeness=0.6,
        )
        assert hs.contradiction_score == 0.7
        assert hs.freshness_score == 0.8
        assert hs.confidence_accuracy_gap == 0.1
        assert hs.provenance_completeness == 0.6

    def test_signature_metrics_at_boundaries(self) -> None:
        hs = make_health_score(
            contradiction_score=0.0,
            freshness_score=1.0,
            confidence_accuracy_gap=0.0,
            provenance_completeness=1.0,
        )
        assert hs.contradiction_score == 0.0
        assert hs.freshness_score == 1.0

    def test_signature_metric_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(contradiction_score=-0.1)

    def test_signature_metric_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(freshness_score=1.1)

    def test_confidence_accuracy_gap_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(confidence_accuracy_gap=1.1)

    def test_provenance_completeness_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_health_score(provenance_completeness=-0.01)


class TestHealthScorePendingReview:
    def test_pending_review_defaults_to_empty_tuple(self) -> None:
        assert make_health_score().pending_review == ()

    def test_pending_review_accepts_conflict_records(self) -> None:
        from engram.core.models import ConflictRecord

        cr = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.9,
        )
        hs = make_health_score(pending_review=(cr,))
        assert len(hs.pending_review) == 1
        assert hs.pending_review[0].memory_a_id == "mem-a"

    def test_pending_review_is_tuple(self) -> None:
        assert isinstance(make_health_score().pending_review, tuple)

    def test_pending_review_survives_round_trip(self) -> None:
        from engram.core.models import ConflictRecord

        cr = ConflictRecord(
            agent_id="agent-1",
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            conflict_type=ConflictType.PARTIAL_CONFLICT,
            confidence=0.85,
        )
        hs = make_health_score(pending_review=(cr,))
        restored = HealthScore.model_validate_json(hs.model_dump_json())
        assert len(restored.pending_review) == 1


# ---------------------------------------------------------------------------
# ConsolidationAction — source_memory_ids tuple, mark_executed, execution coherence
# ---------------------------------------------------------------------------


class TestConsolidationActionSourceIds:
    def test_source_memory_ids_is_tuple(self) -> None:
        a = make_action()
        assert isinstance(a.source_memory_ids, tuple)

    def test_list_input_coerced_to_tuple(self) -> None:
        a = make_action(source_memory_ids=["x", "y"])
        assert a.source_memory_ids == ("x", "y")

    def test_source_memory_ids_frozen(self) -> None:
        a = make_action()
        with pytest.raises(ValidationError):
            a.source_memory_ids = ("mem-c",)  # type: ignore[misc]


class TestConsolidationActionMarkExecuted:
    def test_mark_executed_sets_both_fields(self) -> None:
        a = make_action()
        a.mark_executed("mem-merged")
        assert a.result_memory_id == "mem-merged"
        assert a.executed_at is not None
        assert a.executed_at.tzinfo is not None

    def test_mark_executed_accepts_custom_timestamp(self) -> None:
        a = make_action()
        ts = datetime.now(UTC)
        a.mark_executed("mem-result", executed_at=ts)
        assert a.executed_at == ts

    def test_mark_executed_twice_raises(self) -> None:
        a = make_action()
        a.mark_executed("mem-merged")
        with pytest.raises(ValueError, match="already executed"):
            a.mark_executed("mem-merged-again")

    def test_mark_executed_blank_result_raises(self) -> None:
        a = make_action()
        with pytest.raises(ValueError, match="must not be blank"):
            a.mark_executed("   ")

    def test_executed_action_survives_round_trip(self) -> None:
        a = make_action()
        a.mark_executed("mem-result")
        restored = ConsolidationAction(**a.model_dump(mode="python"))
        assert restored.result_memory_id == "mem-result"
        assert restored.executed_at is not None


class TestConsolidationActionExecutionCoherence:
    def test_executed_at_without_result_id_raises_at_construction(self) -> None:
        base = datetime.now(UTC)
        with pytest.raises(ValidationError, match="result_memory_id must be set"):
            ConsolidationAction(
                agent_id="agent-1",
                action_type=ActionType.MERGE,
                tier=ConsolidationTier.AUTO_MERGE,
                confidence=0.95,
                source_memory_ids=["m1", "m2"],
                created_at=base,
                executed_at=base + timedelta(seconds=1),
                result_memory_id=None,
            )

    def test_executed_at_with_result_id_accepted(self) -> None:
        base = datetime.now(UTC)
        a = ConsolidationAction(
            agent_id="agent-1",
            action_type=ActionType.MERGE,
            tier=ConsolidationTier.AUTO_MERGE,
            confidence=0.95,
            source_memory_ids=["m1", "m2"],
            created_at=base,
            executed_at=base + timedelta(seconds=1),
            result_memory_id="mem-result",
        )
        assert a.executed_at == base + timedelta(seconds=1)
        assert a.result_memory_id == "mem-result"


# ---------------------------------------------------------------------------
# ConsolidationPlan — tuple actions, metadata, properties, all-or-nothing
# ---------------------------------------------------------------------------


class TestConsolidationPlanActions:
    def test_actions_is_tuple(self) -> None:
        p = make_plan()
        assert isinstance(p.actions, tuple)

    def test_list_input_coerced_to_tuple(self) -> None:
        actions = [make_action()]
        p = make_plan(actions=actions)
        assert isinstance(p.actions, tuple)
        assert len(p.actions) == 1

    def test_actions_frozen(self) -> None:
        p = make_plan()
        with pytest.raises(ValidationError):
            p.actions = (make_action(),)  # type: ignore[misc]


class TestConsolidationPlanMetadata:
    def test_is_dry_run_defaults_to_false(self) -> None:
        assert make_plan().is_dry_run is False

    def test_is_dry_run_can_be_set_true(self) -> None:
        p = make_plan(is_dry_run=True)
        assert p.is_dry_run is True

    def test_estimated_health_delta_defaults_to_empty(self) -> None:
        assert make_plan().estimated_health_delta == {}

    def test_estimated_health_delta_accepted(self) -> None:
        p = make_plan(estimated_health_delta={"score": 0.05, "freshness": -0.02})
        assert p.estimated_health_delta["score"] == 0.05


class TestConsolidationPlanProperties:
    def test_would_merge_filters_correctly(self) -> None:
        a_merge = make_action(action_type=ActionType.MERGE, source_memory_ids=["m1", "m2"])
        a_flag = make_action(action_type=ActionType.FLAG, source_memory_ids=["m3"])
        p = make_plan(actions=[a_merge, a_flag])
        assert len(p.would_merge) == 1
        assert p.would_merge[0].action_type == ActionType.MERGE

    def test_would_supersede_filters_correctly(self) -> None:
        a_sup = make_action(action_type=ActionType.SUPERSEDE, source_memory_ids=["m1"])
        a_merge = make_action(action_type=ActionType.MERGE, source_memory_ids=["m2", "m3"])
        p = make_plan(actions=[a_sup, a_merge])
        assert len(p.would_supersede) == 1
        assert len(p.would_merge) == 1

    def test_would_flag_for_review_filters_correctly(self) -> None:
        a_flag = make_action(action_type=ActionType.FLAG, source_memory_ids=["m1"])
        p = make_plan(actions=[a_flag])
        assert len(p.would_flag_for_review) == 1
        assert len(p.would_merge) == 0

    def test_properties_return_empty_tuples_when_no_match(self) -> None:
        p = make_plan()  # default action is MERGE
        assert p.would_supersede == ()
        assert p.would_flag_for_review == ()


class TestConsolidationPlanAllOrNothing:
    def test_partial_execution_raises_at_construction(self) -> None:
        a1 = make_action(source_memory_ids=["m1", "m2"])
        a2 = make_action(source_memory_ids=["m3"])
        a1.mark_executed("mem-merged")
        # a2 is unexecuted — partial → should fail construction
        with pytest.raises(ValidationError, match="all-or-nothing"):
            ConsolidationPlan(agent_id="agent-1", actions=[a1, a2])

    def test_fully_executed_plan_accepted(self) -> None:
        a1 = make_action(source_memory_ids=["m1", "m2"])
        a2 = make_action(source_memory_ids=["m3"])
        a1.mark_executed("mem-merged")
        a2.mark_executed("m3")
        p = ConsolidationPlan(agent_id="agent-1", actions=[a1, a2])
        assert all(a.executed_at is not None for a in p.actions)

    def test_unexecuted_plan_accepted(self) -> None:
        p = make_plan()
        assert all(a.executed_at is None for a in p.actions)


# ---------------------------------------------------------------------------
# SearchResult — conflict fields
# ---------------------------------------------------------------------------


class TestSearchResultConflict:
    def test_conflict_flag_defaults_to_false(self) -> None:
        assert make_search_result().conflict_flag is False

    def test_conflicts_with_defaults_to_empty_tuple(self) -> None:
        assert make_search_result().conflicts_with == ()

    def test_conflict_summary_defaults_to_none(self) -> None:
        assert make_search_result().conflict_summary is None

    def test_recommended_defaults_to_true(self) -> None:
        assert make_search_result().recommended is True

    def test_conflict_flag_true_with_conflicts_accepted(self) -> None:
        sr = make_search_result(
            conflict_flag=True,
            conflicts_with=("mem-old",),
            conflict_summary="Contradicts an earlier fact.",
            recommended=False,
        )
        assert sr.conflict_flag is True
        assert sr.conflicts_with == ("mem-old",)
        assert sr.conflict_summary == "Contradicts an earlier fact."
        assert sr.recommended is False

    def test_conflict_flag_true_empty_conflicts_with_raises(self) -> None:
        with pytest.raises(ValidationError, match="conflicts_with must not be empty"):
            make_search_result(conflict_flag=True, conflicts_with=())

    def test_conflicts_with_non_empty_without_flag_raises(self) -> None:
        with pytest.raises(ValidationError, match="conflict_flag must be True"):
            make_search_result(conflict_flag=False, conflicts_with=("mem-old",))

    def test_blank_conflict_summary_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_search_result(
                conflict_flag=True,
                conflicts_with=("mem-old",),
                conflict_summary="   ",
            )

    def test_conflict_fields_survive_round_trip(self) -> None:
        sr = make_search_result(
            conflict_flag=True,
            conflicts_with=("mem-a", "mem-b"),
            conflict_summary="Temporal conflict.",
        )
        restored = SearchResult.model_validate_json(sr.model_dump_json())
        assert restored.conflict_flag is True
        assert restored.conflicts_with == ("mem-a", "mem-b")
        assert restored.conflict_summary == "Temporal conflict."
