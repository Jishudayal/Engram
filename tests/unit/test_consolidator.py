"""Tests for Consolidator — plan generation, routing, and LLM integration."""

from datetime import UTC, datetime

import pytest

from engram.adapters.memory import InMemoryAdapter
from engram.core.constants import (
    AUTO_FLAG_THRESHOLD,
    AUTO_MERGE_THRESHOLD,
    ActionType,
    ConflictType,
    ConsolidationTier,
    ResolutionStatus,
)
from engram.core.consolidator import _DEFAULT_MODEL, Consolidator, PlanningResult
from engram.core.exceptions import NotFoundError
from engram.core.models import ConflictRecord, ConsolidationAction, ConsolidationPlan, Memory

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_memory(
    text: str,
    *,
    agent_id: str = "agent-1",
    embedding: list[float] | None = None,
) -> Memory:
    return Memory(agent_id=agent_id, text=text, embedding=embedding)


def make_conflict(
    mem_a: Memory,
    mem_b: Memory,
    *,
    conflict_type: ConflictType = ConflictType.DIRECT_CONTRADICTION,
    confidence: float = 0.95,
    description: str | None = "Two memories disagree.",
    agent_id: str = "agent-1",
) -> ConflictRecord:
    return ConflictRecord(
        agent_id=agent_id,
        memory_a_id=mem_a.memory_id,
        memory_b_id=mem_b.memory_id,
        conflict_type=conflict_type,
        confidence=confidence,
        description=description,
    )


def make_consolidator(responses: list[str], **kwargs: object) -> Consolidator:
    """Build a Consolidator backed by a deterministic mock LLM.

    Responses are consumed in order. When exhausted, defaults to flag at 0.0.
    """
    it = iter(responses)

    async def mock_llm(prompt: str) -> str:
        return next(it, '{"action": "flag", "confidence": 0.0, "reasoning": ""}')

    return Consolidator(llm_fn=mock_llm, **kwargs)  # type: ignore[arg-type]


# Reusable LLM response strings
_MERGE_HIGH = '{"action": "merge", "confidence": 0.95, "reasoning": "Clear reconciliation possible."}'
_MERGE_MED = '{"action": "merge", "confidence": 0.85, "reasoning": "Likely reconcilable."}'
_FLAG_LOW = '{"action": "flag", "confidence": 0.60, "reasoning": "Too ambiguous to auto-merge."}'
_FLAG_ZERO = '{"action": "flag", "confidence": 0.0, "reasoning": ""}'


# ---------------------------------------------------------------------------
# PlanningResult
# ---------------------------------------------------------------------------


class TestPlanningResult:
    def test_is_frozen_dataclass(self) -> None:
        r = PlanningResult(conflict_id="c", action_type="merge", confidence=0.9, reasoning="ok")
        with pytest.raises((AttributeError, TypeError)):
            r.action_type = "flag"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        r = PlanningResult(conflict_id="c", action_type="flag", confidence=0.7, reasoning="why")
        assert r.conflict_id == "c"
        assert r.action_type == "flag"
        assert r.confidence == 0.7
        assert r.reasoning == "why"


# ---------------------------------------------------------------------------
# Consolidator — init
# ---------------------------------------------------------------------------


class TestConsolidatorInit:
    def test_default_model_is_known_anthropic_model(self) -> None:
        assert _DEFAULT_MODEL in {
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        }

    def test_custom_llm_fn_accepted(self) -> None:
        async def fn(p: str) -> str:
            return "{}"

        c = Consolidator(llm_fn=fn)
        assert c is not None

    def test_no_llm_fn_no_anthropic_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="anthropic"):
            Consolidator()

    def test_custom_max_concurrency_accepted(self) -> None:
        c = make_consolidator([], max_concurrency=3)
        assert c._max_concurrency == 3


# ---------------------------------------------------------------------------
# _parse_planning_result
# ---------------------------------------------------------------------------


class TestParsePlanningResult:
    def _det(self) -> Consolidator:
        return make_consolidator([])

    def test_merge_parsed(self) -> None:
        r = self._det()._parse_planning_result(_MERGE_HIGH, "c1")
        assert r.action_type == "merge"
        assert r.confidence == 0.95
        assert r.reasoning == "Clear reconciliation possible."

    def test_flag_parsed(self) -> None:
        r = self._det()._parse_planning_result(_FLAG_LOW, "c1")
        assert r.action_type == "flag"
        assert r.confidence == 0.60

    def test_malformed_json_defaults_to_flag(self) -> None:
        r = self._det()._parse_planning_result("not json {{{", "c1")
        assert r.action_type == "flag"
        assert r.confidence == 0.0

    def test_empty_string_defaults_to_flag(self) -> None:
        r = self._det()._parse_planning_result("", "c1")
        assert r.action_type == "flag"

    def test_missing_fields_use_defaults(self) -> None:
        r = self._det()._parse_planning_result("{}", "c1")
        assert r.action_type == "flag"
        assert r.confidence == 0.0
        assert r.reasoning == ""

    def test_confidence_clamped_above_one(self) -> None:
        raw = '{"action": "merge", "confidence": 1.5, "reasoning": "x"}'
        r = self._det()._parse_planning_result(raw, "c1")
        assert r.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        raw = '{"action": "flag", "confidence": -0.3, "reasoning": "x"}'
        r = self._det()._parse_planning_result(raw, "c1")
        assert r.confidence == 0.0

    def test_action_lowercased(self) -> None:
        raw = '{"action": "MERGE", "confidence": 0.9, "reasoning": "x"}'
        r = self._det()._parse_planning_result(raw, "c1")
        assert r.action_type == "merge"

    def test_conflict_id_preserved(self) -> None:
        r = self._det()._parse_planning_result(_MERGE_HIGH, "conflict-xyz")
        assert r.conflict_id == "conflict-xyz"


# ---------------------------------------------------------------------------
# _to_action
# ---------------------------------------------------------------------------


class TestToAction:
    def _det(self) -> Consolidator:
        return make_consolidator([])

    def _result(self, action: str, confidence: float) -> PlanningResult:
        return PlanningResult(
            conflict_id="c1",
            action_type=action,
            confidence=confidence,
            reasoning="test reason",
        )

    def _conflict(self) -> ConflictRecord:
        m1 = make_memory("A")
        m2 = make_memory("B")
        return make_conflict(m1, m2)

    def test_merge_action_type(self) -> None:
        action = self._det()._to_action(self._conflict(), self._result("merge", 0.95), "agent-1")
        assert action is not None
        assert action.action_type == ActionType.MERGE

    def test_flag_action_type(self) -> None:
        action = self._det()._to_action(self._conflict(), self._result("flag", 0.70), "agent-1")
        assert action is not None
        assert action.action_type == ActionType.FLAG

    def test_unknown_action_returns_none(self) -> None:
        action = self._det()._to_action(self._conflict(), self._result("delete", 0.99), "agent-1")
        assert action is None

    def test_tier_auto_merge_for_high_confidence(self) -> None:
        action = self._det()._to_action(
            self._conflict(), self._result("merge", AUTO_MERGE_THRESHOLD), "agent-1"
        )
        assert action is not None
        assert action.tier == ConsolidationTier.AUTO_MERGE

    def test_tier_auto_flag_for_medium_confidence(self) -> None:
        action = self._det()._to_action(
            self._conflict(), self._result("flag", AUTO_FLAG_THRESHOLD), "agent-1"
        )
        assert action is not None
        assert action.tier == ConsolidationTier.AUTO_FLAG

    def test_tier_human_review_for_low_confidence(self) -> None:
        action = self._det()._to_action(
            self._conflict(), self._result("flag", 0.50), "agent-1"
        )
        assert action is not None
        assert action.tier == ConsolidationTier.HUMAN_REVIEW

    def test_reasoning_preserved(self) -> None:
        result = PlanningResult(
            conflict_id="c1", action_type="merge", confidence=0.95, reasoning="Clear synthesis."
        )
        action = self._det()._to_action(self._conflict(), result, "agent-1")
        assert action is not None
        assert action.reasoning == "Clear synthesis."

    def test_blank_reasoning_stored_as_none(self) -> None:
        result = PlanningResult(
            conflict_id="c1", action_type="merge", confidence=0.95, reasoning=""
        )
        action = self._det()._to_action(self._conflict(), result, "agent-1")
        assert action is not None
        assert action.reasoning is None

    def test_source_memory_ids_from_conflict(self) -> None:
        m1 = make_memory("A")
        m2 = make_memory("B")
        conflict = make_conflict(m1, m2)
        action = self._det()._to_action(conflict, self._result("merge", 0.95), "agent-1")
        assert action is not None
        assert set(action.source_memory_ids) == {m1.memory_id, m2.memory_id}

    def test_agent_id_preserved(self) -> None:
        action = self._det()._to_action(
            self._conflict(), self._result("merge", 0.95), "my-agent"
        )
        assert action is not None
        assert action.agent_id == "my-agent"


# ---------------------------------------------------------------------------
# _temporal_supersession_action
# ---------------------------------------------------------------------------


class TestTemporalSupersessionAction:
    def _det(self) -> Consolidator:
        return make_consolidator([])

    async def test_action_type_is_supersede(self) -> None:
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")
        assert action.action_type == ActionType.SUPERSEDE

    async def test_older_memory_is_first_in_source_ids(self) -> None:
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        # m1 created before m2 (guaranteed by sequential construction)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")
        # First source_memory_id is the superseded (older) one
        assert action.source_memory_ids[0] == m1.memory_id
        assert action.source_memory_ids[1] == m2.memory_id

    async def test_conflict_description_used_as_reasoning(self) -> None:
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        conflict = make_conflict(
            m1, m2,
            conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
            description="Rate limit changed from 100 to 500.",
        )
        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")
        assert action.reasoning == "Rate limit changed from 100 to 500."

    async def test_fallback_reasoning_when_no_description(self) -> None:
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        conflict = make_conflict(
            m1, m2,
            conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
            description=None,
        )
        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")
        assert action.reasoning is not None
        assert len(action.reasoning) > 0

    async def test_tier_from_conflict_confidence(self) -> None:
        m1 = make_memory("old")
        m2 = make_memory("new")
        conflict = make_conflict(
            m1, m2,
            conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
            confidence=AUTO_MERGE_THRESHOLD,
        )
        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")
        assert action.tier == ConsolidationTier.AUTO_MERGE

    async def test_tied_created_at_flags_for_human_review(self) -> None:
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        m1 = Memory(agent_id="agent-1", text="fact A", created_at=created_at)
        m2 = Memory(agent_id="agent-1", text="fact B", created_at=created_at)
        conflict = make_conflict(
            m1,
            m2,
            conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
            confidence=AUTO_MERGE_THRESHOLD,
        )

        action = self._det()._temporal_supersession_action(conflict, m1, m2, "agent-1")

        assert action.action_type == ActionType.FLAG
        assert action.tier == ConsolidationTier.HUMAN_REVIEW
        assert action.source_memory_ids == (m1.memory_id, m2.memory_id)
        assert action.reasoning == "Tied created_at; manual review required."


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


class TestPlan:
    async def test_no_pending_conflicts_returns_none(self) -> None:
        det = make_consolidator([])
        result = await det.plan("agent-1", InMemoryAdapter())
        assert result is None

    async def test_temporal_supersession_no_llm_call(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old rate limit is 100")
        m2 = make_memory("new rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        # Empty responses — no LLM calls expected
        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert len(plan.actions) == 1
        assert plan.actions[0].action_type == ActionType.SUPERSEDE

    async def test_direct_contradiction_uses_llm(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        await adapter.store_conflict(conflict)

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert plan.actions[0].action_type == ActionType.MERGE

    async def test_partial_conflict_uses_llm(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("office is in London")
        m2 = make_memory("HQ is in London, branch in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.PARTIAL_CONFLICT)
        await adapter.store_conflict(conflict)

        det = make_consolidator([_FLAG_LOW])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert plan.actions[0].action_type == ActionType.FLAG

    async def test_plan_agent_id_set_correctly(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A", agent_id="my-agent")
        m2 = make_memory("fact B", agent_id="my-agent")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, agent_id="my-agent")
        await adapter.store_conflict(conflict)

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("my-agent", adapter)
        assert plan is not None
        assert plan.agent_id == "my-agent"
        assert plan.actions[0].agent_id == "my-agent"

    async def test_dry_run_flag_propagated(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("agent-1", adapter, dry_run=True)
        assert plan is not None
        assert plan.is_dry_run is True

    async def test_dry_run_false_by_default(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert plan.is_dry_run is False

    async def test_missing_memory_skips_conflict(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        # Only store m1 — m2 is missing from the index
        await adapter.store(m1)
        conflict = make_conflict(m1, m2)
        await adapter.store_conflict(conflict)

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("agent-1", adapter)
        # All conflicts skipped → None
        assert plan is None

    async def test_multiple_conflicts_processed(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        m3 = make_memory("office is in London")
        m4 = make_memory("office is in Berlin")
        for m in [m1, m2, m3, m4]:
            await adapter.store(m)

        conflict1 = make_conflict(m1, m2, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        conflict2 = make_conflict(m3, m4, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        await adapter.store_conflict(conflict1)
        await adapter.store_conflict(conflict2)

        det = make_consolidator([_MERGE_HIGH, _FLAG_LOW])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert len(plan.actions) == 2

    async def test_unknown_llm_action_skips_conflict(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        await adapter.store_conflict(conflict)

        bad_response = '{"action": "delete_all", "confidence": 0.99, "reasoning": "nuke it"}'
        det = make_consolidator([bad_response])
        plan = await det.plan("agent-1", adapter)
        assert plan is None

    async def test_llm_failure_falls_back_to_flag(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        m3 = make_memory("fact C")
        m4 = make_memory("fact D")
        for m in [m1, m2, m3, m4]:
            await adapter.store(m)

        conflict1 = make_conflict(m1, m2, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        conflict2 = make_conflict(m3, m4, conflict_type=ConflictType.DIRECT_CONTRADICTION)
        await adapter.store_conflict(conflict1)
        await adapter.store_conflict(conflict2)

        call_count = 0

        async def failing_then_ok(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM down")
            return _MERGE_HIGH

        det = Consolidator(llm_fn=failing_then_ok)
        plan = await det.plan("agent-1", adapter)
        # LLM failure falls back to flag (not skip) — conflict is preserved for human review.
        # Both conflicts produce actions: one FLAG (from fallback), one MERGE (success).
        assert plan is not None
        assert len(plan.actions) == 2
        action_types = {a.action_type for a in plan.actions}
        assert ActionType.FLAG in action_types
        assert ActionType.MERGE in action_types

    async def test_plan_actions_immutable(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        det = make_consolidator([_MERGE_HIGH])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None
        assert isinstance(plan.actions, tuple)

    async def test_importable_from_root(self) -> None:
        from engram import Consolidator as C
        assert C is not None


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


async def _plan_and_execute(
    adapter: InMemoryAdapter,
    responses: list[str],
    agent_id: str = "agent-1",
    **kwargs: object,
) -> ConsolidationPlan:
    """Helper: plan then execute with a mock LLM."""
    det = make_consolidator(responses, **kwargs)
    plan = await det.plan(agent_id, adapter)
    assert plan is not None
    return await det.execute(plan, adapter)


class TestExecuteSupersede:
    async def test_superseded_memory_gets_status_superseded(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old rate limit is 100")
        m2 = make_memory("new rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        await _plan_and_execute(adapter, [])

        older = await adapter.fetch("agent-1", m1.memory_id)
        assert older is not None
        assert older.status.value == "superseded"

    async def test_keeper_memory_stays_active(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old rate limit is 100")
        m2 = make_memory("new rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        await _plan_and_execute(adapter, [])

        newer = await adapter.fetch("agent-1", m2.memory_id)
        assert newer is not None
        assert newer.status.value == "active"

    async def test_superseded_by_set_to_keeper(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        await _plan_and_execute(adapter, [])

        older = await adapter.fetch("agent-1", m1.memory_id)
        assert older is not None
        assert older.superseded_by == m2.memory_id

    async def test_conflict_marked_auto_resolved(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        await _plan_and_execute(adapter, [])

        pending = await adapter.list_conflicts("agent-1", status=ResolutionStatus.PENDING)
        assert pending == []
        all_conflicts = await adapter.list_conflicts("agent-1")
        assert all_conflicts[0].resolution_status == ResolutionStatus.AUTO_RESOLVED

    async def test_action_marked_executed_with_keeper_id(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        plan = await _plan_and_execute(adapter, [])

        assert plan.actions[0].executed_at is not None
        assert plan.actions[0].result_memory_id == m2.memory_id


class TestExecuteMerge:
    async def test_merged_memory_stored(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        plan = await _plan_and_execute(adapter, [_MERGE_HIGH])

        merged_id = plan.actions[0].result_memory_id
        assert merged_id is not None
        merged = await adapter.fetch("agent-1", merged_id)
        assert merged is not None

    async def test_merged_memory_text_from_reasoning(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        plan = await _plan_and_execute(adapter, [_MERGE_HIGH])

        merged_id = plan.actions[0].result_memory_id
        merged = await adapter.fetch("agent-1", merged_id)
        assert merged is not None
        assert merged.text == "Clear reconciliation possible."

    async def test_both_sources_archived(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        await _plan_and_execute(adapter, [_MERGE_HIGH])

        for mem_id in [m1.memory_id, m2.memory_id]:
            mem = await adapter.fetch("agent-1", mem_id)
            assert mem is not None
            assert mem.status.value == "archived"

    async def test_sources_superseded_by_merged(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        plan = await _plan_and_execute(adapter, [_MERGE_HIGH])
        merged_id = plan.actions[0].result_memory_id

        for mem_id in [m1.memory_id, m2.memory_id]:
            mem = await adapter.fetch("agent-1", mem_id)
            assert mem is not None
            assert mem.superseded_by == merged_id

    async def test_conflict_marked_auto_resolved(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("rate limit is 100")
        m2 = make_memory("rate limit is 500")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        await _plan_and_execute(adapter, [_MERGE_HIGH])

        pending = await adapter.list_conflicts("agent-1", status=ResolutionStatus.PENDING)
        assert pending == []


class TestExecuteFlag:
    async def test_conflict_notes_updated(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("office in London")
        m2 = make_memory("office in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        await _plan_and_execute(adapter, [_FLAG_LOW])

        all_conflicts = await adapter.list_conflicts("agent-1")
        assert all_conflicts[0].resolution_notes == "Too ambiguous to auto-merge."

    async def test_conflict_stays_pending(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("office in London")
        m2 = make_memory("office in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        await _plan_and_execute(adapter, [_FLAG_LOW])

        pending = await adapter.list_conflicts("agent-1", status=ResolutionStatus.PENDING)
        assert len(pending) == 1

    async def test_memories_unchanged(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("office in London")
        m2 = make_memory("office in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(make_conflict(m1, m2))

        await _plan_and_execute(adapter, [_FLAG_LOW])

        for mem_id in [m1.memory_id, m2.memory_id]:
            mem = await adapter.fetch("agent-1", mem_id)
            assert mem is not None
            assert mem.status.value == "active"


class TestExecuteGeneral:
    async def test_plan_executed_at_set(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        plan = await _plan_and_execute(adapter, [])

        assert plan.executed_at is not None

    async def test_all_actions_marked_executed(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("fact A")
        m2 = make_memory("fact B")
        m3 = make_memory("fact C")
        m4 = make_memory("fact D")
        for m in [m1, m2, m3, m4]:
            await adapter.store(m)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )
        await adapter.store_conflict(make_conflict(m3, m4))

        plan = await _plan_and_execute(adapter, [_MERGE_HIGH])

        assert all(a.executed_at is not None for a in plan.actions)
        assert all(a.result_memory_id is not None for a in plan.actions)

    async def test_dry_run_no_memory_mutations(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter, dry_run=True)
        assert plan is not None
        await det.execute(plan, adapter)

        # Memory should be unchanged
        m1_after = await adapter.fetch("agent-1", m1.memory_id)
        assert m1_after is not None
        assert m1_after.status.value == "active"

    async def test_dry_run_marks_all_actions_executed(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter, dry_run=True)
        assert plan is not None
        executed_plan = await det.execute(plan, adapter)

        assert all(a.executed_at is not None for a in executed_plan.actions)
        assert executed_plan.executed_at is not None

    async def test_stale_conflict_raises_not_found(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None

        # Remove the conflict before executing — simulates resolution between plan+execute
        await adapter.delete_conflict("agent-1", conflict.conflict_id)

        with pytest.raises(NotFoundError, match="no longer PENDING"):
            await det.execute(plan, adapter)


# ---------------------------------------------------------------------------
# execute() — ARCHIVE action
# ---------------------------------------------------------------------------


def make_archive_plan(
    agent_id: str,
    source_memory_ids: tuple[str, ...],
) -> ConsolidationPlan:
    action = ConsolidationAction(
        agent_id=agent_id,
        action_type=ActionType.ARCHIVE,
        tier=ConsolidationTier.AUTO_MERGE,
        confidence=1.0,
        source_memory_ids=source_memory_ids,
        reasoning="Direct archive.",
    )
    return ConsolidationPlan(agent_id=agent_id, actions=(action,))


class TestExecuteArchive:
    async def test_memory_gets_archived_status(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("stale fact")
        await adapter.store(m1)
        plan = make_archive_plan("agent-1", (m1.memory_id,))

        det = make_consolidator([])
        await det.execute(plan, adapter)

        fetched = await adapter.fetch("agent-1", m1.memory_id)
        assert fetched is not None
        assert fetched.status.value == "archived"

    async def test_multiple_sources_all_archived(self) -> None:
        adapter = InMemoryAdapter()
        memories = [make_memory(f"stale fact {i}") for i in range(3)]
        for m in memories:
            await adapter.store(m)
        plan = make_archive_plan("agent-1", tuple(m.memory_id for m in memories))

        det = make_consolidator([])
        await det.execute(plan, adapter)

        for m in memories:
            fetched = await adapter.fetch("agent-1", m.memory_id)
            assert fetched is not None
            assert fetched.status.value == "archived"

    async def test_action_marked_executed(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("stale fact")
        await adapter.store(m1)
        plan = make_archive_plan("agent-1", (m1.memory_id,))

        det = make_consolidator([])
        executed = await det.execute(plan, adapter)

        assert executed.actions[0].executed_at is not None
        assert executed.actions[0].result_memory_id is not None

    async def test_missing_source_raises_not_found(self) -> None:
        adapter = InMemoryAdapter()
        plan = make_archive_plan("agent-1", ("nonexistent-id",))

        det = make_consolidator([])
        with pytest.raises(NotFoundError, match="source memories not found"):
            await det.execute(plan, adapter)


# ---------------------------------------------------------------------------
# execute() — preflight all-or-nothing
# ---------------------------------------------------------------------------


class TestExecutePreflight:
    async def test_missing_memory_raises_before_any_write(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        )

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None

        # Delete one memory before executing
        await adapter.delete("agent-1", m1.memory_id)

        with pytest.raises(NotFoundError, match="source memories not found"):
            await det.execute(plan, adapter)

        # The keeper memory must be untouched (no partial writes)
        m2_after = await adapter.fetch("agent-1", m2.memory_id)
        assert m2_after is not None
        assert m2_after.status.value == "active"

    async def test_stale_conflict_raises_before_any_write(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter)
        assert plan is not None

        await adapter.delete_conflict("agent-1", conflict.conflict_id)

        with pytest.raises(NotFoundError, match="no longer PENDING"):
            await det.execute(plan, adapter)

        # Both memories must be untouched
        for mem_id in [m1.memory_id, m2.memory_id]:
            mem = await adapter.fetch("agent-1", mem_id)
            assert mem is not None
            assert mem.status.value == "active"

    async def test_dry_run_skips_preflight(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory("old fact")
        m2 = make_memory("new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = make_conflict(m1, m2, conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        await adapter.store_conflict(conflict)

        det = make_consolidator([])
        plan = await det.plan("agent-1", adapter, dry_run=True)
        assert plan is not None

        # Delete resources that would fail preflight in a live run
        await adapter.delete("agent-1", m1.memory_id)
        await adapter.delete_conflict("agent-1", conflict.conflict_id)

        # Dry run should still complete without raising
        executed = await det.execute(plan, adapter)
        assert executed.executed_at is not None
