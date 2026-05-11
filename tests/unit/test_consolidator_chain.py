"""Unit tests for Phase 4 chain-aware consolidation (step 4.8).

Covers:
  - Consolidator._group_into_chains()   — BFS component detection, deduplication
  - Consolidator._chain_supersession_action() — keeper selection, action shape
  - Consolidator._execute_chain_supersede()   — memory mutations, conflict resolution
  - Consolidator._preflight() for CHAIN_SUPERSEDE — stale-plan guard
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from engram.adapters.memory import InMemoryAdapter
from engram.core.consolidator import Consolidator
from engram.core.constants import (
    ActionType,
    ConflictType,
    ConsolidationTier,
    MemoryStatus,
    ResolutionStatus,
)
from engram.core.exceptions import NotFoundError
from engram.core.models import ConflictRecord, ConsolidationAction, ConsolidationPlan, Memory

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_AGENT = "agent-1"
_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = _T0 + timedelta(days=1)
_T2 = _T0 + timedelta(days=2)


def _mem(text: str, created_at: datetime = _T0, importance: float = 0.5) -> Memory:
    return Memory(agent_id=_AGENT, text=text, created_at=created_at, importance=importance)


def _temporal_conflict(a: Memory, b: Memory) -> ConflictRecord:
    return ConflictRecord(
        agent_id=_AGENT,
        memory_a_id=a.memory_id,
        memory_b_id=b.memory_id,
        conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
        confidence=0.95,
    )


def _direct_conflict(a: Memory, b: Memory) -> ConflictRecord:
    return ConflictRecord(
        agent_id=_AGENT,
        memory_a_id=a.memory_id,
        memory_b_id=b.memory_id,
        conflict_type=ConflictType.DIRECT_CONTRADICTION,
        confidence=0.90,
    )


def _make_consolidator() -> Consolidator:
    async def _noop_llm(prompt: str) -> str:
        return '{"action": "flag", "confidence": 0.0, "reasoning": ""}'

    return Consolidator(llm_fn=_noop_llm)


def _chain_action(
    *memory_ids: str,
    agent_id: str = _AGENT,
) -> ConsolidationAction:
    """Build a minimal CHAIN_SUPERSEDE action with the keeper-last convention."""
    return ConsolidationAction(
        agent_id=agent_id,
        action_type=ActionType.CHAIN_SUPERSEDE,
        tier=ConsolidationTier.AUTO_MERGE,
        confidence=1.0,
        source_memory_ids=memory_ids,
        reasoning="test chain action",
    )


# ---------------------------------------------------------------------------
# _group_into_chains — BFS grouping
# ---------------------------------------------------------------------------


class TestGroupIntoChains:
    def test_triangle_produces_one_chain_zero_solos(self) -> None:
        v1, v2, v3 = _mem("v1"), _mem("v2"), _mem("v3")
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        conflicts = [
            _temporal_conflict(v1, v2),
            _temporal_conflict(v2, v3),
            _temporal_conflict(v1, v3),
        ]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        assert len(chains) == 1
        assert chains[0] == frozenset({v1.memory_id, v2.memory_id, v3.memory_id})
        assert solos == []

    def test_linear_three_produces_one_chain_via_bfs(self) -> None:
        v1, v2, v3 = _mem("v1"), _mem("v2"), _mem("v3")
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        # Only adjacent edges — no v1-v3 direct conflict
        conflicts = [_temporal_conflict(v1, v2), _temporal_conflict(v2, v3)]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        assert len(chains) == 1
        assert chains[0] == frozenset({v1.memory_id, v2.memory_id, v3.memory_id})
        assert solos == []

    def test_pair_only_yields_no_chain_one_solo(self) -> None:
        v1, v2 = _mem("v1"), _mem("v2")
        memory_index = {m.memory_id: m for m in (v1, v2)}
        conflicts = [_temporal_conflict(v1, v2)]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        assert chains == []
        assert len(solos) == 1
        assert solos[0].memory_a_id == v1.memory_id
        assert solos[0].memory_b_id == v2.memory_id

    def test_two_separate_pairs_yield_no_chains_two_solos(self) -> None:
        a1, a2 = _mem("a1"), _mem("a2")
        b1, b2 = _mem("b1"), _mem("b2")
        memory_index = {m.memory_id: m for m in (a1, a2, b1, b2)}
        conflicts = [_temporal_conflict(a1, a2), _temporal_conflict(b1, b2)]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        assert chains == []
        assert len(solos) == 2
        solo_pairs = {frozenset({s.memory_a_id, s.memory_b_id}) for s in solos}
        assert frozenset({a1.memory_id, a2.memory_id}) in solo_pairs
        assert frozenset({b1.memory_id, b2.memory_id}) in solo_pairs

    def test_empty_conflicts_yields_no_chains_no_solos(self) -> None:
        chains, solos = Consolidator._group_into_chains([], {})
        assert chains == []
        assert solos == []

    def test_duplicate_conflict_records_deduplicated_in_solos(self) -> None:
        v1, v2 = _mem("v1"), _mem("v2")
        memory_index = {m.memory_id: m for m in (v1, v2)}
        # Two conflict records for the same pair (simulates stale duplicate PENDING records)
        c1 = _temporal_conflict(v1, v2)
        c2 = _temporal_conflict(v1, v2)
        conflicts = [c1, c2]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        assert chains == []
        assert len(solos) == 1  # deduplicated

    def test_missing_memory_conflict_is_dropped_from_grouping(self) -> None:
        v1, v2, v3 = _mem("v1"), _mem("v2"), _mem("v3")
        # v3 absent from memory_index
        memory_index = {m.memory_id: m for m in (v1, v2)}
        conflicts = [
            _temporal_conflict(v1, v2),
            _temporal_conflict(v2, v3),  # v3 missing — should be dropped
        ]

        chains, solos = Consolidator._group_into_chains(conflicts, memory_index)

        # v1-v2 stays as solo pair; v2-v3 is dropped
        assert chains == []
        assert len(solos) == 1
        assert solos[0].memory_a_id == v1.memory_id


# ---------------------------------------------------------------------------
# plan() — chain routing contract
# ---------------------------------------------------------------------------


class TestPlanChainRouting:
    async def test_triangle_produces_one_chain_supersede_no_pairwise(self) -> None:
        """Triangle of temporal conflicts → exactly one CHAIN_SUPERSEDE, no SUPERSEDE actions."""
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        plan = await _make_consolidator().plan(_AGENT, adapter)

        assert plan is not None
        assert len(plan.would_chain_supersede) == 1
        assert len(plan.would_supersede) == 0

    async def test_solo_pair_produces_pairwise_supersede_not_chain(self) -> None:
        """Two-memory temporal conflict stays a pairwise SUPERSEDE, not CHAIN_SUPERSEDE."""
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        for m in (v1, v2):
            await adapter.store(m)
        await adapter.store_conflict(_temporal_conflict(v1, v2))

        plan = await _make_consolidator().plan(_AGENT, adapter)

        assert plan is not None
        assert len(plan.would_chain_supersede) == 0
        assert len(plan.would_supersede) == 1


# ---------------------------------------------------------------------------
# _chain_supersession_action — keeper selection and action shape
# ---------------------------------------------------------------------------


class TestChainSupersessionAction:
    def test_newest_created_at_is_keeper(self) -> None:
        v1 = _mem("v1", created_at=_T0)
        v2 = _mem("v2", created_at=_T1)
        v3 = _mem("v3", created_at=_T2)
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        cluster = frozenset({v1.memory_id, v2.memory_id, v3.memory_id})

        action = _make_consolidator()._chain_supersession_action(cluster, memory_index, _AGENT)

        # Keeper is last in source_memory_ids
        *stale_ids, keeper_id = action.source_memory_ids
        assert keeper_id == v3.memory_id
        assert set(stale_ids) == {v1.memory_id, v2.memory_id}

    def test_stale_ids_are_sorted(self) -> None:
        v1 = _mem("v1", created_at=_T0)
        v2 = _mem("v2", created_at=_T1)
        v3 = _mem("v3", created_at=_T2)
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        cluster = frozenset({v1.memory_id, v2.memory_id, v3.memory_id})

        action = _make_consolidator()._chain_supersession_action(cluster, memory_index, _AGENT)

        *stale_ids, _ = action.source_memory_ids
        assert stale_ids == sorted(stale_ids)

    def test_tied_timestamps_uses_max_memory_id(self) -> None:
        # All memories share the same created_at — largest memory_id wins.
        ts = datetime(2026, 3, 1, tzinfo=UTC)
        v1 = _mem("v1", created_at=ts)
        v2 = _mem("v2", created_at=ts)
        v3 = _mem("v3", created_at=ts)
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        cluster = frozenset({v1.memory_id, v2.memory_id, v3.memory_id})

        action = _make_consolidator()._chain_supersession_action(cluster, memory_index, _AGENT)

        *_, keeper_id = action.source_memory_ids
        expected_keeper = max(cluster)  # lexicographic max
        assert keeper_id == expected_keeper

    def test_action_type_is_chain_supersede(self) -> None:
        v1, v2, v3 = _mem("v1", _T0), _mem("v2", _T1), _mem("v3", _T2)
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        cluster = frozenset({v1.memory_id, v2.memory_id, v3.memory_id})

        action = _make_consolidator()._chain_supersession_action(cluster, memory_index, _AGENT)

        assert action.action_type == ActionType.CHAIN_SUPERSEDE

    def test_tier_is_auto_merge_confidence_one(self) -> None:
        v1, v2, v3 = _mem("v1", _T0), _mem("v2", _T1), _mem("v3", _T2)
        memory_index = {m.memory_id: m for m in (v1, v2, v3)}
        cluster = frozenset({v1.memory_id, v2.memory_id, v3.memory_id})

        action = _make_consolidator()._chain_supersession_action(cluster, memory_index, _AGENT)

        assert action.tier == ConsolidationTier.AUTO_MERGE
        assert action.confidence == 1.0


# ---------------------------------------------------------------------------
# _execute_chain_supersede — memory mutations
# ---------------------------------------------------------------------------


class TestExecuteChainSupersede:
    async def test_stale_memories_are_superseded(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v1 = await adapter.fetch(_AGENT, v1.memory_id)
        fetched_v2 = await adapter.fetch(_AGENT, v2.memory_id)
        assert fetched_v1 is not None and fetched_v1.status == MemoryStatus.SUPERSEDED
        assert fetched_v2 is not None and fetched_v2.status == MemoryStatus.SUPERSEDED

    async def test_stale_superseded_by_points_to_keeper(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v1 = await adapter.fetch(_AGENT, v1.memory_id)
        fetched_v2 = await adapter.fetch(_AGENT, v2.memory_id)
        assert fetched_v1 is not None and fetched_v1.superseded_by == v3.memory_id
        assert fetched_v2 is not None and fetched_v2.superseded_by == v3.memory_id

    async def test_keeper_supersedes_list_contains_all_stale(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v3 = await adapter.fetch(_AGENT, v3.memory_id)
        assert fetched_v3 is not None
        assert v1.memory_id in fetched_v3.supersedes
        assert v2.memory_id in fetched_v3.supersedes

    async def test_keeper_status_remains_active(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v3 = await adapter.fetch(_AGENT, v3.memory_id)
        assert fetched_v3 is not None
        assert fetched_v3.status == MemoryStatus.ACTIVE

    async def test_temporal_conflicts_are_auto_resolved(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        all_conflicts = await adapter.list_conflicts(_AGENT)
        for c in all_conflicts:
            assert c.resolution_status == ResolutionStatus.AUTO_RESOLVED

    async def test_direct_contradiction_in_cluster_stays_pending(self) -> None:
        """DIRECT_CONTRADICTION between cluster members must NOT be resolved by chain supersede."""
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        # temporal chain
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))
        # direct contradiction between v1 and v3 (different conflict type)
        direct = _direct_conflict(v1, v3)
        await adapter.store_conflict(direct)

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_direct = await adapter.fetch_conflict(_AGENT, direct.conflict_id)
        assert fetched_direct is not None
        assert fetched_direct.resolution_status == ResolutionStatus.PENDING

    async def test_linear_chain_no_direct_conflict_executes_cleanly(self) -> None:
        """Linear chain (v1-v2, v2-v3 only) runs without error despite missing v1-v3 conflict."""
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        # Only adjacent edges
        await adapter.store_conflict(_temporal_conflict(v1, v2))
        await adapter.store_conflict(_temporal_conflict(v2, v3))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        executed = await consolidator.execute(plan, adapter)
        assert executed.executed_at is not None

        fetched_v3 = await adapter.fetch(_AGENT, v3.memory_id)
        assert fetched_v3 is not None
        assert fetched_v3.status == MemoryStatus.ACTIVE

    async def test_keeper_inherits_max_importance_from_cluster(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0, importance=0.9)  # highest importance
        v2 = _mem("v2", _T1, importance=0.5)
        v3 = _mem("v3", _T2, importance=0.3)  # newest = keeper, but lowest importance
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v3 = await adapter.fetch(_AGENT, v3.memory_id)
        assert fetched_v3 is not None
        assert fetched_v3.importance == pytest.approx(0.9)

    async def test_existing_keeper_supersedes_are_preserved(self) -> None:
        """Keeper.supersedes entries that predate execute() must not be cleared."""
        adapter = InMemoryAdapter()
        # v0 was already superseded by v3 before the chain execute
        v0 = _mem("v0", _T0 - timedelta(days=1))
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        # Manually set v3.supersedes to already contain v0
        v3.supersedes = [v0.memory_id]
        for m in (v0, v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        await consolidator.execute(plan, adapter)

        fetched_v3 = await adapter.fetch(_AGENT, v3.memory_id)
        assert fetched_v3 is not None
        assert v0.memory_id in fetched_v3.supersedes
        assert v1.memory_id in fetched_v3.supersedes
        assert v2.memory_id in fetched_v3.supersedes


# ---------------------------------------------------------------------------
# _preflight — stale-plan guard for CHAIN_SUPERSEDE
# ---------------------------------------------------------------------------


class TestPreflightChainSupersede:
    async def test_stale_plan_raises_not_found_error(self) -> None:
        """If all conflicts were resolved between plan() and execute(), preflight raises."""
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None

        # Simulate another execute() resolving all conflicts before we run ours
        first_plan = await consolidator.plan(_AGENT, adapter)
        assert first_plan is not None
        await consolidator.execute(first_plan, adapter)

        # Now the original plan is stale — all conflicts already resolved
        with pytest.raises(NotFoundError, match="no PENDING conflicts remain"):
            await consolidator.execute(plan, adapter)

    async def test_valid_plan_does_not_raise(self) -> None:
        adapter = InMemoryAdapter()
        v1 = _mem("v1", _T0)
        v2 = _mem("v2", _T1)
        v3 = _mem("v3", _T2)
        for m in (v1, v2, v3):
            await adapter.store(m)
        for pair in [(v1, v2), (v2, v3), (v1, v3)]:
            await adapter.store_conflict(_temporal_conflict(*pair))

        consolidator = _make_consolidator()
        plan = await consolidator.plan(_AGENT, adapter)
        assert plan is not None
        executed = await consolidator.execute(plan, adapter)
        assert executed is not None


# ---------------------------------------------------------------------------
# ConsolidationPlan.would_chain_supersede property (step 4.7)
# ---------------------------------------------------------------------------


class TestWouldChainSupersede:
    def test_returns_chain_supersede_actions_only(self) -> None:
        v1, v2, v3 = _mem("v1", _T0), _mem("v2", _T1), _mem("v3", _T2)
        chain_action = _chain_action(v1.memory_id, v2.memory_id, v3.memory_id)
        supersede_action = ConsolidationAction(
            agent_id=_AGENT,
            action_type=ActionType.SUPERSEDE,
            tier=ConsolidationTier.AUTO_MERGE,
            confidence=0.95,
            source_memory_ids=(v1.memory_id, v2.memory_id),
            reasoning="older superseded",
        )
        plan = ConsolidationPlan(
            agent_id=_AGENT,
            actions=(chain_action, supersede_action),
        )

        result = plan.would_chain_supersede

        assert len(result) == 1
        assert result[0].action_type == ActionType.CHAIN_SUPERSEDE

    def test_empty_when_no_chain_supersede_actions(self) -> None:
        v1, v2 = _mem("v1", _T0), _mem("v2", _T1)
        supersede_action = ConsolidationAction(
            agent_id=_AGENT,
            action_type=ActionType.SUPERSEDE,
            tier=ConsolidationTier.AUTO_MERGE,
            confidence=0.95,
            source_memory_ids=(v1.memory_id, v2.memory_id),
            reasoning="supersede only",
        )
        plan = ConsolidationPlan(agent_id=_AGENT, actions=(supersede_action,))

        assert plan.would_chain_supersede == ()

    def test_would_supersede_excludes_chain_supersede(self) -> None:
        v1, v2, v3 = _mem("v1", _T0), _mem("v2", _T1), _mem("v3", _T2)
        chain_action = _chain_action(v1.memory_id, v2.memory_id, v3.memory_id)
        plan = ConsolidationPlan(agent_id=_AGENT, actions=(chain_action,))

        assert plan.would_supersede == ()
        assert len(plan.would_chain_supersede) == 1
