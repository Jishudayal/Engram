"""Tests for the Engram facade class (sub-step 1.8)."""

from typing import Any

import pytest

from engram import (
    AbstractAdapter,
    ActionType,
    AdapterError,
    ConflictType,
    ConsolidationTier,
    Engram,
    EngramError,
    Memory,
    MemoryStatus,
    NotFoundError,
    ResolutionStatus,
    RiskLevel,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Trackable stub adapter
# ---------------------------------------------------------------------------


class _StubAdapter(AbstractAdapter):
    """Trackable no-op adapter for Engram delegation tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._store: dict[str, Memory] = {}
        self.closed = False
        self._search_kwargs: dict[str, Any] = {}
        self._list_all_kwargs: dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return "stub"

    async def __aenter__(self) -> "_StubAdapter":
        self.calls.append("__aenter__")
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.calls.append("__aexit__")
        await self.close()

    async def store(self, memory: Memory) -> None:
        self.calls.append("store")
        self._store[memory.memory_id] = memory

    async def store_batch(self, memories: list[Memory]) -> None:
        self.calls.append("store_batch")
        for memory in memories:
            self._store[memory.memory_id] = memory

    async def update(self, memory: Memory) -> None:
        self.calls.append("update")

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        self.calls.append("delete")
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        self.calls.append("delete_batch")
        count = 0
        for mid in memory_ids:
            if mid in self._store:
                del self._store[mid]
                count += 1
        return count

    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        self.calls.append("fetch")
        return self._store.get(memory_id)

    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        self.calls.append("fetch_batch")
        return {mid: self._store[mid] for mid in memory_ids if mid in self._store}

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.calls.append("search")
        self._search_kwargs = {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "filters": filters,
        }
        return []

    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        self.calls.append("list_all")
        self._list_all_kwargs = {"status": status, "limit": limit, "offset": offset}
        return []

    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        self.calls.append("count")
        return len(self._store)

    async def exists(self, agent_id: str, memory_id: str) -> bool:
        self.calls.append("exists")
        return memory_id in self._store

    async def close(self) -> None:
        self.closed = True


def make_memory(**overrides: object) -> Memory:
    return Memory(**{"agent_id": "agent-1", "text": "fact", **overrides})


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestEngramInstantiation:
    def test_takes_adapter(self) -> None:
        eng = Engram(_StubAdapter())
        assert eng is not None

    def test_adapter_property_returns_adapter(self) -> None:
        adapter = _StubAdapter()
        assert Engram(adapter).adapter is adapter

    def test_adapter_is_abstract_adapter(self) -> None:
        assert isinstance(Engram(_StubAdapter()).adapter, AbstractAdapter)

    def test_non_adapter_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="AbstractAdapter"):
            Engram("not_an_adapter")  # type: ignore[arg-type]

    def test_type_error_includes_actual_type(self) -> None:
        with pytest.raises(TypeError, match="str"):
            Engram("oops")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# backend_name and __repr__
# ---------------------------------------------------------------------------


class TestEngramBackendName:
    def test_delegates_to_adapter(self) -> None:
        assert Engram(_StubAdapter()).backend_name == "stub"


class TestEngramRepr:
    def test_includes_backend_name(self) -> None:
        assert repr(Engram(_StubAdapter())) == "Engram(adapter='stub')"


# ---------------------------------------------------------------------------
# Async context manager — lifecycle and delegation
# ---------------------------------------------------------------------------


class TestEngramContextManager:
    async def test_aenter_returns_engram(self) -> None:
        async with Engram(_StubAdapter()) as eng:
            assert isinstance(eng, Engram)

    async def test_aenter_returns_same_instance(self) -> None:
        eng = Engram(_StubAdapter())
        async with eng as ctx:
            assert ctx is eng

    async def test_aexit_closes_adapter(self) -> None:
        adapter = _StubAdapter()
        async with Engram(adapter):
            pass
        assert adapter.closed is True

    async def test_aexit_closes_adapter_on_exception(self) -> None:
        adapter = _StubAdapter()
        with pytest.raises(RuntimeError):
            async with Engram(adapter):
                raise RuntimeError("crash")
        assert adapter.closed is True


class TestEngramContextManagerDelegation:
    async def test_aenter_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        async with Engram(adapter):
            pass
        assert "__aenter__" in adapter.calls

    async def test_aexit_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        async with Engram(adapter):
            pass
        assert "__aexit__" in adapter.calls

    async def test_aenter_before_aexit(self) -> None:
        adapter = _StubAdapter()
        async with Engram(adapter):
            pass
        assert adapter.calls.index("__aenter__") < adapter.calls.index("__aexit__")


# ---------------------------------------------------------------------------
# Explicit close()
# ---------------------------------------------------------------------------


class TestEngramClose:
    async def test_close_closes_adapter(self) -> None:
        adapter = _StubAdapter()
        eng = Engram(adapter)
        await eng.close()
        assert adapter.closed is True

    async def test_close_is_idempotent(self) -> None:
        adapter = _StubAdapter()
        eng = Engram(adapter)
        await eng.close()
        await eng.close()  # must not raise


# ---------------------------------------------------------------------------
# Delegation — each method calls the corresponding adapter method
# ---------------------------------------------------------------------------


class TestEngramStore:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).store(make_memory())
        assert "store" in adapter.calls

    async def test_returns_none(self) -> None:
        assert await Engram(_StubAdapter()).store(make_memory()) is None

    async def test_memory_reaches_adapter(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        await Engram(adapter).store(m)
        assert m.memory_id in adapter._store


class TestEngramStoreBatch:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).store_batch([make_memory(), make_memory()])
        assert "store_batch" in adapter.calls

    async def test_returns_none(self) -> None:
        assert await Engram(_StubAdapter()).store_batch([make_memory()]) is None

    async def test_memories_reach_adapter(self) -> None:
        adapter = _StubAdapter()
        memories = [make_memory(), make_memory()]
        await Engram(adapter).store_batch(memories)
        for m in memories:
            assert m.memory_id in adapter._store


class TestEngramUpdate:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).update(make_memory())
        assert "update" in adapter.calls

    async def test_returns_none(self) -> None:
        assert await Engram(_StubAdapter()).update(make_memory()) is None


class TestEngramDelete:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).delete("agent-1", "mem-x")
        assert "delete" in adapter.calls

    async def test_returns_true_when_found(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        adapter._store[m.memory_id] = m
        result = await Engram(adapter).delete("agent-1", m.memory_id)
        assert result is True

    async def test_returns_false_when_not_found(self) -> None:
        result = await Engram(_StubAdapter()).delete("agent-1", "missing")
        assert result is False


class TestEngramDeleteBatch:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).delete_batch("agent-1", ["a", "b"])
        assert "delete_batch" in adapter.calls

    async def test_returns_count_deleted(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        adapter._store[m.memory_id] = m
        count = await Engram(adapter).delete_batch("agent-1", [m.memory_id, "missing"])
        assert count == 1

    async def test_returns_zero_when_none_found(self) -> None:
        count = await Engram(_StubAdapter()).delete_batch("agent-1", ["x", "y"])
        assert count == 0


class TestEngramFetch:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).fetch("agent-1", "mem-x")
        assert "fetch" in adapter.calls

    async def test_returns_none_when_absent(self) -> None:
        result = await Engram(_StubAdapter()).fetch("agent-1", "missing")
        assert result is None

    async def test_returns_stored_memory(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        adapter._store[m.memory_id] = m
        result = await Engram(adapter).fetch("agent-1", m.memory_id)
        assert result is m


class TestEngramFetchBatch:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).fetch_batch("agent-1", ["a", "b"])
        assert "fetch_batch" in adapter.calls

    async def test_returns_dict(self) -> None:
        result = await Engram(_StubAdapter()).fetch_batch("agent-1", [])
        assert isinstance(result, dict)

    async def test_absent_ids_omitted(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        adapter._store[m.memory_id] = m
        result = await Engram(adapter).fetch_batch("agent-1", [m.memory_id, "missing"])
        assert m.memory_id in result
        assert "missing" not in result


class TestEngramSearch:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).search("agent-1", [0.1, 0.2])
        assert "search" in adapter.calls

    async def test_returns_list(self) -> None:
        result = await Engram(_StubAdapter()).search("agent-1", [0.1])
        assert isinstance(result, list)

    async def test_passes_kwargs_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).search(
            "agent-1",
            [0.1],
            top_k=5,
            score_threshold=0.8,
            filters={"source": "doc"},
        )
        assert adapter._search_kwargs == {
            "top_k": 5,
            "score_threshold": 0.8,
            "filters": {"source": "doc"},
        }

    async def test_default_kwargs(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).search("agent-1", [0.1])
        assert adapter._search_kwargs["top_k"] == 10
        assert adapter._search_kwargs["score_threshold"] is None
        assert adapter._search_kwargs["filters"] is None


class TestEngramListAll:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).list_all("agent-1")
        assert "list_all" in adapter.calls

    async def test_returns_list(self) -> None:
        result = await Engram(_StubAdapter()).list_all("agent-1")
        assert isinstance(result, list)

    async def test_passes_kwargs_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).list_all(
            "agent-1",
            status=MemoryStatus.ACTIVE,
            limit=50,
            offset=10,
        )
        assert adapter._list_all_kwargs == {
            "status": MemoryStatus.ACTIVE,
            "limit": 50,
            "offset": 10,
        }

    async def test_default_kwargs(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).list_all("agent-1")
        assert adapter._list_all_kwargs["status"] is None
        assert adapter._list_all_kwargs["limit"] is None
        assert adapter._list_all_kwargs["offset"] == 0


class TestEngramCount:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).count("agent-1")
        assert "count" in adapter.calls

    async def test_returns_int(self) -> None:
        result = await Engram(_StubAdapter()).count("agent-1")
        assert isinstance(result, int)

    async def test_reflects_stored_memories(self) -> None:
        adapter = _StubAdapter()
        adapter._store["x"] = make_memory()
        adapter._store["y"] = make_memory()
        assert await Engram(adapter).count("agent-1") == 2


class TestEngramExists:
    async def test_delegates_to_adapter(self) -> None:
        adapter = _StubAdapter()
        await Engram(adapter).exists("agent-1", "mem-x")
        assert "exists" in adapter.calls

    async def test_returns_false_when_absent(self) -> None:
        result = await Engram(_StubAdapter()).exists("agent-1", "missing")
        assert result is False

    async def test_returns_true_when_present(self) -> None:
        adapter = _StubAdapter()
        m = make_memory()
        adapter._store[m.memory_id] = m
        assert await Engram(adapter).exists("agent-1", m.memory_id) is True


# ---------------------------------------------------------------------------
# Stubs for future steps
# ---------------------------------------------------------------------------


class TestEngramNotImplementedStubs:
    async def test_consolidate_without_consolidator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="consolidate\\(\\) requires a Consolidator"):
            await Engram(_StubAdapter()).consolidate("agent-1")


# ---------------------------------------------------------------------------
# Public API — importable from package root
# ---------------------------------------------------------------------------


class TestPublicImports:
    def test_engram_importable_from_root(self) -> None:
        from engram import Engram as E

        assert E is Engram

    def test_exceptions_importable_from_root(self) -> None:
        assert issubclass(NotFoundError, AdapterError)
        assert issubclass(AdapterError, EngramError)

    def test_models_importable_from_root(self) -> None:
        from engram import ConsolidationPlan, HealthScore, Memory, SearchResult

        assert Memory is not None
        assert SearchResult is not None
        assert HealthScore is not None
        assert ConsolidationPlan is not None

    def test_abstract_adapter_importable_from_root(self) -> None:
        from engram import AbstractAdapter as AA

        assert AA is AbstractAdapter

    def test_consolidation_enums_importable_from_root(self) -> None:
        assert ActionType is not None
        assert ConflictType is not None
        assert ResolutionStatus is not None
        assert ConsolidationTier is not None
        assert RiskLevel is not None

    def test_version_available(self) -> None:
        import engram

        assert engram.__version__ == "0.1.0-alpha"

    def test_contradiction_detector_importable_from_root(self) -> None:
        from engram import ClassificationResult, ContradictionDetector, LLMClassifyFn

        assert ContradictionDetector is not None
        assert ClassificationResult is not None
        assert LLMClassifyFn is not None


# ---------------------------------------------------------------------------
# Engram with ContradictionDetector — step 5.2
# ---------------------------------------------------------------------------

# These tests use InMemoryAdapter directly because _StubAdapter does not
# implement conflict storage. InMemoryAdapter is the reference implementation.

from engram.adapters.memory import InMemoryAdapter  # noqa: E402
from engram.core.contradiction import ContradictionDetector  # noqa: E402


def _make_detector(responses: list[str], **kwargs: object) -> ContradictionDetector:
    it = iter(responses)

    async def mock_llm(prompt: str) -> str:
        return next(it, '{"verdict": "unrelated", "confidence": 0.0, "summary": ""}')

    return ContradictionDetector(llm_fn=mock_llm, **kwargs)  # type: ignore[arg-type]


_CONTRADICTION_RESP = (
    '{"verdict": "direct_contradiction", "confidence": 0.93, "summary": "Rate limits conflict"}'
)
_UNRELATED_RESP = (
    '{"verdict": "unrelated", "confidence": 0.90, "summary": ""}'
)


class TestEngramReprWithDetector:
    def test_repr_without_detector(self) -> None:
        assert repr(Engram(_StubAdapter())) == "Engram(adapter='stub')"

    def test_repr_with_detector(self) -> None:
        det = _make_detector([])
        r = repr(Engram(_StubAdapter(), detector=det))
        assert "ContradictionDetector" in r
        assert "stub" in r


class TestEngramStoreTimeDetection:
    async def test_store_without_detector_no_conflict_stored(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        mem_a = make_memory(embedding=[1.0, 0.0])
        mem_b = make_memory(embedding=[0.99, 0.01])
        await eng.store(mem_a)
        await eng.store(mem_b)
        conflicts = await adapter.list_conflicts("agent-1")
        assert conflicts == []

    async def test_store_with_no_embedding_skips_detection(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP])
        eng = Engram(adapter, detector=det)
        mem = make_memory(embedding=None)
        await eng.store(mem)
        assert await adapter.list_conflicts("agent-1") == []

    async def test_store_first_memory_no_candidates(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP])
        eng = Engram(adapter, detector=det)
        await eng.store(make_memory(embedding=[1.0, 0.0]))
        assert await adapter.list_conflicts("agent-1") == []

    async def test_store_triggers_detection_and_stores_conflict(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP], cluster_threshold=0.80)
        eng = Engram(adapter, detector=det)
        mem_a = make_memory(embedding=[1.0, 0.01], text="Rate limit is 100 req/min")
        mem_b = make_memory(embedding=[0.99, 0.02], text="Rate limit is 500 req/min")
        await eng.store(mem_a)
        await eng.store(mem_b)
        conflicts = await adapter.list_conflicts("agent-1")
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert conflicts[0].description == "Rate limits conflict"

    async def test_store_batch_skips_detection(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP], cluster_threshold=0.80)
        eng = Engram(adapter, detector=det)
        mem_a = make_memory(embedding=[1.0, 0.01], text="Rate limit is 100")
        mem_b = make_memory(embedding=[0.99, 0.02], text="Rate limit is 500")
        await eng.store_batch([mem_a, mem_b])
        assert await adapter.list_conflicts("agent-1") == []


class TestEngramSearchEnrichment:
    async def _setup(
        self,
    ) -> tuple[Engram, InMemoryAdapter, Memory, Memory, Memory]:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP], cluster_threshold=0.80)
        eng = Engram(adapter, detector=det)
        mem_a = make_memory(embedding=[1.0, 0.01], text="Rate limit is 100 req/min")
        mem_b = make_memory(embedding=[0.99, 0.02], text="Rate limit is 500 req/min")
        mem_c = make_memory(embedding=[-1.0, 0.0], text="Office is in London")
        await eng.store(mem_a)
        await eng.store(mem_b)
        await eng.store(mem_c)
        return eng, adapter, mem_a, mem_b, mem_c

    async def test_search_without_detector_returns_default_flags(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        mem = make_memory(embedding=[1.0, 0.0])
        await adapter.store(mem)
        results = await eng.search("agent-1", [1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].conflict_flag is False
        assert results[0].conflicts_with == ()
        assert results[0].recommended is True

    async def test_search_no_conflicts_returns_unchanged_results(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([])
        eng = Engram(adapter, detector=det)
        mem = make_memory(embedding=[1.0, 0.0])
        await adapter.store(mem)
        results = await eng.search("agent-1", [1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].conflict_flag is False

    async def test_search_flags_conflicting_memories(self) -> None:
        eng, adapter, mem_a, mem_b, mem_c = await self._setup()
        results = await eng.search("agent-1", [1.0, 0.01], top_k=3)
        by_id = {r.memory.memory_id: r for r in results}
        assert by_id[mem_a.memory_id].conflict_flag is True
        assert by_id[mem_b.memory_id].conflict_flag is True
        assert by_id[mem_c.memory_id].conflict_flag is False

    async def test_search_conflict_summary_populated(self) -> None:
        eng, adapter, mem_a, mem_b, _ = await self._setup()
        results = await eng.search("agent-1", [1.0, 0.01], top_k=3)
        by_id = {r.memory.memory_id: r for r in results}
        assert by_id[mem_a.memory_id].conflict_summary == "Rate limits conflict"

    async def test_search_lower_ranked_conflicting_result_not_recommended(self) -> None:
        eng, adapter, mem_a, mem_b, _ = await self._setup()
        results = await eng.search("agent-1", [1.0, 0.01], top_k=3)
        by_id = {r.memory.memory_id: r for r in results}
        r_a, r_b = by_id[mem_a.memory_id], by_id[mem_b.memory_id]
        # Both conflict with each other; exactly one should be not recommended
        assert r_a.recommended is True or r_b.recommended is True
        assert not (r_a.recommended and r_b.recommended)

    async def test_search_non_conflicting_result_stays_recommended(self) -> None:
        eng, adapter, _, _, mem_c = await self._setup()
        results = await eng.search("agent-1", [1.0, 0.01], top_k=3)
        by_id = {r.memory.memory_id: r for r in results}
        assert by_id[mem_c.memory_id].recommended is True

    async def test_search_conflicts_with_contains_other_id(self) -> None:
        eng, adapter, mem_a, mem_b, _ = await self._setup()
        results = await eng.search("agent-1", [1.0, 0.01], top_k=3)
        by_id = {r.memory.memory_id: r for r in results}
        assert mem_b.memory_id in by_id[mem_a.memory_id].conflicts_with
        assert mem_a.memory_id in by_id[mem_b.memory_id].conflicts_with


class TestEngramScanContradictions:
    async def test_without_detector_raises_value_error(self) -> None:
        eng = Engram(InMemoryAdapter())
        with pytest.raises(ValueError, match="ContradictionDetector"):
            await eng.scan_contradictions("agent-1")

    async def test_with_detector_returns_list(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_UNRELATED_RESP], cluster_threshold=0.80)
        eng = Engram(adapter, detector=det)
        mem_a = make_memory(embedding=[1.0, 0.01], text="fact A")
        mem_b = make_memory(embedding=[0.99, 0.02], text="fact B")
        await adapter.store(mem_a)
        await adapter.store(mem_b)
        result = await eng.scan_contradictions("agent-1")
        assert isinstance(result, list)

    async def test_with_detector_stores_new_conflicts(self) -> None:
        adapter = InMemoryAdapter()
        det = _make_detector([_CONTRADICTION_RESP], cluster_threshold=0.80)
        eng = Engram(adapter, detector=det)
        mem_a = make_memory(embedding=[1.0, 0.01], text="Rate limit is 100")
        mem_b = make_memory(embedding=[0.99, 0.02], text="Rate limit is 500")
        await adapter.store(mem_a)
        await adapter.store(mem_b)
        result = await eng.scan_contradictions("agent-1")
        assert len(result) == 1
        stored = await adapter.list_conflicts("agent-1")
        assert len(stored) == 1


# ---------------------------------------------------------------------------
# Engram.consolidate() — step 6.3
# ---------------------------------------------------------------------------

from engram.core.consolidator import Consolidator  # noqa: E402
from engram.core.constants import ConflictType  # noqa: E402
from engram.core.models import ConflictRecord  # noqa: E402


def _make_consolidator(responses: list[str], **kwargs: object) -> Consolidator:
    it = iter(responses)

    async def mock_llm(prompt: str) -> str:
        return next(it, '{"action": "flag", "confidence": 0.0, "reasoning": ""}')

    return Consolidator(llm_fn=mock_llm, **kwargs)  # type: ignore[arg-type]


def _make_temporal_conflict(mem_a: Memory, mem_b: Memory, agent_id: str = "agent-1") -> ConflictRecord:
    return ConflictRecord(
        agent_id=agent_id,
        memory_a_id=mem_a.memory_id,
        memory_b_id=mem_b.memory_id,
        conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
        confidence=0.95,
    )


class TestEngramConsolidate:
    async def test_without_consolidator_raises_value_error(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        with pytest.raises(ValueError, match="consolidator"):
            await eng.consolidate("agent-1")

    async def test_no_pending_conflicts_returns_none(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter, consolidator=_make_consolidator([]))
        result = await eng.consolidate("agent-1")
        assert result is None

    async def test_with_conflicts_returns_executed_plan(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory(text="old fact")
        m2 = make_memory(text="new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(_make_temporal_conflict(m1, m2))

        eng = Engram(adapter, consolidator=_make_consolidator([]))
        plan = await eng.consolidate("agent-1")

        assert plan is not None
        assert plan.executed_at is not None
        assert len(plan.actions) == 1

    async def test_consolidate_archives_superseded_memory(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory(text="old fact")
        m2 = make_memory(text="new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(_make_temporal_conflict(m1, m2))

        eng = Engram(adapter, consolidator=_make_consolidator([]))
        await eng.consolidate("agent-1")

        older = await adapter.fetch("agent-1", m1.memory_id)
        assert older is not None
        assert older.status.value == "superseded"


# ---------------------------------------------------------------------------
# Engram.pending_review() — step 6.5
# ---------------------------------------------------------------------------

_FLAG_RESP = '{"action": "flag", "confidence": 0.50, "reasoning": "Ambiguous."}'


class TestEngramPendingReview:
    async def test_empty_when_no_conflicts(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        result = await eng.pending_review("agent-1")
        assert result == []

    async def test_returns_pending_conflicts(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory(text="fact A")
        m2 = make_memory(text="fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        conflict = ConflictRecord(
            agent_id="agent-1",
            memory_a_id=m1.memory_id,
            memory_b_id=m2.memory_id,
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.9,
        )
        await adapter.store_conflict(conflict)

        eng = Engram(adapter)
        result = await eng.pending_review("agent-1")
        assert len(result) == 1
        assert result[0].conflict_id == conflict.conflict_id

    async def test_empty_after_full_auto_consolidation(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory(text="old fact")
        m2 = make_memory(text="new fact")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(_make_temporal_conflict(m1, m2))

        eng = Engram(adapter, consolidator=_make_consolidator([]))
        await eng.consolidate("agent-1")

        result = await eng.pending_review("agent-1")
        assert result == []

    async def test_non_empty_after_flag_action(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory(text="office in London")
        m2 = make_memory(text="office in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        await adapter.store_conflict(
            ConflictRecord(
                agent_id="agent-1",
                memory_a_id=m1.memory_id,
                memory_b_id=m2.memory_id,
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                confidence=0.9,
            )
        )

        eng = Engram(adapter, consolidator=_make_consolidator([_FLAG_RESP]))
        await eng.consolidate("agent-1")

        result = await eng.pending_review("agent-1")
        assert len(result) == 1

    async def test_works_without_detector_or_consolidator(self) -> None:
        adapter = InMemoryAdapter()
        eng = Engram(adapter)
        result = await eng.pending_review("agent-1")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Engram.provenance_for() + Engram.lineage() — step 7.2
# ---------------------------------------------------------------------------

from engram.core.models import ProvenanceRecord  # noqa: E402
from engram.core.constants import SourceType     # noqa: E402


def _attach_prov(mem: Memory, *, source_type: SourceType = SourceType.DOCUMENT) -> Memory:
    prov = ProvenanceRecord(memory_id=mem.memory_id, source_type=source_type)
    mem.attach_provenance(prov)
    return mem


class TestEngramProvenanceFor:
    async def test_returns_none_for_missing_memory(self) -> None:
        eng = Engram(InMemoryAdapter())
        result = await eng.provenance_for("agent-1", "nonexistent")
        assert result is None

    async def test_returns_none_when_no_provenance_attached(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="no provenance")
        await adapter.store(mem)
        eng = Engram(adapter)
        result = await eng.provenance_for("agent-1", mem.memory_id)
        assert result is None

    async def test_returns_provenance_record(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="has provenance")
        _attach_prov(mem)
        await adapter.store(mem)
        eng = Engram(adapter)
        result = await eng.provenance_for("agent-1", mem.memory_id)
        assert result is not None
        assert result.memory_id == mem.memory_id
        assert result.source_type == SourceType.DOCUMENT

    async def test_returns_correct_provenance_among_many(self) -> None:
        adapter = InMemoryAdapter()
        m1 = _attach_prov(Memory(agent_id="agent-1", text="fact A"))
        m2 = _attach_prov(Memory(agent_id="agent-1", text="fact B"), source_type=SourceType.API)
        await adapter.store(m1)
        await adapter.store(m2)
        eng = Engram(adapter)
        result = await eng.provenance_for("agent-1", m2.memory_id)
        assert result is not None
        assert result.source_type == SourceType.API


class TestEngramLineage:
    async def test_returns_empty_for_missing_memory(self) -> None:
        eng = Engram(InMemoryAdapter())
        result = await eng.lineage("agent-1", "nonexistent")
        assert result == []

    async def test_single_memory_no_chain(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="standalone")
        await adapter.store(mem)
        eng = Engram(adapter)
        result = await eng.lineage("agent-1", mem.memory_id)
        assert len(result) == 1
        assert result[0].memory_id == mem.memory_id

    async def test_two_link_chain_from_older(self) -> None:
        adapter = InMemoryAdapter()
        old = Memory(agent_id="agent-1", text="old fact")
        await adapter.store(old)
        new = Memory(agent_id="agent-1", text="new fact", supersedes=[old.memory_id])
        old.superseded_by = new.memory_id
        await adapter.store(new)
        await adapter.update(old)

        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", old.memory_id)
        assert [m.memory_id for m in chain] == [old.memory_id, new.memory_id]

    async def test_two_link_chain_from_newer(self) -> None:
        adapter = InMemoryAdapter()
        old = Memory(agent_id="agent-1", text="old fact")
        await adapter.store(old)
        new = Memory(agent_id="agent-1", text="new fact", supersedes=[old.memory_id])
        old.superseded_by = new.memory_id
        await adapter.store(new)
        await adapter.update(old)

        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", new.memory_id)
        assert [m.memory_id for m in chain] == [old.memory_id, new.memory_id]

    async def test_three_link_chain_from_middle(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="v1")
        await adapter.store(m1)
        m2 = Memory(agent_id="agent-1", text="v2", supersedes=[m1.memory_id])
        m1.superseded_by = m2.memory_id
        await adapter.store(m2)
        await adapter.update(m1)
        m3 = Memory(agent_id="agent-1", text="v3", supersedes=[m2.memory_id])
        m2.superseded_by = m3.memory_id
        await adapter.store(m3)
        await adapter.update(m2)

        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", m2.memory_id)
        assert [m.memory_id for m in chain] == [m1.memory_id, m2.memory_id, m3.memory_id]

    async def test_chain_oldest_to_newest(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="v1")
        await adapter.store(m1)
        m2 = Memory(agent_id="agent-1", text="v2", supersedes=[m1.memory_id])
        m1.superseded_by = m2.memory_id
        await adapter.store(m2)
        await adapter.update(m1)

        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", m2.memory_id)
        assert chain[0].text == "v1"
        assert chain[1].text == "v2"

    async def test_stops_at_missing_link(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="orphan", supersedes=["ghost-id"])
        await adapter.store(mem)
        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", mem.memory_id)
        assert len(chain) == 1
        assert chain[0].memory_id == mem.memory_id

    async def test_stops_at_cycle(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="cycle A")
        m2 = Memory(agent_id="agent-1", text="cycle B")
        await adapter.store(m1)
        await adapter.store(m2)
        m1.superseded_by = m2.memory_id
        m2.superseded_by = m1.memory_id  # artificial cycle
        await adapter.update(m1)
        await adapter.update(m2)
        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", m1.memory_id)
        assert len(chain) <= 3  # terminates, doesn't loop forever

    async def test_merge_branch_picks_oldest_parent(self) -> None:
        from datetime import UTC, datetime, timedelta
        adapter = InMemoryAdapter()
        older = Memory(agent_id="agent-1", text="older source")
        newer = Memory(agent_id="agent-1", text="newer source")
        await adapter.store(older)
        await adapter.store(newer)
        # Merged memory supersedes both; older has smaller created_at
        merged = Memory(
            agent_id="agent-1",
            text="merged",
            supersedes=[newer.memory_id, older.memory_id],
        )
        await adapter.store(merged)
        eng = Engram(adapter)
        chain = await eng.lineage("agent-1", merged.memory_id)
        # Root should be the older memory (smallest created_at)
        assert chain[0].memory_id == older.memory_id
        assert chain[-1].memory_id == merged.memory_id

    async def test_lineage_after_real_supersede_via_consolidate(self) -> None:
        """lineage() must include the superseded ancestor after Consolidator.execute()."""
        from datetime import UTC, datetime, timedelta
        adapter = InMemoryAdapter()
        t0 = datetime.now(UTC)
        old = Memory(agent_id="agent-1", text="old fact", created_at=t0)
        new = Memory(agent_id="agent-1", text="new fact", created_at=t0 + timedelta(seconds=1))
        await adapter.store(old)
        await adapter.store(new)
        await adapter.store_conflict(_make_temporal_conflict(old, new))

        eng = Engram(adapter, consolidator=_make_consolidator([]))
        await eng.consolidate("agent-1")

        chain = await eng.lineage("agent-1", new.memory_id)
        chain_ids = [m.memory_id for m in chain]
        assert old.memory_id in chain_ids, "superseded ancestor must appear in lineage"
        assert new.memory_id in chain_ids
        assert chain_ids.index(old.memory_id) < chain_ids.index(new.memory_id)


# ---------------------------------------------------------------------------
# Engram.export_provenance() + Engram.export_provenance_json() — step 7.3
# ---------------------------------------------------------------------------

from engram.core.provenance import ProvenanceManifest  # noqa: E402


class TestEngramExportProvenance:
    async def test_returns_none_for_missing_memory(self) -> None:
        eng = Engram(InMemoryAdapter())
        result = await eng.export_provenance("agent-1", "nonexistent")
        assert result is None

    async def test_returns_none_when_no_provenance(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="no provenance")
        await adapter.store(mem)
        eng = Engram(adapter)
        result = await eng.export_provenance("agent-1", mem.memory_id)
        assert result is None

    async def test_returns_provenance_manifest(self) -> None:
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="has provenance"))
        await adapter.store(mem)
        eng = Engram(adapter)
        result = await eng.export_provenance("agent-1", mem.memory_id)
        assert isinstance(result, ProvenanceManifest)
        assert result.memory_id == mem.memory_id
        assert result.source_type == SourceType.DOCUMENT

    async def test_manifest_fields_match_memory(self) -> None:
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="check fields"), source_type=SourceType.API)
        await adapter.store(mem)
        eng = Engram(adapter)
        manifest = await eng.export_provenance("agent-1", mem.memory_id)
        assert manifest is not None
        assert manifest.agent_id == "agent-1"
        assert manifest.text == "check fields"
        assert manifest.source_type == SourceType.API

    async def test_manifest_is_frozen(self) -> None:
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="frozen"))
        await adapter.store(mem)
        eng = Engram(adapter)
        manifest = await eng.export_provenance("agent-1", mem.memory_id)
        assert manifest is not None
        from pydantic import ValidationError
        with pytest.raises((ValidationError, TypeError)):
            manifest.text = "mutated"  # type: ignore[misc]


class TestEngramExportProvenanceJson:
    async def test_returns_none_for_missing_memory(self) -> None:
        eng = Engram(InMemoryAdapter())
        result = await eng.export_provenance_json("agent-1", "nonexistent")
        assert result is None

    async def test_returns_none_when_no_provenance(self) -> None:
        adapter = InMemoryAdapter()
        mem = Memory(agent_id="agent-1", text="no prov")
        await adapter.store(mem)
        result = await Engram(adapter).export_provenance_json("agent-1", mem.memory_id)
        assert result is None

    async def test_returns_json_string(self) -> None:
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="json export"))
        await adapter.store(mem)
        result = await Engram(adapter).export_provenance_json("agent-1", mem.memory_id)
        assert isinstance(result, str)

    async def test_json_is_valid_and_contains_memory_id(self) -> None:
        import json
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="json valid"))
        await adapter.store(mem)
        raw = await Engram(adapter).export_provenance_json("agent-1", mem.memory_id)
        assert raw is not None
        data = json.loads(raw)
        assert data["memory_id"] == mem.memory_id
        assert data["agent_id"] == "agent-1"
        assert data["text"] == "json valid"

    async def test_json_roundtrip_via_manifest(self) -> None:
        import json
        adapter = InMemoryAdapter()
        mem = _attach_prov(Memory(agent_id="agent-1", text="roundtrip"), source_type=SourceType.API)
        await adapter.store(mem)
        raw = await Engram(adapter).export_provenance_json("agent-1", mem.memory_id)
        assert raw is not None
        data = json.loads(raw)
        assert data["source_type"] == "api"


# ---------------------------------------------------------------------------
# Consolidation integration: merged memory gets ProvenanceRecord — step 7.4
# ---------------------------------------------------------------------------

class TestMergeAttachesProvenance:
    async def test_merged_memory_has_provenance(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="office in London")
        m2 = Memory(agent_id="agent-1", text="office in Berlin")
        await adapter.store(m1)
        await adapter.store(m2)
        _MERGE_RESP = '{"action": "merge", "confidence": 0.95, "reasoning": "Combined office fact."}'
        await adapter.store_conflict(
            ConflictRecord(
                agent_id="agent-1",
                memory_a_id=m1.memory_id,
                memory_b_id=m2.memory_id,
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                confidence=0.9,
            )
        )
        eng = Engram(adapter, consolidator=_make_consolidator([_MERGE_RESP]))
        plan = await eng.consolidate("agent-1")

        assert plan is not None
        merged_id = plan.actions[0].result_memory_id
        assert merged_id is not None
        merged = await adapter.fetch("agent-1", merged_id)
        assert merged is not None
        assert merged.provenance is not None

    async def test_merged_provenance_source_type_is_system(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="fact A")
        m2 = Memory(agent_id="agent-1", text="fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        _MERGE_RESP = '{"action": "merge", "confidence": 0.95, "reasoning": "Merged."}'
        await adapter.store_conflict(
            ConflictRecord(
                agent_id="agent-1",
                memory_a_id=m1.memory_id,
                memory_b_id=m2.memory_id,
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                confidence=0.9,
            )
        )
        eng = Engram(adapter, consolidator=_make_consolidator([_MERGE_RESP]))
        plan = await eng.consolidate("agent-1")

        merged_id = plan.actions[0].result_memory_id
        merged = await adapter.fetch("agent-1", merged_id)
        assert merged.provenance.source_type == SourceType.SYSTEM

    async def test_merged_provenance_derived_from_contains_sources(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="fact A")
        m2 = Memory(agent_id="agent-1", text="fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        _MERGE_RESP = '{"action": "merge", "confidence": 0.95, "reasoning": "Merged."}'
        await adapter.store_conflict(
            ConflictRecord(
                agent_id="agent-1",
                memory_a_id=m1.memory_id,
                memory_b_id=m2.memory_id,
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                confidence=0.9,
            )
        )
        eng = Engram(adapter, consolidator=_make_consolidator([_MERGE_RESP]))
        plan = await eng.consolidate("agent-1")

        merged_id = plan.actions[0].result_memory_id
        merged = await adapter.fetch("agent-1", merged_id)
        assert set(merged.provenance.derived_from) == {m1.memory_id, m2.memory_id}

    async def test_export_provenance_after_merge(self) -> None:
        adapter = InMemoryAdapter()
        m1 = Memory(agent_id="agent-1", text="fact A")
        m2 = Memory(agent_id="agent-1", text="fact B")
        await adapter.store(m1)
        await adapter.store(m2)
        _MERGE_RESP = '{"action": "merge", "confidence": 0.95, "reasoning": "Merged fact."}'
        await adapter.store_conflict(
            ConflictRecord(
                agent_id="agent-1",
                memory_a_id=m1.memory_id,
                memory_b_id=m2.memory_id,
                conflict_type=ConflictType.DIRECT_CONTRADICTION,
                confidence=0.9,
            )
        )
        eng = Engram(adapter, consolidator=_make_consolidator([_MERGE_RESP]))
        plan = await eng.consolidate("agent-1")

        merged_id = plan.actions[0].result_memory_id
        manifest = await eng.export_provenance("agent-1", merged_id)
        assert manifest is not None
        assert set(manifest.derived_from) == {m1.memory_id, m2.memory_id}
        assert manifest.source_type == SourceType.SYSTEM
