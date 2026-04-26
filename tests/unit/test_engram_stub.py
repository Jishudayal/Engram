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
    async def test_health_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="Step 4"):
            await Engram(_StubAdapter()).health("agent-1")

    async def test_consolidate_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="Step 6"):
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
