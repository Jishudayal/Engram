"""Tests for AbstractAdapter contract enforcement and exception hierarchy (sub-step 1.7)."""

from datetime import UTC, datetime
from typing import Any

import pytest

from memnotary.adapters.base import AbstractAdapter
from memnotary.adapters.memory import InMemoryAdapter
from memnotary.core.constants import ConflictType, MemoryStatus, ResolutionStatus
from memnotary.core.exceptions import AdapterError, MemnotaryError, NotFoundError
from memnotary.core.models import ConflictRecord, Memory, SearchResult

# ---------------------------------------------------------------------------
# Minimal concrete adapter used across all tests
# ---------------------------------------------------------------------------


class _MinimalAdapter(AbstractAdapter):
    """Stub that satisfies the full interface contract with no-op implementations."""

    @property
    def backend_name(self) -> str:
        return "test"

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


def make_memory(**overrides: object) -> Memory:
    defaults: dict = {"agent_id": "agent-1", "text": "The refund policy is 30 days."}
    return Memory(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestAbstractAdapterEnforcement:
    def test_cannot_instantiate_abstract_directly(self) -> None:
        with pytest.raises(TypeError):
            AbstractAdapter()  # type: ignore[abstract]

    def test_missing_backend_name_raises(self) -> None:
        class MissingName(AbstractAdapter):
            async def store(self, memory: Memory) -> None:
                pass

            async def update(self, memory: Memory) -> None:
                pass

            async def delete(self, agent_id: str, memory_id: str) -> bool:
                return False

            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return None

            async def search(
                self, agent_id, query_embedding, *, top_k=10, score_threshold=None, filters=None
            ) -> list:
                return []  # type: ignore[override]

            async def list_all(self, agent_id, *, status=None, limit=None, offset=0) -> list:
                return []  # type: ignore[override]

            async def close(self) -> None:
                pass

        with pytest.raises(TypeError):
            MissingName()  # type: ignore[abstract]

    def test_missing_one_method_raises(self) -> None:
        class MissingClose(AbstractAdapter):
            @property
            def backend_name(self) -> str:
                return "x"

            async def store(self, memory: Memory) -> None:
                pass

            async def update(self, memory: Memory) -> None:
                pass

            async def delete(self, agent_id: str, memory_id: str) -> bool:
                return False

            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return None

            async def search(
                self, agent_id, query_embedding, *, top_k=10, score_threshold=None, filters=None
            ) -> list:
                return []  # type: ignore[override]

            async def list_all(self, agent_id, *, status=None, limit=None, offset=0) -> list:
                return []  # type: ignore[override]

            # close() not implemented

        with pytest.raises(TypeError):
            MissingClose()  # type: ignore[abstract]

    def test_full_implementation_instantiates(self) -> None:
        adapter = _MinimalAdapter()
        assert adapter is not None

    def test_full_implementation_is_abstract_adapter(self) -> None:
        assert isinstance(_MinimalAdapter(), AbstractAdapter)


# ---------------------------------------------------------------------------
# backend_name
# ---------------------------------------------------------------------------


class TestBackendName:
    def test_backend_name_accessible(self) -> None:
        assert _MinimalAdapter().backend_name == "test"

    def test_backend_name_is_str(self) -> None:
        assert isinstance(_MinimalAdapter().backend_name, str)


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    async def test_aenter_returns_adapter(self) -> None:
        async with _MinimalAdapter() as adapter:
            assert isinstance(adapter, AbstractAdapter)

    async def test_aexit_calls_close(self) -> None:
        close_count = 0

        class TrackClose(_MinimalAdapter):
            async def close(self) -> None:
                nonlocal close_count
                close_count += 1

        async with TrackClose():
            pass

        assert close_count == 1

    async def test_aexit_calls_close_on_exception(self) -> None:
        close_count = 0

        class TrackClose(_MinimalAdapter):
            async def close(self) -> None:
                nonlocal close_count
                close_count += 1

        with pytest.raises(RuntimeError):
            async with TrackClose():
                raise RuntimeError("something went wrong")

        assert close_count == 1

    async def test_aenter_returns_same_instance(self) -> None:
        adapter = _MinimalAdapter()
        async with adapter as ctx:
            assert ctx is adapter


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_not_found_error_is_adapter_error(self) -> None:
        assert isinstance(NotFoundError("x"), AdapterError)

    def test_not_found_error_is_engram_error(self) -> None:
        assert isinstance(NotFoundError("x"), MemnotaryError)

    def test_adapter_error_is_engram_error(self) -> None:
        assert isinstance(AdapterError("x"), MemnotaryError)

    def test_not_found_error_is_exception(self) -> None:
        assert isinstance(NotFoundError("x"), Exception)

    def test_engram_error_message_preserved(self) -> None:
        err = MemnotaryError("something failed")
        assert "something failed" in str(err)

    def test_adapter_error_message_preserved(self) -> None:
        err = AdapterError("backend unreachable")
        assert "backend unreachable" in str(err)

    def test_not_found_error_message_preserved(self) -> None:
        err = NotFoundError("mem-abc not found")
        assert "mem-abc" in str(err)

    def test_can_catch_not_found_as_adapter_error(self) -> None:
        with pytest.raises(AdapterError):
            raise NotFoundError("gone")

    def test_can_catch_adapter_error_as_engram_error(self) -> None:
        with pytest.raises(MemnotaryError):
            raise AdapterError("io error")


# ---------------------------------------------------------------------------
# Interface method signatures (called via stub to confirm callability)
# ---------------------------------------------------------------------------


class TestStoreInterface:
    async def test_store_accepts_memory(self) -> None:
        await _MinimalAdapter().store(make_memory())

    async def test_store_returns_none(self) -> None:
        result = await _MinimalAdapter().store(make_memory())
        assert result is None


class TestUpdateInterface:
    async def test_update_accepts_memory(self) -> None:
        await _MinimalAdapter().update(make_memory())

    async def test_update_returns_none(self) -> None:
        result = await _MinimalAdapter().update(make_memory())
        assert result is None


class TestDeleteInterface:
    async def test_delete_returns_bool(self) -> None:
        result = await _MinimalAdapter().delete("agent-1", "mem-123")
        assert isinstance(result, bool)

    async def test_delete_false_when_not_found(self) -> None:
        assert await _MinimalAdapter().delete("agent-1", "mem-missing") is False


class TestFetchInterface:
    async def test_fetch_returns_none_when_absent(self) -> None:
        result = await _MinimalAdapter().fetch("agent-1", "mem-123")
        assert result is None

    async def test_fetch_can_return_memory(self) -> None:
        m = make_memory()

        class ReturnMemory(_MinimalAdapter):
            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return m

        result = await ReturnMemory().fetch("agent-1", m.memory_id)
        assert result is m


class TestSearchInterface:
    async def test_search_returns_list(self) -> None:
        result = await _MinimalAdapter().search("agent-1", [0.1, 0.2, 0.3])
        assert result == []

    async def test_search_accepts_all_keyword_args(self) -> None:
        result = await _MinimalAdapter().search(
            "agent-1",
            [0.1, 0.2],
            top_k=5,
            score_threshold=0.7,
            filters={"source": "document"},
        )
        assert isinstance(result, list)

    async def test_search_default_top_k(self) -> None:
        result = await _MinimalAdapter().search("agent-1", [0.0])
        assert isinstance(result, list)


class TestListAllInterface:
    async def test_list_all_returns_list(self) -> None:
        result = await _MinimalAdapter().list_all("agent-1")
        assert result == []

    async def test_list_all_accepts_status_filter(self) -> None:
        result = await _MinimalAdapter().list_all("agent-1", status=MemoryStatus.ACTIVE)
        assert isinstance(result, list)

    async def test_list_all_accepts_pagination_args(self) -> None:
        result = await _MinimalAdapter().list_all("agent-1", limit=50, offset=10)
        assert isinstance(result, list)

    async def test_list_all_no_status_returns_all(self) -> None:
        result = await _MinimalAdapter().list_all("agent-1", status=None)
        assert isinstance(result, list)


class TestCloseInterface:
    async def test_close_returns_none(self) -> None:
        result = await _MinimalAdapter().close()
        assert result is None

    async def test_close_idempotent(self) -> None:
        adapter = _MinimalAdapter()
        await adapter.close()
        await adapter.close()  # must not raise


# ---------------------------------------------------------------------------
# Bulk helpers — concrete default implementations
# ---------------------------------------------------------------------------


class TestStoreBatch:
    async def test_store_batch_calls_store_for_each(self) -> None:
        stored: list[str] = []

        class TrackStore(_MinimalAdapter):
            async def store(self, memory: Memory) -> None:
                stored.append(memory.memory_id)

        memories = [make_memory(text=f"fact {i}") for i in range(3)]
        await TrackStore().store_batch(memories)
        assert len(stored) == 3

    async def test_store_batch_returns_none(self) -> None:
        result = await _MinimalAdapter().store_batch([make_memory()])
        assert result is None

    async def test_store_batch_empty_list_is_noop(self) -> None:
        await _MinimalAdapter().store_batch([])  # must not raise

    async def test_store_batch_can_be_overridden(self) -> None:
        called_with: list[list[Memory]] = []

        class NativeBulk(_MinimalAdapter):
            async def store_batch(self, memories: list[Memory]) -> None:
                called_with.append(memories)

        memories = [make_memory(), make_memory()]
        await NativeBulk().store_batch(memories)
        assert called_with == [memories]


class TestFetchBatch:
    async def test_fetch_batch_returns_dict(self) -> None:
        result = await _MinimalAdapter().fetch_batch("agent-1", ["mem-a", "mem-b"])
        assert isinstance(result, dict)

    async def test_fetch_batch_omits_absent_ids(self) -> None:
        result = await _MinimalAdapter().fetch_batch("agent-1", ["mem-missing"])
        assert result == {}

    async def test_fetch_batch_includes_found_records(self) -> None:
        m = make_memory()

        class ReturnOne(_MinimalAdapter):
            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return m if memory_id == m.memory_id else None

        result = await ReturnOne().fetch_batch("agent-1", [m.memory_id, "absent"])
        assert m.memory_id in result
        assert result[m.memory_id] is m
        assert "absent" not in result

    async def test_fetch_batch_empty_list_returns_empty_dict(self) -> None:
        result = await _MinimalAdapter().fetch_batch("agent-1", [])
        assert result == {}

    async def test_fetch_batch_calls_fetch_with_agent_id(self) -> None:
        seen_agents: list[str] = []

        class TrackFetch(_MinimalAdapter):
            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                seen_agents.append(agent_id)
                return None

        await TrackFetch().fetch_batch("tenant-x", ["m1", "m2"])
        assert seen_agents == ["tenant-x", "tenant-x"]


class TestDeleteBatch:
    async def test_delete_batch_returns_int(self) -> None:
        result = await _MinimalAdapter().delete_batch("agent-1", ["m1", "m2"])
        assert isinstance(result, int)

    async def test_delete_batch_counts_deleted(self) -> None:
        delete_counts = {"mem-a": True, "mem-b": False, "mem-c": True}

        class PartialDelete(_MinimalAdapter):
            async def delete(self, agent_id: str, memory_id: str) -> bool:
                return delete_counts.get(memory_id, False)

        deleted = await PartialDelete().delete_batch("agent-1", list(delete_counts))
        assert deleted == 2

    async def test_delete_batch_zero_when_none_found(self) -> None:
        assert await _MinimalAdapter().delete_batch("agent-1", ["x", "y"]) == 0

    async def test_delete_batch_empty_list_returns_zero(self) -> None:
        assert await _MinimalAdapter().delete_batch("agent-1", []) == 0


# ---------------------------------------------------------------------------
# count and exists — concrete defaults delegate to list_all / fetch
# ---------------------------------------------------------------------------


class TestCount:
    async def test_count_returns_int(self) -> None:
        result = await _MinimalAdapter().count("agent-1")
        assert isinstance(result, int)

    async def test_count_zero_when_no_memories(self) -> None:
        assert await _MinimalAdapter().count("agent-1") == 0

    async def test_count_reflects_list_all_result(self) -> None:
        memories = [make_memory(text=f"fact {i}") for i in range(5)]

        class ReturnFive(_MinimalAdapter):
            async def list_all(
                self,
                agent_id: str,
                *,
                status: MemoryStatus | None = None,
                limit: int | None = None,
                offset: int = 0,
            ) -> list[Memory]:
                return memories

        assert await ReturnFive().count("agent-1") == 5

    async def test_count_passes_status_to_list_all(self) -> None:
        seen_status: list[MemoryStatus | None] = []

        class TrackListAll(_MinimalAdapter):
            async def list_all(
                self,
                agent_id: str,
                *,
                status: MemoryStatus | None = None,
                limit: int | None = None,
                offset: int = 0,
            ) -> list[Memory]:
                seen_status.append(status)
                return []

        await TrackListAll().count("agent-1", status=MemoryStatus.ACTIVE)
        assert seen_status == [MemoryStatus.ACTIVE]

    async def test_count_can_be_overridden(self) -> None:
        class FastCount(_MinimalAdapter):
            async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
                return 42

        assert await FastCount().count("agent-1") == 42


class TestExists:
    async def test_exists_returns_bool(self) -> None:
        result = await _MinimalAdapter().exists("agent-1", "mem-123")
        assert isinstance(result, bool)

    async def test_exists_false_when_fetch_returns_none(self) -> None:
        assert await _MinimalAdapter().exists("agent-1", "mem-missing") is False

    async def test_exists_true_when_fetch_returns_memory(self) -> None:
        m = make_memory()

        class ReturnMemory(_MinimalAdapter):
            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                return m

        assert await ReturnMemory().exists("agent-1", m.memory_id) is True

    async def test_exists_delegates_agent_id_to_fetch(self) -> None:
        seen: list[tuple[str, str]] = []

        class TrackFetch(_MinimalAdapter):
            async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
                seen.append((agent_id, memory_id))
                return None

        await TrackFetch().exists("agent-x", "mem-y")
        assert seen == [("agent-x", "mem-y")]


# ---------------------------------------------------------------------------
# SearchResult conflict fields — adapters return defaults; Engram enriches
# ---------------------------------------------------------------------------


class TestSearchResultConflictDefaults:
    async def test_search_result_conflict_flag_defaults_false(self) -> None:
        m = make_memory()
        sr = SearchResult(memory=m, score=0.9, rank=1)
        assert sr.conflict_flag is False

    async def test_search_result_conflicts_with_defaults_empty(self) -> None:
        m = make_memory()
        sr = SearchResult(memory=m, score=0.9, rank=1)
        assert sr.conflicts_with == ()

    async def test_search_result_recommended_defaults_true(self) -> None:
        m = make_memory()
        sr = SearchResult(memory=m, score=0.9, rank=1)
        assert sr.recommended is True


# ---------------------------------------------------------------------------
# TestUpdateConflict — InMemoryAdapter.update_conflict (Step 6.2)
# ---------------------------------------------------------------------------


def _make_conflict(mem_a: Memory, mem_b: Memory, *, agent_id: str = "agent-1") -> ConflictRecord:
    return ConflictRecord(
        agent_id=agent_id,
        memory_a_id=mem_a.memory_id,
        memory_b_id=mem_b.memory_id,
        conflict_type=ConflictType.DIRECT_CONTRADICTION,
        confidence=0.92,
    )


class TestUpdateConflict:
    async def test_update_replaces_existing_record(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory()
        m2 = make_memory()
        conflict = _make_conflict(m1, m2)
        await adapter.store_conflict(conflict)

        resolved = conflict.model_copy(
            update={
                "resolution_status": ResolutionStatus.AUTO_RESOLVED,
                "resolved_at": datetime.now(UTC),
            }
        )
        await adapter.update_conflict(resolved)

        fetched = await adapter.fetch_conflict("agent-1", conflict.conflict_id)
        assert fetched is not None
        assert fetched.resolution_status == ResolutionStatus.AUTO_RESOLVED

    async def test_update_not_found_raises(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory()
        m2 = make_memory()
        conflict = _make_conflict(m1, m2)
        # Never stored — should raise
        with pytest.raises(NotFoundError):
            await adapter.update_conflict(conflict)

    async def test_update_wrong_agent_raises(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory()
        m2 = make_memory()
        conflict = _make_conflict(m1, m2, agent_id="agent-a")
        await adapter.store_conflict(conflict)

        # Build a copy scoped to a different agent — treated as absent
        other = ConflictRecord(
            conflict_id=conflict.conflict_id,
            agent_id="agent-b",
            memory_a_id=m1.memory_id,
            memory_b_id=m2.memory_id,
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.92,
        )
        with pytest.raises(NotFoundError):
            await adapter.update_conflict(other)

    async def test_update_deep_copy_isolation(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory()
        m2 = make_memory()
        conflict = _make_conflict(m1, m2)
        await adapter.store_conflict(conflict)

        resolved = conflict.model_copy(
            update={
                "resolution_status": ResolutionStatus.AUTO_RESOLVED,
                "resolved_at": datetime.now(UTC),
            }
        )
        await adapter.update_conflict(resolved)

        # Mutate the instance that was passed to update_conflict — stored record must not change.
        resolved.resolution_notes = "mutated after store"
        fetched = await adapter.fetch_conflict("agent-1", conflict.conflict_id)
        assert fetched is not None
        assert fetched.resolution_notes is None

    async def test_update_reflected_in_list_conflicts(self) -> None:
        adapter = InMemoryAdapter()
        m1 = make_memory()
        m2 = make_memory()
        conflict = _make_conflict(m1, m2)
        await adapter.store_conflict(conflict)

        resolved = conflict.model_copy(
            update={
                "resolution_status": ResolutionStatus.AUTO_RESOLVED,
                "resolved_at": datetime.now(UTC),
            }
        )
        await adapter.update_conflict(resolved)

        # PENDING list is now empty; AUTO_RESOLVED appears in unfiltered list
        pending = await adapter.list_conflicts("agent-1", status=ResolutionStatus.PENDING)
        assert pending == []
        all_conflicts = await adapter.list_conflicts("agent-1")
        assert len(all_conflicts) == 1
        assert all_conflicts[0].resolution_status == ResolutionStatus.AUTO_RESOLVED

    async def test_base_adapter_update_conflict_raises_not_implemented(self) -> None:
        m1 = make_memory()
        m2 = make_memory()
        adapter = _MinimalAdapter()
        conflict = _make_conflict(m1, m2)
        with pytest.raises(NotImplementedError, match="update_conflict"):
            await adapter.update_conflict(conflict)
