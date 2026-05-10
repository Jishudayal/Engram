"""Adapter contract test suite.

Any AbstractAdapter implementation can run this full suite by subclassing
AdapterContractTests and implementing make_adapter():

    from tests.adapters.contract import AdapterContractTests

    class TestMyAdapter(AdapterContractTests):
        def make_adapter(self) -> MyAdapter:
            return MyAdapter(...)

pytest does NOT collect AdapterContractTests directly — its name does not
start with 'Test'. Only concrete subclasses in test modules are collected.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from engram import AbstractAdapter, Memory, MemoryStatus, NotFoundError
from engram.core.constants import SourceType
from engram.core.models import ProvenanceRecord, SearchResult

# Fixed base time so ordering tests are not sensitive to the system clock.
_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _m(agent_id: str = "a1", *, text: str = "hello world", **kw: Any) -> Memory:
    return Memory(agent_id=agent_id, text=text, **kw)  # type: ignore[arg-type]


class AdapterContractTests:
    """Mixin: subclass and implement make_adapter() to run the full contract suite.

    A fresh adapter is opened before each test and closed after via the
    autouse _open fixture.
    """

    adapter: AbstractAdapter  # set by the autouse _open fixture before each test

    def make_adapter(self) -> AbstractAdapter:
        raise NotImplementedError("Subclasses must implement make_adapter()")

    @pytest.fixture(autouse=True)
    async def _open(self) -> AsyncGenerator[None, None]:
        async with self.make_adapter() as adapter:
            self.adapter = adapter
            yield

    # ------------------------------------------------------------------
    # store / fetch / delete
    # ------------------------------------------------------------------

    async def test_store_then_fetch_round_trip(self) -> None:
        m = _m()
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.memory_id == m.memory_id

    async def test_fetch_unknown_returns_none(self) -> None:
        assert await self.adapter.fetch("a1", "no-such-id") is None

    async def test_fetch_wrong_agent_returns_none(self) -> None:
        m = _m("a1")
        await self.adapter.store(m)
        assert await self.adapter.fetch("a2", m.memory_id) is None

    async def test_store_same_memory_id_different_agents_isolated(self) -> None:
        """Two agents storing the same memory_id must not overwrite each other."""
        shared_id = "00000000-0000-0000-0000-000000000001"
        m1 = Memory(memory_id=shared_id, agent_id="a1", text="agent1 text")
        m2 = Memory(memory_id=shared_id, agent_id="a2", text="agent2 text")
        await self.adapter.store(m1)
        await self.adapter.store(m2)
        result1 = await self.adapter.fetch("a1", shared_id)
        result2 = await self.adapter.fetch("a2", shared_id)
        assert result1 is not None and result1.text == "agent1 text"
        assert result2 is not None and result2.text == "agent2 text"

    async def test_store_is_upsert(self) -> None:
        m = _m()
        await self.adapter.store(m)
        overwrite = Memory(memory_id=m.memory_id, agent_id="a1", text="overwritten")
        await self.adapter.store(overwrite)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.text == "overwritten"

    async def test_delete_existing_returns_true(self) -> None:
        m = _m()
        await self.adapter.store(m)
        assert await self.adapter.delete("a1", m.memory_id) is True

    async def test_delete_missing_returns_false(self) -> None:
        assert await self.adapter.delete("a1", "no-such-id") is False

    async def test_delete_removes_record(self) -> None:
        m = _m()
        await self.adapter.store(m)
        await self.adapter.delete("a1", m.memory_id)
        assert await self.adapter.fetch("a1", m.memory_id) is None

    async def test_delete_wrong_agent_returns_false(self) -> None:
        m = _m("a1")
        await self.adapter.store(m)
        assert await self.adapter.delete("a2", m.memory_id) is False

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    async def test_update_replaces_record(self) -> None:
        m = _m()
        await self.adapter.store(m)
        updated = Memory(memory_id=m.memory_id, agent_id="a1", text="new text")
        await self.adapter.update(updated)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.text == "new text"

    async def test_update_missing_raises_not_found(self) -> None:
        with pytest.raises(NotFoundError):
            await self.adapter.update(_m())

    async def test_update_wrong_agent_raises_not_found(self) -> None:
        m = _m("a1")
        await self.adapter.store(m)
        m_other = Memory(memory_id=m.memory_id, agent_id="a2", text="wrong tenant")
        with pytest.raises(NotFoundError):
            await self.adapter.update(m_other)

    # ------------------------------------------------------------------
    # list_all
    # ------------------------------------------------------------------

    async def test_list_all_returns_memories(self) -> None:
        await self.adapter.store(_m())
        await self.adapter.store(_m())
        assert len(await self.adapter.list_all("a1")) == 2

    async def test_list_all_empty_for_unknown_agent(self) -> None:
        await self.adapter.store(_m("a1"))  # store something so the adapter is non-empty
        assert await self.adapter.list_all("nobody") == []

    async def test_list_all_tenant_isolated(self) -> None:
        await self.adapter.store(_m("a1"))
        await self.adapter.store(_m("a2"))
        assert len(await self.adapter.list_all("a1")) == 1
        assert len(await self.adapter.list_all("a2")) == 1

    async def test_list_all_ordered_by_created_at(self) -> None:
        m3 = _m(created_at=_T0 + timedelta(seconds=2))
        m1 = _m(created_at=_T0)
        m2 = _m(created_at=_T0 + timedelta(seconds=1))
        for m in (m3, m1, m2):  # intentionally out of order
            await self.adapter.store(m)
        result = await self.adapter.list_all("a1")
        assert [r.memory_id for r in result] == [m1.memory_id, m2.memory_id, m3.memory_id]

    async def test_list_all_status_filter(self) -> None:
        active = _m(status=MemoryStatus.ACTIVE)
        archived = _m(status=MemoryStatus.ARCHIVED)
        await self.adapter.store(active)
        await self.adapter.store(archived)
        result = await self.adapter.list_all("a1", status=MemoryStatus.ACTIVE)
        assert len(result) == 1
        assert result[0].memory_id == active.memory_id

    async def test_list_all_limit(self) -> None:
        for i in range(5):
            await self.adapter.store(_m(created_at=_T0 + timedelta(seconds=i)))
        assert len(await self.adapter.list_all("a1", limit=3)) == 3

    async def test_list_all_offset(self) -> None:
        memories = [_m(created_at=_T0 + timedelta(seconds=i)) for i in range(4)]
        for m in memories:
            await self.adapter.store(m)
        result = await self.adapter.list_all("a1", offset=2)
        assert len(result) == 2
        assert result[0].memory_id == memories[2].memory_id

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    async def test_search_returns_list(self) -> None:
        assert isinstance(await self.adapter.search("a1", [1.0, 0.0]), list)

    async def test_search_empty_store_returns_empty(self) -> None:
        assert await self.adapter.search("a1", [1.0, 0.0]) == []

    async def test_search_results_ordered_by_score_desc(self) -> None:
        # Memories at known angles to query [1.0, 0.0]:
        #   high → cosine ≈ 1.0   mid → cosine ≈ 0.707   low → cosine = 0.0
        high = _m(embedding=[1.0, 0.0])
        mid = _m(embedding=[0.5, 0.5])
        low = _m(embedding=[0.0, 1.0])
        for m in (low, mid, high):  # store out of expected order
            await self.adapter.store(m)
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert results[0].memory.memory_id == high.memory_id
        assert results[1].memory.memory_id == mid.memory_id
        assert results[2].memory.memory_id == low.memory_id

    async def test_search_rank_starts_at_1(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert results[0].rank == 1

    async def test_search_ranks_contiguous(self) -> None:
        for i in range(3):
            await self.adapter.store(_m(embedding=[float(i + 1), 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    async def test_search_score_in_range(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert all(0.0 <= r.score <= 1.0 for r in results)

    async def test_search_respects_top_k(self) -> None:
        for i in range(5):
            await self.adapter.store(_m(embedding=[float(i + 1), 0.0]))
        assert len(await self.adapter.search("a1", [1.0, 0.0], top_k=3)) == 3

    async def test_search_respects_score_threshold(self) -> None:
        # [1.0, 0.0] vs [1.0, 0.0] → score 1.0 (above 0.5)
        # [1.0, 0.0] vs [0.0, 1.0] → score 0.0 (below 0.5)
        await self.adapter.store(_m(text="match", embedding=[1.0, 0.0]))
        await self.adapter.store(_m(text="no match", embedding=[0.0, 1.0]))
        results = await self.adapter.search("a1", [1.0, 0.0], score_threshold=0.5)
        assert len(results) == 1
        assert results[0].memory.text == "match"

    async def test_search_tenant_isolated(self) -> None:
        await self.adapter.store(_m("a1", embedding=[1.0, 0.0]))
        await self.adapter.store(_m("a2", embedding=[1.0, 0.0]))
        assert len(await self.adapter.search("a1", [1.0, 0.0])) == 1
        assert len(await self.adapter.search("a2", [1.0, 0.0])) == 1

    async def test_search_skips_no_embedding(self) -> None:
        await self.adapter.store(_m(text="no embedding"))
        await self.adapter.store(_m(text="has embedding", embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].memory.text == "has embedding"

    async def test_search_result_type(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_search_top_k_zero_returns_empty(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        assert await self.adapter.search("a1", [1.0, 0.0], top_k=0) == []

    # ------------------------------------------------------------------
    # Round-trip fidelity
    # ------------------------------------------------------------------

    async def test_store_fetch_memory_without_embedding(self) -> None:
        m = _m()  # embedding=None
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.embedding is None

    async def test_store_fetch_preserves_all_fields(self) -> None:
        """Full field equality — the primary serialization-correctness test for each adapter."""
        m = _m(
            text="complex memory",
            embedding=[0.1, 0.2],
            metadata={"nested": {"k": [1, 2, 3]}},
            supersedes=["id1", "id2"],
            importance=0.87,
        )
        prov = ProvenanceRecord(memory_id=m.memory_id, source_type=SourceType.DOCUMENT)
        m.attach_provenance(prov)
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result == m

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    async def test_store_batch_stores_all(self) -> None:
        memories = [_m() for _ in range(3)]
        await self.adapter.store_batch(memories)
        for m in memories:
            assert await self.adapter.fetch("a1", m.memory_id) is not None

    async def test_store_batch_is_upsert(self) -> None:
        m = _m()
        await self.adapter.store(m)
        replaced = Memory(memory_id=m.memory_id, agent_id="a1", text="replaced")
        await self.adapter.store_batch([replaced])
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.text == "replaced"

    async def test_fetch_batch_returns_dict(self) -> None:
        memories = [_m() for _ in range(2)]
        await self.adapter.store_batch(memories)
        ids = [m.memory_id for m in memories]
        result = await self.adapter.fetch_batch("a1", ids)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(ids)

    async def test_fetch_batch_omits_absent(self) -> None:
        m = _m()
        await self.adapter.store(m)
        result = await self.adapter.fetch_batch("a1", [m.memory_id, "missing"])
        assert m.memory_id in result
        assert "missing" not in result

    async def test_delete_batch_returns_count(self) -> None:
        memories = [_m() for _ in range(3)]
        await self.adapter.store_batch(memories)
        count = await self.adapter.delete_batch(
            "a1", [memories[0].memory_id, memories[1].memory_id, "missing"]
        )
        assert count == 2

    async def test_delete_batch_removes_records(self) -> None:
        memories = [_m() for _ in range(2)]
        await self.adapter.store_batch(memories)
        await self.adapter.delete_batch("a1", [m.memory_id for m in memories])
        for m in memories:
            assert await self.adapter.fetch("a1", m.memory_id) is None

    async def test_count_returns_total(self) -> None:
        await self.adapter.store_batch([_m() for _ in range(4)])
        assert await self.adapter.count("a1") == 4

    async def test_count_with_status_filter(self) -> None:
        await self.adapter.store(_m(status=MemoryStatus.ACTIVE))
        await self.adapter.store(_m(status=MemoryStatus.ACTIVE))
        await self.adapter.store(_m(status=MemoryStatus.ARCHIVED))
        assert await self.adapter.count("a1", status=MemoryStatus.ACTIVE) == 2

    async def test_exists_true(self) -> None:
        m = _m()
        await self.adapter.store(m)
        assert await self.adapter.exists("a1", m.memory_id) is True

    async def test_exists_false(self) -> None:
        assert await self.adapter.exists("a1", "no-such-id") is False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def test_close_is_idempotent(self) -> None:
        await self.adapter.close()
        await self.adapter.close()  # must not raise
