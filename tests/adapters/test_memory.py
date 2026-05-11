"""Tests for InMemoryAdapter.

Runs the full contract suite via AdapterContractTests, then adds
InMemory-specific tests: cosine accuracy, metadata filter behaviour,
and backend_name.
"""

from __future__ import annotations

import pytest

from memnotary import MemoryStatus
from memnotary.adapters.memory import InMemoryAdapter
from tests.adapters.contract import AdapterContractTests, _m


class TestInMemoryAdapter(AdapterContractTests):
    """Full contract suite wired to InMemoryAdapter."""

    def make_adapter(self) -> InMemoryAdapter:
        return InMemoryAdapter()


class TestInMemorySpecific:
    """Behaviour specific to InMemoryAdapter not covered by the contract suite."""

    @pytest.fixture(autouse=True)
    async def _open(self) -> None:
        async with InMemoryAdapter() as adapter:
            self.adapter = adapter
            yield

    def test_backend_name(self) -> None:
        assert InMemoryAdapter().backend_name == "memory"

    async def test_search_identical_vector_score_is_one(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert abs(results[0].score - 1.0) < 1e-9

    async def test_search_orthogonal_vector_score_is_zero(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [0.0, 1.0])
        assert abs(results[0].score - 0.0) < 1e-9

    async def test_search_metadata_filter_includes_match(self) -> None:
        match = _m(embedding=[1.0, 0.0], metadata={"type": "doc"})
        other = _m(embedding=[1.0, 0.0], metadata={"type": "api"})
        await self.adapter.store(match)
        await self.adapter.store(other)
        results = await self.adapter.search("a1", [1.0, 0.0], filters={"type": "doc"})
        assert len(results) == 1
        assert results[0].memory.memory_id == match.memory_id

    async def test_search_metadata_filter_excludes_all(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0], metadata={"type": "doc"}))
        results = await self.adapter.search("a1", [1.0, 0.0], filters={"type": "none"})
        assert results == []

    async def test_search_negative_cosine_clamped_to_zero(self) -> None:
        """Opposite-direction vectors produce cosine=-1; must be clamped, not crash."""
        await self.adapter.store(_m(embedding=[-1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].score == 0.0

    async def test_store_aliasing_protected(self) -> None:
        """Mutating the original Memory after store() must not affect stored state."""
        m = _m()
        await self.adapter.store(m)
        m.status = MemoryStatus.ARCHIVED
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.status == MemoryStatus.ACTIVE

    async def test_fetch_aliasing_protected(self) -> None:
        """Mutating a fetched Memory must not affect stored state."""
        m = _m()
        await self.adapter.store(m)
        fetched = await self.adapter.fetch("a1", m.memory_id)
        assert fetched is not None
        fetched.status = MemoryStatus.ARCHIVED
        result2 = await self.adapter.fetch("a1", m.memory_id)
        assert result2 is not None
        assert result2.status == MemoryStatus.ACTIVE

    async def test_importable_from_root(self) -> None:
        from memnotary import InMemoryAdapter as IMA

        assert IMA is InMemoryAdapter
