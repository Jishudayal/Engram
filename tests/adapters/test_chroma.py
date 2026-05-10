"""Tests for ChromaAdapter.

Runs the full contract suite via AdapterContractTests, then adds
Chroma-specific tests: null-embedding sentinel, score conversion,
backend_name, and metadata filter behaviour.

Uses chromadb.EphemeralClient() so no external Chroma server is needed.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from engram.adapters.chroma import ChromaAdapter
from engram.core.exceptions import NotFoundError
from tests.adapters.contract import AdapterContractTests, _m

_VECTOR_SIZE = 2  # all contract tests use 2D embeddings


def _make_adapter() -> ChromaAdapter:
    # EphemeralClient instances share state within the same process (Chroma
    # uses a process-wide SQLite). Use a unique collection name per adapter
    # instance to ensure tests are fully isolated from each other.
    collection = f"test_{uuid.uuid4().hex}"
    return ChromaAdapter(chromadb.EphemeralClient(), collection, vector_size=_VECTOR_SIZE)


class TestChromaAdapter(AdapterContractTests):
    """Full contract suite wired to ChromaAdapter in-process (EphemeralClient)."""

    def make_adapter(self) -> ChromaAdapter:
        return _make_adapter()


class TestChromaSpecific:
    """Behaviour specific to ChromaAdapter not covered by the contract suite."""

    @pytest.fixture(autouse=True)
    async def _open(self) -> None:
        async with _make_adapter() as adapter:
            self.adapter = adapter
            yield

    def test_backend_name(self) -> None:
        assert _make_adapter().backend_name == "chroma"

    def test_invalid_vector_size_raises(self) -> None:
        with pytest.raises(ValueError, match="vector_size"):
            ChromaAdapter(chromadb.EphemeralClient(), "test", vector_size=0)

    async def test_null_embedding_stored_and_retrieved(self) -> None:
        m = _m(text="no embedding")  # embedding=None
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.embedding is None

    async def test_null_embedding_excluded_from_search(self) -> None:
        await self.adapter.store(_m(text="no emb"))
        await self.adapter.store(_m(text="has emb", embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].memory.text == "has emb"

    async def test_search_identical_vector_score_is_one(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert abs(results[0].score - 1.0) < 1e-6

    async def test_search_orthogonal_vector_score_near_zero(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [0.0, 1.0])
        assert results[0].score < 0.01

    async def test_search_score_always_in_range(self) -> None:
        await self.adapter.store(_m(embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [0.0, 1.0])
        assert all(0.0 <= r.score <= 1.0 for r in results)

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

    async def test_update_wrong_agent_raises_not_found(self) -> None:
        m = _m("a1")
        await self.adapter.store(m)
        from engram.core.models import Memory

        m_other = Memory(memory_id=m.memory_id, agent_id="a2", text="wrong tenant")
        with pytest.raises(NotFoundError):
            await self.adapter.update(m_other)

    async def test_open_is_idempotent(self) -> None:
        await self.adapter.open()  # already open — must not raise or create duplicate
        m = _m()
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None

    async def test_importable_from_root(self) -> None:
        from engram import ChromaAdapter as CA

        assert CA is ChromaAdapter
