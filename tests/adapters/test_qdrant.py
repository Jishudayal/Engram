"""Tests for QdrantAdapter.

Runs the full contract suite via AdapterContractTests, then adds
Qdrant-specific tests: null-embedding sentinel, backend_name, and
metadata filter behaviour.

Uses QdrantAdapter(":memory:", ...) so no external Qdrant server is needed.
"""

from __future__ import annotations

import pytest

from engram.adapters.qdrant import QdrantAdapter
from engram.core.exceptions import NotFoundError
from tests.adapters.contract import AdapterContractTests, _m

_COLLECTION = "test_engram"
_VECTOR_SIZE = 2  # all contract tests use 2D embeddings


class TestQdrantAdapter(AdapterContractTests):
    """Full contract suite wired to QdrantAdapter in-process (:memory:)."""

    def make_adapter(self) -> QdrantAdapter:
        return QdrantAdapter(":memory:", _COLLECTION, vector_size=_VECTOR_SIZE)


class TestQdrantSpecific:
    """Behaviour specific to QdrantAdapter not covered by the contract suite."""

    @pytest.fixture(autouse=True)
    async def _open(self) -> None:
        async with QdrantAdapter(":memory:", _COLLECTION, vector_size=_VECTOR_SIZE) as adapter:
            self.adapter = adapter
            yield

    def test_backend_name(self) -> None:
        assert QdrantAdapter(":memory:", _COLLECTION, vector_size=2).backend_name == "qdrant"

    def test_invalid_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            QdrantAdapter(":memory:", _COLLECTION, vector_size=2, distance="Bad")

    async def test_open_creates_collection_idempotent(self) -> None:
        # second open on same in-memory client — collection already exists, must not raise
        async with QdrantAdapter(":memory:", _COLLECTION, vector_size=_VECTOR_SIZE):
            pass

    async def test_null_embedding_stored_and_retrieved(self) -> None:
        m = _m(text="no embedding")  # embedding=None
        await self.adapter.store(m)
        result = await self.adapter.fetch("a1", m.memory_id)
        assert result is not None
        assert result.embedding is None

    async def test_null_embedding_excluded_from_search(self) -> None:
        await self.adapter.store(_m(text="no emb"))  # no embedding
        await self.adapter.store(_m(text="has emb", embedding=[1.0, 0.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].memory.text == "has emb"

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

    async def test_count_with_no_embedding_sentinel(self) -> None:
        await self.adapter.store(_m(text="no emb"))
        await self.adapter.store(_m(text="has emb", embedding=[1.0, 0.0]))
        # count should include both — sentinel is a storage detail, not a status
        assert await self.adapter.count("a1") == 2

    async def test_search_score_clamped_to_zero(self) -> None:
        # Qdrant Cosine similarity can return slightly negative values for
        # nearly-orthogonal vectors; score must stay in [0, 1].
        await self.adapter.store(_m(embedding=[0.0, 1.0]))
        results = await self.adapter.search("a1", [1.0, 0.0])
        assert len(results) == 1
        assert results[0].score >= 0.0

    async def test_importable_from_root(self) -> None:
        from engram import QdrantAdapter as QA

        assert QA is QdrantAdapter
