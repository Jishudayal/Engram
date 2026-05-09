"""LangChain integration for Engram.

Exposes Engram as a LangChain VectorStore so it can be used anywhere
LangChain expects a VectorStore — retrieval chains, as_retriever(),
ConversationalRetrievalChain, etc.

Engram reliability features apply transparently:
  - similarity_search returns conflict-enriched results when a detector is configured.
  - aadd_texts uses store_batch() semantics — per-document contradiction detection is
    skipped for bulk adds. Call engram.scan_contradictions(agent_id) after ingestion
    to detect conflicts across the batch in one pass.

Install:
    pip install "engram[langchain]"

Sync methods (add_texts, similarity_search, …) work only outside a running
event loop. In async applications use the async variants instead:
aadd_texts, asimilarity_search, asimilarity_search_with_score.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable
from uuid import uuid4

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "EngramVectorStore requires langchain-core. "
        'Install it with: pip install "engram[langchain]"'
    ) from exc

from engram.core.models import Memory
from engram.engram import Engram

__all__ = ["EngramVectorStore"]


def _run(coro: Any) -> Any:
    """Run a coroutine synchronously from a non-async context.

    Raises RuntimeError when called from a running event loop — the caller
    must use the async (a*) variant instead. asyncio.run() creates a fresh
    event loop; adapter state (e.g. connection pools) must therefore be
    accessible from that loop. For persistent-pool adapters in long-running
    applications, always use async methods directly.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Cannot call synchronous VectorStore methods from a running event loop. "
            "Use the async variants (aadd_texts, asimilarity_search, etc.) instead."
        )
    return asyncio.run(coro)


class EngramVectorStore(VectorStore):
    """LangChain VectorStore backed by an Engram adapter.

    Wraps an already-open Engram instance so it can be plugged into any
    LangChain component that consumes a VectorStore.

    Args:
        engram:     An already-open Engram instance.
        embeddings: LangChain Embeddings for text → vector conversion.
        agent_id:   Tenant scope — all documents are stored under this agent.

    Example (async)::

        from engram import Engram, InMemoryAdapter
        from engram.integrations.langchain import EngramVectorStore
        from langchain_openai import OpenAIEmbeddings

        async with Engram(InMemoryAdapter()) as eng:
            store = EngramVectorStore(eng, OpenAIEmbeddings(), agent_id="my-agent")
            await store.aadd_texts(["Paris is the capital of France."])
            docs = await store.asimilarity_search("capital of France", k=3)
    """

    def __init__(self, engram: Engram, embeddings: Embeddings, agent_id: str) -> None:
        self._engram = engram
        self._embeddings = embeddings
        self._agent_id = agent_id

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    # ------------------------------------------------------------------
    # Internal mapping helpers
    # ------------------------------------------------------------------

    def _to_memory(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
        doc_id: str,
    ) -> Memory:
        return Memory(
            memory_id=doc_id,
            agent_id=self._agent_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

    @staticmethod
    def _to_document(memory: Memory) -> Document:
        # Mirror memory_id into metadata so callers can round-trip back to Engram.
        meta = {**memory.metadata, "_memory_id": memory.memory_id}
        return Document(page_content=memory.text, metadata=meta)

    # ------------------------------------------------------------------
    # Async primary implementations
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and store texts. Returns the IDs assigned to each document.

        Uses store_batch() — per-document contradiction detection is skipped.
        Call engram.scan_contradictions(agent_id) after ingestion to detect
        conflicts across the batch.
        """
        text_list = list(texts)
        meta_list = metadatas if metadatas is not None else [{} for _ in text_list]
        id_list = ids if ids is not None else [str(uuid4()) for _ in text_list]

        vectors = await self._embeddings.aembed_documents(text_list)
        memories = [
            self._to_memory(t, v, m, did)
            for t, v, m, did in zip(text_list, vectors, meta_list, id_list, strict=True)
        ]
        await self._engram.store_batch(memories)
        return id_list

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents. Returns Documents without scores."""
        pairs = await self.asimilarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in pairs]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents. Returns (Document, score) pairs."""
        vector = await self._embeddings.aembed_query(query)
        score_threshold: float | None = kwargs.get("score_threshold")
        # LangChain passes metadata filters as "filter" (singular); also accept "filters".
        filters: dict[str, Any] | None = kwargs.get("filter") or kwargs.get("filters")
        results = await self._engram.search(
            self._agent_id,
            vector,
            top_k=k,
            score_threshold=score_threshold,
            filters=filters,
        )
        return [(self._to_document(r.memory), r.score) for r in results]

    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID. Returns True if at least one was deleted."""
        if not ids:
            return False
        count = await self._engram.delete_batch(self._agent_id, ids)
        return count > 0

    # ------------------------------------------------------------------
    # Sync implementations (shim → async primary)
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed and store texts synchronously. Use aadd_texts() in async contexts."""
        return _run(self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs))

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents synchronously. Use asimilarity_search() in async contexts."""
        return _run(self.asimilarity_search(query, k=k, **kwargs))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search synchronously. Returns (Document, score) pairs. Use async variant in async contexts."""
        return _run(self.asimilarity_search_with_score(query, k=k, **kwargs))

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete documents by ID synchronously. Use adelete() in async contexts."""
        return _run(self.adelete(ids=ids, **kwargs))

    # ------------------------------------------------------------------
    # from_texts classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> EngramVectorStore:
        """Create a store and immediately add texts.

        Required kwargs:
            engram (Engram):   An already-open Engram instance.
            agent_id (str):    Tenant scope for all documents.

        Example::

            store = EngramVectorStore.from_texts(
                ["Paris is the capital of France."],
                OpenAIEmbeddings(),
                engram=eng,
                agent_id="my-agent",
            )
        """
        engram: Engram | None = kwargs.pop("engram", None)
        agent_id: str | None = kwargs.pop("agent_id", None)
        if engram is None or agent_id is None:
            raise ValueError(
                "from_texts() requires 'engram' and 'agent_id' keyword arguments."
            )
        store = cls(engram, embedding, agent_id)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
