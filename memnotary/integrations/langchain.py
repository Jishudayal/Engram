"""LangChain integration for Memnotary.

Provides two classes:

  MemnotaryVectorStore         — langchain_core VectorStore backed by any Memnotary adapter.
  MemnotaryChatMessageHistory  — BaseChatMessageHistory for per-session conversation storage.

Memnotary reliability features apply transparently:
  - similarity_search returns conflict-enriched results when a detector is configured.
  - aadd_texts uses store_batch() semantics — per-document contradiction detection is
    skipped for bulk adds. Call engram.scan_contradictions(agent_id) after ingestion
    to detect conflicts across the batch in one pass.
  - MemnotaryChatMessageHistory stores messages without embeddings; vector contradiction
    detection does not fire. Call engram.scan_contradictions(session_id) explicitly
    if you embed messages separately and want conflict detection over chat history.

Install:
    pip install "memnotary[langchain]"

Sync methods work only outside a running event loop. In async applications use the
async variants: aadd_texts, asimilarity_search, aget_messages, aadd_messages, etc.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any, TypeVar
from uuid import uuid4

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:
    raise ImportError(
        "MemnotaryVectorStore and MemnotaryChatMessageHistory require langchain-core. "
        'Install it with: pip install "memnotary[langchain]"'
    ) from exc

from memnotary.core.models import Memory
from memnotary.memnotary import Memnotary

__all__ = ["MemnotaryVectorStore", "MemnotaryChatMessageHistory"]

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
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
        # Close the coroutine before raising so it is never left unawaited.
        coro.close()
        raise RuntimeError(
            "Cannot call synchronous VectorStore methods from a running event loop. "
            "Use the async variants (aadd_texts, asimilarity_search, etc.) instead."
        )
    return asyncio.run(coro)


class MemnotaryVectorStore(VectorStore):
    """LangChain VectorStore backed by a Memnotary adapter.

    Wraps an already-open Memnotary instance so it can be plugged into any
    LangChain component that consumes a VectorStore.

    Args:
        engram:     An already-open Memnotary instance.
        embeddings: LangChain Embeddings for text → vector conversion.
        agent_id:   Tenant scope — all documents are stored under this agent.

    Example (async)::

        from memnotary import Memnotary, InMemoryAdapter
        from memnotary.integrations.langchain import MemnotaryVectorStore
        from langchain_openai import OpenAIEmbeddings

        async with Memnotary(InMemoryAdapter()) as eng:
            store = MemnotaryVectorStore(eng, OpenAIEmbeddings(), agent_id="my-agent")
            await store.aadd_texts(["Paris is the capital of France."])
            docs = await store.asimilarity_search("capital of France", k=3)
    """

    def __init__(self, engram: Memnotary, embeddings: Embeddings, agent_id: str) -> None:
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
        # Mirror memory_id into metadata so callers can round-trip back to Memnotary.
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

        if len(meta_list) != len(text_list):
            raise ValueError(
                f"metadatas length ({len(meta_list)}) must match texts length ({len(text_list)})"
            )
        if len(id_list) != len(text_list):
            raise ValueError(
                f"ids length ({len(id_list)}) must match texts length ({len(text_list)})"
            )

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
    ) -> MemnotaryVectorStore:
        """Create a store and immediately add texts.

        Required kwargs:
            engram (Memnotary):   An already-open Memnotary instance.
            agent_id (str):    Tenant scope for all documents.

        Example::

            store = MemnotaryVectorStore.from_texts(
                ["Paris is the capital of France."],
                OpenAIEmbeddings(),
                engram=eng,
                agent_id="my-agent",
            )
        """
        engram: Memnotary | None = kwargs.pop("engram", None)
        agent_id: str | None = kwargs.pop("agent_id", None)
        if engram is None or agent_id is None:
            raise ValueError("from_texts() requires 'engram' and 'agent_id' keyword arguments.")
        store = cls(engram, embedding, agent_id)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


class MemnotaryChatMessageHistory(BaseChatMessageHistory):
    """LangChain BaseChatMessageHistory backed by a Memnotary adapter.

    Stores each conversation message as a separate Memory under a session_id
    (used as the Memnotary agent_id). Messages are retrieved in insertion order
    (list_all returns oldest-first by created_at).

    Chat-history memories are stored without an embedding, so Memnotary's
    store-time vector contradiction detection does not fire. To detect conflicts
    across a conversation, embed the memories separately and call
    engram.scan_contradictions(session_id) explicitly.

    Args:
        engram:     An already-open Memnotary instance.
        session_id: Conversation scope — all messages are stored under this ID.

    Example (async)::

        from memnotary import Memnotary, InMemoryAdapter
        from memnotary.integrations.langchain import MemnotaryChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage

        async with Memnotary(InMemoryAdapter()) as eng:
            history = MemnotaryChatMessageHistory(eng, session_id="chat-42")
            await history.aadd_messages([
                HumanMessage(content="What is the capital of France?"),
                AIMessage(content="Paris."),
            ])
            msgs = await history.aget_messages()
    """

    def __init__(self, engram: Memnotary, session_id: str) -> None:
        self._engram = engram
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Internal mapping helpers
    # ------------------------------------------------------------------

    def _to_memory(self, message: BaseMessage) -> Memory:
        content = message.content
        meta: dict[str, Any] = {"lc_type": message.type}
        if not isinstance(content, str):
            # Multimodal/structured content — serialise and mark so _to_message can restore it.
            content = json.dumps(content)
            meta["lc_content_json"] = True
        if message.additional_kwargs:
            # Keep each key independently so a single non-serializable value
            # (e.g. a raw provider object) does not drop useful keys like tool_calls.
            safe_kwargs = {}
            for k, v in message.additional_kwargs.items():
                try:
                    json.dumps(v)
                    safe_kwargs[k] = v
                except (TypeError, ValueError):
                    pass
            if safe_kwargs:
                meta["lc_kwargs"] = safe_kwargs
        if isinstance(message, FunctionMessage):
            meta["lc_name"] = message.name
        if isinstance(message, ToolMessage):
            meta["lc_tool_call_id"] = message.tool_call_id
        return Memory(
            agent_id=self._session_id,
            text=content,
            metadata=meta,
        )

    @staticmethod
    def _to_message(memory: Memory) -> BaseMessage:
        lc_type = memory.metadata.get("lc_type", "chat")
        content: str | list[Any] = memory.text
        if memory.metadata.get("lc_content_json"):
            content = json.loads(memory.text)
        additional_kwargs: dict[str, Any] = memory.metadata.get("lc_kwargs", {})
        if lc_type == "human":
            return HumanMessage(content=content, additional_kwargs=additional_kwargs)
        if lc_type == "ai":
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        if lc_type == "system":
            return SystemMessage(content=content, additional_kwargs=additional_kwargs)
        if lc_type == "function":
            return FunctionMessage(
                content=content,
                name=memory.metadata.get("lc_name", ""),
                additional_kwargs=additional_kwargs,
            )
        if lc_type == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=memory.metadata.get("lc_tool_call_id", ""),
                additional_kwargs=additional_kwargs,
            )
        return ChatMessage(content=content, role=lc_type, additional_kwargs=additional_kwargs)

    # ------------------------------------------------------------------
    # Async primary implementations
    # ------------------------------------------------------------------

    async def aget_messages(self) -> list[BaseMessage]:
        """Return all messages for the session in insertion order."""
        memories = await self._engram.list_all(self._session_id)
        return [self._to_message(m) for m in memories]

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Store messages as Memnotary Memories (no embedding — vector detection does not fire)."""
        for message in messages:
            await self._engram.store(self._to_memory(message))

    async def aclear(self) -> None:
        """Delete all messages for the session."""
        memories = await self._engram.list_all(self._session_id)
        ids = [m.memory_id for m in memories]
        if ids:
            await self._engram.delete_batch(self._session_id, ids)

    # ------------------------------------------------------------------
    # Sync implementations (shim → async primary)
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages synchronously. Use aget_messages() in async contexts."""
        return _run(self.aget_messages())

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        raise NotImplementedError("Use aadd_messages() to add messages.")

    def add_message(self, message: BaseMessage) -> None:
        """Store a single message synchronously. Use aadd_messages() in async contexts."""
        _run(self.aadd_messages([message]))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Store multiple messages synchronously. Use aadd_messages() in async contexts."""
        _run(self.aadd_messages(messages))

    def clear(self) -> None:
        """Delete all messages for the session synchronously. Use aclear() in async contexts."""
        _run(self.aclear())
