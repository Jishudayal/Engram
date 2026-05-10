"""Mock-based tests for the LangChain bridge (steps 8.2.1–8.2.3).

Covers EngramVectorStore and EngramChatMessageHistory. No real LangChain
chains or backends are exercised — Engram and Embeddings are replaced with
AsyncMock / MagicMock so the suite runs without any backend or LLM.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.vectorstores import VectorStore

from engram.core.models import Memory, SearchResult
from engram.integrations.langchain import EngramChatMessageHistory, EngramVectorStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_engram() -> AsyncMock:
    eng = AsyncMock()
    eng.store_batch.return_value = None
    eng.store.return_value = None
    eng.search.return_value = []
    eng.list_all.return_value = []
    eng.delete_batch.return_value = 0
    return eng


def _make_embeddings() -> MagicMock:
    emb = MagicMock()
    emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    emb.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return emb


def _make_store(engram=None, embeddings=None, agent_id="agent-1") -> EngramVectorStore:
    return EngramVectorStore(
        engram or _make_engram(),
        embeddings or _make_embeddings(),
        agent_id=agent_id,
    )


def _make_history(engram=None, session_id="sess-1") -> EngramChatMessageHistory:
    return EngramChatMessageHistory(engram or _make_engram(), session_id=session_id)


def _search_result(
    text: str = "hello",
    memory_id: str = "mem-1",
    score: float = 0.9,
    rank: int = 1,
) -> SearchResult:
    return SearchResult(
        memory=Memory(memory_id=memory_id, agent_id="agent-1", text=text),
        score=score,
        rank=rank,
    )


def _mem_with_meta(
    lc_type: str,
    text: str = "hi",
    memory_id: str | None = None,
    **extra_meta: object,
) -> Memory:
    return Memory(
        memory_id=memory_id or "mem-1",
        agent_id="sess-1",
        text=text,
        metadata={"lc_type": lc_type, **extra_meta},
    )


# ===========================================================================
# EngramVectorStore — add_texts
# ===========================================================================


class TestEngramVectorStoreAddTexts:
    async def test_embeds_then_stores_batch(self) -> None:
        eng = _make_engram()
        emb = _make_embeddings()
        emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        store = _make_store(eng, emb)

        await store.aadd_texts(["hello", "world"], metadatas=[{"k": "v"}, {}])

        emb.aembed_documents.assert_awaited_once_with(["hello", "world"])
        eng.store_batch.assert_awaited_once()
        stored: list[Memory] = eng.store_batch.call_args[0][0]
        assert len(stored) == 2
        assert stored[0].text == "hello"
        assert stored[0].metadata == {"k": "v"}
        assert stored[1].text == "world"

    async def test_returns_provided_ids(self) -> None:
        emb = _make_embeddings()
        emb.aembed_documents = AsyncMock(return_value=[[0.1], [0.2]])
        store = _make_store(embeddings=emb)
        result = await store.aadd_texts(["a", "b"], ids=["id-1", "id-2"])
        assert result == ["id-1", "id-2"]

    async def test_generates_uuid_ids_when_none(self) -> None:
        store = _make_store()
        result = await store.aadd_texts(["a"])
        assert len(result) == 1
        assert len(result[0]) == 36  # UUID4 format

    async def test_agent_id_stamped_on_each_memory(self) -> None:
        eng = _make_engram()
        store = _make_store(eng, agent_id="my-agent")
        await store.aadd_texts(["x"])
        stored: list[Memory] = eng.store_batch.call_args[0][0]
        assert stored[0].agent_id == "my-agent"

    async def test_length_mismatch_metadatas_raises(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError, match="metadatas length"):
            await store.aadd_texts(["a", "b"], metadatas=[{"k": "v"}])

    async def test_length_mismatch_ids_raises(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError, match="ids length"):
            await store.aadd_texts(["a", "b"], ids=["only-one"])


# ===========================================================================
# EngramVectorStore — search
# ===========================================================================


class TestEngramVectorStoreSearch:
    async def test_similarity_search_returns_documents(self) -> None:
        eng = _make_engram()
        eng.search.return_value = [_search_result("Paris is a city", "m1", 0.95)]
        store = _make_store(eng)

        docs = await store.asimilarity_search("capital of France", k=3)

        assert len(docs) == 1
        assert docs[0].page_content == "Paris is a city"
        assert docs[0].metadata["_memory_id"] == "m1"

    async def test_similarity_search_with_score_returns_pairs(self) -> None:
        eng = _make_engram()
        eng.search.return_value = [_search_result("hello", score=0.88)]
        store = _make_store(eng)

        pairs = await store.asimilarity_search_with_score("hi", k=5)

        assert len(pairs) == 1
        doc, score = pairs[0]
        assert doc.page_content == "hello"
        assert score == 0.88

    async def test_empty_search_returns_empty_list(self) -> None:
        store = _make_store()
        assert await store.asimilarity_search("anything") == []

    async def test_score_threshold_forwarded_to_engram(self) -> None:
        eng = _make_engram()
        store = _make_store(eng)
        await store.asimilarity_search("query", k=4, score_threshold=0.7)
        _, kwargs = eng.search.call_args
        assert kwargs["score_threshold"] == 0.7
        assert kwargs["top_k"] == 4

    async def test_filter_kwarg_forwarded(self) -> None:
        eng = _make_engram()
        store = _make_store(eng)
        await store.asimilarity_search("q", filter={"status": "active"})
        _, kwargs = eng.search.call_args
        assert kwargs["filters"] == {"status": "active"}

    async def test_filters_kwarg_forwarded(self) -> None:
        eng = _make_engram()
        store = _make_store(eng)
        await store.asimilarity_search("q", filters={"tag": "news"})
        _, kwargs = eng.search.call_args
        assert kwargs["filters"] == {"tag": "news"}

    async def test_memory_id_in_document_metadata(self) -> None:
        eng = _make_engram()
        eng.search.return_value = [_search_result(memory_id="abc-123")]
        store = _make_store(eng)
        docs = await store.asimilarity_search("q")
        assert docs[0].metadata["_memory_id"] == "abc-123"


# ===========================================================================
# EngramVectorStore — delete
# ===========================================================================


class TestEngramVectorStoreDelete:
    async def test_adelete_calls_delete_batch_with_ids(self) -> None:
        eng = _make_engram()
        eng.delete_batch.return_value = 2
        store = _make_store(eng)

        result = await store.adelete(["id-1", "id-2"])

        eng.delete_batch.assert_awaited_once_with("agent-1", ["id-1", "id-2"])
        assert result is True

    async def test_adelete_returns_false_for_none(self) -> None:
        store = _make_store()
        assert await store.adelete(None) is False

    async def test_adelete_returns_false_for_empty_list(self) -> None:
        store = _make_store()
        assert await store.adelete([]) is False

    async def test_adelete_returns_false_when_count_is_zero(self) -> None:
        eng = _make_engram()
        eng.delete_batch.return_value = 0
        store = _make_store(eng)
        assert await store.adelete(["gone"]) is False


# ===========================================================================
# EngramVectorStore — from_texts classmethod
# ===========================================================================


class TestEngramVectorStoreFromTexts:
    def test_creates_store_and_calls_add_texts(self) -> None:
        eng = _make_engram()
        emb = _make_embeddings()
        with patch.object(EngramVectorStore, "add_texts", return_value=["id-1"]) as mock_add:
            store = EngramVectorStore.from_texts(
                ["hello"], emb, engram=eng, agent_id="agent-1"
            )
        mock_add.assert_called_once_with(["hello"], metadatas=None, ids=None)
        assert isinstance(store, EngramVectorStore)
        assert store._agent_id == "agent-1"

    def test_missing_engram_raises(self) -> None:
        with pytest.raises(ValueError, match="engram"):
            EngramVectorStore.from_texts(["x"], _make_embeddings(), agent_id="a")

    def test_missing_agent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_id"):
            EngramVectorStore.from_texts(["x"], _make_embeddings(), engram=_make_engram())


# ===========================================================================
# EngramVectorStore — sync shims
# ===========================================================================


class TestEngramVectorStoreSyncShims:
    def test_add_texts_delegates_to_aadd_texts(self) -> None:
        store = _make_store()
        with patch.object(store, "aadd_texts", new=AsyncMock(return_value=["x"])) as mock:
            result = store.add_texts(["hello"])
        mock.assert_awaited_once()
        assert result == ["x"]

    def test_similarity_search_delegates(self) -> None:
        from langchain_core.documents import Document
        store = _make_store()
        doc = Document(page_content="hi", metadata={})
        with patch.object(store, "asimilarity_search", new=AsyncMock(return_value=[doc])) as mock:
            result = store.similarity_search("hi")
        mock.assert_awaited_once()
        assert result == [doc]

    def test_delete_delegates_to_adelete(self) -> None:
        store = _make_store()
        with patch.object(store, "adelete", new=AsyncMock(return_value=True)) as mock:
            result = store.delete(["id-1"])
        mock.assert_awaited_once_with(ids=["id-1"])
        assert result is True


# ===========================================================================
# EngramVectorStore — ABC contract
# ===========================================================================


class TestEngramVectorStoreContract:
    def test_is_vectorstore_subclass(self) -> None:
        assert issubclass(EngramVectorStore, VectorStore)

    def test_no_unimplemented_abstract_methods(self) -> None:
        assert not EngramVectorStore.__abstractmethods__

    def test_embeddings_property(self) -> None:
        emb = _make_embeddings()
        store = _make_store(embeddings=emb)
        assert store.embeddings is emb


# ===========================================================================
# EngramChatMessageHistory — add_messages
# ===========================================================================


class TestEngramChatMessageHistoryAddMessages:
    async def test_human_message_stored_with_lc_type(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([HumanMessage(content="Hi")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.text == "Hi"
        assert stored.metadata["lc_type"] == "human"

    async def test_ai_message_stored(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([AIMessage(content="Hello!")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_type"] == "ai"

    async def test_system_message_stored(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([SystemMessage(content="Be helpful")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_type"] == "system"

    async def test_function_message_preserves_name(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([FunctionMessage(content="42", name="get_answer")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_type"] == "function"
        assert stored.metadata["lc_name"] == "get_answer"

    async def test_tool_message_preserves_tool_call_id(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([ToolMessage(content="result", tool_call_id="call_abc")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_type"] == "tool"
        assert stored.metadata["lc_tool_call_id"] == "call_abc"

    async def test_multiple_messages_each_stored_separately(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([HumanMessage(content="Q"), AIMessage(content="A")])
        assert eng.store.await_count == 2

    async def test_session_id_used_as_agent_id(self) -> None:
        eng = _make_engram()
        history = _make_history(eng, session_id="my-session")
        await history.aadd_messages([HumanMessage(content="x")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.agent_id == "my-session"


# ===========================================================================
# EngramChatMessageHistory — get_messages
# ===========================================================================


class TestEngramChatMessageHistoryGetMessages:
    async def test_returns_messages_in_insertion_order(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = [
            _mem_with_meta("human", "Hello"),
            _mem_with_meta("ai", "World", memory_id="mem-2"),
        ]
        history = _make_history(eng)
        msgs = await history.aget_messages()
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "World"

    async def test_empty_session_returns_empty_list(self) -> None:
        history = _make_history()  # list_all returns []
        assert await history.aget_messages() == []

    async def test_function_message_name_restored(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = [_mem_with_meta("function", "result", lc_name="my_fn")]
        history = _make_history(eng)
        msgs = await history.aget_messages()
        assert isinstance(msgs[0], FunctionMessage)
        assert msgs[0].name == "my_fn"

    async def test_tool_message_tool_call_id_restored(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = [
            _mem_with_meta("tool", "output", lc_tool_call_id="call_xyz")
        ]
        history = _make_history(eng)
        msgs = await history.aget_messages()
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == "call_xyz"

    async def test_unknown_type_becomes_chat_message(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = [_mem_with_meta("custom_role", "hi")]
        history = _make_history(eng)
        msgs = await history.aget_messages()
        assert isinstance(msgs[0], ChatMessage)
        assert msgs[0].role == "custom_role"


# ===========================================================================
# EngramChatMessageHistory — structured content + kwargs safety
# ===========================================================================


class TestEngramChatMessageHistoryContentHandling:
    async def test_structured_content_serialised_with_marker(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        structured = [{"type": "text", "text": "describe"}, {"type": "image_url"}]
        await history.aadd_messages([HumanMessage(content=structured)])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata.get("lc_content_json") is True
        assert stored.text == json.dumps(structured)

    async def test_structured_content_restored_on_read(self) -> None:
        structured = [{"type": "text", "text": "describe"}, {"type": "image_url"}]
        mem = _mem_with_meta("human", json.dumps(structured), lc_content_json=True)
        eng = _make_engram()
        eng.list_all.return_value = [mem]
        history = _make_history(eng)
        msgs = await history.aget_messages()
        assert msgs[0].content == structured

    async def test_plain_string_has_no_json_marker(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([HumanMessage(content="plain text")])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata.get("lc_content_json") is None

    async def test_non_serializable_additional_kwargs_dropped(self) -> None:
        import datetime
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([
            AIMessage(content="hi", additional_kwargs={"ts": datetime.datetime.now()})
        ])
        stored: Memory = eng.store.call_args[0][0]
        assert "lc_kwargs" not in stored.metadata

    async def test_serializable_additional_kwargs_preserved(self) -> None:
        eng = _make_engram()
        history = _make_history(eng)
        safe_kwargs = {"tool_calls": [{"id": "c1", "type": "function"}]}
        await history.aadd_messages([AIMessage(content="hi", additional_kwargs=safe_kwargs)])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_kwargs"] == safe_kwargs

    async def test_mixed_kwargs_preserves_serializable_keys(self) -> None:
        import datetime
        eng = _make_engram()
        history = _make_history(eng)
        await history.aadd_messages([
            AIMessage(
                content="hi",
                additional_kwargs={
                    "tool_calls": [{"id": "c1"}],  # serializable — must be kept
                    "raw": datetime.datetime.now(),  # non-serializable — must be dropped
                },
            )
        ])
        stored: Memory = eng.store.call_args[0][0]
        assert stored.metadata["lc_kwargs"] == {"tool_calls": [{"id": "c1"}]}


# ===========================================================================
# EngramChatMessageHistory — clear
# ===========================================================================


class TestEngramChatMessageHistoryClear:
    async def test_deletes_all_messages_by_id(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = [
            _mem_with_meta("human", "A", "m1"),
            _mem_with_meta("ai", "B", "m2"),
        ]
        history = _make_history(eng, session_id="sess-1")
        await history.aclear()
        eng.delete_batch.assert_awaited_once_with("sess-1", ["m1", "m2"])

    async def test_empty_session_skips_delete_batch(self) -> None:
        eng = _make_engram()
        eng.list_all.return_value = []
        history = _make_history(eng)
        await history.aclear()
        eng.delete_batch.assert_not_awaited()


# ===========================================================================
# EngramChatMessageHistory — sync shims
# ===========================================================================


class TestEngramChatMessageHistorySyncShims:
    def test_add_message_delegates_to_aadd_messages(self) -> None:
        history = _make_history()
        msg = HumanMessage(content="test")
        with patch.object(history, "aadd_messages", new=AsyncMock()) as mock:
            history.add_message(msg)
        mock.assert_awaited_once_with([msg])

    def test_add_messages_delegates_to_aadd_messages(self) -> None:
        history = _make_history()
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        with patch.object(history, "aadd_messages", new=AsyncMock()) as mock:
            history.add_messages(msgs)
        mock.assert_awaited_once_with(msgs)

    def test_messages_property_delegates_to_aget_messages(self) -> None:
        history = _make_history()
        expected = [HumanMessage(content="hi")]
        with patch.object(history, "aget_messages", new=AsyncMock(return_value=expected)):
            result = history.messages
        assert result == expected

    def test_clear_delegates_to_aclear(self) -> None:
        history = _make_history()
        with patch.object(history, "aclear", new=AsyncMock()) as mock:
            history.clear()
        mock.assert_awaited_once()


# ===========================================================================
# EngramChatMessageHistory — ABC contract
# ===========================================================================


class TestEngramChatMessageHistoryContract:
    def test_is_base_chat_message_history_subclass(self) -> None:
        assert issubclass(EngramChatMessageHistory, BaseChatMessageHistory)

    def test_no_unimplemented_abstract_methods(self) -> None:
        assert not EngramChatMessageHistory.__abstractmethods__


# ===========================================================================
# Pyproject wiring (step 8.2.3)
# ===========================================================================


class TestPyprojectWiring:
    def test_both_classes_in_root_all(self) -> None:
        import engram
        assert "EngramVectorStore" in engram.__all__
        assert "EngramChatMessageHistory" in engram.__all__

    def test_root_import_resolves_both_classes(self) -> None:
        from engram import EngramChatMessageHistory as ECMH
        from engram import EngramVectorStore as EVS
        assert issubclass(EVS, VectorStore)
        assert issubclass(ECMH, BaseChatMessageHistory)

    def test_missing_dep_gives_install_hint(self) -> None:
        import engram

        # Null only langchain_core so the integration module re-imports and
        # hits our ImportError rewrite in __getattr__ (not Python's own "halted" error).
        lc_null = {k: None for k in sys.modules if k.startswith("langchain_core")}
        saved_integration = sys.modules.pop("engram.integrations.langchain", None)
        try:
            with patch.dict(sys.modules, lc_null):
                try:
                    engram.__getattr__("EngramVectorStore")
                    raise AssertionError("should have raised ImportError")
                except ImportError as exc:
                    assert "engram[langchain]" in str(exc)
        finally:
            if saved_integration is not None:
                sys.modules["engram.integrations.langchain"] = saved_integration
