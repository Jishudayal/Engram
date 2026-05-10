"""Unit tests for PgVectorAdapter — no real database required.

Covers: constructor validation, helper functions (_row_id, _to_float_list,
_validate_table, _conflict_row_id), mock-based behavioural tests for every
CRUD and conflict method, open() error/cleanup paths, and pyproject wiring.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from engram.adapters._utils import POINT_NAMESPACE, memory_to_payload
from engram.adapters.pgvector import (
    PgVectorAdapter,
    _conflict_row_id,
    _row_id,
    _to_float_list,
    _validate_table,
)
from engram.core.constants import ConflictType, MemoryStatus, ResolutionStatus
from engram.core.exceptions import AdapterError, NotFoundError
from engram.core.models import ConflictRecord, Memory

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _mem(**kw: object) -> Memory:
    return Memory(**{"agent_id": "agent-1", "text": "fact", **kw})  # type: ignore[arg-type]


def _conflict(mem_a: Memory, mem_b: Memory) -> ConflictRecord:
    return ConflictRecord(
        agent_id="agent-1",
        memory_a_id=mem_a.memory_id,
        memory_b_id=mem_b.memory_id,
        conflict_type=ConflictType.TEMPORAL_SUPERSESSION,
        confidence=0.9,
    )


class _Row(dict):
    """Minimal asyncpg Record substitute — subscriptable dict."""


def _mem_row(mem: Memory, *, score: float = 0.8) -> _Row:
    return _Row(payload=memory_to_payload(mem), embedding=mem.embedding, score=score)


def _conflict_row(conflict: ConflictRecord) -> _Row:
    return _Row(payload=conflict.model_dump(mode="json"))


def _make_pool() -> AsyncMock:
    pool = AsyncMock()
    pool.execute.return_value = "INSERT 0 1"
    pool.fetch.return_value = []
    pool.fetchrow.return_value = None
    pool.fetchval.return_value = None
    return pool


def _open_adapter(pool: AsyncMock, vector_size: int = 3) -> PgVectorAdapter:
    """Return a PgVectorAdapter that skips open() by injecting a mock pool."""
    adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=vector_size)
    adapter._pool = pool
    return adapter


# ---------------------------------------------------------------------------
# _validate_table
# ---------------------------------------------------------------------------


class TestValidateTable:
    def test_accepts_simple_name(self) -> None:
        _validate_table("engram_memories")  # must not raise

    def test_accepts_leading_underscore(self) -> None:
        _validate_table("_private")

    def test_accepts_mixed_case_and_digits(self) -> None:
        _validate_table("Agent42_Memories")

    def test_rejects_leading_digit(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            _validate_table("1bad")

    def test_rejects_hyphen(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            _validate_table("my-table")

    def test_rejects_dot(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            _validate_table("schema.table")

    def test_rejects_space(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            _validate_table("my table")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            _validate_table("")


# ---------------------------------------------------------------------------
# _row_id
# ---------------------------------------------------------------------------


class TestRowId:
    def test_returns_uuid(self) -> None:
        result = _row_id("agent-1", "mem-1")
        assert isinstance(result, uuid.UUID)

    def test_deterministic(self) -> None:
        assert _row_id("agent-1", "mem-1") == _row_id("agent-1", "mem-1")

    def test_different_agents_give_different_ids(self) -> None:
        assert _row_id("agent-1", "mem-1") != _row_id("agent-2", "mem-1")

    def test_different_memories_give_different_ids(self) -> None:
        assert _row_id("agent-1", "mem-1") != _row_id("agent-1", "mem-2")

    def test_uses_shared_point_namespace(self) -> None:
        expected = uuid.uuid5(POINT_NAMESPACE, "agent-1\x00mem-1")
        assert _row_id("agent-1", "mem-1") == expected


# ---------------------------------------------------------------------------
# _to_float_list
# ---------------------------------------------------------------------------


class TestToFloatList:
    def test_none_returns_none(self) -> None:
        assert _to_float_list(None) is None

    def test_list_of_floats_passes_through(self) -> None:
        assert _to_float_list([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]

    def test_converts_ints_to_float(self) -> None:
        result = _to_float_list([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)

    def test_empty_list(self) -> None:
        assert _to_float_list([]) == []

    def test_works_with_tuple(self) -> None:
        assert _to_float_list((0.5, 0.6)) == [0.5, 0.6]


# ---------------------------------------------------------------------------
# PgVectorAdapter — constructor validation
# ---------------------------------------------------------------------------


class TestPgVectorAdapterInit:
    def test_accepts_valid_arguments(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter is not None

    def test_backend_name(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter.backend_name == "pgvector"

    def test_rejects_zero_vector_size(self) -> None:
        with pytest.raises(ValueError, match="vector_size"):
            PgVectorAdapter("postgresql://localhost/db", vector_size=0)

    def test_rejects_negative_vector_size(self) -> None:
        with pytest.raises(ValueError, match="vector_size"):
            PgVectorAdapter("postgresql://localhost/db", vector_size=-1)

    def test_rejects_invalid_table_name(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            PgVectorAdapter("postgresql://localhost/db", vector_size=4, table="bad-name")

    def test_custom_table_name(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4, table="my_memories")
        assert adapter._table == "my_memories"

    def test_default_table_name(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter._table == "engram_memories"

    def test_table_name_lowercased(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4, table="MyMemories")
        assert adapter._table == "mymemories"

    def test_pool_is_none_before_open(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter._pool is None

    def test_property_raises_before_open(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        with pytest.raises(RuntimeError, match="not open"):
            _ = adapter._p

    def test_conflicts_table_is_sidecar(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter._conflicts_table == "engram_memories_conflicts"

    def test_conflicts_table_follows_custom_table(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4, table="agent_mem")
        assert adapter._conflicts_table == "agent_mem_conflicts"

    def test_conflicts_table_lowercased_with_table(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4, table="MyMem")
        assert adapter._conflicts_table == "mymem_conflicts"


# ---------------------------------------------------------------------------
# _conflict_row_id
# ---------------------------------------------------------------------------


class TestConflictRowId:
    def test_returns_uuid(self) -> None:
        result = _conflict_row_id("agent-1", "conflict-1")
        assert isinstance(result, uuid.UUID)

    def test_deterministic(self) -> None:
        assert _conflict_row_id("agent-1", "c1") == _conflict_row_id("agent-1", "c1")

    def test_differs_from_memory_row_id(self) -> None:
        assert _conflict_row_id("agent-1", "x") != _row_id("agent-1", "x")

    def test_different_agents_differ(self) -> None:
        assert _conflict_row_id("agent-1", "c1") != _conflict_row_id("agent-2", "c1")

    def test_different_conflicts_differ(self) -> None:
        assert _conflict_row_id("agent-1", "c1") != _conflict_row_id("agent-1", "c2")


# ---------------------------------------------------------------------------
# open() — error and cleanup paths
# ---------------------------------------------------------------------------


class TestOpen:
    async def test_pool_creation_failure_raises_adapter_error(self) -> None:
        with patch(
            "asyncpg.create_pool",
            side_effect=asyncpg.PostgresError("connection refused"),
        ), patch("engram.adapters.pgvector.register_vector"):
            adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=3)
            with pytest.raises(AdapterError, match="connection refused"):
                await adapter.open()

    async def test_pool_closed_and_cleared_when_setup_fails(self) -> None:
        # async with pool.acquire() as conn: calls pool.acquire() synchronously
        # and uses the result as an async context manager — so acquire must be
        # a plain MagicMock (not AsyncMock) returning a CM object.
        mock_conn = AsyncMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        from unittest.mock import MagicMock
        mock_pool = AsyncMock()
        mock_pool.acquire = MagicMock(return_value=ctx)  # sync call → CM object

        # CREATE EXTENSION succeeds, EXISTS returns False, CREATE TABLE raises.
        mock_conn.fetchval.return_value = False
        mock_conn.execute.side_effect = [
            None,                                   # CREATE EXTENSION
            asyncpg.PostgresError("disk full"),     # CREATE TABLE
        ]

        with patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)), \
             patch("engram.adapters.pgvector.register_vector"):
            adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=3)
            with pytest.raises(asyncpg.PostgresError):
                await adapter.open()

        mock_pool.close.assert_called_once()
        assert adapter._pool is None

    async def test_open_is_idempotent(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        # Calling open() again must not replace the pool.
        await adapter.open()
        assert adapter._pool is pool


# ---------------------------------------------------------------------------
# store / update / delete
# ---------------------------------------------------------------------------


class TestStore:
    async def test_store_calls_execute(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        await adapter.store(_mem())
        pool.execute.assert_called_once()

    async def test_store_postgres_error_raises_adapter_error(self) -> None:
        pool = _make_pool()
        pool.execute.side_effect = asyncpg.PostgresError("write failed")
        adapter = _open_adapter(pool)
        with pytest.raises(AdapterError):
            await adapter.store(_mem())


class TestUpdate:
    async def test_update_success(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "UPDATE 1"
        adapter = _open_adapter(pool)
        await adapter.update(_mem())  # must not raise

    async def test_update_raises_not_found_when_no_rows(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "UPDATE 0"
        adapter = _open_adapter(pool)
        with pytest.raises(NotFoundError):
            await adapter.update(_mem())


class TestDelete:
    async def test_delete_returns_true_when_row_deleted(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "DELETE 1"
        adapter = _open_adapter(pool)
        assert await adapter.delete("agent-1", "mem-1") is True

    async def test_delete_returns_false_when_not_found(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "DELETE 0"
        adapter = _open_adapter(pool)
        assert await adapter.delete("agent-1", "missing") is False


# ---------------------------------------------------------------------------
# fetch / fetch_batch
# ---------------------------------------------------------------------------


class TestFetch:
    async def test_returns_none_for_missing_row(self) -> None:
        pool = _make_pool()
        pool.fetchrow.return_value = None
        adapter = _open_adapter(pool)
        assert await adapter.fetch("agent-1", "missing") is None

    async def test_deserializes_payload(self) -> None:
        pool = _make_pool()
        mem = _mem(text="hello world")
        pool.fetchrow.return_value = _mem_row(mem)
        adapter = _open_adapter(pool)
        result = await adapter.fetch("agent-1", mem.memory_id)
        assert result is not None
        assert result.text == "hello world"
        assert result.agent_id == "agent-1"
        assert result.memory_id == mem.memory_id

    async def test_returns_none_embedding_when_stored_null(self) -> None:
        pool = _make_pool()
        mem = _mem()  # no embedding
        row = _Row(payload=memory_to_payload(mem), embedding=None)
        pool.fetchrow.return_value = row
        adapter = _open_adapter(pool)
        result = await adapter.fetch("agent-1", mem.memory_id)
        assert result is not None
        assert result.embedding is None


class TestFetchBatch:
    async def test_empty_ids_returns_empty_dict(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        result = await adapter.fetch_batch("agent-1", [])
        assert result == {}
        pool.fetch.assert_not_called()

    async def test_returns_keyed_by_memory_id(self) -> None:
        pool = _make_pool()
        m1 = _mem(text="fact A")
        m2 = _mem(text="fact B")
        pool.fetch.return_value = [_mem_row(m1), _mem_row(m2)]
        adapter = _open_adapter(pool)
        result = await adapter.fetch_batch("agent-1", [m1.memory_id, m2.memory_id])
        assert set(result.keys()) == {m1.memory_id, m2.memory_id}
        assert result[m1.memory_id].text == "fact A"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_top_k_zero_returns_empty_without_db_call(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        result = await adapter.search("agent-1", [1.0, 0.0, 0.0], top_k=0)
        assert result == []
        pool.fetch.assert_not_called()

    async def test_score_clamped_above_one(self) -> None:
        pool = _make_pool()
        mem = _mem()
        pool.fetch.return_value = [_mem_row(mem, score=1.5)]
        adapter = _open_adapter(pool)
        results = await adapter.search("agent-1", [1.0, 0.0, 0.0])
        assert results[0].score == 1.0

    async def test_score_clamped_below_zero(self) -> None:
        pool = _make_pool()
        mem = _mem()
        pool.fetch.return_value = [_mem_row(mem, score=-0.3)]
        adapter = _open_adapter(pool)
        results = await adapter.search("agent-1", [1.0, 0.0, 0.0])
        assert results[0].score == 0.0

    async def test_ranks_start_at_one(self) -> None:
        pool = _make_pool()
        m1, m2 = _mem(text="a"), _mem(text="b")
        pool.fetch.return_value = [_mem_row(m1, score=0.9), _mem_row(m2, score=0.7)]
        adapter = _open_adapter(pool)
        results = await adapter.search("agent-1", [1.0, 0.0, 0.0])
        assert results[0].rank == 1
        assert results[1].rank == 2

    async def test_metadata_filter_applied_in_python(self) -> None:
        pool = _make_pool()
        m_match = _mem(text="match", metadata={"env": "prod"})
        m_skip = _mem(text="skip", metadata={"env": "dev"})
        pool.fetch.return_value = [_mem_row(m_match), _mem_row(m_skip)]
        adapter = _open_adapter(pool)
        results = await adapter.search("agent-1", [1.0, 0.0, 0.0], filters={"env": "prod"})
        assert len(results) == 1
        assert results[0].memory.text == "match"

    async def test_score_threshold_included_in_sql(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        await adapter.search("agent-1", [1.0, 0.0, 0.0], score_threshold=0.8)
        call_args = pool.fetch.call_args
        sql = call_args[0][0]
        assert "<=" in sql  # distance bound pushed into SQL


# ---------------------------------------------------------------------------
# list_all / count / exists / delete_batch
# ---------------------------------------------------------------------------


class TestListAll:
    async def test_returns_empty_list_when_no_rows(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        assert await adapter.list_all("agent-1") == []

    async def test_status_filter_included_in_sql(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        await adapter.list_all("agent-1", status=MemoryStatus.ACTIVE)
        sql = pool.fetch.call_args[0][0]
        assert "status" in sql

    async def test_deserializes_rows(self) -> None:
        pool = _make_pool()
        mem = _mem(text="listed")
        pool.fetch.return_value = [_mem_row(mem)]
        adapter = _open_adapter(pool)
        result = await adapter.list_all("agent-1")
        assert len(result) == 1
        assert result[0].text == "listed"


class TestCount:
    async def test_returns_integer(self) -> None:
        pool = _make_pool()
        pool.fetchval.return_value = 7
        adapter = _open_adapter(pool)
        assert await adapter.count("agent-1") == 7

    async def test_status_filter_changes_sql(self) -> None:
        pool = _make_pool()
        pool.fetchval.return_value = 2
        adapter = _open_adapter(pool)
        await adapter.count("agent-1", status=MemoryStatus.SUPERSEDED)
        sql = pool.fetchval.call_args[0][0]
        assert "status" in sql


class TestExists:
    async def test_returns_true_when_row_exists(self) -> None:
        pool = _make_pool()
        pool.fetchval.return_value = True
        adapter = _open_adapter(pool)
        assert await adapter.exists("agent-1", "mem-1") is True

    async def test_returns_false_when_missing(self) -> None:
        pool = _make_pool()
        pool.fetchval.return_value = False
        adapter = _open_adapter(pool)
        assert await adapter.exists("agent-1", "missing") is False


class TestDeleteBatch:
    async def test_empty_ids_returns_zero_without_db_call(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        assert await adapter.delete_batch("agent-1", []) == 0
        pool.execute.assert_not_called()

    async def test_returns_deleted_count(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "DELETE 3"
        adapter = _open_adapter(pool)
        assert await adapter.delete_batch("agent-1", ["a", "b", "c"]) == 3


# ---------------------------------------------------------------------------
# Conflict methods
# ---------------------------------------------------------------------------


class TestStoreConflict:
    async def test_calls_execute(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        m1, m2 = _mem(text="A"), _mem(text="B")
        await adapter.store_conflict(_conflict(m1, m2))
        pool.execute.assert_called_once()


class TestFetchConflict:
    async def test_returns_none_when_missing(self) -> None:
        pool = _make_pool()
        pool.fetchrow.return_value = None
        adapter = _open_adapter(pool)
        assert await adapter.fetch_conflict("agent-1", "no-such") is None

    async def test_deserializes_conflict_record(self) -> None:
        pool = _make_pool()
        m1, m2 = _mem(text="A"), _mem(text="B")
        c = _conflict(m1, m2)
        pool.fetchrow.return_value = _conflict_row(c)
        adapter = _open_adapter(pool)
        result = await adapter.fetch_conflict("agent-1", c.conflict_id)
        assert result is not None
        assert result.conflict_id == c.conflict_id


class TestListConflicts:
    async def test_returns_empty_list_when_no_rows(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        assert await adapter.list_conflicts("agent-1") == []

    async def test_status_filter_in_sql(self) -> None:
        pool = _make_pool()
        adapter = _open_adapter(pool)
        await adapter.list_conflicts("agent-1", status=ResolutionStatus.PENDING)
        sql = pool.fetch.call_args[0][0]
        assert "resolution_status" in sql

    async def test_deserializes_conflict_records(self) -> None:
        pool = _make_pool()
        m1, m2 = _mem(text="A"), _mem(text="B")
        c = _conflict(m1, m2)
        pool.fetch.return_value = [_conflict_row(c)]
        adapter = _open_adapter(pool)
        result = await adapter.list_conflicts("agent-1")
        assert len(result) == 1
        assert result[0].conflict_id == c.conflict_id


class TestUpdateConflict:
    async def test_success(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "UPDATE 1"
        adapter = _open_adapter(pool)
        m1, m2 = _mem(text="A"), _mem(text="B")
        await adapter.update_conflict(_conflict(m1, m2))  # must not raise

    async def test_raises_not_found_when_no_rows(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "UPDATE 0"
        adapter = _open_adapter(pool)
        m1, m2 = _mem(text="A"), _mem(text="B")
        with pytest.raises(NotFoundError):
            await adapter.update_conflict(_conflict(m1, m2))


class TestDeleteConflict:
    async def test_returns_true_on_success(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "DELETE 1"
        adapter = _open_adapter(pool)
        assert await adapter.delete_conflict("agent-1", "c1") is True

    async def test_returns_false_when_not_found(self) -> None:
        pool = _make_pool()
        pool.execute.return_value = "DELETE 0"
        adapter = _open_adapter(pool)
        assert await adapter.delete_conflict("agent-1", "missing") is False


# ---------------------------------------------------------------------------
# pyproject wiring — PgVectorAdapter importable from engram root
# ---------------------------------------------------------------------------


class TestPyprojectWiring:
    def test_importable_from_engram_root(self) -> None:
        from engram import PgVectorAdapter as PVA
        assert PVA is PgVectorAdapter

    def test_lazy_import_error_has_install_hint(self) -> None:
        import sys

        # Temporarily hide pgvector so the lazy importer's error path fires.
        pgvector_mod = sys.modules.pop("pgvector", None)
        pgvector_asyncpg_mod = sys.modules.pop("pgvector.asyncpg", None)
        adapter_mod = sys.modules.pop("engram.adapters.pgvector", None)
        try:
            sys.modules["pgvector"] = None  # type: ignore[assignment]
            sys.modules["pgvector.asyncpg"] = None  # type: ignore[assignment]
            import engram
            with pytest.raises(ImportError, match="engram\\[pgvector\\]"):
                engram.__getattr__("PgVectorAdapter")
        finally:
            # Restore everything
            for key, mod in [
                ("pgvector", pgvector_mod),
                ("pgvector.asyncpg", pgvector_asyncpg_mod),
                ("engram.adapters.pgvector", adapter_mod),
            ]:
                if mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = mod
