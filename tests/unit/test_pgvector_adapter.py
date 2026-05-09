"""Unit tests for PgVectorAdapter — no real database required.

Covers: constructor validation, helper functions (_row_id, _to_float_list,
_validate_table), and the guard that raises RuntimeError before open().
Behavioural tests (store/fetch/search round-trips with a mock asyncpg pool)
are in step 8.1.3.
"""

from __future__ import annotations

import uuid

import pytest

from engram.adapters._utils import POINT_NAMESPACE
from engram.adapters.pgvector import (
    PgVectorAdapter,
    _row_id,
    _to_float_list,
    _validate_table,
)


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

    def test_pool_is_none_before_open(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        assert adapter._pool is None

    def test_property_raises_before_open(self) -> None:
        adapter = PgVectorAdapter("postgresql://localhost/db", vector_size=4)
        with pytest.raises(RuntimeError, match="not open"):
            _ = adapter._p
