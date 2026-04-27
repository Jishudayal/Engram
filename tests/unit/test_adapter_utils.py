"""Tests for engram.adapters._utils — serialization and error-mapping utilities."""

from __future__ import annotations

import json
from typing import Any

import pytest

from engram import Memory, MemoryStatus, MemoryType
from engram.adapters._utils import map_adapter_errors, memory_to_payload, payload_to_memory
from engram.core.constants import SourceType
from engram.core.exceptions import AdapterError, NotFoundError
from engram.core.models import ProvenanceRecord


def _m(**kw: Any) -> Memory:
    return Memory(**{"agent_id": "agent-1", "text": "fact", **kw})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# memory_to_payload
# ---------------------------------------------------------------------------


class TestMemoryToPayload:
    def test_excludes_embedding(self) -> None:
        m = _m(embedding=[1.0, 2.0])
        assert "embedding" not in memory_to_payload(m)

    def test_includes_text(self) -> None:
        assert memory_to_payload(_m())["text"] == "fact"

    def test_includes_memory_id(self) -> None:
        m = _m()
        assert memory_to_payload(m)["memory_id"] == m.memory_id

    def test_includes_agent_id(self) -> None:
        assert memory_to_payload(_m())["agent_id"] == "agent-1"

    def test_datetimes_are_iso_strings(self) -> None:
        payload = memory_to_payload(_m())
        assert isinstance(payload["created_at"], str)
        assert isinstance(payload["updated_at"], str)

    def test_enums_are_string_values(self) -> None:
        m = _m(status=MemoryStatus.ARCHIVED, memory_type=MemoryType.CONVERSATION)
        payload = memory_to_payload(m)
        assert payload["status"] == "archived"
        assert payload["memory_type"] == "conversation"

    def test_none_optional_fields_preserved(self) -> None:
        payload = memory_to_payload(_m())
        assert payload["expires_at"] is None
        assert payload["last_accessed"] is None
        assert payload["superseded_by"] is None

    def test_metadata_preserved(self) -> None:
        m = _m(metadata={"key": "val", "num": 42})
        assert memory_to_payload(m)["metadata"] == {"key": "val", "num": 42}

    def test_supersedes_preserved(self) -> None:
        m = _m(supersedes=["id1", "id2"])
        assert memory_to_payload(m)["supersedes"] == ["id1", "id2"]

    def test_provenance_none_preserved(self) -> None:
        assert memory_to_payload(_m())["provenance"] is None

    def test_provenance_serialized_as_nested_dict(self) -> None:
        m = _m()
        prov = ProvenanceRecord(memory_id=m.memory_id, source_type=SourceType.DOCUMENT)
        m.attach_provenance(prov)
        payload = memory_to_payload(m)
        assert isinstance(payload["provenance"], dict)
        assert payload["provenance"]["source_type"] == "document"
        assert payload["provenance"]["memory_id"] == m.memory_id

    def test_result_is_json_serializable(self) -> None:
        m = _m(metadata={"nested": {"k": [1, 2]}})
        json.dumps(memory_to_payload(m))  # must not raise


# ---------------------------------------------------------------------------
# payload_to_memory
# ---------------------------------------------------------------------------


class TestPayloadToMemory:
    def test_round_trip_basic(self) -> None:
        m = _m()
        assert payload_to_memory(memory_to_payload(m)) == m

    def test_round_trip_with_embedding(self) -> None:
        m = _m(embedding=[0.1, 0.2, 0.3])
        assert payload_to_memory(memory_to_payload(m), embedding=m.embedding) == m

    def test_round_trip_full_fields(self) -> None:
        m = _m(
            text="complex",
            embedding=[0.5, 0.5],
            metadata={"nested": {"k": 1}},
            supersedes=["id1"],
            importance=0.9,
            status=MemoryStatus.FLAGGED,
            memory_type=MemoryType.SKILL,
        )
        assert payload_to_memory(memory_to_payload(m), embedding=m.embedding) == m

    def test_round_trip_with_provenance(self) -> None:
        m = _m()
        prov = ProvenanceRecord(memory_id=m.memory_id, source_type=SourceType.API)
        m.attach_provenance(prov)
        assert payload_to_memory(memory_to_payload(m)) == m

    def test_default_embedding_is_none(self) -> None:
        result = payload_to_memory(memory_to_payload(_m()))
        assert result.embedding is None

    def test_preserves_memory_id(self) -> None:
        m = _m()
        assert payload_to_memory(memory_to_payload(m)).memory_id == m.memory_id

    def test_preserves_created_at(self) -> None:
        m = _m()
        assert payload_to_memory(memory_to_payload(m)).created_at == m.created_at

    def test_embedding_kwarg_applied(self) -> None:
        vec = [1.0, 0.0]
        result = payload_to_memory(memory_to_payload(_m()), embedding=vec)
        assert result.embedding == vec

    def test_embedding_kwarg_not_aliased(self) -> None:
        vec = [1.0, 0.0]
        result = payload_to_memory(memory_to_payload(_m()), embedding=vec)
        vec[0] = 999.0
        assert result.embedding is not None
        assert result.embedding[0] == 1.0


# ---------------------------------------------------------------------------
# map_adapter_errors
# ---------------------------------------------------------------------------


class TestMapAdapterErrors:
    async def test_passthrough_on_success(self) -> None:
        @map_adapter_errors()
        async def fn() -> int:
            return 42

        assert await fn() == 42

    async def test_maps_to_not_found(self) -> None:
        @map_adapter_errors(not_found=(ValueError,))
        async def fn() -> None:
            raise ValueError("gone")

        with pytest.raises(NotFoundError):
            await fn()

    async def test_maps_to_adapter_error(self) -> None:
        @map_adapter_errors(error=(RuntimeError,))
        async def fn() -> None:
            raise RuntimeError("backend down")

        with pytest.raises(AdapterError):
            await fn()

    async def test_engram_error_passes_through_unchanged(self) -> None:
        @map_adapter_errors(error=(Exception,))
        async def fn() -> None:
            raise NotFoundError("already wrapped")

        with pytest.raises(NotFoundError):
            await fn()

    async def test_adapter_error_passes_through_unchanged(self) -> None:
        @map_adapter_errors(error=(Exception,))
        async def fn() -> None:
            raise AdapterError("already an adapter error")

        with pytest.raises(AdapterError) as exc_info:
            await fn()
        assert type(exc_info.value) is AdapterError  # not double-wrapped

    async def test_unmapped_exception_propagates(self) -> None:
        @map_adapter_errors(not_found=(ValueError,))
        async def fn() -> None:
            raise RuntimeError("not in mapping")

        with pytest.raises(RuntimeError):
            await fn()

    async def test_not_found_takes_precedence_over_error(self) -> None:
        @map_adapter_errors(not_found=(ValueError,), error=(ValueError,))
        async def fn() -> None:
            raise ValueError("ambiguous")

        with pytest.raises(NotFoundError):
            await fn()

    async def test_empty_mappings_propagate_unchanged(self) -> None:
        @map_adapter_errors()
        async def fn() -> None:
            raise RuntimeError("no mapping")

        with pytest.raises(RuntimeError):
            await fn()

    async def test_preserves_function_name(self) -> None:
        @map_adapter_errors()
        async def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    async def test_original_exception_is_cause(self) -> None:
        original = ValueError("root cause")

        @map_adapter_errors(error=(ValueError,))
        async def fn() -> None:
            raise original

        with pytest.raises(AdapterError) as exc_info:
            await fn()
        assert exc_info.value.__cause__ is original

    async def test_not_found_message_includes_qualname(self) -> None:
        @map_adapter_errors(not_found=(ValueError,))
        async def my_op() -> None:
            raise ValueError("gone")

        with pytest.raises(NotFoundError, match="my_op"):
            await my_op()

    async def test_error_message_includes_qualname(self) -> None:
        @map_adapter_errors(error=(RuntimeError,))
        async def my_op() -> None:
            raise RuntimeError("down")

        with pytest.raises(AdapterError, match="my_op"):
            await my_op()

    async def test_works_as_method_decorator(self) -> None:
        class FakeAdapter:
            @map_adapter_errors(error=(RuntimeError,))
            async def store(self, value: int) -> int:
                if value < 0:
                    raise RuntimeError("negative")
                return value

        adapter = FakeAdapter()
        assert await adapter.store(1) == 1
        with pytest.raises(AdapterError):
            await adapter.store(-1)
