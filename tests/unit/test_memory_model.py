"""Tests for engram.core.models.Memory."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engram.core.constants import DEFAULT_IMPORTANCE, MemoryStatus, MemoryType
from engram.core.models import Memory


def make_memory(**overrides: object) -> Memory:
    """Return a minimal valid Memory, with optional field overrides."""
    defaults: dict = {"agent_id": "agent-1", "text": "The refund policy is 30 days."}
    return Memory(**{**defaults, **overrides})


class TestInstantiation:
    def test_minimal_required_fields(self) -> None:
        m = make_memory()
        assert m.agent_id == "agent-1"
        assert m.text == "The refund policy is 30 days."

    def test_memory_id_auto_generated(self) -> None:
        m = make_memory()
        assert m.memory_id
        assert len(m.memory_id) == 36  # UUID4 format

    def test_two_memories_get_different_ids(self) -> None:
        assert make_memory().memory_id != make_memory().memory_id

    def test_default_memory_type(self) -> None:
        assert make_memory().memory_type == MemoryType.CUSTOM

    def test_default_status(self) -> None:
        assert make_memory().status == MemoryStatus.ACTIVE

    def test_default_importance(self) -> None:
        assert make_memory().importance == DEFAULT_IMPORTANCE

    def test_default_access_count(self) -> None:
        assert make_memory().access_count == 0

    def test_default_embedding_is_none(self) -> None:
        assert make_memory().embedding is None

    def test_default_expires_at_is_none(self) -> None:
        assert make_memory().expires_at is None

    def test_default_superseded_by_is_none(self) -> None:
        assert make_memory().superseded_by is None

    def test_default_supersedes_is_empty_list(self) -> None:
        assert make_memory().supersedes == []

    def test_timestamps_are_utc_aware(self) -> None:
        m = make_memory()
        assert m.created_at.tzinfo is not None
        assert m.updated_at.tzinfo is not None

    def test_metadata_defaults_to_empty_dict(self) -> None:
        assert make_memory().metadata == {}

    def test_metadata_instances_are_independent(self) -> None:
        m1, m2 = make_memory(), make_memory()
        m1.metadata["key"] = "value"
        assert "key" not in m2.metadata


class TestValidators:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError, match="text must not be empty"):
            make_memory(text="")

    def test_whitespace_only_text_raises(self) -> None:
        with pytest.raises(ValidationError, match="text must not be empty"):
            make_memory(text="   ")

    def test_text_is_stripped(self) -> None:
        assert make_memory(text="  hello  ").text == "hello"

    def test_empty_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="agent_id must not be empty"):
            make_memory(agent_id="")

    def test_whitespace_only_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="agent_id must not be empty"):
            make_memory(agent_id="   ")

    def test_agent_id_is_stripped(self) -> None:
        assert make_memory(agent_id="  agent-1  ").agent_id == "agent-1"

    def test_importance_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_memory(importance=-0.1)

    def test_importance_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_memory(importance=1.1)

    def test_importance_at_boundaries_accepted(self) -> None:
        assert make_memory(importance=0.0).importance == 0.0
        assert make_memory(importance=1.0).importance == 1.0

    def test_negative_access_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_memory(access_count=-1)

    def test_validate_assignment_rejects_bad_status(self) -> None:
        m = make_memory()
        with pytest.raises(ValidationError):
            m.status = "not_a_valid_status"  # type: ignore[assignment]

    def test_empty_embedding_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty vector"):
            make_memory(embedding=[])

    def test_valid_embedding_accepted(self) -> None:
        m = make_memory(embedding=[0.1, 0.2, 0.3])
        assert m.embedding == [0.1, 0.2, 0.3]

    def test_naive_expires_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_memory(expires_at=datetime.utcnow())  # noqa: DTZ003

    def test_aware_expires_at_accepted(self) -> None:
        future = datetime.now(UTC) + timedelta(days=30)
        assert make_memory(expires_at=future).expires_at == future

    def test_metadata_non_json_serializable_raises(self) -> None:
        with pytest.raises(ValidationError, match="JSON-serializable"):
            make_memory(metadata={"fn": lambda x: x})

    def test_metadata_json_serializable_accepted(self) -> None:
        m = make_memory(metadata={"key": "value", "count": 1, "nested": {"a": True}})
        assert m.metadata["key"] == "value"

    def test_self_supersede_in_supersedes_raises(self) -> None:
        mid = "abc-123"
        with pytest.raises(ValidationError, match="cannot supersede itself"):
            make_memory(memory_id=mid, supersedes=[mid])

    def test_self_superseded_by_raises(self) -> None:
        mid = "abc-123"
        with pytest.raises(ValidationError, match="cannot supersede itself"):
            make_memory(memory_id=mid, superseded_by=mid)

    def test_different_supersedes_id_accepted(self) -> None:
        m = make_memory(memory_id="new-id", supersedes=["old-id"])
        assert m.supersedes == ["old-id"]


class TestIsExpired:
    def test_no_expiry_never_expires(self) -> None:
        assert make_memory(expires_at=None).is_expired() is False

    def test_future_expiry_not_expired(self) -> None:
        future = datetime.now(UTC) + timedelta(days=30)
        assert make_memory(expires_at=future).is_expired() is False

    def test_past_expiry_is_expired(self) -> None:
        past = datetime.now(UTC) - timedelta(seconds=1)
        assert make_memory(expires_at=past).is_expired() is True


class TestIsUsable:
    def test_active_status_no_expiry(self) -> None:
        assert make_memory().is_usable() is True

    def test_superseded_status_not_usable(self) -> None:
        m = make_memory(status=MemoryStatus.SUPERSEDED)
        assert m.is_usable() is False

    def test_flagged_status_not_usable(self) -> None:
        assert make_memory(status=MemoryStatus.FLAGGED).is_usable() is False

    def test_active_but_expired_not_usable(self) -> None:
        past = datetime.now(UTC) - timedelta(seconds=1)
        assert make_memory(expires_at=past).is_usable() is False


class TestTouch:
    def test_touch_increments_access_count(self) -> None:
        m = make_memory()
        m.touch()
        assert m.access_count == 1

    def test_touch_multiple_times(self) -> None:
        m = make_memory()
        m.touch()
        m.touch()
        m.touch()
        assert m.access_count == 3

    def test_touch_sets_last_accessed(self) -> None:
        m = make_memory()
        assert m.last_accessed is None
        m.touch()
        assert m.last_accessed is not None
        assert m.last_accessed.tzinfo is not None

    def test_touch_updates_last_accessed_on_repeat(self) -> None:
        m = make_memory()
        m.touch()
        first = m.last_accessed
        m.touch()
        assert m.last_accessed >= first  # type: ignore[operator]


class TestUpdateFields:
    def test_update_fields_changes_value(self) -> None:
        m = make_memory()
        m.update_fields(text="Updated text")
        assert m.text == "Updated text"

    def test_update_fields_bumps_updated_at(self) -> None:
        m = make_memory()
        before = m.updated_at
        m.update_fields(text="Changed")
        assert m.updated_at >= before

    def test_update_fields_validates_assignment(self) -> None:
        m = make_memory()
        with pytest.raises(ValidationError):
            m.update_fields(importance=2.0)

    def test_update_fields_multiple_keys(self) -> None:
        m = make_memory()
        m.update_fields(text="New text", importance=0.9)
        assert m.text == "New text"
        assert m.importance == 0.9

    def test_update_fields_invalid_leaves_updated_at_unchanged(self) -> None:
        m = make_memory()
        before = m.updated_at
        with pytest.raises(ValidationError):
            m.update_fields(importance=99.0)
        assert m.updated_at == before


class TestSerialization:
    def test_model_dump_round_trip(self) -> None:
        m = make_memory(memory_type=MemoryType.CONVERSATION)
        data = m.model_dump()
        restored = Memory(**data)
        assert restored.memory_id == m.memory_id
        assert restored.text == m.text
        assert restored.memory_type == MemoryType.CONVERSATION

    def test_model_dump_json_round_trip(self) -> None:
        m = make_memory()
        json_str = m.model_dump_json()
        restored = Memory.model_validate_json(json_str)
        assert restored.memory_id == m.memory_id

    def test_status_serializes_as_string(self) -> None:
        m = make_memory()
        data = m.model_dump()
        assert data["status"] == "active"

    def test_memory_type_serializes_as_string(self) -> None:
        m = make_memory(memory_type=MemoryType.SKILL)
        data = m.model_dump()
        assert data["memory_type"] == "skill"
