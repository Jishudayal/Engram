"""Tests for ProvenanceRecord and ConflictRecord models (sub-step 1.5)."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from engram.core.constants import ConflictType, MemoryType, ResolutionStatus, SourceType
from engram.core.models import ConflictRecord, Memory, ProvenanceRecord


def make_provenance(**overrides: object) -> ProvenanceRecord:
    defaults: dict = {"memory_id": "mem-1", "source_type": SourceType.CONVERSATION}
    return ProvenanceRecord(**{**defaults, **overrides})


def make_conflict(**overrides: object) -> ConflictRecord:
    defaults: dict = {
        "agent_id": "agent-1",
        "memory_a_id": "mem-a",
        "memory_b_id": "mem-b",
        "conflict_type": ConflictType.DIRECT_CONTRADICTION,
        "confidence": 0.9,
    }
    return ConflictRecord(**{**defaults, **overrides})


def make_memory(**overrides: object) -> Memory:
    defaults: dict = {"agent_id": "agent-1", "text": "The refund policy is 30 days."}
    return Memory(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------


class TestProvenanceRecordInstantiation:
    def test_minimal_required_fields(self) -> None:
        pr = make_provenance()
        assert pr.memory_id == "mem-1"
        assert pr.source_type == SourceType.CONVERSATION

    def test_record_id_auto_generated(self) -> None:
        pr = make_provenance()
        assert pr.record_id
        assert len(pr.record_id) == 36  # UUID4 format

    def test_two_records_get_different_ids(self) -> None:
        assert make_provenance().record_id != make_provenance().record_id

    def test_ingested_at_defaults_to_utc_aware(self) -> None:
        pr = make_provenance()
        assert pr.ingested_at.tzinfo is not None

    def test_optional_fields_default_to_none(self) -> None:
        pr = make_provenance()
        assert pr.source_id is None
        assert pr.ingested_by is None
        assert pr.raw_content is None
        assert pr.chunk_index is None

    def test_all_fields_accepted(self) -> None:
        pr = make_provenance(
            source_id="conv-99",
            ingested_by="agent-1",
            raw_content="Original text here.",
            chunk_index=2,
        )
        assert pr.source_id == "conv-99"
        assert pr.chunk_index == 2


class TestProvenanceRecordValidators:
    def test_blank_memory_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="memory_id must not be empty"):
            make_provenance(memory_id="")

    def test_whitespace_memory_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="memory_id must not be empty"):
            make_provenance(memory_id="   ")

    def test_memory_id_is_stripped(self) -> None:
        pr = make_provenance(memory_id="  mem-1  ")
        assert pr.memory_id == "mem-1"

    def test_blank_source_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_provenance(source_id="")

    def test_blank_ingested_by_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_provenance(ingested_by="   ")

    def test_optional_strings_are_stripped(self) -> None:
        pr = make_provenance(source_id="  doc-1  ", ingested_by="  agent-1  ")
        assert pr.source_id == "doc-1"
        assert pr.ingested_by == "agent-1"

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            make_provenance(chunk_index=-1)

    def test_zero_chunk_index_accepted(self) -> None:
        assert make_provenance(chunk_index=0).chunk_index == 0

    def test_naive_ingested_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_provenance(ingested_at=datetime.utcnow())  # noqa: DTZ003


class TestProvenanceRecordFrozen:
    def test_assignment_after_creation_raises(self) -> None:
        pr = make_provenance()
        with pytest.raises(ValidationError):
            pr.source_id = "new-source"  # type: ignore[misc]

    def test_record_id_immutable(self) -> None:
        pr = make_provenance()
        with pytest.raises(ValidationError):
            pr.record_id = "tampered"  # type: ignore[misc]


class TestProvenanceRecordSerialization:
    def test_model_dump_round_trip(self) -> None:
        pr = make_provenance(source_type=SourceType.DOCUMENT, chunk_index=3)
        data = pr.model_dump()
        restored = ProvenanceRecord(**data)
        assert restored.record_id == pr.record_id
        assert restored.source_type == SourceType.DOCUMENT
        assert restored.chunk_index == 3

    def test_source_type_serializes_as_string(self) -> None:
        data = make_provenance(source_type=SourceType.API).model_dump()
        assert data["source_type"] == "api"

    def test_derived_from_round_trips(self) -> None:
        pr = make_provenance(derived_from=("mem-a", "mem-b"))
        data = pr.model_dump()
        restored = ProvenanceRecord(**data)
        assert restored.derived_from == ("mem-a", "mem-b")


class TestProvenanceRecordDerivedFrom:
    def test_defaults_to_empty_tuple(self) -> None:
        assert make_provenance().derived_from == ()

    def test_single_source_accepted(self) -> None:
        pr = make_provenance(derived_from=("mem-x",))
        assert pr.derived_from == ("mem-x",)

    def test_multiple_sources_accepted(self) -> None:
        pr = make_provenance(derived_from=("mem-a", "mem-b", "mem-c"))
        assert len(pr.derived_from) == 3

    def test_blank_id_in_derived_from_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not contain blank"):
            make_provenance(derived_from=("mem-a", "  "))

    def test_derived_from_immutable(self) -> None:
        pr = make_provenance(derived_from=("mem-a",))
        with pytest.raises((ValidationError, TypeError)):
            pr.derived_from = ("mem-b",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProvenanceManifest
# ---------------------------------------------------------------------------

from engram.core.provenance import ProvenanceManifest  # noqa: E402


def make_memory_with_provenance(**mem_overrides: object) -> Memory:
    mem = Memory(agent_id="agent-1", text="The refund policy is 30 days.", **mem_overrides)  # type: ignore[arg-type]
    prov = ProvenanceRecord(
        memory_id=mem.memory_id,
        source_type=SourceType.DOCUMENT,
        source_id="policy-v3.pdf",
        ingested_by="agent-1",
    )
    mem.attach_provenance(prov)
    return mem


class TestProvenanceManifest:
    def test_from_memory_with_provenance_returns_manifest(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None

    def test_from_memory_without_provenance_returns_none(self) -> None:
        mem = Memory(agent_id="agent-1", text="No provenance here.")
        assert ProvenanceManifest.from_memory(mem) is None

    def test_manifest_memory_fields(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.memory_id == mem.memory_id
        assert manifest.agent_id == "agent-1"
        assert manifest.text == "The refund policy is 30 days."

    def test_manifest_provenance_fields(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.source_type == SourceType.DOCUMENT
        assert manifest.source_id == "policy-v3.pdf"
        assert manifest.ingested_by == "agent-1"
        assert manifest.record_id == mem.provenance.record_id  # type: ignore[union-attr]

    def test_manifest_lineage_defaults(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.supersedes == ()
        assert manifest.superseded_by is None
        assert manifest.derived_from == ()

    def test_manifest_reflects_supersedes_chain(self) -> None:
        mem = make_memory_with_provenance(supersedes=["mem-old-1", "mem-old-2"])
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.supersedes == ("mem-old-1", "mem-old-2")

    def test_manifest_reflects_derived_from(self) -> None:
        mem = Memory(agent_id="agent-1", text="Merged fact.")
        prov = ProvenanceRecord(
            memory_id=mem.memory_id,
            source_type=SourceType.SYSTEM,
            derived_from=("mem-a", "mem-b"),
        )
        mem.attach_provenance(prov)
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.derived_from == ("mem-a", "mem-b")

    def test_manifest_includes_raw_content(self) -> None:
        mem = Memory(agent_id="agent-1", text="Processed fact.")
        prov = ProvenanceRecord(
            memory_id=mem.memory_id,
            source_type=SourceType.DOCUMENT,
            raw_content="Original paragraph text before chunking.",
        )
        mem.attach_provenance(prov)
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.raw_content == "Original paragraph text before chunking."

    def test_manifest_includes_chunk_index(self) -> None:
        mem = Memory(agent_id="agent-1", text="Chunk fact.")
        prov = ProvenanceRecord(
            memory_id=mem.memory_id,
            source_type=SourceType.DOCUMENT,
            chunk_index=4,
        )
        mem.attach_provenance(prov)
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        assert manifest.chunk_index == 4

    def test_manifest_raw_content_defaults_none(self) -> None:
        manifest = ProvenanceManifest.from_memory(make_memory_with_provenance())
        assert manifest is not None
        assert manifest.raw_content is None

    def test_manifest_chunk_index_defaults_none(self) -> None:
        manifest = ProvenanceManifest.from_memory(make_memory_with_provenance())
        assert manifest is not None
        assert manifest.chunk_index is None

    def test_manifest_is_frozen(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        with pytest.raises((ValidationError, TypeError)):
            manifest.text = "tampered"  # type: ignore[misc]

    def test_manifest_serializes_to_dict(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        data = manifest.model_dump()
        assert "memory_id" in data
        assert "source_type" in data
        assert data["source_type"] == "document"

    def test_manifest_round_trips_json(self) -> None:
        mem = make_memory_with_provenance()
        manifest = ProvenanceManifest.from_memory(mem)
        assert manifest is not None
        json_str = manifest.model_dump_json()
        restored = ProvenanceManifest.model_validate_json(json_str)
        assert restored.memory_id == manifest.memory_id
        assert restored.source_type == manifest.source_type

    def test_importable_from_engram_core(self) -> None:
        from engram.core import ProvenanceManifest as PM

        assert PM is ProvenanceManifest

    def test_importable_from_engram_root(self) -> None:
        from engram import ProvenanceManifest as PM

        assert PM is ProvenanceManifest


# ---------------------------------------------------------------------------
# ConflictRecord
# ---------------------------------------------------------------------------


class TestConflictRecordInstantiation:
    def test_minimal_required_fields(self) -> None:
        cr = make_conflict()
        assert cr.agent_id == "agent-1"
        assert cr.memory_a_id == "mem-a"
        assert cr.memory_b_id == "mem-b"
        assert cr.conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert cr.confidence == 0.9

    def test_conflict_id_auto_generated(self) -> None:
        cr = make_conflict()
        assert cr.conflict_id
        assert len(cr.conflict_id) == 36

    def test_two_records_get_different_ids(self) -> None:
        assert make_conflict().conflict_id != make_conflict().conflict_id

    def test_default_resolution_status_is_pending(self) -> None:
        assert make_conflict().resolution_status == ResolutionStatus.PENDING

    def test_detected_at_defaults_to_utc_aware(self) -> None:
        cr = make_conflict()
        assert cr.detected_at.tzinfo is not None

    def test_resolved_at_defaults_to_none(self) -> None:
        assert make_conflict().resolved_at is None

    def test_resolution_notes_defaults_to_none(self) -> None:
        assert make_conflict().resolution_notes is None


class TestConflictRecordValidators:
    def test_blank_agent_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="ID fields must not be empty"):
            make_conflict(agent_id="")

    def test_blank_memory_a_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="ID fields must not be empty"):
            make_conflict(memory_a_id="   ")

    def test_blank_memory_b_id_raises(self) -> None:
        with pytest.raises(ValidationError, match="ID fields must not be empty"):
            make_conflict(memory_b_id="")

    def test_ids_are_stripped(self) -> None:
        cr = make_conflict(agent_id="  agent-1  ", memory_a_id="  mem-a  ")
        assert cr.agent_id == "agent-1"
        assert cr.memory_a_id == "mem-a"

    def test_same_memory_ids_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be different"):
            make_conflict(memory_a_id="mem-x", memory_b_id="mem-x")

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_conflict(confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_conflict(confidence=1.1)

    def test_confidence_at_boundaries_accepted(self) -> None:
        assert make_conflict(confidence=0.0).confidence == 0.0
        assert make_conflict(confidence=1.0).confidence == 1.0

    def test_naive_detected_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_conflict(detected_at=datetime.utcnow())  # noqa: DTZ003

    def test_naive_resolved_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_conflict(resolved_at=datetime.utcnow())  # noqa: DTZ003

    def test_resolved_at_before_detected_at_raises(self) -> None:
        base = datetime.now(UTC)
        with pytest.raises(ValidationError, match="must not be before detected_at"):
            make_conflict(
                detected_at=base,
                resolved_at=base - timedelta(seconds=1),
            )

    def test_resolved_at_equal_to_detected_at_accepted(self) -> None:
        base = datetime.now(UTC)
        cr = make_conflict(detected_at=base, resolved_at=base)
        assert cr.resolved_at == cr.detected_at

    def test_resolved_status_without_resolved_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires resolved_at"):
            make_conflict(
                resolution_status=ResolutionStatus.AUTO_RESOLVED,
                resolved_at=None,
            )

    def test_human_reviewed_without_resolved_at_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires resolved_at"):
            make_conflict(
                resolution_status=ResolutionStatus.HUMAN_REVIEWED,
                resolved_at=None,
            )

    def test_blank_resolution_notes_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_conflict(resolution_notes="")

    def test_whitespace_resolution_notes_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not be blank"):
            make_conflict(resolution_notes="   ")


class TestConflictRecordResolution:
    def test_resolve_sets_status_and_timestamp(self) -> None:
        cr = make_conflict()
        cr.resolve(ResolutionStatus.AUTO_RESOLVED)
        assert cr.resolution_status == ResolutionStatus.AUTO_RESOLVED
        assert cr.resolved_at is not None
        assert cr.resolved_at.tzinfo is not None

    def test_resolve_accepts_explicit_timestamp(self) -> None:
        cr = make_conflict()
        ts = datetime.now(UTC)
        cr.resolve(ResolutionStatus.HUMAN_REVIEWED, resolved_at=ts)
        assert cr.resolved_at == ts
        assert cr.resolution_status == ResolutionStatus.HUMAN_REVIEWED

    def test_resolve_sets_notes(self) -> None:
        cr = make_conflict()
        cr.resolve(ResolutionStatus.AUTO_RESOLVED, notes="Newer memory supersedes.")
        assert cr.resolution_notes == "Newer memory supersedes."

    def test_resolve_to_pending_raises(self) -> None:
        cr = make_conflict()
        with pytest.raises(ValueError, match="cannot set status back to PENDING"):
            cr.resolve(ResolutionStatus.PENDING)

    def test_validate_assignment_rejects_bad_resolution_status(self) -> None:
        cr = make_conflict()
        with pytest.raises(ValidationError):
            cr.resolution_status = "not_a_status"  # type: ignore[assignment]

    def test_can_set_resolved_at_while_pending(self) -> None:
        cr = make_conflict()
        now = datetime.now(UTC)
        cr.resolved_at = now
        assert cr.resolved_at == now


class TestConflictRecordSerialization:
    def test_model_dump_round_trip(self) -> None:
        cr = make_conflict(conflict_type=ConflictType.TEMPORAL_SUPERSESSION)
        data = cr.model_dump()
        restored = ConflictRecord(**data)
        assert restored.conflict_id == cr.conflict_id
        assert restored.conflict_type == ConflictType.TEMPORAL_SUPERSESSION

    def test_resolution_status_serializes_as_string(self) -> None:
        data = make_conflict().model_dump()
        assert data["resolution_status"] == "pending"

    def test_conflict_type_serializes_as_string(self) -> None:
        data = make_conflict(conflict_type=ConflictType.PARTIAL_CONFLICT).model_dump()
        assert data["conflict_type"] == "partial_conflict"


# ---------------------------------------------------------------------------
# Memory.provenance integration
# ---------------------------------------------------------------------------


class TestMemoryWithProvenance:
    def test_provenance_defaults_to_none(self) -> None:
        assert make_memory().provenance is None

    def test_provenance_memory_id_mismatch_raises(self) -> None:
        pr = make_provenance(memory_id="mem-a")
        with pytest.raises(ValidationError, match="does not match memory_id"):
            make_memory(memory_id="mem-b", provenance=pr)

    def test_provenance_memory_id_match_accepted(self) -> None:
        make_memory(memory_id="mem-x")
        pr = make_provenance(memory_id="mem-x", source_type=SourceType.DOCUMENT)
        m2 = make_memory(memory_id="mem-x", provenance=pr)
        assert m2.provenance is not None
        assert m2.provenance.source_type == SourceType.DOCUMENT

    def test_provenance_included_in_model_dump(self) -> None:
        make_memory(memory_id="mem-x")
        pr = make_provenance(memory_id="mem-x", source_type=SourceType.API, source_id="req-42")
        m2 = make_memory(memory_id="mem-x", provenance=pr)
        data = m2.model_dump()
        assert data["provenance"] is not None
        assert data["provenance"]["source_type"] == "api"
        assert data["provenance"]["source_id"] == "req-42"

    def test_provenance_round_trips_through_json(self) -> None:
        make_memory(memory_id="mem-x")
        pr = make_provenance(memory_id="mem-x", source_type=SourceType.MANUAL)
        m2 = make_memory(memory_id="mem-x", provenance=pr)
        restored = Memory.model_validate_json(m2.model_dump_json())
        assert restored.provenance is not None
        assert restored.provenance.record_id == pr.record_id
        assert restored.provenance.source_type == SourceType.MANUAL

    def test_memory_type_unaffected_by_provenance(self) -> None:
        make_memory(memory_id="mem-x")
        pr = make_provenance(memory_id="mem-x")
        m2 = make_memory(memory_id="mem-x", provenance=pr, memory_type=MemoryType.CONVERSATION)
        assert m2.memory_type == MemoryType.CONVERSATION


class TestAttachProvenance:
    def test_attach_provenance_sets_field(self) -> None:
        m = make_memory()
        pr = make_provenance(memory_id=m.memory_id)
        m.attach_provenance(pr)
        assert m.provenance is pr

    def test_direct_provenance_assignment_raises(self) -> None:
        m = make_memory()
        pr = make_provenance(memory_id=m.memory_id)
        with pytest.raises(ValidationError):
            m.provenance = pr  # type: ignore[misc]

    def test_attach_provenance_set_once_raises(self) -> None:
        m = make_memory()
        pr = make_provenance(memory_id=m.memory_id)
        m.attach_provenance(pr)
        pr2 = make_provenance(memory_id=m.memory_id)
        with pytest.raises(ValueError, match="already attached"):
            m.attach_provenance(pr2)

    def test_attach_provenance_memory_id_mismatch_raises(self) -> None:
        m = make_memory(memory_id="mem-correct")
        pr = make_provenance(memory_id="mem-wrong")
        with pytest.raises(ValueError, match="does not match"):
            m.attach_provenance(pr)


# ---------------------------------------------------------------------------
# Memory immutability and update_fields atomicity
# ---------------------------------------------------------------------------


class TestMemoryImmutableFields:
    def test_memory_id_frozen(self) -> None:
        m = make_memory()
        with pytest.raises(ValidationError):
            m.memory_id = "new-id"  # type: ignore[misc]

    def test_created_at_frozen(self) -> None:
        m = make_memory()
        with pytest.raises(ValidationError):
            m.created_at = datetime.now(UTC)  # type: ignore[misc]

    def test_update_fields_rejects_memory_id(self) -> None:
        m = make_memory()
        with pytest.raises(ValueError, match="immutable"):
            m.update_fields(memory_id="new-id")

    def test_update_fields_rejects_created_at(self) -> None:
        m = make_memory()
        with pytest.raises(ValueError, match="immutable"):
            m.update_fields(created_at=datetime.now(UTC))

    def test_update_fields_rejects_provenance(self) -> None:
        m = make_memory()
        pr = make_provenance(memory_id=m.memory_id)
        with pytest.raises(ValueError, match="immutable"):
            m.update_fields(provenance=pr)


class TestUpdateFieldsAtomicity:
    def test_partial_failure_leaves_all_fields_unchanged(self) -> None:
        m = make_memory()
        original_agent_id = m.agent_id
        with pytest.raises(ValidationError):
            # agent_id change is valid, importance is not — neither should apply
            m.update_fields(agent_id="new-agent", importance=99.0)
        assert m.agent_id == original_agent_id

    def test_partial_failure_leaves_updated_at_unchanged(self) -> None:
        m = make_memory()
        before = m.updated_at
        with pytest.raises(ValidationError):
            m.update_fields(importance=99.0)
        assert m.updated_at == before

    def test_successful_update_bumps_updated_at(self) -> None:
        m = make_memory()
        before = m.updated_at
        m.update_fields(text="New text")
        assert m.updated_at >= before

    def test_successful_multi_field_update(self) -> None:
        m = make_memory()
        m.update_fields(text="Updated", importance=0.8)
        assert m.text == "Updated"
        assert m.importance == 0.8
