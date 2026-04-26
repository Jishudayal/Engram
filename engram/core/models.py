"""Core data models for Engram.

Models are built incrementally across sub-steps 1.4 → 1.6:
  1.4 — Memory
  1.5 — ProvenanceRecord, ConflictRecord
  1.6 — HealthScore, ConsolidationAction, ConsolidationPlan, SearchResult

Classes are ordered by dependency: ProvenanceRecord and ConflictRecord are
defined before Memory because Memory references ProvenanceRecord.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar
from uuid import uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from engram.core.constants import (
    DEFAULT_IMPORTANCE,
    ConflictType,
    MemoryStatus,
    MemoryType,
    ResolutionStatus,
    SourceType,
)

__all__ = [
    "UTCDatetime",
    "OptionalUTCDatetime",
    "ProvenanceRecord",
    "ConflictRecord",
    "Memory",
]


def _require_utc_aware(v: datetime | None) -> datetime | None:
    """Reject naive datetimes — shared across all datetime fields."""
    if v is not None and v.tzinfo is None:
        raise ValueError("datetime must be timezone-aware; use datetime.now(UTC)")
    return v


# Reusable annotated datetime types — used in all three models and available
# to sub-step 1.6 models without duplicating the validator.
UTCDatetime = Annotated[datetime, AfterValidator(_require_utc_aware)]
OptionalUTCDatetime = Annotated[datetime | None, AfterValidator(_require_utc_aware)]


class ProvenanceRecord(BaseModel):
    """Immutable audit record tracking the origin of a Memory.

    Created once at ingestion time and never mutated. Stores enough context
    to answer: where did this memory come from, who created it, and when?

    Fields are grouped by concern:
      Identity — record and memory IDs
      Source   — channel, external reference, and ingesting agent
      Content  — original text and position within a chunked document
      Timing   — when ingestion occurred
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    memory_id: str

    # --- Source ---
    source_type: SourceType
    source_id: str | None = None  # conversation_id, document_id, etc.
    ingested_by: str | None = None  # agent_id or user who triggered ingestion

    # --- Content ---
    raw_content: str | None = None  # text before chunking/processing
    chunk_index: int | None = None  # position within a chunked document

    # --- Timing ---
    ingested_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- Validators ---

    @field_validator("memory_id")
    @classmethod
    def memory_id_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("memory_id must not be empty")
        return stripped

    @field_validator("source_id", "ingested_by")
    @classmethod
    def optional_strings_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("string fields must not be blank when provided")
        return v.strip() if v is not None else v

    @field_validator("chunk_index")
    @classmethod
    def chunk_index_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("chunk_index must be non-negative")
        return v


class ConflictRecord(BaseModel):
    """A detected conflict between two Memory records.

    Created by the contradiction detector and updated as the conflict moves
    through the resolution pipeline (PENDING → AUTO_RESOLVED / HUMAN_REVIEWED).

    Use resolve() to transition out of PENDING — it sets resolved_at and
    resolution_status atomically, satisfying the cross-field invariant.

    Fields are grouped by concern:
      Identity       — record and owning agent
      Conflict       — which memories conflict and how
      Classification — LLM confidence in the detection
      Resolution     — current status, when resolved, and optional notes
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    conflict_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str

    # --- Conflict ---
    memory_a_id: str
    memory_b_id: str
    conflict_type: ConflictType

    # --- Classification ---
    confidence: float = Field(ge=0.0, le=1.0)

    # --- Resolution ---
    resolution_status: ResolutionStatus = ResolutionStatus.PENDING
    detected_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: OptionalUTCDatetime = None
    resolution_notes: str | None = None

    # --- Validators ---

    @field_validator("agent_id", "memory_a_id", "memory_b_id")
    @classmethod
    def ids_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("ID fields must not be empty")
        return stripped

    @field_validator("resolution_notes")
    @classmethod
    def resolution_notes_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("resolution_notes must not be blank when provided")
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> ConflictRecord:
        if self.memory_a_id == self.memory_b_id:
            raise ValueError("memory_a_id and memory_b_id must be different")
        if self.resolved_at is not None and self.resolved_at < self.detected_at:
            raise ValueError("resolved_at must not be before detected_at")
        if self.resolution_status != ResolutionStatus.PENDING and self.resolved_at is None:
            raise ValueError(
                f"resolution_status={self.resolution_status.value!r} requires resolved_at"
            )
        return self

    # --- Helper methods ---

    def resolve(
        self,
        status: ResolutionStatus,
        *,
        resolved_at: datetime | None = None,
        notes: str | None = None,
    ) -> None:
        """Transition to a resolved status atomically.

        Sets resolved_at before resolution_status so the cross-field invariant
        (resolved status requires a timestamp) is satisfied at each step.
        Raises ValueError if status is PENDING.
        """
        if status == ResolutionStatus.PENDING:
            raise ValueError("resolve() cannot set status back to PENDING")
        at = resolved_at if resolved_at is not None else datetime.now(UTC)
        self.resolved_at = at
        self.resolution_status = status
        if notes is not None:
            self.resolution_notes = notes


class Memory(BaseModel):
    """A single stored memory record.

    The fundamental unit in Engram. Every piece of knowledge stored in a
    backend — whether it came from a conversation, a document, or a custom
    ingestion — is represented as a Memory.

    Fields are grouped by concern:
      Identity      — who owns this memory and where it came from
      Content       — the raw text, its embedding, and arbitrary metadata
      Lifecycle     — status, timestamps, and optional expiration
      Scoring       — importance and access frequency for retrieval ranking
      Relationships — links to superseded or superseding memories
      Provenance    — optional origin record, attached via attach_provenance()

    Notes:
      - memory_id and created_at are frozen after construction
      - metadata values must be JSON-serializable (adapters serialize to JSON at storage)
      - expires_at must be timezone-aware; use datetime.now(UTC) + timedelta(...)
      - text and agent_id are stripped of leading/trailing whitespace on input
      - use attach_provenance() rather than direct assignment to enforce set-once semantics
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    memory_id: str = Field(default_factory=lambda: str(uuid4()), frozen=True)
    agent_id: str
    memory_type: MemoryType = MemoryType.CUSTOM

    # --- Content ---
    text: str
    embedding: list[float] | None = Field(default=None, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Lifecycle ---
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    updated_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: OptionalUTCDatetime = None

    # --- Scoring ---
    importance: float = Field(default=DEFAULT_IMPORTANCE, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: OptionalUTCDatetime = None

    # --- Relationships ---
    # superseded_by points to the memory_id that replaced this one;
    # supersedes lists what this record merged.
    superseded_by: str | None = None
    supersedes: list[str] = Field(default_factory=list)

    # --- Provenance ---
    provenance: ProvenanceRecord | None = Field(default=None, frozen=True)

    # Fields that update_fields() refuses to touch — use dedicated helpers instead.
    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"memory_id", "created_at", "provenance"}
    )

    # --- Validators ---

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace")
        return stripped

    @field_validator("agent_id")
    @classmethod
    def agent_id_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("agent_id must not be empty or whitespace")
        return stripped

    @field_validator("embedding")
    @classmethod
    def embedding_not_empty(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) == 0:
            raise ValueError("embedding must be None or a non-empty vector")
        return v

    @field_validator("metadata")
    @classmethod
    def metadata_is_json_serializable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metadata must be JSON-serializable: {e}") from e
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> Memory:
        if self.memory_id in self.supersedes or self.superseded_by == self.memory_id:
            raise ValueError("memory cannot supersede itself")
        if self.provenance is not None and self.provenance.memory_id != self.memory_id:
            raise ValueError(
                f"provenance.memory_id ({self.provenance.memory_id!r}) "
                f"does not match memory_id ({self.memory_id!r})"
            )
        return self

    # --- Helper methods ---

    def is_expired(self) -> bool:
        """Return True if this memory has passed its expiration datetime."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    def is_usable(self) -> bool:
        """Return True if status is ACTIVE and the memory has not expired.

        Distinct from `status == MemoryStatus.ACTIVE`: an ACTIVE memory that
        has passed its expiration datetime is not usable for retrieval.
        """
        return self.status == MemoryStatus.ACTIVE and not self.is_expired()

    def touch(self) -> None:
        """Record an access — increments access_count and sets last_accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)

    def attach_provenance(self, record: ProvenanceRecord) -> None:
        """Attach provenance. Raises if already set or if memory_id mismatches.

        Preferred over direct assignment: enforces set-once semantics and
        validates that the record belongs to this memory.
        """
        if self.provenance is not None:
            raise ValueError("provenance already attached; cannot be reassigned")
        if record.memory_id != self.memory_id:
            raise ValueError(
                f"provenance.memory_id ({record.memory_id!r}) does not match "
                f"memory_id ({self.memory_id!r})"
            )
        object.__setattr__(self, "provenance", record)

    def update_fields(self, **changes: Any) -> None:
        """Apply field updates and bump updated_at atomically.

        Pre-validates all changes as a unit before applying any — if validation
        fails, self is untouched and updated_at is not bumped.
        Raises ValueError for immutable fields; use attach_provenance() for provenance.
        """
        immutable = self._IMMUTABLE_FIELDS & changes.keys()
        if immutable:
            raise ValueError(
                f"cannot update immutable fields via update_fields: {sorted(immutable)}"
            )
        # Validate all changes as a unit before touching self.
        # model_dump(mode="python") keeps datetime objects intact so model_validate
        # can re-parse them without lossy ISO-string conversion.
        candidate = {**self.model_dump(mode="python"), **changes}
        type(self).model_validate(candidate)
        # All valid — apply
        for key, value in changes.items():
            setattr(self, key, value)
        self.updated_at = datetime.now(UTC)
