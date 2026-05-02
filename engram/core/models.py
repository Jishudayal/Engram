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
    ActionType,
    ConflictType,
    ConsolidationTier,
    MemoryStatus,
    MemoryType,
    ResolutionStatus,
    RiskLevel,
    SourceType,
)

__all__ = [
    "UTCDatetime",
    "OptionalUTCDatetime",
    "NonEmptyStr",
    "ProvenanceRecord",
    "ConflictRecord",
    "Memory",
    "HealthScore",
    "ConsolidationAction",
    "ConsolidationPlan",
    "SearchResult",
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


def _require_non_empty_stripped(v: str) -> str:
    """Strip whitespace and reject blank strings — shared across required string fields."""
    stripped = v.strip()
    if not stripped:
        raise ValueError("must not be empty or whitespace")
    return stripped


NonEmptyStr = Annotated[str, AfterValidator(_require_non_empty_stripped)]


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


# ---------------------------------------------------------------------------
# Sub-step 1.6 models
# ---------------------------------------------------------------------------


class HealthScore(BaseModel):
    """Point-in-time health snapshot for an agent's memory collection.

    Produced by the health scorer on demand. Immutable after creation — to
    refresh, compute a new HealthScore rather than mutating an existing one.

    Fields are grouped by concern:
      Identity   — record and owning agent
      Counts     — active memory and issue counts
      Metrics    — derived aggregates over the collection
      Signatures — per-dimension health signals (0–1 scale; polarity varies)
      Assessment — overall score and risk tier
      Conflicts  — unresolved conflict records queued for human review
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    score_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: NonEmptyStr
    computed_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- Counts ---
    total_memories: int = Field(
        ge=0,
        description="Number of currently usable ACTIVE memories in the collection.",
    )
    stale_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)

    # --- Metrics ---
    avg_importance: float = Field(ge=0.0, le=1.0)
    oldest_memory_age_days: float | None = Field(default=None, ge=0.0)

    # --- Signatures ---
    contradiction_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Higher is worse: fraction of embedded memories in candidate conflict pairs.",
    )
    freshness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Higher is better: freshness of currently usable memories.",
    )
    confidence_accuracy_gap: float = Field(
        ge=0.0,
        le=1.0,
        description="Higher is worse: retrieval confidence vs measured precision gap.",
    )
    provenance_completeness: float = Field(
        ge=0.0,
        le=1.0,
        description="Higher is better: provenance coverage of currently usable memories.",
    )

    # --- Assessment ---
    score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel

    # --- Conflicts ---
    pending_review: tuple[ConflictRecord, ...] = Field(default_factory=tuple)

    # --- Validators ---

    @model_validator(mode="after")
    def cross_field_invariants(self) -> HealthScore:
        if self.stale_count > self.total_memories:
            raise ValueError(
                f"stale_count ({self.stale_count}) cannot exceed "
                f"total_memories ({self.total_memories})"
            )
        return self


class ConsolidationAction(BaseModel):
    """A single action planned or executed by the consolidation engine.

    Created when the engine classifies a memory cluster and decides what to do.
    Mutable until executed — use mark_executed() to record execution atomically.

    The source_memory_ids tuple is frozen after construction: the set of inputs
    to an action cannot change once the action is created.

    Fields are grouped by concern:
      Identity   — record and owning agent
      Decision   — what to do, why, and at what confidence
      Inputs     — memory IDs being consolidated (immutable after construction)
      Result     — the surviving or merged memory ID, set post-execution
      Timing     — when planned and when executed
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: NonEmptyStr

    # --- Decision ---
    action_type: ActionType
    tier: ConsolidationTier
    confidence: float = Field(ge=0.0, le=1.0)

    # --- Inputs ---
    source_memory_ids: tuple[str, ...] = Field(frozen=True)  # non-empty; set at creation

    # --- Result (set after execution) ---
    result_memory_id: str | None = None
    reasoning: str | None = None

    # --- Timing ---
    created_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    executed_at: OptionalUTCDatetime = None

    # --- Validators ---

    @field_validator("source_memory_ids")
    @classmethod
    def source_memory_ids_valid(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("source_memory_ids must not be empty")
        for mid in v:
            if not mid.strip():
                raise ValueError("source_memory_ids must not contain blank strings")
        return v

    @field_validator("result_memory_id", "reasoning")
    @classmethod
    def optional_strings_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("string fields must not be blank when provided")
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> ConsolidationAction:
        if self.executed_at is not None and self.executed_at < self.created_at:
            raise ValueError("executed_at must not be before created_at")
        if self.executed_at is not None and self.result_memory_id is None:
            raise ValueError("result_memory_id must be set when action is executed")
        return self

    # --- Helper methods ---

    def mark_executed(self, result_memory_id: str, *, executed_at: datetime | None = None) -> None:
        """Record execution atomically — sets result_memory_id then executed_at.

        result_memory_id is set first so that if this action is later serialized
        and deserialized, the construction-time invariant (executed_at requires
        result_memory_id) is satisfied.
        Raises if already executed or if result_memory_id is blank.
        """
        if self.executed_at is not None:
            raise ValueError("action already executed; mark_executed cannot be called again")
        if not result_memory_id.strip():
            raise ValueError("result_memory_id must not be blank")
        at = executed_at if executed_at is not None else datetime.now(UTC)
        self.result_memory_id = result_memory_id
        self.executed_at = at


class ConsolidationPlan(BaseModel):
    """An ordered batch of ConsolidationActions for a single agent.

    Created by the consolidation engine after clustering and classification.
    All actions in a plan must belong to the same agent. The actions tuple is
    frozen after construction to prevent mid-plan mutations.

    Execution is all-or-nothing: a plan deserialized with partial execution is
    rejected by the cross-field invariant.

    Fields are grouped by concern:
      Identity  — plan ID and owning agent
      Actions   — ordered tuple of actions to execute (immutable after construction)
      Metadata  — dry-run flag and estimated health impact
      Timing    — when planned and when executed
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    plan_id: str = Field(default_factory=lambda: str(uuid4()), frozen=True)
    agent_id: NonEmptyStr

    # --- Actions ---
    actions: tuple[ConsolidationAction, ...] = Field(frozen=True)

    # --- Metadata ---
    is_dry_run: bool = False
    estimated_health_delta: dict[str, float] = Field(default_factory=dict)

    # --- Timing ---
    created_at: UTCDatetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    executed_at: OptionalUTCDatetime = None

    # --- Validators ---

    @field_validator("actions")
    @classmethod
    def actions_not_empty(
        cls, v: tuple[ConsolidationAction, ...]
    ) -> tuple[ConsolidationAction, ...]:
        if not v:
            raise ValueError("actions must not be empty")
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> ConsolidationPlan:
        if self.executed_at is not None and self.executed_at < self.created_at:
            raise ValueError("executed_at must not be before created_at")
        for action in self.actions:
            if action.agent_id != self.agent_id:
                raise ValueError(
                    f"action {action.action_id!r} has agent_id={action.agent_id!r} "
                    f"but plan agent_id={self.agent_id!r}"
                )
        executed_count = sum(1 for a in self.actions if a.executed_at is not None)
        if executed_count and executed_count != len(self.actions):
            raise ValueError(
                f"plan execution is all-or-nothing: "
                f"{executed_count}/{len(self.actions)} actions are executed"
            )
        return self

    # --- Properties ---

    @property
    def would_merge(self) -> tuple[ConsolidationAction, ...]:
        """Actions in this plan that merge memories."""
        return tuple(a for a in self.actions if a.action_type == ActionType.MERGE)

    @property
    def would_supersede(self) -> tuple[ConsolidationAction, ...]:
        """Actions in this plan that supersede memories."""
        return tuple(a for a in self.actions if a.action_type == ActionType.SUPERSEDE)

    @property
    def would_flag_for_review(self) -> tuple[ConsolidationAction, ...]:
        """Actions in this plan that flag memories for human review."""
        return tuple(a for a in self.actions if a.action_type == ActionType.FLAG)


class SearchResult(BaseModel):
    """A single result from a memory search query.

    Wraps the matched Memory with relevance metadata. Immutable — search
    results are computed outputs, not mutable state.

    Note: frozen=True prevents field reassignment but does not prevent in-place
    mutation of nested mutable objects (e.g., memory.metadata). Treat search
    results as value objects.

    Fields are grouped by concern:
      Match    — the retrieved memory
      Ranking  — similarity score and position in the result set
      Conflict — conflict-flagging metadata surfaced to the caller
      Context  — optional reference to the query that produced this result
    """

    model_config = ConfigDict(frozen=True)

    # --- Match ---
    memory: Memory

    # --- Ranking ---
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)

    # --- Conflict ---
    conflict_flag: bool = False
    conflicts_with: tuple[str, ...] = Field(default_factory=tuple)
    conflict_summary: str | None = None
    recommended: bool = True

    # --- Context ---
    query_id: str | None = None

    # --- Validators ---

    @field_validator("query_id", "conflict_summary")
    @classmethod
    def optional_strings_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("string fields must not be blank when provided")
        return v

    @model_validator(mode="after")
    def cross_field_invariants(self) -> SearchResult:
        if self.conflict_flag and not self.conflicts_with:
            raise ValueError("conflicts_with must not be empty when conflict_flag is True")
        if self.conflicts_with and not self.conflict_flag:
            raise ValueError("conflict_flag must be True when conflicts_with is non-empty")
        return self
