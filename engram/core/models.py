"""Core data models for Engram.

Models are built incrementally across sub-steps 1.4 → 1.6:
  1.4 — Memory
  1.5 — ProvenanceRecord, ConflictRecord
  1.6 — HealthScore, ConsolidationAction, ConsolidationPlan, SearchResult
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engram.core.constants import DEFAULT_IMPORTANCE, MemoryStatus, MemoryType

__all__ = ["Memory"]


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

    Notes:
      - metadata values must be JSON-serializable (adapters serialize to JSON at storage)
      - expires_at must be timezone-aware; pass datetime.now(UTC) + timedelta(...)
      - text and agent_id are stripped of leading/trailing whitespace on input
    """

    model_config = ConfigDict(validate_assignment=True)

    # --- Identity ---
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    memory_type: MemoryType = MemoryType.CUSTOM

    # --- Content ---
    text: str
    embedding: list[float] | None = Field(default=None, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Lifecycle ---
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    # --- Scoring ---
    importance: float = Field(default=DEFAULT_IMPORTANCE, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime | None = None

    # --- Relationships ---
    # Populated by the consolidation engine. superseded_by points to the
    # memory_id that replaced this one; supersedes lists what this record merged.
    superseded_by: str | None = None
    supersedes: list[str] = Field(default_factory=list)

    # provenance: ProvenanceRecord | None added in sub-step 1.5

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

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "expires_at must be a timezone-aware datetime; "
                "use datetime.now(UTC) + timedelta(...)"
            )
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
    def no_self_supersede(self) -> Memory:
        if self.memory_id in self.supersedes or self.superseded_by == self.memory_id:
            raise ValueError("memory cannot supersede itself")
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

    def update_fields(self, **changes: Any) -> None:
        """Apply field updates and bump updated_at atomically.

        Each assignment triggers validate_assignment, so invalid values raise
        ValidationError before updated_at is touched.
        """
        for key, value in changes.items():
            setattr(self, key, value)
        self.updated_at = datetime.now(UTC)
