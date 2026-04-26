"""Shared enums and threshold constants used across Engram."""

from enum import StrEnum

__all__ = [
    # Enums
    "MemoryType",
    "MemoryStatus",
    "SourceType",
    "ConflictType",
    "ResolutionStatus",
    "ActionType",
    "ConsolidationTier",
    "RiskLevel",
    # Thresholds
    "CLUSTER_SIMILARITY_THRESHOLD",
    "AUTO_MERGE_THRESHOLD",
    "AUTO_FLAG_THRESHOLD",
    "CONTRADICTION_CONFIDENCE_THRESHOLD",
    # Defaults
    "DEFAULT_IMPORTANCE",
    "DEFAULT_EXPIRATION_DAYS",
]


class MemoryType(StrEnum):
    """The origin type of a stored memory."""

    CONVERSATION = "conversation"
    SKILL = "skill"
    CUSTOM = "custom"


class MemoryStatus(StrEnum):
    """Lifecycle state of a memory record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FLAGGED = "flagged"
    ARCHIVED = "archived"
    PENDING_REVIEW = "pending_review"


class SourceType(StrEnum):
    """Origin channel of a ProvenanceRecord."""

    CONVERSATION = "conversation"
    DOCUMENT = "document"
    API = "api"
    MANUAL = "manual"
    SYSTEM = "system"


class ConflictType(StrEnum):
    """How two memories conflict with each other."""

    DIRECT_CONTRADICTION = "direct_contradiction"
    TEMPORAL_SUPERSESSION = "temporal_supersession"
    PARTIAL_CONFLICT = "partial_conflict"


class ResolutionStatus(StrEnum):
    """Resolution state of a detected memory conflict."""

    PENDING = "pending"
    AUTO_RESOLVED = "auto_resolved"
    HUMAN_REVIEWED = "human_reviewed"


class ActionType(StrEnum):
    """Action taken by the consolidation engine on a memory cluster."""

    MERGE = "merge"
    SUPERSEDE = "supersede"
    FLAG = "flag"
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------
# Consolidation thresholds — defined before ConsolidationTier so the
# from_confidence() classmethod can reference them directly.
# ---------------------------------------------------------------------------

# Cosine similarity above which two memories enter the consolidation pipeline.
CLUSTER_SIMILARITY_THRESHOLD: float = 0.82

# LLM classification confidence above which a merge/supersede runs automatically.
AUTO_MERGE_THRESHOLD: float = 0.92

# LLM classification confidence above which a memory is auto-flagged.
# Below this threshold the cluster is queued for human review.
AUTO_FLAG_THRESHOLD: float = 0.78

# Minimum LLM confidence for the contradiction detector to emit a conflict record.
# Separate from consolidation thresholds — contradiction detection is a binary
# yes/no classification, not a tiered merge decision.
# See: engram/core/contradiction.py (Step 5).
CONTRADICTION_CONFIDENCE_THRESHOLD: float = 0.80


class ConsolidationTier(StrEnum):
    """Decision tier assigned to a memory cluster during consolidation.

    Tier is determined by the LLM classification confidence score:
      AUTO_MERGE   — confidence >= AUTO_MERGE_THRESHOLD
      AUTO_FLAG    — AUTO_FLAG_THRESHOLD <= confidence < AUTO_MERGE_THRESHOLD
      HUMAN_REVIEW — confidence < AUTO_FLAG_THRESHOLD, or genuine conflict
    """

    AUTO_MERGE = "auto_merge"
    AUTO_FLAG = "auto_flag"
    HUMAN_REVIEW = "human_review"

    @classmethod
    def from_confidence(cls, confidence: float) -> "ConsolidationTier":
        """Return the consolidation tier for a given confidence score."""
        if confidence >= AUTO_MERGE_THRESHOLD:
            return cls.AUTO_MERGE
        if confidence >= AUTO_FLAG_THRESHOLD:
            return cls.AUTO_FLAG
        return cls.HUMAN_REVIEW


class RiskLevel(StrEnum):
    """Overall memory health risk level surfaced by the health scorer."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Memory defaults
# ---------------------------------------------------------------------------

DEFAULT_IMPORTANCE: float = 0.5

# None means no expiration is applied unless the caller provides one explicitly.
DEFAULT_EXPIRATION_DAYS: int | None = None
