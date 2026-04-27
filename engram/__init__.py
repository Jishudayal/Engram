"""Engram — memory traces you can trust."""

from engram.adapters.base import AbstractAdapter
from engram.adapters.memory import InMemoryAdapter
from engram.core.constants import (
    ActionType,
    ConflictType,
    ConsolidationTier,
    MemoryStatus,
    MemoryType,
    ResolutionStatus,
    RiskLevel,
    SourceType,
)
from engram.core.exceptions import AdapterError, EngramError, NotFoundError
from engram.core.models import (
    ConflictRecord,
    ConsolidationAction,
    ConsolidationPlan,
    HealthScore,
    Memory,
    ProvenanceRecord,
    SearchResult,
)
from engram.engram import Engram

__version__ = "0.1.0-alpha"

__all__ = [
    "__version__",
    # Main class
    "Engram",
    # Adapters
    "AbstractAdapter",
    "InMemoryAdapter",
    # Exceptions
    "EngramError",
    "AdapterError",
    "NotFoundError",
    # Models
    "Memory",
    "ProvenanceRecord",
    "ConflictRecord",
    "HealthScore",
    "ConsolidationAction",
    "ConsolidationPlan",
    "SearchResult",
    # Enums
    "MemoryType",
    "MemoryStatus",
    "SourceType",
    "ConflictType",
    "ResolutionStatus",
    "ActionType",
    "ConsolidationTier",
    "RiskLevel",
]
