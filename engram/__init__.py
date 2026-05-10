"""Engram — memory traces you can trust."""

from engram.adapters.base import AbstractAdapter
from engram.adapters.memory import InMemoryAdapter
from engram.core.consolidator import Consolidator, LLMConsolidateFn, PlanningResult
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
from engram.core.contradiction import ClassificationResult, ContradictionDetector, LLMClassifyFn
from engram.core.exceptions import AdapterError, EngramError, NotFoundError
from engram.core.health import HealthScorer
from engram.core.models import (
    ConflictRecord,
    ConsolidationAction,
    ConsolidationPlan,
    HealthScore,
    Memory,
    ProvenanceRecord,
    SearchResult,
)
from engram.core.provenance import ProvenanceManifest
from engram.engram import Engram

__version__ = "0.1.0-alpha"

_OPTIONAL_ADAPTERS: dict[str, tuple[str, str, str]] = {
    # name: (module_path, dep_module_prefix, install_hint)
    "QdrantAdapter": ("engram.adapters.qdrant", "qdrant_client", "engram[qdrant]"),
    "ChromaAdapter": ("engram.adapters.chroma", "chromadb", "engram[chroma]"),
    "PgVectorAdapter": ("engram.adapters.pgvector", "pgvector", "engram[pgvector]"),
    "EngramVectorStore": ("engram.integrations.langchain", "langchain_core", "engram[langchain]"),
    "EngramChatMessageHistory": ("engram.integrations.langchain", "langchain_core", "engram[langchain]"),
}


def __getattr__(name: str) -> object:
    if name in _OPTIONAL_ADAPTERS:
        module_path, dep_module, install_hint = _OPTIONAL_ADAPTERS[name]
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return getattr(mod, name)
        except ModuleNotFoundError as exc:
            # Only rewrite the error if the missing module is the optional
            # dependency. Re-raise import errors that originate from our own
            # code (e.g. a bug in the adapter module itself).
            if (exc.name or "").startswith(dep_module):
                raise ImportError(
                    f"{name} requires an optional dependency. "
                    f"Install it with: pip install \"{install_hint}\""
                ) from exc
            raise
    raise AttributeError(f"module 'engram' has no attribute {name!r}")

__all__ = [
    "__version__",
    # Main class
    "Engram",
    "HealthScorer",
    "ContradictionDetector",
    "ClassificationResult",
    "LLMClassifyFn",
    "Consolidator",
    "LLMConsolidateFn",
    "PlanningResult",
    # Adapters
    "AbstractAdapter",
    "InMemoryAdapter",
    "QdrantAdapter",
    "ChromaAdapter",
    "PgVectorAdapter",
    "EngramVectorStore",
    "EngramChatMessageHistory",
    # Exceptions
    "EngramError",
    "AdapterError",
    "NotFoundError",
    # Models
    "Memory",
    "ProvenanceRecord",
    "ProvenanceManifest",
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
