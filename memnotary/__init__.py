"""Memnotary — memory traces you can trust."""

from memnotary.adapters.base import AbstractAdapter
from memnotary.adapters.memory import InMemoryAdapter
from memnotary.core.consolidator import Consolidator, LLMConsolidateFn, PlanningResult
from memnotary.core.constants import (
    ActionType,
    ConflictType,
    ConsolidationTier,
    MemoryStatus,
    MemoryType,
    ResolutionStatus,
    RiskLevel,
    SourceType,
)
from memnotary.core.contradiction import ClassificationResult, ContradictionDetector, LLMClassifyFn
from memnotary.core.exceptions import AdapterError, MemnotaryError, NotFoundError
from memnotary.core.health import HealthScorer
from memnotary.core.models import (
    ConflictRecord,
    ConsolidationAction,
    ConsolidationPlan,
    HealthScore,
    Memory,
    ProvenanceRecord,
    SearchResult,
)
from memnotary.core.provenance import ProvenanceManifest
from memnotary.memnotary import Memnotary

__version__ = "0.1.2"

_OPTIONAL_ADAPTERS: dict[str, tuple[str, str, str]] = {
    # name: (module_path, dep_module_prefix, install_hint)
    "QdrantAdapter": ("memnotary.adapters.qdrant", "qdrant_client", "memnotary[qdrant]"),
    "ChromaAdapter": ("memnotary.adapters.chroma", "chromadb", "memnotary[chroma]"),
    "PgVectorAdapter": ("memnotary.adapters.pgvector", "pgvector", "memnotary[pgvector]"),
    "MemnotaryVectorStore": (
        "memnotary.integrations.langchain",
        "langchain_core",
        "memnotary[langchain]",
    ),
    "MemnotaryChatMessageHistory": (
        "memnotary.integrations.langchain",
        "langchain_core",
        "memnotary[langchain]",
    ),
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
                    f'Install it with: pip install "{install_hint}"'
                ) from exc
            raise
    raise AttributeError(f"module 'memnotary' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Main class
    "Memnotary",
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
    "MemnotaryVectorStore",
    "MemnotaryChatMessageHistory",
    # Exceptions
    "MemnotaryError",
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
