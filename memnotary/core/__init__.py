"""Core data models, adapter interface, and exceptions."""

from memnotary.core.consolidator import Consolidator, LLMConsolidateFn, PlanningResult
from memnotary.core.contradiction import ClassificationResult, ContradictionDetector, LLMClassifyFn
from memnotary.core.health import HealthScorer
from memnotary.core.provenance import ProvenanceManifest

__all__ = [
    "ClassificationResult",
    "ContradictionDetector",
    "LLMClassifyFn",
    "Consolidator",
    "LLMConsolidateFn",
    "PlanningResult",
    "HealthScorer",
    "ProvenanceManifest",
]
