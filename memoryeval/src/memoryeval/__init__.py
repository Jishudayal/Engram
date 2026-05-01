"""MemoryEval — benchmark suite for measuring AI memory system reliability."""

from memoryeval.case import TestCase
from memoryeval.config import BackendConfig, MemoryEvalConfig
from memoryeval.report import ReportRenderer
from memoryeval.scorer import BenchmarkScorer
from memoryeval.types import (
    BenchmarkCategory,
    BenchmarkReport,
    CaseResult,
    CategoryScore,
    HallucinationRisk,
)

__version__ = "0.1.0-alpha"

__all__ = [
    # Output types
    "BenchmarkCategory",
    "HallucinationRisk",
    "CaseResult",
    "CategoryScore",
    "BenchmarkReport",
    # Test case interface
    "TestCase",
    # Scoring
    "BenchmarkScorer",
    # Reporting
    "ReportRenderer",
    # Config
    "BackendConfig",
    "MemoryEvalConfig",
    # Metadata
    "__version__",
]
