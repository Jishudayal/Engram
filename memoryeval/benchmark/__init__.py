"""Benchmark test cases across all five evaluation categories."""

from memoryeval.benchmark.contradiction import CONTRADICTION_CASES
from memoryeval.benchmark.crosstype import CROSSTYPE_CASES
from memoryeval.benchmark.importance import IMPORTANCE_CASES
from memoryeval.benchmark.multihop import MULTIHOP_CASES
from memoryeval.benchmark.temporal import TEMPORAL_CASES
from memoryeval.case import TestCase

__all__ = [
    "TEMPORAL_CASES",
    "CONTRADICTION_CASES",
    "MULTIHOP_CASES",
    "IMPORTANCE_CASES",
    "CROSSTYPE_CASES",
    "ALL_CASES",
]

# Display order matches the five BenchmarkCategory enum order.
ALL_CASES: list[type[TestCase]] = [
    *TEMPORAL_CASES,
    *CONTRADICTION_CASES,
    *MULTIHOP_CASES,
    *IMPORTANCE_CASES,
    *CROSSTYPE_CASES,
]
