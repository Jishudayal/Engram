"""Benchmark test cases across all five evaluation categories."""

from memoryeval.benchmark.contradiction import CONTRADICTION_CASES
from memoryeval.benchmark.temporal import TEMPORAL_CASES
from memoryeval.case import TestCase

__all__ = [
    "TEMPORAL_CASES",
    "CONTRADICTION_CASES",
    "ALL_CASES",
]

# Ordered: temporal first, contradiction second — matches the five-category
# display order in reports (multi-hop, importance, cross-type added in 3.3).
ALL_CASES: list[type[TestCase]] = [
    *TEMPORAL_CASES,
    *CONTRADICTION_CASES,
]
