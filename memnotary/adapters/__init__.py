"""Storage backend adapters for Memnotary."""

from memnotary.adapters.base import AbstractAdapter
from memnotary.adapters.memory import InMemoryAdapter

__all__ = ["AbstractAdapter", "InMemoryAdapter"]
