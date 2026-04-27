"""Storage backend adapters for Engram."""

from engram.adapters.base import AbstractAdapter
from engram.adapters.memory import InMemoryAdapter

__all__ = ["AbstractAdapter", "InMemoryAdapter"]
