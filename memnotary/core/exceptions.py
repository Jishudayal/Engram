"""Memnotary exception hierarchy.

Exceptions raised by Memnotary's adapter and core layers are subclasses of
MemnotaryError. Note: model construction raises Pydantic's ValidationError and
helper methods raise ValueError — those are not wrapped here.

Hierarchy:
  MemnotaryError
    AdapterError          — any backend storage failure
      NotFoundError       — requested record does not exist in the backend
"""

__all__ = [
    "MemnotaryError",
    "AdapterError",
    "NotFoundError",
]


class MemnotaryError(Exception):
    """Base class for all Memnotary exceptions."""


class AdapterError(MemnotaryError):
    """Raised when a backend adapter operation fails.

    Covers connection failures, I/O errors, and serialization problems.
    Backends should wrap native client exceptions in AdapterError so callers
    don't need to import backend-specific exception types.
    """


class NotFoundError(AdapterError):
    """Raised when an expected record is absent from the backend.

    Distinct from a None return value: fetch() returns None for absent records,
    while update() and other operations that require the record to exist raise
    NotFoundError.
    """
