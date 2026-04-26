"""Engram exception hierarchy.

Exceptions raised by Engram's adapter and core layers are subclasses of
EngramError. Note: model construction raises Pydantic's ValidationError and
helper methods raise ValueError — those are not wrapped here.

Hierarchy:
  EngramError
    AdapterError          — any backend storage failure
      NotFoundError       — requested record does not exist in the backend
"""

__all__ = [
    "EngramError",
    "AdapterError",
    "NotFoundError",
]


class EngramError(Exception):
    """Base class for all Engram exceptions."""


class AdapterError(EngramError):
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
