"""Shared serialization and error-mapping utilities for storage adapters.

Internal module — import directly from engram.adapters._utils; not re-exported
from the package root.

memory_to_payload / payload_to_memory
    Serialize a Memory to a JSON-compatible payload dict (embedding excluded)
    and reconstruct it. Used when writing to / reading from a vector store.

map_adapter_errors
    Decorator factory for async adapter methods. Catches configured backend
    exceptions and re-raises as AdapterError or NotFoundError so callers
    never need to import a backend client's exception types.
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from engram.core.exceptions import AdapterError, EngramError, NotFoundError
from engram.core.models import Memory

__all__ = ["POINT_NAMESPACE", "map_adapter_errors", "memory_to_payload", "payload_to_memory"]

# Engram-specific UUID5 namespace for deriving per-backend point IDs from
# (agent_id, memory_id). Using a project-specific namespace makes generated
# IDs distinguishable from UUIDv5s produced under other namespaces (e.g., DNS)
# and keeps cross-adapter IDs consistent when data is migrated.
POINT_NAMESPACE = uuid.UUID("7f8da9c2-3e5b-4a1f-b6d8-9c2e5f0a3d7e")

_P = ParamSpec("_P")
_T = TypeVar("_T")


def memory_to_payload(memory: Memory) -> dict[str, Any]:
    """Serialize a Memory to a JSON-compatible dict, embedding excluded.

    Datetimes are ISO-8601 strings; enums are their string values; nested
    models (e.g. ProvenanceRecord) become nested dicts. The embedding field
    is excluded because each backend stores it as a dense vector, not as part
    of the payload.
    """
    return memory.model_dump(mode="json", exclude={"embedding"})


def payload_to_memory(payload: dict[str, Any], *, embedding: list[float] | None = None) -> Memory:
    """Reconstruct a Memory from a serialized payload dict.

    Pass the embedding separately — it is not stored in the payload.
    Unknown keys in the payload are silently ignored by Pydantic.
    Raises pydantic.ValidationError if the payload is malformed.
    """
    return Memory.model_validate({**payload, "embedding": embedding})


def map_adapter_errors(
    *,
    not_found: tuple[type[Exception], ...] = (),
    error: tuple[type[Exception], ...] = (),
) -> Callable[[Callable[_P, Coroutine[Any, Any, _T]]], Callable[_P, Coroutine[Any, Any, _T]]]:
    """Decorator factory: map backend exceptions to Engram's exception hierarchy.

    not_found   — re-raise matching exceptions as NotFoundError
    error       — re-raise matching exceptions as AdapterError
    not_found is checked first, so it takes precedence when a type appears in both.
    EngramError subclasses pass through unchanged (never double-wrapped).
    Unmapped exceptions propagate as-is.

    Error messages include fn.__qualname__ for fast triage in logs:
        "QdrantAdapter.store: Connection refused"

    Usage::

        @map_adapter_errors(error=(SomeBackendError,))
        async def store(self, memory: Memory) -> None:
            ...
    """

    def decorator(
        fn: Callable[_P, Coroutine[Any, Any, _T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T]]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            try:
                return await fn(*args, **kwargs)
            except EngramError:
                raise  # already in hierarchy; never double-wrap
            except Exception as exc:
                if not_found and isinstance(exc, not_found):
                    raise NotFoundError(f"{fn.__qualname__}: {exc}") from exc
                if error and isinstance(exc, error):
                    raise AdapterError(f"{fn.__qualname__}: {exc}") from exc
                raise

        return wrapper

    return decorator
