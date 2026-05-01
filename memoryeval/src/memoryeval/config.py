"""Configuration schema for MemoryEval CLI runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memoryeval.types import BenchmarkCategory


__all__ = ["BackendConfig", "MemoryEvalConfig"]


class BackendConfig(BaseModel):
    """Connection parameters for the storage backend being evaluated."""

    name: str  # "qdrant" | "chroma" | "memory"
    collection: str = "memoryeval"
    # 16 matches the synthetic test fixtures bundled with memoryeval.
    # Override for real backends (common values: 768, 1536, 3072).
    vector_size: int = Field(default=16, gt=0)
    connection: dict[str, Any] = Field(default_factory=dict)


class MemoryEvalConfig(BaseModel):
    """Top-level configuration for a benchmark run.

    Load from a YAML file via ``MemoryEvalConfig.from_yaml(path)``::

        backend:
          name: qdrant
          collection: memoryeval
          vector_size: 1536
          connection:
            url: http://localhost:6333

        pass_threshold: 0.7
        categories:        # omit to run all five
          - temporal
          - contradiction

    ``categories: null`` (or omitting the key entirely) runs all five categories.
    """

    backend: BackendConfig
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    categories: list[BenchmarkCategory] | None = None  # None → run all

    @classmethod
    def from_yaml(cls, path: str | Path) -> MemoryEvalConfig:
        """Load config from a YAML file.

        Raises ImportError if pyyaml is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required to load YAML configs: pip install pyyaml"
            ) from exc

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
