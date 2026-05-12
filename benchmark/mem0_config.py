"""
Confirmed Mem0 2.x configuration for Track 2 behavioral benchmarks.

All Track 2 scripts import MEM0_CONFIG and build_mem0() from here
rather than repeating config inline.

Mem0 2.x API notes
------------------
- Memory.from_config(config_dict)         — initialise
- memory.add(messages, user_id="...")     — user_id still top-level for add()
- memory.search(query, filters={"user_id": "..."})  — filters kwarg, NOT user_id=
- memory.get_all(filters={"user_id": "..."})        — same pattern
- memory.delete_all(user_id="...")        — user_id still top-level for delete

Warnings at startup (non-blocking, safe to ignore):
  "Failed to load spaCy..."     — BM25 optional feature, not used
  "fastembed not installed..."  — BM25 optional feature, not used
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Confirmed working config (Mem0 2.0.2 + gpt-4o-mini + text-embedding-3-small)
# ---------------------------------------------------------------------------

_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333  # memnotary-qdrant-mem0 Docker container


def build_mem0_config(collection: str) -> dict:
    """Return a Mem0 config dict for the given Qdrant collection name.

    Reads OPENAI_API_KEY from the environment — set it before calling.
    Each Track 2 scenario uses its own collection to avoid cross-test pollution.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": api_key,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",  # 1536-dim
                "api_key": api_key,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "host": _QDRANT_HOST,
                "port": _QDRANT_PORT,
                "embedding_model_dims": 1536,
            },
        },
    }
