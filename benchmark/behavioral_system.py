"""
Track 2 — BehavioralSystem Protocol and concrete implementations.

Four systems under comparison:
  - Mem0System          : Mem0 2.x with OpenAI extraction + embeddings
  - NaiveQdrantSystem   : raw Qdrant, cosine-only, no data model
  - EngramSystem        : Engram + QdrantAdapter + ContradictionDetector (OpenAI gpt-4o-mini)
                          use_consolidator=False  → detect-only
                          use_consolidator=True   → detect + consolidate

All systems embed via OpenAI text-embedding-3-small (1536-dim) so comparisons
are apples-to-apples on the retrieval side.

Usage:
    systems = build_systems(openai_api_key="sk-...")
    for sys in systems:
        await sys.reset("scenario_b1")
        await sys.add_fact("scenario_b1", "Refund window is 30 days.")
        await sys.add_fact("scenario_b1", "Refund window updated to 60 days.")
        hits = await sys.search("scenario_b1", "refund window")
        count = await sys.retained_count("scenario_b1")
        await sys.consolidate("scenario_b1")
        await sys.close()
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import openai
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from engram.adapters.qdrant import QdrantAdapter
from engram.core.contradiction import ContradictionDetector
from engram.core.consolidator import Consolidator
from engram.core.constants import MemoryStatus
from engram.core.models import Memory
from engram.engram import Engram

_QDRANT_URL = "http://localhost:6333"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIMS = 1536
_LLM_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class BehavioralHit:
    text: str
    score: float | None = None
    conflict_flag: bool = False
    recommended: bool = True


@runtime_checkable
class BehavioralSystem(Protocol):
    name: str

    async def reset(self, namespace: str) -> None:
        """Wipe all stored memories for this namespace and prepare for a fresh run."""
        ...

    async def add_fact(self, namespace: str, text: str) -> None:
        """Store one fact into the namespace."""
        ...

    async def search(self, namespace: str, query: str, top_k: int = 5) -> list[BehavioralHit]:
        """Return top_k hits for query."""
        ...

    async def retained_count(self, namespace: str) -> int:
        """Return the number of active (non-archived) memories for this namespace."""
        ...

    async def consolidate(self, namespace: str) -> None:
        """Resolve pending conflicts (no-op for systems without a consolidator)."""
        ...

    async def close(self) -> None:
        """Release any open connections."""
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _embed(text: str, client: openai.AsyncOpenAI) -> list[float]:
    resp = await client.embeddings.create(model=_EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _make_llm_fn(client: openai.AsyncOpenAI) -> Callable[[str], Awaitable[str]]:
    async def _fn(prompt: str) -> str:
        resp = await client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    return _fn


# ---------------------------------------------------------------------------
# Mem0System
# ---------------------------------------------------------------------------


class Mem0System:
    """Mem0 2.x with OpenAI extraction + OpenAI embeddings."""

    name = "mem0"

    def __init__(self, api_key: str) -> None:
        from mem0 import Memory as Mem0Memory

        from mem0_config import build_mem0_config

        self._m = Mem0Memory.from_config(build_mem0_config("behavioral_track2"))

    async def reset(self, namespace: str) -> None:
        self._m.delete_all(user_id=namespace)

    async def add_fact(self, namespace: str, text: str) -> None:
        self._m.add(
            [{"role": "user", "content": text}],
            user_id=namespace,
        )

    async def search(self, namespace: str, query: str, top_k: int = 5) -> list[BehavioralHit]:
        raw: dict[str, Any] = (
            self._m.search(query, filters={"user_id": namespace}, limit=top_k) or {}
        )
        hits = []
        for r in raw.get("results", []):
            hits.append(
                BehavioralHit(
                    text=r.get("memory", ""),
                    score=r.get("score"),
                    conflict_flag=False,
                    recommended=True,
                )
            )
        return hits

    async def retained_count(self, namespace: str) -> int:
        raw: dict[str, Any] = self._m.get_all(filters={"user_id": namespace}) or {}
        return len(raw.get("results", []))

    async def consolidate(self, namespace: str) -> None:
        pass  # Mem0 handles internally during add()

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# NaiveQdrantSystem
# ---------------------------------------------------------------------------


class NaiveQdrantSystem:
    """Raw Qdrant: cosine search only, no Engram data model."""

    name = "naive-qdrant"

    def __init__(self, api_key: str) -> None:
        self._oai = openai.AsyncOpenAI(api_key=api_key)
        self._client: AsyncQdrantClient | None = None

    @property
    def _c(self) -> AsyncQdrantClient:
        assert self._client is not None, "call reset() before using NaiveQdrantSystem"
        return self._client

    def _collection(self, namespace: str) -> str:
        return f"naive_{namespace}"

    async def reset(self, namespace: str) -> None:
        if self._client is None:
            self._client = AsyncQdrantClient(location=_QDRANT_URL)
        coll = self._collection(namespace)
        import contextlib

        with contextlib.suppress(Exception):
            await self._c.delete_collection(coll)
        await self._c.create_collection(
            coll,
            vectors_config=VectorParams(size=_EMBED_DIMS, distance=Distance.COSINE),
        )

    async def add_fact(self, namespace: str, text: str) -> None:
        vector = await _embed(text, self._oai)
        import uuid

        point_id = str(uuid.uuid4())
        await self._c.upsert(
            self._collection(namespace),
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": text, "agent_id": namespace},
                )
            ],
        )

    async def search(self, namespace: str, query: str, top_k: int = 5) -> list[BehavioralHit]:
        vector = await _embed(query, self._oai)
        results = await self._c.query_points(
            self._collection(namespace),
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=namespace))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [
            BehavioralHit(
                text=(p.payload or {}).get("text", ""),
                score=p.score,
                conflict_flag=False,
                recommended=True,
            )
            for p in results.points
        ]

    async def retained_count(self, namespace: str) -> int:
        records, _ = await self._c.scroll(
            self._collection(namespace),
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=namespace))]
            ),
            limit=1000,
            with_payload=False,
        )
        return len(records)

    async def consolidate(self, namespace: str) -> None:
        pass  # no-op: naive adapter has no conflict layer

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# EngramSystem
# ---------------------------------------------------------------------------


class EngramSystem:
    """Engram + QdrantAdapter + ContradictionDetector (OpenAI gpt-4o-mini).

    use_consolidator=False  → name "engram-detect"
    use_consolidator=True   → name "engram-consolidated"
    """

    def __init__(self, api_key: str, *, use_consolidator: bool = False) -> None:
        self.name = "engram-consolidated" if use_consolidator else "engram-detect"
        self._use_consolidator = use_consolidator
        self._oai = openai.AsyncOpenAI(api_key=api_key)
        self._adapter: QdrantAdapter | None = None
        self._engram: Engram | None = None

    @property
    def _e(self) -> Engram:
        assert self._engram is not None, "call reset() before using EngramSystem"
        return self._engram

    def _collection(self, namespace: str) -> str:
        prefix = "engramcon" if self._use_consolidator else "engramdet"
        return f"{prefix}_{namespace}"

    async def reset(self, namespace: str) -> None:
        if self._adapter is not None:
            await self._adapter.close()
            self._adapter = None
            self._engram = None

        import contextlib

        coll = self._collection(namespace)
        # Drop old collection so we start clean
        cleanup = AsyncQdrantClient(location=_QDRANT_URL)
        with contextlib.suppress(Exception):
            await cleanup.delete_collection(coll)
        with contextlib.suppress(Exception):
            await cleanup.delete_collection(f"{coll}_conflicts")
        await cleanup.close()

        llm_fn = _make_llm_fn(self._oai)
        detector = ContradictionDetector(llm_fn=llm_fn)
        consolidator = Consolidator(llm_fn=llm_fn) if self._use_consolidator else None

        self._adapter = QdrantAdapter(
            location=_QDRANT_URL,
            collection=coll,
            vector_size=_EMBED_DIMS,
        )
        await self._adapter.open()

        self._engram = Engram(
            adapter=self._adapter,
            detector=detector,
            consolidator=consolidator,
        )

    async def add_fact(self, namespace: str, text: str) -> None:
        vector = await _embed(text, self._oai)
        import uuid

        memory = Memory(
            agent_id=namespace,
            memory_id=str(uuid.uuid4()),
            text=text,
            embedding=vector,
        )
        await self._e.store(memory)

    async def search(self, namespace: str, query: str, top_k: int = 5) -> list[BehavioralHit]:
        vector = await _embed(query, self._oai)
        results = await self._e.search(namespace, vector, top_k=top_k)
        return [
            BehavioralHit(
                text=r.memory.text,
                score=r.score,
                conflict_flag=r.conflict_flag,
                recommended=r.recommended,
            )
            for r in results
        ]

    async def retained_count(self, namespace: str) -> int:
        memories = await self._e.list_all(namespace, status=MemoryStatus.ACTIVE)
        return len(memories)

    async def consolidate(self, namespace: str) -> None:
        if self._use_consolidator:
            await self._e.consolidate(namespace)

    async def close(self) -> None:
        if self._adapter is not None:
            await self._adapter.close()
            self._adapter = None
            self._engram = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_systems(openai_api_key: str | None = None) -> list[BehavioralSystem]:
    api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
    return [
        Mem0System(api_key=api_key),
        NaiveQdrantSystem(api_key=api_key),
        EngramSystem(api_key=api_key, use_consolidator=False),
        EngramSystem(api_key=api_key, use_consolidator=True),
    ]
