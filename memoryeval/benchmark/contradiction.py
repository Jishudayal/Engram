"""Contradiction reliability test cases for MemoryEval.

Ten cases that measure how well a memory backend handles conflicting facts:
detection via cosine similarity, status-based resolution, linkage integrity,
and multi-claim management.

Scoring convention
------------------
- 1.0  — contradiction is correctly handled (e.g. only one ACTIVE version)
- 0.0  — contradiction exists unresolved (both conflicting facts are ACTIVE)
- 0.x  — partial correctness

C1 (BothClaimsActiveIsContradiction) intentionally scores 0.0 for any
naive adapter — it represents the bad state that Engram's consolidation
engine is designed to fix. Its low score on unmanaged systems is the
benchmark's most important signal.
"""

from __future__ import annotations

from uuid import uuid4

from engram.core.constants import MemoryStatus
from engram.core.models import Memory, SearchResult

from memoryeval.benchmark._embeddings import cosine, vec
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory


# ---------------------------------------------------------------------------
# C1 — Both conflicting claims ACTIVE is a contradiction (bad baseline)
# ---------------------------------------------------------------------------

class BothClaimsActiveIsContradiction(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "both_claims_active_is_contradiction"
    description = "Two conflicting facts both ACTIVE is the unresolved-contradiction baseline"
    pass_threshold = 0.5  # easier bar — this case is designed to expose raw systems

    # INTENTIONAL: scores 0.0 on naive adapters — that gap is the benchmark's core signal.
    # Scores 1.0 only when exactly one ACTIVE version remains (contradiction resolved).

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        await adapter.store(Memory(
            agent_id=self.agent_id,
            text="API rate limit is 100 requests/min.",
            embedding=embedding,
        ))
        await adapter.store(Memory(
            agent_id=self.agent_id,
            text="API rate limit is 500 requests/min.",
            embedding=embedding,
        ))

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: list[Memory]) -> float:
        # Exactly 1 active = resolved. 0 = data loss. >1 = unresolved contradiction.
        return 1.0 if len(result) == 1 else 0.0


# ---------------------------------------------------------------------------
# C2 — After resolving, only one ACTIVE version remains
# ---------------------------------------------------------------------------

class ResolvedContradictionHasOneActive(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "resolved_contradiction_has_one_active"
    description = "After marking the old claim SUPERSEDED, exactly one ACTIVE version should remain"

    _new_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        new_id = str(uuid4())

        old = Memory(
            agent_id=self.agent_id,
            text="API rate limit is 100 requests/min.",
            embedding=embedding,
        )
        await adapter.store(old)
        old.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=new_id)
        await adapter.update(old)

        new = Memory(
            memory_id=new_id,
            agent_id=self.agent_id,
            text="API rate limit is 500 requests/min.",
            embedding=embedding,
            supersedes=[old.memory_id],
        )
        await adapter.store(new)
        self._new_id = new_id

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: list[Memory]) -> float:
        return 1.0 if len(result) == 1 and result[0].memory_id == self._new_id else 0.0


# ---------------------------------------------------------------------------
# C3 — Stored embeddings survive write-read with near-perfect cosine fidelity
# ---------------------------------------------------------------------------

class EmbeddingFidelityHighPrecision(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "embedding_fidelity_high_precision"
    description = "Stored embeddings must survive write-read with near-perfect cosine fidelity (> 0.9999)"

    _memory_id: str
    _original_embedding: list[float]

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit", "request")
        m = Memory(
            agent_id=self.agent_id,
            text="API rate limit is 100 requests/min.",
            embedding=embedding,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id
        self._original_embedding = embedding

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None or result.embedding is None:
            return 0.0
        sim = cosine(self._original_embedding, result.embedding)
        # Near-perfect fidelity is required for reliable contradiction detection via cosine
        return 1.0 if sim > 0.9999 else sim


# ---------------------------------------------------------------------------
# C4 — Three claims resolved to one ACTIVE
# ---------------------------------------------------------------------------

class ThreeClaimsOneActive(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "three_claims_one_active"
    description = "After superseding two of three competing claims, exactly one should be ACTIVE"

    _final_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("price", "plan", "subscription")
        final_id = str(uuid4())

        for text in ("Starter plan is $9/mo.", "Starter plan is $19/mo."):
            m = Memory(agent_id=self.agent_id, text=text, embedding=embedding)
            await adapter.store(m)
            m.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=final_id)
            await adapter.update(m)

        final = Memory(
            memory_id=final_id,
            agent_id=self.agent_id,
            text="Starter plan is $29/mo.",
            embedding=embedding,
        )
        await adapter.store(final)
        self._final_id = final_id

    async def run(self, adapter) -> int:
        return await adapter.count(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: int) -> float:
        if result == 1:
            return 1.0
        if result == 0:
            return 0.0  # data loss is not a resolution
        return max(0.0, 1.0 - abs(result - 1) * 0.5)


# ---------------------------------------------------------------------------
# C5 — FLAGGED status persists and is filterable
# ---------------------------------------------------------------------------

class FlaggedStatusPersistable(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "flagged_status_persistable"
    description = "A memory stored with status=FLAGGED must appear in list_all(status=FLAGGED)"

    _flagged_id: str

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Conflicting policy claim — needs human review.",
            embedding=vec("policy", "requirement"),
            status=MemoryStatus.FLAGGED,
        )
        await adapter.store(m)
        self._flagged_id = m.memory_id

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.FLAGGED)

    def score(self, result: list[Memory]) -> float:
        return 1.0 if any(m.memory_id == self._flagged_id for m in result) else 0.0


# ---------------------------------------------------------------------------
# C6 — Both conflicting claims must be co-retrievable via search
# ---------------------------------------------------------------------------

class ContradictionDetectableViaSearch(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "contradiction_detectable_via_search"
    description = "Both conflicting claims must be co-retrievable via search for detection to be possible"

    _claim_ids: set[str]

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        m1 = Memory(agent_id=self.agent_id, text="API limit is 100 req/min.", embedding=embedding)
        m2 = Memory(agent_id=self.agent_id, text="API limit is 500 req/min.", embedding=embedding)
        await adapter.store(m1)
        await adapter.store(m2)
        self._claim_ids = {m1.memory_id, m2.memory_id}

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("api", "rate", "limit"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        matched = len(self._claim_ids & found)
        return matched / len(self._claim_ids)


# ---------------------------------------------------------------------------
# C7 — supersedes / superseded_by links are both intact after resolution
# ---------------------------------------------------------------------------

class SupersessionLinksIntact(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "supersession_links_intact"
    description = "Both supersedes[] on the new memory and superseded_by on the old must be correct"

    _old_id: str
    _new_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("password", "security", "requirement")
        new_id = str(uuid4())

        old = Memory(
            agent_id=self.agent_id,
            text="Password requirement: 8 characters minimum.",
            embedding=embedding,
        )
        await adapter.store(old)
        self._old_id = old.memory_id

        old.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=new_id)
        await adapter.update(old)

        new = Memory(
            memory_id=new_id,
            agent_id=self.agent_id,
            text="Password requirement: 16 characters minimum.",
            embedding=embedding,
            supersedes=[old.memory_id],
        )
        await adapter.store(new)
        self._new_id = new_id

    async def run(self, adapter) -> tuple[Memory | None, Memory | None]:
        old = await adapter.fetch(self.agent_id, self._old_id)
        new = await adapter.fetch(self.agent_id, self._new_id)
        return old, new

    def score(self, result: tuple[Memory | None, Memory | None]) -> float:
        old, new = result
        if old is None or new is None:
            return 0.0
        old_link_ok = old.superseded_by == self._new_id
        new_link_ok = self._old_id in new.supersedes
        if old_link_ok and new_link_ok:
            return 1.0
        # Equal weighting: both links are necessary for a complete audit trail.
        # superseded_by enables "is this current?" checks; supersedes enables chain reconstruction.
        return 0.5 if old_link_ok or new_link_ok else 0.0


# ---------------------------------------------------------------------------
# C8 — Distinct embeddings remain distinguishable after adapter round-trip
# ---------------------------------------------------------------------------

class EmbeddingFidelityDistinct(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "embedding_fidelity_distinct"
    description = "Orthogonal embeddings must remain distinct (cosine < 0.05) after adapter round-trip"

    _m1_id: str
    _m2_id: str
    _original_m1: list[float]
    _original_m2: list[float]

    async def setup(self, adapter) -> None:
        # Completely different topic clusters → orthogonal in 16D (zero shared dimensions)
        emb_api = vec("api", "rate", "limit")
        emb_pass = vec("password", "security", "requirement")
        m1 = Memory(agent_id=self.agent_id, text="API rate limit is 100 req/min.", embedding=emb_api)
        m2 = Memory(agent_id=self.agent_id, text="Password requirement: 12+ chars.", embedding=emb_pass)
        await adapter.store(m1)
        await adapter.store(m2)
        self._m1_id = m1.memory_id
        self._m2_id = m2.memory_id
        self._original_m1 = emb_api
        self._original_m2 = emb_pass

    async def run(self, adapter) -> tuple[Memory | None, Memory | None]:
        m1 = await adapter.fetch(self.agent_id, self._m1_id)
        m2 = await adapter.fetch(self.agent_id, self._m2_id)
        return m1, m2

    def score(self, result: tuple[Memory | None, Memory | None]) -> float:
        m1, m2 = result
        if m1 is None or m2 is None:
            return 0.0
        if m1.embedding is None or m2.embedding is None:
            return 0.0
        sim = cosine(m1.embedding, m2.embedding)
        # Near-zero cosine confirms the adapter didn't corrupt or collapse distinct vectors
        return 1.0 if sim < 0.05 else max(0.0, 1.0 - sim / 0.05)


# ---------------------------------------------------------------------------
# C9 — Resolving conflicts reduces the ACTIVE count
# ---------------------------------------------------------------------------

class ResolutionReducesActiveCount(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "resolution_reduces_active_count"
    description = "Resolving two of three conflicting claims must reduce active count to one"

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy", "days")
        winner_id = str(uuid4())

        for text in ("Refund window is 14 days.", "Refund window is 21 days."):
            loser = Memory(agent_id=self.agent_id, text=text, embedding=embedding)
            await adapter.store(loser)
            loser.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=winner_id)
            await adapter.update(loser)

        winner = Memory(
            memory_id=winner_id,
            agent_id=self.agent_id,
            text="Refund window is 30 days.",
            embedding=embedding,
        )
        await adapter.store(winner)

    async def run(self, adapter) -> int:
        return await adapter.count(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: int) -> float:
        return 1.0 if result == 1 else 0.0


# ---------------------------------------------------------------------------
# C10 — All contradicting claims are searchable (detectable by retrieval)
# ---------------------------------------------------------------------------

class AllConflictingClaimsSearchable(TestCase):
    category    = BenchmarkCategory.CONTRADICTION
    name        = "all_conflicting_claims_searchable"
    description = "All versions of a contradicting fact must be retrievable via search"

    _claim_ids: list[str]

    async def setup(self, adapter) -> None:
        embedding = vec("price", "plan", "subscription")
        claims = [
            Memory(agent_id=self.agent_id, text="Enterprise plan: $199/mo.", embedding=embedding),
            Memory(agent_id=self.agent_id, text="Enterprise plan: $299/mo.", embedding=embedding),
            Memory(agent_id=self.agent_id, text="Enterprise plan: $399/mo.", embedding=embedding),
        ]
        for m in claims:
            await adapter.store(m)
        self._claim_ids = [m.memory_id for m in claims]

    async def run(self, adapter) -> list[SearchResult]:
        query = vec("price", "plan", "subscription")
        return await adapter.search(self.agent_id, query, top_k=len(self._claim_ids) + 2)

    def score(self, result: list) -> float:
        if not result:
            return 0.0
        found_ids = {r.memory.memory_id for r in result}
        matched = sum(1 for cid in self._claim_ids if cid in found_ids)
        return matched / len(self._claim_ids)


# ---------------------------------------------------------------------------
# Public list — used by the benchmark runner and __init__.py
# ---------------------------------------------------------------------------

CONTRADICTION_CASES: list[type[TestCase]] = [
    BothClaimsActiveIsContradiction,
    ResolvedContradictionHasOneActive,
    EmbeddingFidelityHighPrecision,
    ThreeClaimsOneActive,
    FlaggedStatusPersistable,
    ContradictionDetectableViaSearch,
    SupersessionLinksIntact,
    EmbeddingFidelityDistinct,
    ResolutionReducesActiveCount,
    AllConflictingClaimsSearchable,
]
