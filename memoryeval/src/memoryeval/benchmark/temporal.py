"""Temporal reliability test cases for MemoryEval.

Ten cases that measure whether a memory backend correctly handles the
lifecycle of facts over time: updates, supersession, provenance, access
tracking, and staleness detection.

All cases use 16D synthetic embeddings from ``_embeddings.py`` — no
external API or embedding model is required.

Scoring convention
------------------
- 1.0  — backend handles this temporal property correctly
- 0.0  — backend fails or the property is not preserved
- 0.x  — partial correctness (used where fractional measurement makes sense)

Cases that expose a raw adapter's known limitation (e.g. no importance-
based ranking) intentionally score < 1.0 on naive systems. That gap is
exactly what Engram's consolidation engine closes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from engram.core.constants import MemoryStatus, SourceType
from engram.core.models import Memory, ProvenanceRecord
from memoryeval.benchmark._embeddings import vec
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# T1 — After updating a fact, only the new version is ACTIVE
# ---------------------------------------------------------------------------

class NewerVersionActive(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "newer_version_active"
    description = "After a fact update, only the newer version should be ACTIVE"

    _v2_text: str

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy", "days")
        v2_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="Refund policy is 30 days.", embedding=embedding)
        await adapter.store(v1)
        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="Refund policy is 60 days.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        await adapter.store(v2)
        self._v2_text = v2.text

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: list[Memory]) -> float:
        if len(result) != 1:
            return 0.0
        return 1.0 if self._v2_text in result[0].text else 0.0


# ---------------------------------------------------------------------------
# T2 — SUPERSEDED memories do not appear in the ACTIVE list
# ---------------------------------------------------------------------------

class SupersededExcludedFromActive(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "superseded_excluded_from_active"
    description = "SUPERSEDED memories must not appear in the ACTIVE list; the replacement must still be there"

    _superseded_id: str
    _fresh_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        old = Memory(agent_id=self.agent_id, text="Rate limit is 100 req/min.", embedding=embedding)
        await adapter.store(old)
        old.update_fields(status=MemoryStatus.SUPERSEDED)
        await adapter.update(old)
        self._superseded_id = old.memory_id

        fresh = Memory(agent_id=self.agent_id, text="Rate limit is 500 req/min.", embedding=embedding)
        await adapter.store(fresh)
        self._fresh_id = fresh.memory_id

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: list[Memory]) -> float:
        ids = {m.memory_id for m in result}
        if self._superseded_id in ids:
            return 0.0  # old memory is still active — wrong
        if self._fresh_id not in ids:
            return 0.0  # fresh memory was lost — data loss
        return 1.0


# ---------------------------------------------------------------------------
# T3 — After multiple updates only the final version is ACTIVE
# ---------------------------------------------------------------------------

class ThreeVersionsOnlyLatestActive(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "three_versions_only_latest_active"
    description = "With three fact versions, only the latest should remain ACTIVE"

    _v3_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("password", "security", "requirement")
        v3_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="Password must be 6+ chars.", embedding=embedding)
        await adapter.store(v1)
        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v3_id)
        await adapter.update(v1)

        v2 = Memory(agent_id=self.agent_id, text="Password must be 8+ chars.", embedding=embedding)
        await adapter.store(v2)
        v2.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v3_id)
        await adapter.update(v2)

        v3 = Memory(
            memory_id=v3_id,
            agent_id=self.agent_id,
            text="Password must be 12+ chars.",
            embedding=embedding,
            supersedes=[v1.memory_id, v2.memory_id],
        )
        await adapter.store(v3)
        self._v3_id = v3_id

    async def run(self, adapter) -> list[Memory]:
        return await adapter.list_all(self.agent_id, status=MemoryStatus.ACTIVE)

    def score(self, result: list[Memory]) -> float:
        if len(result) != 1:
            return 0.0
        return 1.0 if result[0].memory_id == self._v3_id else 0.0


# ---------------------------------------------------------------------------
# T4 — New memory correctly records what it supersedes
# ---------------------------------------------------------------------------

class SupersedesLinkRecorded(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "supersedes_link_recorded"
    description = "The supersedes[] list on a new memory must reference the memory it replaced"

    _v1_id: str
    _v2_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("price", "plan", "subscription")
        v2_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="Pro plan costs $49/month.", embedding=embedding)
        await adapter.store(v1)
        self._v1_id = v1.memory_id

        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="Pro plan costs $79/month.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        await adapter.store(v2)
        self._v2_id = v2_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._v2_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if self._v1_id in result.supersedes else 0.0


# ---------------------------------------------------------------------------
# T5 — Old memory correctly records which memory superseded it
# ---------------------------------------------------------------------------

class SupersededByLinkRecorded(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "superseded_by_link_recorded"
    description = "The superseded_by field on an old memory must point to its replacement"

    _v1_id: str
    _v2_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "request")
        v2_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="API limit: 60 requests/min.", embedding=embedding)
        await adapter.store(v1)
        self._v1_id = v1.memory_id

        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="API limit: 600 requests/min.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        await adapter.store(v2)
        self._v2_id = v2_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._v1_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if result.superseded_by == self._v2_id else 0.0


# ---------------------------------------------------------------------------
# T6 — importance field survives storage and retrieval
# ---------------------------------------------------------------------------

class ImportanceFieldRoundtrips(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "importance_field_roundtrips"
    description = "A custom importance score must survive write-read roundtrip"

    _memory_id: str
    _expected_importance: float = 0.93

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Critical security policy: all passwords expire every 90 days.",
            embedding=vec("password", "security", "expire"),
            importance=self._expected_importance,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if abs(result.importance - self._expected_importance) < 1e-6 else 0.0


# ---------------------------------------------------------------------------
# T7 — memories with a past expires_at are identifiable as expired
# ---------------------------------------------------------------------------

class ExpirationDetectable(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "expiration_detectable"
    description = "A past expires_at must flag as expired; a future expires_at must not"

    _past_id: str
    _future_id: str

    async def setup(self, adapter) -> None:
        past = Memory(
            agent_id=self.agent_id,
            text="Temporary promotion: 50% off all plans — expired.",
            embedding=vec("price", "plan"),
            expires_at=datetime.now(UTC) - timedelta(minutes=1),
        )
        future = Memory(
            agent_id=self.agent_id,
            text="Temporary promotion: 50% off all plans — still active.",
            embedding=vec("price", "plan"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        await adapter.store(past)
        await adapter.store(future)
        self._past_id = past.memory_id
        self._future_id = future.memory_id

    async def run(self, adapter) -> tuple[Memory | None, Memory | None]:
        past = await adapter.fetch(self.agent_id, self._past_id)
        future = await adapter.fetch(self.agent_id, self._future_id)
        return past, future

    def score(self, result: tuple[Memory | None, Memory | None]) -> float:
        past, future = result
        if past is None or future is None:
            return 0.0
        past_ok = past.is_expired()
        future_ok = not future.is_expired()
        if past_ok and future_ok:
            return 1.0
        return 0.5 if past_ok or future_ok else 0.0


# ---------------------------------------------------------------------------
# T8 — touch() increments access_count and persists after update
# ---------------------------------------------------------------------------

class AccessTrackingPersists(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "access_tracking_persists"
    description = "touch() must increment access_count and persist after adapter.update()"

    _memory_id: str

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Return window for all products is 30 days.",
            embedding=vec("return", "window", "days"),
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        m = await adapter.fetch(self.agent_id, self._memory_id)
        if m is None:
            return None
        m.touch()
        m.touch()
        await adapter.update(m)
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if result.access_count == 2 and result.last_accessed is not None else 0.0


# ---------------------------------------------------------------------------
# T9 — provenance metadata survives storage and retrieval
# ---------------------------------------------------------------------------

class ProvenanceRoundtrips(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "provenance_roundtrips"
    description = "Provenance metadata (source_type, ingested_at) must survive write-read"

    _memory_id: str
    _ingested_at: datetime

    async def setup(self, adapter) -> None:
        self._ingested_at = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        m = Memory(
            agent_id=self.agent_id,
            text="SLA uptime guarantee is 99.9%.",
            embedding=vec("policy", "requirement"),
        )
        prov = ProvenanceRecord(
            memory_id=m.memory_id,
            source_type=SourceType.DOCUMENT,
            source_id="sla_v2.pdf",
            ingested_at=self._ingested_at,
        )
        m.attach_provenance(prov)
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None or result.provenance is None:
            return 0.0
        prov = result.provenance
        source_ok = prov.source_type == SourceType.DOCUMENT
        time_ok = prov.ingested_at == self._ingested_at
        return 1.0 if source_ok and time_ok else 0.5


# ---------------------------------------------------------------------------
# T10 — updated_at is preserved exactly through a write-read round-trip
# ---------------------------------------------------------------------------

class UpdatedAtPreservedOnRoundtrip(TestCase):
    category    = BenchmarkCategory.TEMPORAL
    name        = "updated_at_preserved_on_roundtrip"
    description = "A manually set updated_at timestamp must survive write-read roundtrip unchanged"

    _memory_id: str
    _expected_updated_at: datetime

    async def setup(self, adapter) -> None:
        # Fixed historical datetime — deterministic, independent of when the test runs
        self._expected_updated_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        m = Memory(
            agent_id=self.agent_id,
            text="Archived policy document from Q2 2024.",
            embedding=vec("policy", "security"),
            updated_at=self._expected_updated_at,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        return 1.0 if result.updated_at == self._expected_updated_at else 0.0


# ---------------------------------------------------------------------------
# Public list — used by the benchmark runner and __init__.py
# ---------------------------------------------------------------------------

TEMPORAL_CASES: list[type[TestCase]] = [
    NewerVersionActive,
    SupersededExcludedFromActive,
    ThreeVersionsOnlyLatestActive,
    SupersedesLinkRecorded,
    SupersededByLinkRecorded,
    ImportanceFieldRoundtrips,
    ExpirationDetectable,
    AccessTrackingPersists,
    ProvenanceRoundtrips,
    UpdatedAtPreservedOnRoundtrip,
]
