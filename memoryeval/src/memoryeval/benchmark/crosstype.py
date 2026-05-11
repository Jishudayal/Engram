"""Cross-type interaction test cases for MemoryEval.

Ten cases that measure how well a memory backend handles memories where
multiple properties interact simultaneously: status × count, importance ×
expiry, provenance × supersession, touch × field stability, memory_type ×
searchability, expiry × searchability, and holistic round-trip fidelity.

Scoring convention
------------------
- 1.0  — all cross-property interactions are correctly handled
- 0.0  — any interaction fails
- 0.x  — partial correctness where fractional measurement applies
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from memnotary.core.constants import MemoryStatus, MemoryType, SourceType
from memnotary.core.models import Memory, ProvenanceRecord, SearchResult
from memoryeval.benchmark._embeddings import vec
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# X1 — After supersession, status counts are exactly correct
# ---------------------------------------------------------------------------


class ActiveAndSupersededCountCorrect(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "active_and_superseded_count_correct"
    description = "After supersession, count(ACTIVE)=1 and count(SUPERSEDED)=1 must both hold"

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        v2_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="Rate limit: 100 req/min.", embedding=embedding)
        await adapter.store(v1)
        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="Rate limit: 500 req/min.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        await adapter.store(v2)

    async def run(self, adapter) -> tuple[int, int]:
        active = await adapter.count(self.agent_id, status=MemoryStatus.ACTIVE)
        superseded = await adapter.count(self.agent_id, status=MemoryStatus.SUPERSEDED)
        return active, superseded

    def score(self, result: tuple[int, int]) -> float:
        active, superseded = result
        active_ok = active == 1
        superseded_ok = superseded == 1
        if active_ok and superseded_ok:
            return 1.0
        return 0.5 if active_ok or superseded_ok else 0.0


# ---------------------------------------------------------------------------
# X2 — All four MemoryStatus values are independently storable and filterable
# ---------------------------------------------------------------------------


class AllFourStatusesStorable(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "all_four_statuses_storable"
    description = "One memory of each status must be independently storable and filterable"

    async def setup(self, adapter) -> None:
        embedding = vec("policy", "requirement")
        for status in (
            MemoryStatus.ACTIVE,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.FLAGGED,
            MemoryStatus.PENDING_REVIEW,
        ):
            await adapter.store(
                Memory(
                    agent_id=self.agent_id,
                    text=f"Policy claim with status {status}.",
                    embedding=embedding,
                    status=status,
                )
            )

    async def run(self, adapter) -> dict[str, int]:
        return {
            s.value: await adapter.count(self.agent_id, status=s)
            for s in (
                MemoryStatus.ACTIVE,
                MemoryStatus.SUPERSEDED,
                MemoryStatus.FLAGGED,
                MemoryStatus.PENDING_REVIEW,
            )
        }

    def score(self, result: dict[str, int]) -> float:
        correct = sum(1 for v in result.values() if v == 1)
        return correct / 4


# ---------------------------------------------------------------------------
# X3 — Provenance and supersession links coexist on the same memory
# ---------------------------------------------------------------------------


class ProvenanceAndSupersessionCombined(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "provenance_and_supersession_combined"
    description = (
        "A memory with both provenance and supersedes[] set must preserve both after round-trip"
    )

    _memory_id: str
    _old_id: str
    _ingested_at: datetime

    async def setup(self, adapter) -> None:
        self._ingested_at = datetime(2025, 3, 1, 9, 0, 0, tzinfo=UTC)
        old = Memory(
            agent_id=self.agent_id,
            text="Old SLA: 99.5% uptime.",
            embedding=vec("policy", "requirement"),
        )
        await adapter.store(old)
        self._old_id = old.memory_id

        new = Memory(
            agent_id=self.agent_id,
            text="Updated SLA: 99.9% uptime.",
            embedding=vec("policy", "requirement"),
            supersedes=[old.memory_id],
        )
        prov = ProvenanceRecord(
            memory_id=new.memory_id,
            source_type=SourceType.DOCUMENT,
            source_id="sla_v3.pdf",
            ingested_at=self._ingested_at,
        )
        new.attach_provenance(prov)
        await adapter.store(new)
        self._memory_id = new.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        supersedes_ok = self._old_id in result.supersedes
        provenance_ok = (
            result.provenance is not None
            and result.provenance.source_type == SourceType.DOCUMENT
            and result.provenance.ingested_at == self._ingested_at
        )
        if supersedes_ok and provenance_ok:
            return 1.0
        return 0.5 if supersedes_ok or provenance_ok else 0.0


# ---------------------------------------------------------------------------
# X4 — expires_at and importance coexist on the same memory
# ---------------------------------------------------------------------------


class ExpiryAndImportanceBothPreserved(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "expiry_and_importance_both_preserved"
    description = (
        "A memory with both expires_at and importance must preserve both fields after round-trip"
    )

    _memory_id: str
    _expected_importance: float = 0.82
    _expires_at: datetime

    async def setup(self, adapter) -> None:
        self._expires_at = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
        m = Memory(
            agent_id=self.agent_id,
            text="Temporary price promotion: 20% off Enterprise until year-end.",
            embedding=vec("price", "plan"),
            importance=self._expected_importance,
            expires_at=self._expires_at,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        importance_ok = abs(result.importance - self._expected_importance) < 1e-6
        expiry_ok = result.expires_at == self._expires_at
        if importance_ok and expiry_ok:
            return 1.0
        return 0.5 if importance_ok or expiry_ok else 0.0


# ---------------------------------------------------------------------------
# X5 — Supersession chain where both ends have provenance
# ---------------------------------------------------------------------------


class SupersessionChainWithProvenance(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "supersession_chain_with_provenance"
    description = "A v1→v2 chain where both memories have provenance must fully preserve all fields"

    _v1_id: str
    _v2_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("policy", "security")
        v2_id = str(uuid4())

        v1 = Memory(
            agent_id=self.agent_id, text="Password policy v1: 8 chars.", embedding=embedding
        )
        prov1 = ProvenanceRecord(
            memory_id=v1.memory_id,
            source_type=SourceType.DOCUMENT,
            source_id="policy_v1.pdf",
            ingested_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        v1.attach_provenance(prov1)
        await adapter.store(v1)
        self._v1_id = v1.memory_id

        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="Password policy v2: 12 chars.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        prov2 = ProvenanceRecord(
            memory_id=v2.memory_id,
            source_type=SourceType.DOCUMENT,
            source_id="policy_v2.pdf",
            ingested_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        v2.attach_provenance(prov2)
        await adapter.store(v2)
        self._v2_id = v2_id

    async def run(self, adapter) -> tuple[Memory | None, Memory | None]:
        v1 = await adapter.fetch(self.agent_id, self._v1_id)
        v2 = await adapter.fetch(self.agent_id, self._v2_id)
        return v1, v2

    def score(self, result: tuple[Memory | None, Memory | None]) -> float:
        v1, v2 = result
        if v1 is None or v2 is None:
            return 0.0
        v1_prov_ok = v1.provenance is not None and v1.provenance.source_id == "policy_v1.pdf"
        v2_prov_ok = v2.provenance is not None and v2.provenance.source_id == "policy_v2.pdf"
        chain_ok = v1.superseded_by == self._v2_id and self._v1_id in v2.supersedes
        checks = [v1_prov_ok, v2_prov_ok, chain_ok]
        passed = sum(checks)
        if passed == 3:
            return 1.0
        return passed / 3


# ---------------------------------------------------------------------------
# X6 — Search surfaces memories of all statuses (no implicit status filter)
# ---------------------------------------------------------------------------


class SearchReturnsAllStatusesByDefault(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "search_returns_all_statuses_by_default"
    description = "Search without a status filter must return memories of any status"

    _ids: dict[str, str]  # status → memory_id

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        self._ids = {}
        for status in (MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED, MemoryStatus.FLAGGED):
            m = Memory(
                agent_id=self.agent_id,
                text=f"Rate limit claim with status {status}.",
                embedding=embedding,
                status=status,
            )
            await adapter.store(m)
            self._ids[status.value] = m.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("api", "rate", "limit"), top_k=10)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        matched = sum(1 for mid in self._ids.values() if mid in found)
        return matched / len(self._ids)


# ---------------------------------------------------------------------------
# X7 — A memory past its expires_at is still returned by search
#
# Replaces CountMatchesListAll (tautological contract: count==len(list_all)
# is tested by the adapter contract suite, not a reliability property).
# Expiry is informational metadata — adapters must not silently filter it.
# ---------------------------------------------------------------------------


class ExpiredMemoryStillSearchable(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "expired_memory_still_searchable"
    description = "A memory with expires_at in the past must still appear in search results (expiry is metadata, not a filter)"

    _expired_id: str
    _fresh_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy")
        expired = Memory(
            agent_id=self.agent_id,
            text="Expired refund policy (lapsed promotion).",
            embedding=embedding,
            expires_at=datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC),
        )
        fresh = Memory(
            agent_id=self.agent_id,
            text="Current refund policy.",
            embedding=embedding,
        )
        await adapter.store(expired)
        await adapter.store(fresh)
        self._expired_id = expired.memory_id
        self._fresh_id = fresh.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        found = {r.memory.memory_id for r in result}
        expired_ok = self._expired_id in found
        fresh_ok = self._fresh_id in found
        if expired_ok and fresh_ok:
            return 1.0
        return 0.5 if expired_ok or fresh_ok else 0.0


# ---------------------------------------------------------------------------
# X8 — touch() only changes access fields; all other fields are stable
# ---------------------------------------------------------------------------


class TouchOnlyChangesAccessFields(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "touch_only_changes_access_fields"
    description = (
        "touch() must only increment access_count and set last_accessed; no other fields change"
    )

    _memory_id: str
    _expected_importance: float = 0.77
    _expected_status: MemoryStatus = MemoryStatus.ACTIVE

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="Multi-field memory for cross-property stability check.",
            embedding=vec("security", "policy"),
            importance=self._expected_importance,
            status=self._expected_status,
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> Memory | None:
        m = await adapter.fetch(self.agent_id, self._memory_id)
        if m is None:
            return None
        m.touch()
        await adapter.update(m)
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        importance_ok = abs(result.importance - self._expected_importance) < 1e-6
        status_ok = result.status == self._expected_status
        access_ok = result.access_count == 1 and result.last_accessed is not None
        checks = [importance_ok, status_ok, access_ok]
        return sum(checks) / 3


# ---------------------------------------------------------------------------
# X9 — Memories of all three MemoryType values are searchable in one result
#
# Replaces AgentIsolationEnforced (agent isolation is tested by the adapter
# contract suite). MemoryType (CONVERSATION/SKILL/CUSTOM) was not exercised
# anywhere in the benchmark; this case closes that gap.
# ---------------------------------------------------------------------------


class MemoryTypeCoexistAllSearchable(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "memory_type_coexist_all_searchable"
    description = (
        "Memories of CONVERSATION, SKILL, and CUSTOM type must all appear in a single search result"
    )

    _type_to_id: dict[str, str]

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        self._type_to_id = {}
        for mtype in (MemoryType.CONVERSATION, MemoryType.SKILL, MemoryType.CUSTOM):
            m = Memory(
                agent_id=self.agent_id,
                text=f"API rate limit memory ({mtype}).",
                embedding=embedding,
                memory_type=mtype,
            )
            await adapter.store(m)
            self._type_to_id[mtype.value] = m.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("api", "rate", "limit"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        matched = sum(1 for mid in self._type_to_id.values() if mid in found)
        return matched / len(self._type_to_id)


# ---------------------------------------------------------------------------
# X10 — Memory with every optional field populated survives full round-trip
# ---------------------------------------------------------------------------


class AllOptionalFieldsRoundtrip(TestCase):
    category = BenchmarkCategory.CROSS_TYPE
    name = "all_optional_fields_roundtrip"
    description = "A memory with importance, expires_at, provenance, and supersedes all set must preserve every field"

    _memory_id: str
    _old_id: str
    _expected_importance: float = 0.85
    _expires_at: datetime
    _ingested_at: datetime

    async def setup(self, adapter) -> None:
        self._expires_at = datetime(2027, 6, 30, 0, 0, 0, tzinfo=UTC)
        self._ingested_at = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)

        old = Memory(
            agent_id=self.agent_id,
            text="Old comprehensive policy v1.",
            embedding=vec("policy", "security"),
        )
        await adapter.store(old)
        self._old_id = old.memory_id

        new = Memory(
            agent_id=self.agent_id,
            text="Comprehensive policy v2 with all metadata set.",
            embedding=vec("policy", "security"),
            importance=self._expected_importance,
            expires_at=self._expires_at,
            supersedes=[old.memory_id],
        )
        prov = ProvenanceRecord(
            memory_id=new.memory_id,
            source_type=SourceType.DOCUMENT,
            source_id="comprehensive_policy_v2.pdf",
            ingested_at=self._ingested_at,
        )
        new.attach_provenance(prov)
        await adapter.store(new)
        self._memory_id = new.memory_id

    async def run(self, adapter) -> Memory | None:
        return await adapter.fetch(self.agent_id, self._memory_id)

    def score(self, result: Memory | None) -> float:
        if result is None:
            return 0.0
        checks = [
            abs(result.importance - self._expected_importance) < 1e-6,
            result.expires_at == self._expires_at,
            self._old_id in result.supersedes,
            result.provenance is not None
            and result.provenance.source_type == SourceType.DOCUMENT
            and result.provenance.ingested_at == self._ingested_at,
        ]
        passed = sum(checks)
        return passed / len(checks)


# ---------------------------------------------------------------------------
# Public list — used by the benchmark runner and __init__.py
# ---------------------------------------------------------------------------

CROSSTYPE_CASES: list[type[TestCase]] = [
    ActiveAndSupersededCountCorrect,
    AllFourStatusesStorable,
    ProvenanceAndSupersessionCombined,
    ExpiryAndImportanceBothPreserved,
    SupersessionChainWithProvenance,
    SearchReturnsAllStatusesByDefault,
    ExpiredMemoryStillSearchable,
    TouchOnlyChangesAccessFields,
    MemoryTypeCoexistAllSearchable,
    AllOptionalFieldsRoundtrip,
]
