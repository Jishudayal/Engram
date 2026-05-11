"""Multi-hop retrieval test cases for MemoryEval.

Ten cases that measure whether a memory backend can surface all the memories
required for multi-step reasoning.

**Tier 1 — Adapter fidelity (7 cases):** verify that search returns expected
results based on cosine similarity, threshold filtering, and version history.
These should score 1.0 on any compliant adapter.

**Tier 2 — Chained retrieval (3 cases, M5/M7/M10):** verify that memories
connected by bridging keywords are all reachable across two sequential searches.
These tests simulate the retrieval pattern required for multi-hop reasoning,
where hop N's result informs the query for hop N+1. These also score 1.0 on
InMemoryAdapter because keyword bridging works with cosine similarity alone.

Scoring convention
------------------
- 1.0  — adapter correctly handles this retrieval property
- 0.0  — property fails (wrong count, wrong order, wrong scope)
- 0.x  — partial correctness (fraction of expected results found)
"""

from __future__ import annotations

from uuid import uuid4

from memnotary.core.constants import MemoryStatus
from memnotary.core.models import Memory, SearchResult
from memoryeval.benchmark._embeddings import vec
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# M1 — All same-topic memories are co-retrievable in a single search
# ---------------------------------------------------------------------------


class RelatedFactsAllRetrievable(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "related_facts_all_retrievable"
    description = "All same-topic memories must appear together in a single search result set"

    _claim_ids: list[str]

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy", "days")
        claims = [
            Memory(
                agent_id=self.agent_id,
                text="Return window is 30 days for all items.",
                embedding=embedding,
            ),
            Memory(
                agent_id=self.agent_id,
                text="Electronics have a 15-day return window.",
                embedding=embedding,
            ),
            Memory(
                agent_id=self.agent_id,
                text="Software is non-refundable after download.",
                embedding=embedding,
            ),
        ]
        for m in claims:
            await adapter.store(m)
        self._claim_ids = [m.memory_id for m in claims]

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy", "days"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        return sum(1 for cid in self._claim_ids if cid in found) / len(self._claim_ids)


# ---------------------------------------------------------------------------
# M2 — A partial keyword query still surfaces the full-keyword memory
# ---------------------------------------------------------------------------


class PartialQuerySurfacesRelated(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "partial_query_surfaces_related"
    description = (
        "A 2-keyword query must surface a memory that matches 4 keywords (partial overlap)"
    )

    _memory_id: str

    async def setup(self, adapter) -> None:
        m = Memory(
            agent_id=self.agent_id,
            text="API rate limit is 1000 requests per minute for paid plans.",
            embedding=vec("api", "rate", "limit", "request"),
        )
        await adapter.store(m)
        self._memory_id = m.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        # 2-keyword query has cosine ≈ 0.71 with the 4-keyword memory — above 0.5 threshold
        return await adapter.search(self.agent_id, vec("api", "rate"), top_k=5, score_threshold=0.5)

    def score(self, result: list[SearchResult]) -> float:
        found = {r.memory.memory_id for r in result}
        return 1.0 if self._memory_id in found else 0.0


# ---------------------------------------------------------------------------
# M3 — More relevant memory ranks above less relevant in cosine order
# ---------------------------------------------------------------------------


class RelevanceRankingCorrect(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "relevance_ranking_correct"
    description = "A memory with higher keyword overlap must rank above one with lower overlap"

    _high_id: str
    _low_id: str

    async def setup(self, adapter) -> None:
        high = Memory(
            agent_id=self.agent_id,
            text="API rate limit: 1000 requests/min for all endpoints.",
            embedding=vec("api", "rate", "limit", "request"),
        )
        low = Memory(
            agent_id=self.agent_id,
            text="API documentation is available at docs.example.com.",
            embedding=vec("api", "rate"),
        )
        await adapter.store(high)
        await adapter.store(low)
        self._high_id = high.memory_id
        self._low_id = low.memory_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("api", "rate", "limit", "request"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        if len(result) < 2:
            return 0.0
        ids = [r.memory.memory_id for r in result]
        if self._high_id not in ids or self._low_id not in ids:
            return 0.0
        return 1.0 if ids.index(self._high_id) < ids.index(self._low_id) else 0.0


# ---------------------------------------------------------------------------
# M4 — Score threshold isolates on-topic memories from off-topic ones
# ---------------------------------------------------------------------------


class TopicIsolationByThreshold(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "topic_isolation_by_threshold"
    description = "A score threshold must exclude off-topic memories while retaining on-topic ones"

    _api_ids: set[str]

    async def setup(self, adapter) -> None:
        api_emb = vec("api", "rate", "limit")
        api_ids: list[str] = []
        for text in (
            "Rate limit: 100 req/min.",
            "Rate limit: 500 req/min.",
            "Rate limit: 1000 req/min.",
        ):
            m = Memory(agent_id=self.agent_id, text=text, embedding=api_emb)
            await adapter.store(m)
            api_ids.append(m.memory_id)
        self._api_ids = set(api_ids)

        price_emb = vec("price", "plan", "subscription")
        for text in ("Starter plan: $9/mo.", "Pro plan: $49/mo.", "Enterprise: $199/mo."):
            await adapter.store(Memory(agent_id=self.agent_id, text=text, embedding=price_emb))

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(
            self.agent_id, vec("api", "rate", "limit"), top_k=10, score_threshold=0.5
        )

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        all_api_found = self._api_ids.issubset(found)
        no_contamination = all(mid in self._api_ids for mid in found)
        if all_api_found and no_contamination:
            return 1.0
        if all_api_found:
            return 0.5  # right coverage but contaminated
        return 0.0


# ---------------------------------------------------------------------------
# M5 — A 2-hop bridge chain: Start→Bridge→End, all 3 reachable across hops
#
# Replaces TopKRespected (pure contract test, already in adapter suite).
# Bridge memory shares "policy" with Start and "security" with End.
# hop1 finds Start+Bridge; hop2 finds Bridge+End.
# ---------------------------------------------------------------------------


class TwoHopBridgeReachable(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "two_hop_bridge_reachable"
    description = "All three memories in a Start→Bridge→End chain must be reachable across two hops"

    _start_id: str
    _bridge_id: str
    _end_id: str

    async def setup(self, adapter) -> None:
        # refund+policy cluster → policy+security bridge → security+password cluster
        # cosine(Start, Bridge) = 0.5; cosine(Bridge, End) = 0.5; cosine(Start, End) = 0
        start = Memory(
            agent_id=self.agent_id, text="Refund policy updated.", embedding=vec("refund", "policy")
        )
        bridge = Memory(
            agent_id=self.agent_id,
            text="Policy and security review.",
            embedding=vec("policy", "security"),
        )
        end = Memory(
            agent_id=self.agent_id,
            text="Security password requirements.",
            embedding=vec("security", "password"),
        )
        for m in (start, bridge, end):
            await adapter.store(m)
        self._start_id = start.memory_id
        self._bridge_id = bridge.memory_id
        self._end_id = end.memory_id

    async def run(self, adapter) -> tuple[list[SearchResult], list[SearchResult]]:
        hop1 = await adapter.search(
            self.agent_id, vec("refund", "policy"), top_k=5, score_threshold=0.3
        )
        hop2 = await adapter.search(
            self.agent_id, vec("security", "password"), top_k=5, score_threshold=0.3
        )
        return hop1, hop2

    def score(self, result: tuple[list[SearchResult], list[SearchResult]]) -> float:
        hop1, hop2 = result
        hop1_ids = {r.memory.memory_id for r in hop1}
        hop2_ids = {r.memory.memory_id for r in hop2}
        start_found = self._start_id in hop1_ids
        bridge_found = self._bridge_id in hop1_ids and self._bridge_id in hop2_ids
        end_found = self._end_id in hop2_ids
        return sum([start_found, bridge_found, end_found]) / 3


# ---------------------------------------------------------------------------
# M6 — Search surfaces both ACTIVE and SUPERSEDED versions (no implicit filter)
# ---------------------------------------------------------------------------


class AllVersionsSurfacedBySearch(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "all_versions_surfaced_by_search"
    description = (
        "Search must return ACTIVE and SUPERSEDED versions when no status filter is applied"
    )

    _v1_id: str
    _v2_id: str

    async def setup(self, adapter) -> None:
        embedding = vec("refund", "policy", "days")
        v2_id = str(uuid4())

        v1 = Memory(agent_id=self.agent_id, text="Refund policy: 30 days.", embedding=embedding)
        await adapter.store(v1)
        v1.update_fields(status=MemoryStatus.SUPERSEDED, superseded_by=v2_id)
        await adapter.update(v1)
        self._v1_id = v1.memory_id

        v2 = Memory(
            memory_id=v2_id,
            agent_id=self.agent_id,
            text="Refund policy: 60 days.",
            embedding=embedding,
            supersedes=[v1.memory_id],
        )
        await adapter.store(v2)
        self._v2_id = v2_id

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(self.agent_id, vec("refund", "policy", "days"), top_k=5)

    def score(self, result: list[SearchResult]) -> float:
        found = {r.memory.memory_id for r in result}
        v1_found = self._v1_id in found
        v2_found = self._v2_id in found
        if v1_found and v2_found:
            return 1.0
        return 0.5 if v1_found or v2_found else 0.0


# ---------------------------------------------------------------------------
# M7 — A 3-hop chain: A→B→C→D, all 4 memories reachable across two searches
#
# Replaces MultiTopicStoreAllRetrievable (pure contract test).
# Chain: refund+return → return+window → window+days → days+expire
# hop1 (query: refund+return) surfaces A (1.0) + B (0.5)
# hop2 (query: window+days) surfaces C (1.0) + D (0.5)
# ---------------------------------------------------------------------------


class ChainedFactsThreeHopSurface(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "chained_facts_three_hop_surface"
    description = "All 4 memories in a 3-hop chain must surface across two targeted searches"

    _a_id: str
    _b_id: str
    _c_id: str
    _d_id: str

    async def setup(self, adapter) -> None:
        # Each adjacent pair shares one keyword → cosine = 0.5 between neighbours
        a = Memory(
            agent_id=self.agent_id, text="Refund return notice.", embedding=vec("refund", "return")
        )
        b = Memory(
            agent_id=self.agent_id, text="Return window policy.", embedding=vec("return", "window")
        )
        c = Memory(
            agent_id=self.agent_id, text="Window days calculation.", embedding=vec("window", "days")
        )
        d = Memory(
            agent_id=self.agent_id,
            text="Days until expiry notice.",
            embedding=vec("days", "expire"),
        )
        for m in (a, b, c, d):
            await adapter.store(m)
        self._a_id = a.memory_id
        self._b_id = b.memory_id
        self._c_id = c.memory_id
        self._d_id = d.memory_id

    async def run(self, adapter) -> tuple[list[SearchResult], list[SearchResult]]:
        hop1 = await adapter.search(
            self.agent_id, vec("refund", "return"), top_k=5, score_threshold=0.3
        )
        hop2 = await adapter.search(
            self.agent_id, vec("window", "days"), top_k=5, score_threshold=0.3
        )
        return hop1, hop2

    def score(self, result: tuple[list[SearchResult], list[SearchResult]]) -> float:
        hop1, hop2 = result
        hop1_ids = {r.memory.memory_id for r in hop1}
        hop2_ids = {r.memory.memory_id for r in hop2}
        a_found = self._a_id in hop1_ids
        b_found = self._b_id in hop1_ids
        c_found = self._c_id in hop2_ids
        d_found = self._d_id in hop2_ids
        return sum([a_found, b_found, c_found, d_found]) / 4


# ---------------------------------------------------------------------------
# M8 — A high score threshold still captures exact-match memories
# ---------------------------------------------------------------------------


class HighThresholdCapturesExactMatches(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "high_threshold_captures_exact_matches"
    description = "score_threshold=0.99 must still return memories whose cosine similarity is ≈ 1.0"

    _ids: list[str]

    async def setup(self, adapter) -> None:
        embedding = vec("api", "rate", "limit")
        ids: list[str] = []
        for text in ("Rate limit A.", "Rate limit B.", "Rate limit C."):
            m = Memory(agent_id=self.agent_id, text=text, embedding=embedding)
            await adapter.store(m)
            ids.append(m.memory_id)
        self._ids = ids

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(
            self.agent_id, vec("api", "rate", "limit"), top_k=10, score_threshold=0.99
        )

    def score(self, result: list[SearchResult]) -> float:
        if not result:
            return 0.0
        found = {r.memory.memory_id for r in result}
        return sum(1 for mid in self._ids if mid in found) / len(self._ids)


# ---------------------------------------------------------------------------
# M9 — Orthogonal query with high threshold returns no results
# ---------------------------------------------------------------------------


class OrthogonalQueryReturnsEmpty(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "orthogonal_query_returns_empty"
    description = "A query on an unrelated topic with score_threshold=0.99 must return no results"

    async def setup(self, adapter) -> None:
        sec_emb = vec("password", "security", "requirement")
        for text in ("Password must be 12+ chars.", "MFA required for all accounts."):
            await adapter.store(Memory(agent_id=self.agent_id, text=text, embedding=sec_emb))

    async def run(self, adapter) -> list[SearchResult]:
        return await adapter.search(
            self.agent_id, vec("price", "plan", "subscription"), top_k=10, score_threshold=0.99
        )

    def score(self, result: list[SearchResult]) -> float:
        return 1.0 if len(result) == 0 else 0.0


# ---------------------------------------------------------------------------
# M10 — Cross-cluster bridged retrieval: orthogonal clusters linked by a bridge
#
# Replaces ListAllCompleteCoverage (pure contract test).
# API cluster (dims 4-7) and Price cluster (dims 12-15) have zero cosine.
# Bridge memory vec("limit","subscription") connects them:
#   cosine(api+rate+limit query, bridge) ≈ 0.41
#   cosine(price+plan+subscription query, bridge) ≈ 0.41
# Without the bridge memory, there is no path from api→price in one hop.
# ---------------------------------------------------------------------------


class CrossClusterBridgedRetrieval(TestCase):
    category = BenchmarkCategory.MULTIHOP
    name = "cross_cluster_bridged_retrieval"
    description = (
        "A bridge memory must make an otherwise-unreachable cluster discoverable in a 2-hop chain"
    )

    _api_id: str
    _bridge_id: str
    _price_id: str

    async def setup(self, adapter) -> None:
        api_mem = Memory(
            agent_id=self.agent_id,
            text="API rate limit is 1000 requests/min.",
            embedding=vec("api", "rate", "limit"),
        )
        bridge = Memory(
            agent_id=self.agent_id,
            text="API subscription plan pricing.",
            # limit (dim 6) bridges api cluster; subscription (dim 14) bridges price cluster
            embedding=vec("limit", "subscription"),
        )
        price_mem = Memory(
            agent_id=self.agent_id,
            text="Pro plan costs $79/month.",
            embedding=vec("price", "plan", "subscription"),
        )
        for m in (api_mem, bridge, price_mem):
            await adapter.store(m)
        self._api_id = api_mem.memory_id
        self._bridge_id = bridge.memory_id
        self._price_id = price_mem.memory_id

    async def run(self, adapter) -> tuple[list[SearchResult], list[SearchResult]]:
        hop1 = await adapter.search(
            self.agent_id, vec("api", "rate", "limit"), top_k=5, score_threshold=0.2
        )
        hop2 = await adapter.search(
            self.agent_id, vec("price", "plan", "subscription"), top_k=5, score_threshold=0.2
        )
        return hop1, hop2

    def score(self, result: tuple[list[SearchResult], list[SearchResult]]) -> float:
        hop1, hop2 = result
        hop1_ids = {r.memory.memory_id for r in hop1}
        hop2_ids = {r.memory.memory_id for r in hop2}
        api_found = self._api_id in hop1_ids
        bridge_in_hop1 = self._bridge_id in hop1_ids
        bridge_in_hop2 = self._bridge_id in hop2_ids
        price_found = self._price_id in hop2_ids
        return sum([api_found, bridge_in_hop1, bridge_in_hop2, price_found]) / 4


# ---------------------------------------------------------------------------
# Public list — used by the benchmark runner and __init__.py
# ---------------------------------------------------------------------------

MULTIHOP_CASES: list[type[TestCase]] = [
    RelatedFactsAllRetrievable,
    PartialQuerySurfacesRelated,
    RelevanceRankingCorrect,
    TopicIsolationByThreshold,
    TwoHopBridgeReachable,
    AllVersionsSurfacedBySearch,
    ChainedFactsThreeHopSurface,
    HighThresholdCapturesExactMatches,
    OrthogonalQueryReturnsEmpty,
    CrossClusterBridgedRetrieval,
]
