"""Tests for ContradictionDetector — classification, detection, and batch scan."""

import pytest

from memnotary.adapters.memory import InMemoryAdapter
from memnotary.core.constants import (
    CLUSTER_SIMILARITY_THRESHOLD,
    CONTRADICTION_CONFIDENCE_THRESHOLD,
    ConflictType,
    ResolutionStatus,
)
from memnotary.core.contradiction import _DEFAULT_MODEL, ClassificationResult, ContradictionDetector
from memnotary.core.models import ConflictRecord, Memory

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_memory(
    text: str,
    *,
    embedding: list[float] | None = None,
    agent_id: str = "agent-1",
) -> Memory:
    return Memory(agent_id=agent_id, text=text, embedding=embedding)


def make_detector(
    responses: list[str],
    **kwargs: object,
) -> ContradictionDetector:
    """Build a detector backed by a deterministic mock LLM.

    Responses are consumed in order. When exhausted, defaults to "unrelated".
    """
    it = iter(responses)

    async def mock_llm(prompt: str) -> str:
        return next(it, '{"verdict": "unrelated", "confidence": 0.0, "summary": ""}')

    return ContradictionDetector(llm_fn=mock_llm, **kwargs)  # type: ignore[arg-type]


# Reusable LLM response strings
_CONTRADICTION = (
    '{"verdict": "direct_contradiction", "confidence": 0.95, "summary": "Rate limits conflict"}'
)
_SUPERSESSION = (
    '{"verdict": "temporal_supersession", "confidence": 0.88, "summary": "Old value superseded"}'
)
_PARTIAL = '{"verdict": "partial_conflict", "confidence": 0.84, "summary": "Partial overlap"}'
_UNRELATED = '{"verdict": "unrelated", "confidence": 0.91, "summary": "Different topics"}'


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_is_frozen_dataclass(self) -> None:
        r = ClassificationResult(
            memory_a_id="a",
            memory_b_id="b",
            verdict="unrelated",
            confidence=0.5,
            summary="test",
        )
        with pytest.raises((AttributeError, TypeError)):
            r.verdict = "direct_contradiction"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        r = ClassificationResult(
            memory_a_id="a",
            memory_b_id="b",
            verdict="direct_contradiction",
            confidence=0.91,
            summary="conflict found",
        )
        assert r.memory_a_id == "a"
        assert r.memory_b_id == "b"
        assert r.verdict == "direct_contradiction"
        assert r.confidence == 0.91
        assert r.summary == "conflict found"


# ---------------------------------------------------------------------------
# ContradictionDetector — init and properties
# ---------------------------------------------------------------------------


class TestContradictionDetectorInit:
    def test_default_model_is_known_anthropic_model(self) -> None:
        assert _DEFAULT_MODEL in {
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        }

    def test_custom_llm_fn_accepted(self) -> None:
        async def fn(p: str) -> str:
            return "{}"

        det = ContradictionDetector(llm_fn=fn)
        assert det is not None

    def test_no_llm_fn_no_anthropic_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="anthropic"):
            ContradictionDetector()

    def test_cluster_threshold_property_reflects_kwarg(self) -> None:
        det = make_detector([], cluster_threshold=0.75)
        assert det.cluster_threshold == 0.75

    def test_default_cluster_threshold(self) -> None:
        det = make_detector([])
        assert det.cluster_threshold == CLUSTER_SIMILARITY_THRESHOLD

    def test_custom_max_concurrency_accepted(self) -> None:
        det = make_detector([], max_concurrency=3)
        assert det._max_concurrency == 3


# ---------------------------------------------------------------------------
# _parse_classification
# ---------------------------------------------------------------------------


class TestParseClassification:
    def _det(self) -> ContradictionDetector:
        return make_detector([])

    def test_direct_contradiction(self) -> None:
        r = self._det()._parse_classification(_CONTRADICTION, "a", "b")
        assert r.verdict == "direct_contradiction"
        assert r.confidence == 0.95
        assert r.summary == "Rate limits conflict"

    def test_temporal_supersession(self) -> None:
        r = self._det()._parse_classification(_SUPERSESSION, "a", "b")
        assert r.verdict == "temporal_supersession"
        assert r.confidence == 0.88

    def test_partial_conflict(self) -> None:
        r = self._det()._parse_classification(_PARTIAL, "a", "b")
        assert r.verdict == "partial_conflict"

    def test_unrelated(self) -> None:
        r = self._det()._parse_classification(_UNRELATED, "a", "b")
        assert r.verdict == "unrelated"

    def test_malformed_json_defaults_to_unrelated(self) -> None:
        r = self._det()._parse_classification("not json {{{", "a", "b")
        assert r.verdict == "unrelated"
        assert r.confidence == 0.0

    def test_empty_string_defaults_to_unrelated(self) -> None:
        r = self._det()._parse_classification("", "a", "b")
        assert r.verdict == "unrelated"

    def test_missing_fields_use_defaults(self) -> None:
        r = self._det()._parse_classification("{}", "a", "b")
        assert r.verdict == "unrelated"
        assert r.confidence == 0.0
        assert r.summary == ""

    def test_confidence_clamped_above_one(self) -> None:
        raw = '{"verdict": "direct_contradiction", "confidence": 1.5, "summary": ""}'
        r = self._det()._parse_classification(raw, "a", "b")
        assert r.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        raw = '{"verdict": "partial_conflict", "confidence": -0.3, "summary": ""}'
        r = self._det()._parse_classification(raw, "a", "b")
        assert r.confidence == 0.0

    def test_verdict_lowercased(self) -> None:
        raw = '{"verdict": "DIRECT_CONTRADICTION", "confidence": 0.9, "summary": ""}'
        r = self._det()._parse_classification(raw, "a", "b")
        assert r.verdict == "direct_contradiction"

    def test_ids_preserved_in_result(self) -> None:
        r = self._det()._parse_classification(_CONTRADICTION, "mem-x", "mem-y")
        assert r.memory_a_id == "mem-x"
        assert r.memory_b_id == "mem-y"


# ---------------------------------------------------------------------------
# _to_conflict_record
# ---------------------------------------------------------------------------


class TestToConflictRecord:
    def _det(self, threshold: float = CONTRADICTION_CONFIDENCE_THRESHOLD) -> ContradictionDetector:
        return make_detector([], confidence_threshold=threshold)

    def _result(
        self,
        verdict: str,
        confidence: float,
        summary: str = "",
    ) -> ClassificationResult:
        return ClassificationResult(
            memory_a_id="mem-a",
            memory_b_id="mem-b",
            verdict=verdict,
            confidence=confidence,
            summary=summary,
        )

    def test_direct_contradiction_above_threshold(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("direct_contradiction", 0.95), "agent-1"
        )
        assert record is not None
        assert record.conflict_type == ConflictType.DIRECT_CONTRADICTION

    def test_temporal_supersession(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("temporal_supersession", 0.90), "agent-1"
        )
        assert record is not None
        assert record.conflict_type == ConflictType.TEMPORAL_SUPERSESSION

    def test_partial_conflict(self) -> None:
        record = self._det()._to_conflict_record(self._result("partial_conflict", 0.85), "agent-1")
        assert record is not None
        assert record.conflict_type == ConflictType.PARTIAL_CONFLICT

    def test_unrelated_returns_none(self) -> None:
        assert self._det()._to_conflict_record(self._result("unrelated", 0.99), "agent-1") is None

    def test_unknown_verdict_returns_none(self) -> None:
        assert self._det()._to_conflict_record(self._result("foo_bar", 0.99), "agent-1") is None

    def test_below_threshold_returns_none(self) -> None:
        det = self._det(threshold=0.80)
        assert (
            det._to_conflict_record(self._result("direct_contradiction", 0.79), "agent-1") is None
        )

    def test_at_threshold_emits_record(self) -> None:
        det = self._det(threshold=0.80)
        record = det._to_conflict_record(self._result("direct_contradiction", 0.80), "agent-1")
        assert record is not None

    def test_description_preserved(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("direct_contradiction", 0.95, "Rate limits conflict"), "agent-1"
        )
        assert record is not None
        assert record.description == "Rate limits conflict"

    def test_blank_summary_stored_as_none(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("direct_contradiction", 0.95, ""), "agent-1"
        )
        assert record is not None
        assert record.description is None

    def test_resolution_status_is_pending(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("direct_contradiction", 0.95), "agent-1"
        )
        assert record is not None
        assert record.resolution_status == ResolutionStatus.PENDING

    def test_memory_ids_preserved(self) -> None:
        record = self._det()._to_conflict_record(
            self._result("direct_contradiction", 0.95), "agent-1"
        )
        assert record is not None
        assert record.memory_a_id == "mem-a"
        assert record.memory_b_id == "mem-b"
        assert record.agent_id == "agent-1"


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


class TestDetect:
    async def test_no_embedding_returns_empty(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem = make_memory("fact", embedding=None)
        result = await det.detect(mem, [make_memory("other", embedding=[1.0, 0.0])])
        assert result == []

    async def test_skips_self(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem = make_memory("fact", embedding=[1.0, 0.0])
        result = await det.detect(mem, [mem])
        assert result == []

    async def test_skips_candidate_without_embedding(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem = make_memory("fact A", embedding=[1.0, 0.0])
        no_emb = make_memory("fact B", embedding=None)
        result = await det.detect(mem, [no_emb])
        assert result == []

    async def test_empty_candidates_returns_empty(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem = make_memory("fact", embedding=[1.0, 0.0])
        result = await det.detect(mem, [])
        assert result == []

    async def test_returns_conflict_record_for_confirmed_pair(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.0])
        mem_b = make_memory("Rate limit is 500", embedding=[0.99, 0.01])
        records = await det.detect(mem_a, [mem_b])
        assert len(records) == 1
        assert records[0].conflict_type == ConflictType.DIRECT_CONTRADICTION
        assert records[0].confidence == 0.95
        assert records[0].description == "Rate limits conflict"

    async def test_returns_empty_for_unrelated_pair(self) -> None:
        det = make_detector([_UNRELATED])
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.0])
        mem_b = make_memory("Office is in London", embedding=[-1.0, 0.0])
        records = await det.detect(mem_a, [mem_b])
        assert records == []

    async def test_multiple_candidates_processed(self) -> None:
        det = make_detector([_CONTRADICTION, _UNRELATED])
        mem = make_memory("fact", embedding=[1.0, 0.0])
        c1 = make_memory("conflict", embedding=[0.99, 0.0])
        c2 = make_memory("unrelated", embedding=[-1.0, 0.0])
        records = await det.detect(mem, [c1, c2])
        assert len(records) == 1

    async def test_all_candidates_unrelated_returns_empty(self) -> None:
        det = make_detector([_UNRELATED, _UNRELATED])
        mem = make_memory("fact", embedding=[1.0, 0.0])
        c1 = make_memory("a", embedding=[0.5, 0.5])
        c2 = make_memory("b", embedding=[0.6, 0.4])
        records = await det.detect(mem, [c1, c2])
        assert records == []

    async def test_agent_id_from_new_memory(self) -> None:
        det = make_detector([_CONTRADICTION])
        mem_a = make_memory("fact A", embedding=[1.0, 0.0], agent_id="my-agent")
        mem_b = make_memory("fact B", embedding=[0.99, 0.01], agent_id="my-agent")
        records = await det.detect(mem_a, [mem_b])
        assert len(records) == 1
        assert records[0].agent_id == "my-agent"


# ---------------------------------------------------------------------------
# scan()
# ---------------------------------------------------------------------------


class TestScan:
    async def test_empty_collection_returns_empty(self) -> None:
        det = make_detector([])
        result = await det.scan("agent-1", InMemoryAdapter())
        assert result == []

    async def test_single_memory_returns_empty(self) -> None:
        det = make_detector([])
        adapter = InMemoryAdapter()
        await adapter.store(make_memory("only one", embedding=[1.0, 0.0]))
        result = await det.scan("agent-1", adapter)
        assert result == []

    async def test_skips_known_pending_pairs(self) -> None:
        adapter = InMemoryAdapter()
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.01])
        mem_b = make_memory("Rate limit is 500", embedding=[0.99, 0.02])
        await adapter.store(mem_a)
        await adapter.store(mem_b)

        existing = ConflictRecord(
            agent_id="agent-1",
            memory_a_id=mem_a.memory_id,
            memory_b_id=mem_b.memory_id,
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            confidence=0.92,
        )
        await adapter.store_conflict(existing)

        det = make_detector([_CONTRADICTION])
        result = await det.scan("agent-1", adapter)
        assert result == []

    async def test_stores_new_conflicts_in_adapter(self) -> None:
        adapter = InMemoryAdapter()
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.01])
        mem_b = make_memory("Rate limit is 500", embedding=[0.99, 0.02])
        await adapter.store(mem_a)
        await adapter.store(mem_b)

        det = make_detector([_CONTRADICTION], cluster_threshold=0.80)
        new_records = await det.scan("agent-1", adapter)

        assert len(new_records) == 1
        stored = await adapter.list_conflicts("agent-1")
        assert len(stored) == 1
        assert stored[0].conflict_id == new_records[0].conflict_id

    async def test_returns_only_newly_detected_records(self) -> None:
        adapter = InMemoryAdapter()
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.01])
        mem_b = make_memory("Rate limit is 500", embedding=[0.99, 0.02])
        await adapter.store(mem_a)
        await adapter.store(mem_b)

        det = make_detector([_CONTRADICTION], cluster_threshold=0.80)
        result = await det.scan("agent-1", adapter)
        assert len(result) == 1
        assert result[0].conflict_type == ConflictType.DIRECT_CONTRADICTION

    async def test_uses_instance_cluster_threshold(self) -> None:
        adapter = InMemoryAdapter()
        # Embeddings with moderate similarity (~0.71) — below 0.99 threshold
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.01])
        mem_b = make_memory("Rate limit is 500", embedding=[0.5, 0.5])
        await adapter.store(mem_a)
        await adapter.store(mem_b)

        det = make_detector([_CONTRADICTION], cluster_threshold=0.99)
        result = await det.scan("agent-1", adapter)
        assert result == []

    async def test_unrelated_pairs_produce_no_records(self) -> None:
        adapter = InMemoryAdapter()
        mem_a = make_memory("Rate limit is 100", embedding=[1.0, 0.01])
        mem_b = make_memory("Rate limit is 500", embedding=[0.99, 0.02])
        await adapter.store(mem_a)
        await adapter.store(mem_b)

        det = make_detector([_UNRELATED], cluster_threshold=0.80)
        result = await det.scan("agent-1", adapter)
        assert result == []
        stored = await adapter.list_conflicts("agent-1")
        assert stored == []
