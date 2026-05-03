"""Contradiction detector — LLM-powered conflict classification for memory pairs.

Step 5.1 — ContradictionDetector core and storage contract.

The detection pipeline has two stages:

  Stage 1 (embedding): cosine similarity clustering surfaces candidate pairs.
    HealthScorer.contradiction_score() identifies pairs above
    cluster_threshold. This is cheap and runs in embedding space.

  Stage 2 (LLM): a classify_fn call classifies each candidate pair as one of:
    direct_contradiction | temporal_supersession | partial_conflict | unrelated.
    Only pairs with a conflict verdict AND confidence >= confidence_threshold
    produce a ConflictRecord.

The LLM call is abstracted behind a single async callable::

    LLMClassifyFn = Callable[[str], Awaitable[str]]

This makes the detector provider-agnostic. The default callable uses the
Anthropic SDK (requires the 'anthropic' extra), but any LLM that accepts a
prompt string and returns text works::

    # Default — uses ANTHROPIC_API_KEY from environment
    detector = ContradictionDetector()

    # Custom provider (OpenAI, Ollama, proxy, etc.)
    async def my_llm(prompt: str) -> str:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    detector = ContradictionDetector(llm_fn=my_llm)

Two entry points:

  detect(new_memory, candidate_memories)
    Pure classification — no adapter I/O. For store-time use in Step 5.2:
    the caller (Engram.store) supplies pre-filtered candidates and stores
    the returned ConflictRecords.

  scan(agent_id, adapter)
    Batch scan — fetches all memories, runs Stage 1 clustering, classifies
    candidate pairs concurrently (bounded by max_concurrency), stores new
    ConflictRecords, and skips pairs where a PENDING ConflictRecord already
    exists (or that were resolved and re-scanned intentionally).

Requires the Anthropic extra for the default LLM callable::

    pip install "engram[llm]"
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.core.constants import (
    CLUSTER_SIMILARITY_THRESHOLD,
    CONTRADICTION_CONFIDENCE_THRESHOLD,
    ConflictType,
    ResolutionStatus,
)
from engram.core.models import ConflictRecord

if TYPE_CHECKING:
    from engram.adapters.base import AbstractAdapter
    from engram.core.models import Memory

__all__ = ["ClassificationResult", "ContradictionDetector", "LLMClassifyFn"]

logger = logging.getLogger(__name__)

# Any async callable that takes a prompt string and returns the model's text response.
LLMClassifyFn = Callable[[str], Awaitable[str]]

# Current Anthropic model IDs — override via model= or llm_fn= for other providers.
_DEFAULT_MODEL = "claude-sonnet-4-5"

_CLASSIFY_PROMPT = """\
You are analyzing two memories stored by an AI agent for potential contradictions.

Memory A (id={id_a}):
{text_a}

Memory B (id={id_b}):
{text_b}

Classify the relationship between Memory A and Memory B. Choose exactly one label:

- "direct_contradiction": They assert conflicting facts about the same subject \
(e.g. different prices, opposite policies, incompatible states).
- "temporal_supersession": One clearly supersedes the other — a newer update \
of the same fact; the older is now stale.
- "partial_conflict": They overlap topically but differ on specific details \
without being fully contradictory.
- "unrelated": They are about different subjects with no meaningful conflict. \
Note: two memories that agree on the same topic are also "unrelated".

Respond with valid JSON only — no explanation outside the JSON object. \
The summary must be 15 words or fewer:
{{"verdict": "<label>", "confidence": <float 0.0-1.0>, \
"summary": "<≤15-word explanation>"}}
"""


@dataclass(frozen=True)
class ClassificationResult:
    """Output of classifying a single memory pair.

    Produced by ContradictionDetector.classify_pair(). Use verdict and
    confidence to decide whether to emit a ConflictRecord.
    """

    memory_a_id: str
    memory_b_id: str
    verdict: str  # "direct_contradiction" | "temporal_supersession" | "partial_conflict" | "unrelated"
    confidence: float  # 0.0–1.0; higher = more certain of the verdict
    summary: str  # ≤15-word explanation of the relationship


def _build_anthropic_fn(model: str) -> LLMClassifyFn:
    """Build the default Anthropic-backed classify function.

    Lazy-imports anthropic so the package is only required when no custom
    llm_fn is provided. Raises ImportError with a helpful message if missing.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "ContradictionDetector requires the 'anthropic' package when no "
            "llm_fn is provided. Install it with: pip install \"engram[llm]\" "
            "or supply your own: ContradictionDetector(llm_fn=my_async_fn)"
        ) from exc

    client = anthropic.AsyncAnthropic()

    async def _classify(prompt: str) -> str:
        message = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    return _classify


class ContradictionDetector:
    """LLM-powered contradiction detector for memory pairs.

    Uses cosine similarity clustering (Stage 1) to pre-filter candidate pairs,
    then an LLM call (Stage 2) to classify each pair. Only pairs with a
    conflict verdict and confidence >= confidence_threshold produce a
    ConflictRecord.

    The LLM call is abstracted behind an async callable so any provider works.
    The default uses the Anthropic SDK; pass llm_fn= to use any other model.

    Args:
        llm_fn:               Async callable ``(prompt: str) -> str`` that
                              returns the model's raw text response. If None,
                              builds a default Anthropic-backed callable using
                              ANTHROPIC_API_KEY from the environment.
        model:                Claude model ID for the default Anthropic callable.
                              Ignored when llm_fn is provided.
        cluster_threshold:    Cosine similarity cutoff for Stage 1 filtering.
                              Pairs below this threshold skip the LLM entirely.
        confidence_threshold: Minimum LLM confidence to emit a ConflictRecord.
        max_concurrency:      Maximum parallel LLM calls during scan().
                              Prevents rate-limit exhaustion on large batches.
    """

    _VERDICT_TO_TYPE: dict[str, ConflictType] = {
        "direct_contradiction": ConflictType.DIRECT_CONTRADICTION,
        "temporal_supersession": ConflictType.TEMPORAL_SUPERSESSION,
        "partial_conflict": ConflictType.PARTIAL_CONFLICT,
    }

    def __init__(
        self,
        *,
        llm_fn: LLMClassifyFn | None = None,
        model: str = _DEFAULT_MODEL,
        cluster_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
        confidence_threshold: float = CONTRADICTION_CONFIDENCE_THRESHOLD,
        max_concurrency: int = 5,
    ) -> None:
        self._llm_fn: LLMClassifyFn = llm_fn or _build_anthropic_fn(model)
        self._model = model
        self._cluster_threshold = cluster_threshold
        self._confidence_threshold = confidence_threshold
        self._max_concurrency = max_concurrency

    @property
    def cluster_threshold(self) -> float:
        """Cosine similarity cutoff used for Stage 1 candidate filtering."""
        return self._cluster_threshold

    async def classify_pair(
        self,
        memory_a: Memory,
        memory_b: Memory,
    ) -> ClassificationResult:
        """Classify the relationship between two memories using the LLM.

        Returns a ClassificationResult for every pair regardless of verdict.
        On LLM or network failure, logs a warning and returns verdict="unrelated"
        with confidence=0.0 so the pipeline never crashes.
        """
        prompt = _CLASSIFY_PROMPT.format(
            id_a=memory_a.memory_id,
            text_a=memory_a.text,
            id_b=memory_b.memory_id,
            text_b=memory_b.text,
        )
        try:
            raw = await self._llm_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "classify_pair: LLM call failed for pair (%s, %s): %s — defaulting to unrelated",
                memory_a.memory_id,
                memory_b.memory_id,
                exc,
            )
            raw = '{"verdict": "unrelated", "confidence": 0.0, "summary": ""}'

        return self._parse_classification(raw, memory_a.memory_id, memory_b.memory_id)

    def _parse_classification(
        self,
        raw: str,
        memory_a_id: str,
        memory_b_id: str,
    ) -> ClassificationResult:
        """Parse the LLM text response into a ClassificationResult.

        Falls back to verdict="unrelated" / confidence=0.0 on any parse error.
        """
        try:
            data = json.loads(raw)
            verdict = str(data.get("verdict", "unrelated")).lower().strip()
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            summary = str(data.get("summary", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "classify_pair: failed to parse LLM response for pair (%s, %s): %s — raw=%r",
                memory_a_id,
                memory_b_id,
                exc,
                raw,
            )
            verdict = "unrelated"
            confidence = 0.0
            summary = ""

        return ClassificationResult(
            memory_a_id=memory_a_id,
            memory_b_id=memory_b_id,
            verdict=verdict,
            confidence=confidence,
            summary=summary,
        )

    def _to_conflict_record(
        self,
        result: ClassificationResult,
        agent_id: str,
    ) -> ConflictRecord | None:
        """Convert a ClassificationResult to a ConflictRecord if it qualifies.

        Returns None when verdict is "unrelated" or confidence is below threshold.
        The LLM summary is preserved in ConflictRecord.description for use
        by Step 5.2's search-time enrichment (SearchResult.conflict_summary).
        """
        conflict_type = self._VERDICT_TO_TYPE.get(result.verdict)
        if conflict_type is None:
            return None
        if result.confidence < self._confidence_threshold:
            logger.debug(
                "classify_pair: pair (%s, %s) verdict=%r confidence=%.2f "
                "below threshold=%.2f — skipping",
                result.memory_a_id,
                result.memory_b_id,
                result.verdict,
                result.confidence,
                self._confidence_threshold,
            )
            return None

        return ConflictRecord(
            agent_id=agent_id,
            memory_a_id=result.memory_a_id,
            memory_b_id=result.memory_b_id,
            conflict_type=conflict_type,
            confidence=result.confidence,
            description=result.summary or None,
            resolution_status=ResolutionStatus.PENDING,
        )

    async def detect(
        self,
        new_memory: Memory,
        candidate_memories: list[Memory],
    ) -> list[ConflictRecord]:
        """Classify new_memory against a pre-filtered list of candidate memories.

        Pure classification — no adapter I/O. Designed for store-time use in
        Step 5.2: the caller (Engram.store) supplies candidates already filtered
        by cosine similarity and stores the returned ConflictRecords.

        Skips candidates that have no embedding, or that are the same memory
        as new_memory. Returns a ConflictRecord for each pair where the LLM
        confirms a conflict above the confidence threshold.

        LLM calls run concurrently bounded by max_concurrency.
        """
        if not new_memory.embedding:
            logger.debug(
                "detect: new_memory %s has no embedding — skipping detection",
                new_memory.memory_id,
            )
            return []

        valid_candidates = [
            c
            for c in candidate_memories
            if c.memory_id != new_memory.memory_id and c.embedding is not None
        ]
        if not valid_candidates:
            return []

        sem = asyncio.Semaphore(self._max_concurrency)

        async def _classify_one(candidate: Memory) -> ConflictRecord | None:
            async with sem:
                result = await self.classify_pair(new_memory, candidate)
                record = self._to_conflict_record(result, new_memory.agent_id)
                if record is not None:
                    logger.info(
                        "detect: conflict — agent=%s pair=(%s, %s) type=%s confidence=%.2f",
                        new_memory.agent_id,
                        result.memory_a_id,
                        result.memory_b_id,
                        result.verdict,
                        result.confidence,
                    )
                return record

        results = await asyncio.gather(*[_classify_one(c) for c in valid_candidates])
        records = [r for r in results if r is not None]

        logger.debug(
            "detect: agent=%s new_memory=%s candidates=%d conflicts_found=%d",
            new_memory.agent_id,
            new_memory.memory_id,
            len(valid_candidates),
            len(records),
        )
        return records

    async def scan(
        self,
        agent_id: str,
        adapter: AbstractAdapter,
    ) -> list[ConflictRecord]:
        """Batch contradiction scan across all ACTIVE memories for an agent.

        Fetches all memories, runs embedding clustering (Stage 1) respecting
        self._cluster_threshold, LLM-classifies candidate pairs concurrently
        (bounded by max_concurrency), stores new ConflictRecords, and skips
        pairs where a PENDING ConflictRecord already exists.

        Note: pairs with RESOLVED status are re-scanned — if the underlying
        memories changed, a previously resolved conflict may have re-emerged.

        Returns the list of newly emitted and stored ConflictRecords.
        Raises NotImplementedError if the adapter does not implement conflict storage.
        """
        from engram.core.health import HealthScorer

        all_memories = await adapter.list_all(agent_id)
        active = [m for m in all_memories if m.is_usable()]

        if len(active) < 2:
            logger.debug(
                "scan: fewer than 2 active memories for agent %s — nothing to scan", agent_id
            )
            return []

        scorer = HealthScorer()
        scorer.CONTRADICTION_CLUSTER_THRESHOLD = self._cluster_threshold
        _, candidate_pairs, _ = scorer.contradiction_score(active)

        if not candidate_pairs:
            logger.debug("scan: no candidate pairs above threshold for agent %s", agent_id)
            return []

        existing = await adapter.list_conflicts(agent_id, status=ResolutionStatus.PENDING)
        known_pairs: set[frozenset[str]] = {
            frozenset({c.memory_a_id, c.memory_b_id}) for c in existing
        }

        memory_index: dict[str, Memory] = {m.memory_id: m for m in active}
        new_pairs = [
            (id_a, id_b)
            for id_a, id_b in candidate_pairs
            if frozenset({id_a, id_b}) not in known_pairs
            and id_a in memory_index
            and id_b in memory_index
        ]

        if not new_pairs:
            logger.debug(
                "scan: all %d candidate pairs already known for agent %s",
                len(candidate_pairs),
                agent_id,
            )
            return []

        sem = asyncio.Semaphore(self._max_concurrency)

        async def _classify_one(id_a: str, id_b: str) -> ConflictRecord | None:
            async with sem:
                mem_a = memory_index[id_a]
                mem_b = memory_index[id_b]
                result = await self.classify_pair(mem_a, mem_b)
                return self._to_conflict_record(result, agent_id)

        raw_results = await asyncio.gather(
            *[_classify_one(a, b) for a, b in new_pairs],
            return_exceptions=True,
        )

        new_records: list[ConflictRecord] = []
        failures = 0
        for pair, outcome in zip(new_pairs, raw_results, strict=True):
            if isinstance(outcome, BaseException):
                failures += 1
                logger.warning(
                    "scan: classification failed for pair %s: %s", pair, outcome
                )
                continue
            if outcome is not None:
                await adapter.store_conflict(outcome)
                new_records.append(outcome)
                logger.info(
                    "scan: new conflict — agent=%s pair=(%s, %s) type=%s confidence=%.2f",
                    agent_id,
                    outcome.memory_a_id,
                    outcome.memory_b_id,
                    outcome.conflict_type,
                    outcome.confidence,
                )

        logger.debug(
            "scan: agent=%s candidate_pairs=%d skipped_known=%d classified=%d "
            "new_conflicts=%d failures=%d",
            agent_id,
            len(candidate_pairs),
            len(candidate_pairs) - len(new_pairs),
            len(new_pairs),
            len(new_records),
            failures,
        )
        return new_records
