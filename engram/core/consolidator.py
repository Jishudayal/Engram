"""Consolidation engine — plan memory consolidation actions from detected conflicts.

Step 6.1 — Consolidator.plan(): LLM-driven action planning over PENDING ConflictRecords.

The consolidation pipeline has two phases:

  Phase 1 (plan — this file): For each PENDING ConflictRecord, decide what to do:
    - TEMPORAL_SUPERSESSION → SUPERSEDE heuristically; the older memory loses.
      No LLM call: the ContradictionDetector already classified this with confidence.
    - DIRECT_CONTRADICTION / PARTIAL_CONFLICT → LLM prompt returns action + confidence.
      "merge"  — synthesise both memories into a single reconciled fact.
      "flag"   — keep both; surface for human review (use when ambiguous).
    The output is a ConsolidationPlan: an ordered, immutable tuple of ConsolidationActions.

  Phase 2 (execute — Step 6.3): Apply each action against the adapter.

The LLM call is abstracted behind a single async callable::

    LLMConsolidateFn = Callable[[str], Awaitable[str]]

This makes the Consolidator provider-agnostic. The default callable uses the
Anthropic SDK (requires the 'anthropic' extra), but any LLM that accepts a
prompt string and returns text works::

    # Default — uses ANTHROPIC_API_KEY from environment
    consolidator = Consolidator()

    # Custom provider (OpenAI, Ollama, proxy, etc.)
    async def my_llm(prompt: str) -> str:
        ...

    consolidator = Consolidator(llm_fn=my_llm)

Requires the Anthropic extra for the default LLM callable::

    pip install "engram[llm]"
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.core.constants import (
    ActionType,
    ConflictType,
    ConsolidationTier,
    ResolutionStatus,
)
from engram.core.models import ConsolidationAction, ConsolidationPlan

if TYPE_CHECKING:
    from engram.adapters.base import AbstractAdapter
    from engram.core.models import ConflictRecord, Memory

__all__ = ["Consolidator", "LLMConsolidateFn", "PlanningResult"]

logger = logging.getLogger(__name__)

# Any async callable that takes a prompt string and returns the model's text response.
LLMConsolidateFn = Callable[[str], Awaitable[str]]

# Current Anthropic model IDs — override via model= or llm_fn= for other providers.
_DEFAULT_MODEL = "claude-sonnet-4-5"

_CONSOLIDATE_PROMPT = """\
You are deciding how to consolidate two conflicting AI agent memories.

Conflict type: {conflict_type}
Conflict summary: {description}

Memory A (id={id_a}):
{text_a}

Memory B (id={id_b}):
{text_b}

Choose exactly one action:
- "merge": The two memories can be synthesised into a single, reconciled fact. \
Use when the conflict is resolvable and a clear combined statement is possible.
- "flag": Keep both memories; surface for human review. \
Use when the conflict is genuinely ambiguous or requires domain knowledge to resolve.

Respond with valid JSON only — no explanation outside the JSON object. \
Reasoning must be 20 words or fewer:
{{"action": "<merge|flag>", "confidence": <float 0.0-1.0>, \
"reasoning": "<≤20-word explanation>"}}
"""


@dataclass(frozen=True)
class PlanningResult:
    """Output of classifying a single ConflictRecord for a consolidation action.

    Produced by Consolidator._parse_planning_result(). Intermediate representation
    that _to_action() maps to a ConsolidationAction.
    """

    conflict_id: str
    action_type: str   # "merge" | "flag"
    confidence: float  # 0.0–1.0; higher = more certain of the chosen action
    reasoning: str     # ≤20-word explanation; may be empty on parse failure


def _build_anthropic_fn(model: str) -> LLMConsolidateFn:
    """Build the default Anthropic-backed consolidation function.

    Lazy-imports anthropic so the package is only required when no custom
    llm_fn is provided. Raises ImportError with a helpful message if missing.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Consolidator requires the 'anthropic' package when no "
            "llm_fn is provided. Install it with: pip install \"engram[llm]\" "
            "or supply your own: Consolidator(llm_fn=my_async_fn)"
        ) from exc

    client = anthropic.AsyncAnthropic()

    async def _consolidate(prompt: str) -> str:
        message = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    return _consolidate


class Consolidator:
    """LLM-powered consolidation planner for PENDING memory conflicts.

    Reads PENDING ConflictRecords from the adapter and produces a ConsolidationPlan
    — an ordered tuple of ConsolidationActions describing what to do with each
    conflict. Execution is handled separately in Step 6.3.

    Routing logic:
      - TEMPORAL_SUPERSESSION  → SUPERSEDE (heuristic, no LLM; older memory loses).
      - DIRECT_CONTRADICTION   → LLM decides: merge or flag.
      - PARTIAL_CONFLICT       → LLM decides: merge or flag.

    The LLM decision maps to a ConsolidationTier via ConsolidationTier.from_confidence():
      >= AUTO_MERGE_THRESHOLD  → AUTO_MERGE  (act automatically in execute)
      >= AUTO_FLAG_THRESHOLD   → AUTO_FLAG   (act automatically in execute)
      < AUTO_FLAG_THRESHOLD    → HUMAN_REVIEW (surfaced for manual decision)

    Args:
        llm_fn:          Async callable ``(prompt: str) -> str`` that returns the
                         model's raw text response. If None, builds a default
                         Anthropic-backed callable using ANTHROPIC_API_KEY from env.
        model:           Claude model ID for the default Anthropic callable.
                         Ignored when llm_fn is provided.
        max_concurrency: Maximum parallel LLM calls during plan().
                         Prevents rate-limit exhaustion on large conflict sets.
    """

    _ACTION_TYPE_MAP: dict[str, ActionType] = {
        "merge": ActionType.MERGE,
        "flag": ActionType.FLAG,
    }

    def __init__(
        self,
        *,
        llm_fn: LLMConsolidateFn | None = None,
        model: str = _DEFAULT_MODEL,
        max_concurrency: int = 5,
    ) -> None:
        self._llm_fn: LLMConsolidateFn = llm_fn or _build_anthropic_fn(model)
        self._model = model
        self._max_concurrency = max_concurrency

    async def plan(
        self,
        agent_id: str,
        adapter: AbstractAdapter,
        *,
        dry_run: bool = False,
    ) -> ConsolidationPlan | None:
        """Build a ConsolidationPlan from all PENDING conflicts for an agent.

        Flow:
          1. Fetch all PENDING ConflictRecords.
          2. Bulk-fetch the involved memories via adapter.fetch_batch().
          3. Route each conflict: heuristic for TEMPORAL_SUPERSESSION, LLM for
             DIRECT_CONTRADICTION / PARTIAL_CONFLICT.
          4. Collect actionable ConsolidationActions and wrap in a ConsolidationPlan.

        Returns None when there are no PENDING conflicts or no actionable actions
        (i.e., all conflicts were skipped due to missing memories or unknown verdicts).
        Sets is_dry_run=True on the plan when dry_run=True — execute() respects this
        flag and skips all adapter writes.

        LLM calls run concurrently bounded by max_concurrency. Failures are logged
        and skipped; they do not abort the entire plan.
        """
        pending = await adapter.list_conflicts(agent_id, status=ResolutionStatus.PENDING)
        if not pending:
            logger.debug("plan: no PENDING conflicts for agent %s — nothing to plan", agent_id)
            return None

        # Bulk-fetch all memories referenced by the conflict records.
        all_memory_ids = list(
            {mid for c in pending for mid in (c.memory_a_id, c.memory_b_id)}
        )
        memory_index = await adapter.fetch_batch(agent_id, all_memory_ids)

        sem = asyncio.Semaphore(self._max_concurrency)

        async def _plan_one(conflict: ConflictRecord) -> ConsolidationAction | None:
            mem_a = memory_index.get(conflict.memory_a_id)
            mem_b = memory_index.get(conflict.memory_b_id)
            if mem_a is None or mem_b is None:
                logger.warning(
                    "plan: skipping conflict %s — memory not found "
                    "(a=%s present=%s, b=%s present=%s)",
                    conflict.conflict_id,
                    conflict.memory_a_id,
                    mem_a is not None,
                    conflict.memory_b_id,
                    mem_b is not None,
                )
                return None
            return await self._classify_conflict(conflict, mem_a, mem_b, agent_id, sem)

        raw_results = await asyncio.gather(
            *[_plan_one(c) for c in pending],
            return_exceptions=True,
        )

        actions: list[ConsolidationAction] = []
        for conflict, outcome in zip(pending, raw_results, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "plan: classification failed for conflict %s: %s",
                    conflict.conflict_id,
                    outcome,
                )
                continue
            if outcome is not None:
                actions.append(outcome)

        if not actions:
            logger.debug(
                "plan: no actionable conflicts for agent %s (%d PENDING, all skipped)",
                agent_id,
                len(pending),
            )
            return None

        plan = ConsolidationPlan(
            agent_id=agent_id,
            actions=tuple(actions),
            is_dry_run=dry_run,
        )
        action_counts = Counter(action.action_type.value for action in actions)
        logger.info(
            "plan: agent=%s pending=%d actions=%d by_type=%s dry_run=%s",
            agent_id,
            len(pending),
            len(actions),
            dict(action_counts),
            dry_run,
        )
        return plan

    # ------------------------------------------------------------------
    # Internal — routing
    # ------------------------------------------------------------------

    async def _classify_conflict(
        self,
        conflict: ConflictRecord,
        mem_a: Memory,
        mem_b: Memory,
        agent_id: str,
        sem: asyncio.Semaphore,
    ) -> ConsolidationAction | None:
        """Route a single conflict to the right classification strategy."""
        if conflict.conflict_type == ConflictType.TEMPORAL_SUPERSESSION:
            return self._temporal_supersession_action(conflict, mem_a, mem_b, agent_id)

        # DIRECT_CONTRADICTION and PARTIAL_CONFLICT both require an LLM decision.
        async with sem:
            result = await self._llm_classify(conflict, mem_a, mem_b)
        return self._to_action(conflict, result, agent_id)

    def _temporal_supersession_action(
        self,
        conflict: ConflictRecord,
        mem_a: Memory,
        mem_b: Memory,
        agent_id: str,
    ) -> ConsolidationAction:
        """Build a SUPERSEDE action without an LLM call.

        The older memory (lower created_at) is the superseded source; the newer
        one is the keeper. source_memory_ids=(superseded, keeper) encodes this
        for the executor in Step 6.3.

        Tier is derived from the detector's original classification confidence:
        a moderately-confident supersession should preserve that uncertainty for
        executor review instead of becoming automatically executable.
        """
        if mem_a.created_at == mem_b.created_at:
            return ConsolidationAction(
                agent_id=agent_id,
                action_type=ActionType.FLAG,
                tier=ConsolidationTier.HUMAN_REVIEW,
                confidence=conflict.confidence,
                source_memory_ids=(mem_a.memory_id, mem_b.memory_id),
                reasoning="Tied created_at; manual review required.",
            )

        if mem_a.created_at < mem_b.created_at:
            superseded_id, keeper_id = mem_a.memory_id, mem_b.memory_id
        else:
            superseded_id, keeper_id = mem_b.memory_id, mem_a.memory_id

        return ConsolidationAction(
            agent_id=agent_id,
            action_type=ActionType.SUPERSEDE,
            tier=ConsolidationTier.from_confidence(conflict.confidence),
            confidence=conflict.confidence,
            source_memory_ids=(superseded_id, keeper_id),
            reasoning=conflict.description or "Newer memory supersedes older conflicting fact.",
        )

    async def _llm_classify(
        self,
        conflict: ConflictRecord,
        mem_a: Memory,
        mem_b: Memory,
    ) -> PlanningResult:
        """Ask the LLM to choose an action for a DIRECT_CONTRADICTION / PARTIAL_CONFLICT."""
        prompt = _CONSOLIDATE_PROMPT.format(
            conflict_type=conflict.conflict_type.value,
            description=conflict.description or "No summary available.",
            id_a=mem_a.memory_id,
            text_a=mem_a.text,
            id_b=mem_b.memory_id,
            text_b=mem_b.text,
        )
        try:
            raw = await self._llm_fn(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_llm_classify: LLM call failed for conflict %s: %s — defaulting to flag",
                conflict.conflict_id,
                exc,
            )
            raw = '{"action": "flag", "confidence": 0.0, "reasoning": ""}'

        return self._parse_planning_result(raw, conflict.conflict_id)

    def _parse_planning_result(self, raw: str, conflict_id: str) -> PlanningResult:
        """Parse the LLM text response into a PlanningResult.

        Falls back to action="flag" / confidence=0.0 on any parse error.
        """
        try:
            data = json.loads(raw)
            action = str(data.get("action", "flag")).lower().strip()
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(data.get("reasoning", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "_parse_planning_result: failed to parse LLM response for conflict %s: %s — raw=%r",
                conflict_id,
                exc,
                raw,
            )
            action = "flag"
            confidence = 0.0
            reasoning = ""

        return PlanningResult(
            conflict_id=conflict_id,
            action_type=action,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _to_action(
        self,
        conflict: ConflictRecord,
        result: PlanningResult,
        agent_id: str,
    ) -> ConsolidationAction | None:
        """Map a PlanningResult to a ConsolidationAction.

        Returns None for unknown action_type strings (logged as warning).
        For MERGE/FLAG actions, source_memory_ids order is not semantically meaningful.
        """
        action_type = self._ACTION_TYPE_MAP.get(result.action_type)
        if action_type is None:
            logger.warning(
                "_to_action: unknown action_type %r for conflict %s — skipping",
                result.action_type,
                conflict.conflict_id,
            )
            return None

        return ConsolidationAction(
            agent_id=agent_id,
            action_type=action_type,
            tier=ConsolidationTier.from_confidence(result.confidence),
            confidence=result.confidence,
            source_memory_ids=(conflict.memory_a_id, conflict.memory_b_id),
            reasoning=result.reasoning or None,
        )
