"""
Customer support bot with six months of accumulated memories.

This is the scenario that motivated Engram.

A support bot ingests company policies and product facts as they're written.
Over six months the team updates pricing, revises policies, and fixes mistakes.
Each update gets stored — but nothing removes the old version. By month six,
the bot has three live contradictions it has no way to know about.

Engram detects them, scores the memory health, and resolves them in one pass.

Run:
    pip install engram
    python examples/customer_support_bot.py
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta

from engram import (
    Consolidator,
    ContradictionDetector,
    Engram,
    InMemoryAdapter,
    Memory,
    MemoryStatus,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


async def my_llm(prompt: str) -> str:
    """
    Stub — returns temporal_supersession so this runs without an API key.
    Replace with your real LLM for production use.
    """
    return json.dumps({
        "verdict": "temporal_supersession",
        "confidence": 0.91,
        "summary": "Newer memory supersedes the outdated one.",
    })


async def my_consolidate_llm(prompt: str) -> str:
    return json.dumps({
        "action": "flag",
        "confidence": 0.80,
        "reasoning": "Requires human review.",
    })


# Topic-specific unit vectors so only same-topic memories are flagged as candidates.
# In production, a real embedding model produces these automatically.
_EMBED: dict[str, list[float]] = {
    "shipping": [1.0, 0.0, 0.0, 0.0],
    "returns":  [0.0, 1.0, 0.0, 0.0],
    "pricing":  [0.0, 0.0, 1.0, 0.0],
    "general":  [0.0, 0.0, 0.0, 1.0],
}


def months_ago(n: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=30 * n)


# ---------------------------------------------------------------------------
# Six months of documentation
# ---------------------------------------------------------------------------

# (text, topic, months_ago)
MEMORIES = [
    # Month 6 — initial documentation
    ("Standard shipping takes 5–7 business days.",              "shipping", 6),
    ("We offer a 30-day return window on all products.",        "returns",  6),
    ("The Pro plan costs $49 per month.",                       "pricing",  6),

    # Month 4 — first round of policy updates
    ("Shipping times improved. Standard is now 3–5 days.",     "shipping", 4),
    ("Our return window has been extended to 60 days.",         "returns",  4),

    # Month 2 — carrier issue + pricing change
    ("Due to carrier delays, shipping is back to 5–7 days.",   "shipping", 2),
    ("The Pro plan is now $39 per month.",                      "pricing",  2),

    # Month 1 — permanent pricing discount
    ("Permanent discount applied. Pro plan is now $29/month.", "pricing",  1),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    eng = Engram(
        InMemoryAdapter(),
        detector=ContradictionDetector(llm_fn=my_llm),
        consolidator=Consolidator(llm_fn=my_consolidate_llm),
    )

    async with eng:

        # --- Ingest ---

        print("Six months of support documentation:\n")
        for text, topic, months in MEMORIES:
            mem = Memory(
                agent_id="support-bot",
                text=text,
                embedding=_EMBED[topic],
                created_at=months_ago(months),
            )
            await eng.store(mem)
            print(f"  [{months}mo ago]  {text}")
        print()

        # --- Health before ---

        before = await eng.health("support-bot")
        print("Memory health right now:\n")
        print(f"  Memories in store:   {before.total_memories}")
        print(f"  Detected conflicts:  {before.conflict_count}")
        print(f"  Contradiction score: {before.contradiction_score:.2f}  ← this is the problem")
        print(f"  Overall score:       {before.score:.2f}")
        print()

        # --- Show the pending conflicts ---

        conflicts = await eng.pending_review("support-bot")
        print(f"Conflicts queued for resolution ({len(conflicts)} total):\n")
        for c in conflicts:
            print(f"  ⚠  [{c.conflict_type.value}]  {c.description}")
        print()

        # --- Customer asks a question into this mess ---

        print("Customer asks: 'How much does the Pro plan cost?'\n")
        results = await eng.search("support-bot", _EMBED["pricing"], top_k=5, score_threshold=0.5)
        for r in results:
            flag = "⚠  conflict" if r.conflict_flag else "✓  ok      "
            print(f"  [{flag}]  {r.memory.text}")
        print()
        print("  The bot has three different prices in memory.")
        print("  Without Engram, it picks one at random.\n")

        # --- Consolidate ---

        print("Running consolidation...\n")
        plan = await eng.consolidate("support-bot")
        if plan:
            print(f"  Actions taken: {len(plan.actions)}")
            print(f"  Superseded:    {len(plan.would_supersede)}  (resolved automatically)")
            print(f"  Flagged:       {len(plan.would_flag_for_review)}  (sent to review queue)")
        print()

        # --- Health after ---

        after = await eng.health("support-bot")
        print("Memory health after consolidation:\n")
        print(f"  Memories in store:   {after.total_memories}  (one per topic — no duplicates)")
        print(f"  Pending conflicts:   {after.conflict_count}")
        print(f"  Contradiction score: {after.contradiction_score:.2f}")
        print(f"  Overall score:       {after.score:.2f}  "
              f"(was {before.score:.2f}, +{after.score - before.score:.2f})")
        print()

        # --- What does the bot say now? ---

        print("Same question after consolidation:\n")
        results = await eng.search("support-bot", _EMBED["pricing"], top_k=5, score_threshold=0.5)
        active = [r for r in results if r.memory.status == MemoryStatus.ACTIVE]
        for r in active:
            print(f"  ✓  {r.memory.text}")
        print()
        print("  One answer. No conflicts. No guessing.")


if __name__ == "__main__":
    asyncio.run(main())
