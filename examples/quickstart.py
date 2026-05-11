"""
Memnotary quickstart.

Scenario: A support bot stores company policies over time. The refund policy
changes, but both versions end up in memory. Without Memnotary, the bot gives
inconsistent answers. With Memnotary, the contradiction is caught and resolved.

Run:
    pip install memnotary
    python examples/quickstart.py

No API key needed — this example uses stub LLM functions so you can see
the full loop without any setup. Swap them out for your real LLM to use
Memnotary in production.
"""

import asyncio
import json

from memnotary import (
    Consolidator,
    ContradictionDetector,
    InMemoryAdapter,
    Memnotary,
    Memory,
)

# ---------------------------------------------------------------------------
# Stubs — replace these with your real LLM and embedding model
# ---------------------------------------------------------------------------


async def my_llm(prompt: str) -> str:
    """
    In production: call OpenAI, Anthropic, Gemini, Ollama — anything.
    The signature is just: async (prompt: str) -> str.
    """
    return json.dumps(
        {
            "verdict": "temporal_supersession",
            "confidence": 0.93,
            "summary": "Newer 14-day policy supersedes the outdated 30-day policy.",
        }
    )


async def my_consolidate_llm(prompt: str) -> str:
    """Only called for DIRECT_CONTRADICTION / PARTIAL_CONFLICT. Not needed here."""
    return json.dumps(
        {
            "action": "flag",
            "confidence": 0.85,
            "reasoning": "Ambiguous conflict — flag for human review.",
        }
    )


def embed(text: str) -> list[float]:
    """
    In production: use sentence-transformers, OpenAI embeddings, etc.
    Here we return the same unit vector so any two memories score as similar
    (cosine = 1.0) and pass the contradiction detector's Stage 1 filter.
    """
    _ = text
    return [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    eng = Memnotary(
        InMemoryAdapter(),
        detector=ContradictionDetector(llm_fn=my_llm),
        consolidator=Consolidator(llm_fn=my_consolidate_llm),
    )

    async with eng:
        # --- 1. Store what the bot learned over time ---

        print("Storing what the bot learned...\n")

        await eng.store(
            Memory(
                agent_id="support-bot",
                text="Our refund policy is 30 days from the date of purchase.",
                embedding=embed("refund policy 30 days"),
            )
        )
        print("  [Jan]  Stored: 'Our refund policy is 30 days from the date of purchase.'")

        await eng.store(
            Memory(
                agent_id="support-bot",
                text="As of March 1st, the refund window is now 14 days.",
                embedding=embed("refund policy 14 days"),
            )
        )
        print("  [Mar]  Stored: 'As of March 1st, the refund window is now 14 days.'")
        print("         └─ Memnotary detected a conflict on this store.\n")

        # --- 2. See what the bot sees ---

        print("Searching 'refund policy'...\n")
        results = await eng.search("support-bot", embed("refund policy"), top_k=5)

        for r in results:
            flag = "⚠  conflict" if r.conflict_flag else "✓  ok      "
            print(f"  [{flag}]  {r.memory.text}")
        print()
        # Both are flagged. When conflicting results are returned together, Memnotary also
        # sets recommended=False on the lower-ranked result so you know which to suppress.

        # --- 3. Fix it ---

        print("Running consolidation...\n")
        plan = await eng.consolidate("support-bot")
        if plan:
            for action in plan.actions:
                print(
                    f"  {action.action_type.value.upper()}: older memory superseded by "
                    f"{action.result_memory_id[:8]}…"
                )
        print()

        # --- 4. Check the health score ---

        health = await eng.health("support-bot")
        print("Memory health after consolidation:\n")
        print(f"  Active memories:     {health.total_memories}")
        print(f"  Pending conflicts:   {health.conflict_count}")
        print(f"  Contradiction score: {health.contradiction_score:.2f}  (lower is better)")
        print(f"  Overall score:       {health.score:.2f}  (higher is better)")
        print()
        print("The bot will now consistently answer with the current 14-day policy.")


if __name__ == "__main__":
    asyncio.run(main())
