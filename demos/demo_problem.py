"""Scene 1 — The problem: naive storage with no conflict signal.

Run:
    python demos/demo_problem.py

Shows two contradicting memories stored and retrieved with equal authority.
No flag. No signal. Your LLM has to guess.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import (
    BOLD, DIM, GREEN, RESET, YELLOW,
    EMBED_QUERY, EMBED_REFUND_14, EMBED_REFUND_30,
    pause, section,
)

from memnotary import InMemoryAdapter, Memory


async def main() -> None:
    adapter = InMemoryAdapter()

    section("Naive storage — no MemNotary")

    print("Storing memories...")
    pause(0.4)

    m1 = Memory(agent_id="bot", text="Refund policy is 30 days",
                embedding=EMBED_REFUND_30)
    await adapter.store(m1)
    print(f'  {GREEN}✓{RESET} stored  "Refund policy is 30 days"')
    pause(0.8)

    m2 = Memory(agent_id="bot", text="Refund policy changed to 14 days",
                embedding=EMBED_REFUND_14)
    await adapter.store(m2)
    print(f'  {GREEN}✓{RESET} stored  "Refund policy changed to 14 days"')
    pause(1.0)

    print(f'\nSearching: {DIM}"what is our refund policy?"{RESET}')
    pause(0.8)

    results = await adapter.search("bot", EMBED_QUERY, top_k=3)

    print()
    for i, r in enumerate(results, 1):
        print(f'  {i}. "{r.memory.text}"')
        print(f"     similarity: {r.score:.2f}")
        pause(0.6)

    print()
    pause(0.4)
    print(f"  {YELLOW}⚠  Both answers returned.{RESET}")
    print(f"  {YELLOW}⚠  No conflict flag. No signal.{RESET}")
    print(f"  {DIM}     Your LLM sees both. It picks one. Silently.{RESET}")
    print()
    pause(1.5)


if __name__ == "__main__":
    asyncio.run(main())
