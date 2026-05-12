"""Scene 2 — The fix: same scenario with MemNotary.

Run:
    python demos/demo_memnotary.py

Shows contradiction detected on store, and enriched search results with
recommended / stale signals. This is the GIF hero moment.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import (
    BOLD, DIM, GREEN, RESET, YELLOW,
    EMBED_QUERY, EMBED_REFUND_14, EMBED_REFUND_30,
    mock_llm, pause, section,
)

from memnotary import ContradictionDetector, InMemoryAdapter, Memory, Memnotary
from memnotary.core.constants import ResolutionStatus


async def main() -> None:
    adapter  = InMemoryAdapter()
    detector = ContradictionDetector(llm_fn=mock_llm)
    mn       = Memnotary(adapter, detector=detector)

    section("Same scenario — with MemNotary")

    print("Storing memories...")
    pause(0.4)

    m1 = Memory(agent_id="bot", text="Refund policy is 30 days",
                embedding=EMBED_REFUND_30)
    await mn.store(m1)
    print(f'  {GREEN}✓{RESET} stored  "Refund policy is 30 days"')
    pause(0.8)

    m2 = Memory(agent_id="bot", text="Refund policy changed to 14 days",
                embedding=EMBED_REFUND_14)
    await mn.store(m2)

    # Show what the detector found
    conflicts = await adapter.list_conflicts("bot", status=ResolutionStatus.PENDING)
    if conflicts:
        c = conflicts[0]
        short_id = c.memory_a_id[:6]
        print(f"  {YELLOW}⚠{RESET} ConflictRecord created  "
              f"direct_contradiction  confidence: {c.confidence:.2f}")
        print(f'  {DIM}   "{c.description}"{RESET}')
        pause(0.5)
    print(f'  {GREEN}✓{RESET} stored  "Refund policy changed to 14 days"')
    pause(1.0)

    print(f'\nSearching: {DIM}"what is our refund policy?"{RESET}')
    pause(0.8)

    results = await mn.search("bot", EMBED_QUERY, top_k=3)

    print()
    for r in results:
        if r.recommended:
            tag = f"{GREEN}✓ recommended{RESET}"
        else:
            tag = f"{YELLOW}⚠ stale{RESET}  — superseded"
        print(f'  "{r.memory.text}"')
        print(f"   {tag}")
        if r.conflict_summary:
            print(f'   {DIM}conflict: "{r.conflict_summary}"{RESET}')
        pause(0.7)

    print()
    pause(1.5)


if __name__ == "__main__":
    asyncio.run(main())
