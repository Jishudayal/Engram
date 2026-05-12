"""Scene 3 — Health check: measuring and fixing memory reliability.

Run:
    python demos/demo_health.py

Shows health() before and after consolidate() — contradiction_score
drops from 1.00 to 0.00, risk level improves from HIGH to MEDIUM.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import (
    BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    EMBED_QUERY, EMBED_REFUND_14, EMBED_REFUND_30,
    mock_llm, mock_consolidate_llm, pause, section,
)

from memnotary import ContradictionDetector, InMemoryAdapter, Memory, Memnotary
from memnotary.core.consolidator import Consolidator
from memnotary.core.constants import RiskLevel


_RISK_COLOR = {
    RiskLevel.LOW:      GREEN,
    RiskLevel.MEDIUM:   YELLOW,
    RiskLevel.HIGH:     YELLOW,
    RiskLevel.CRITICAL: RED,
}


def _print_health(label: str, h) -> None:
    risk_col = _RISK_COLOR.get(h.risk_level, RESET)
    print(f"  {BOLD}{label}{RESET}")
    print(f"    score              {h.score:.2f}")
    print(f"    risk_level         {risk_col}{h.risk_level.upper()}{RESET}")
    print(f"    contradiction_score {h.contradiction_score:.2f}"
          f"  {DIM}({h.conflict_count} conflict(s) pending){RESET}")
    print(f"    freshness_score    {h.freshness_score:.2f}")
    print(f"    total_memories     {h.total_memories}")


async def main() -> None:
    adapter      = InMemoryAdapter()
    detector     = ContradictionDetector(llm_fn=mock_llm)
    consolidator = Consolidator(llm_fn=mock_consolidate_llm)
    mn           = Memnotary(adapter, detector=detector, consolidator=consolidator)

    section("Health check — before and after consolidation")

    # Set up the conflicting state (same as demo_memnotary).
    m1 = Memory(agent_id="bot", text="Refund policy is 30 days",
                embedding=EMBED_REFUND_30)
    m2 = Memory(agent_id="bot", text="Refund policy changed to 14 days",
                embedding=EMBED_REFUND_14)
    await mn.store(m1)
    await mn.store(m2)
    print(f'  {DIM}(stored 2 conflicting memories){RESET}')
    pause(0.6)

    print()
    h_before = await mn.health("bot")
    _print_health("health() — before consolidation", h_before)
    pause(1.2)

    print()
    print(f"  Running {CYAN}mn.consolidate(){RESET} ...")
    pause(0.8)

    plan = await mn.consolidate("bot")
    if plan:
        action = plan.actions[0]
        print(f"  {GREEN}✓{RESET} plan executed  "
              f"action={action.action_type}  "
              f"tier={action.tier}")
    pause(1.0)

    print()
    h_after = await mn.health("bot")
    _print_health("health() — after consolidation", h_after)
    pause(1.0)

    # Summary delta
    delta = h_after.score - h_before.score
    print()
    print(f"  {GREEN}↑ score +{delta:.2f}  "
          f"contradiction_score {h_before.contradiction_score:.2f} → "
          f"{h_after.contradiction_score:.2f}{RESET}")
    risk_before_col = _RISK_COLOR.get(h_before.risk_level, RESET)
    risk_after_col  = _RISK_COLOR.get(h_after.risk_level, RESET)
    print(f"  {GREEN}↑ risk             "
          f"{risk_before_col}{h_before.risk_level.upper()}{RESET}"
          f" → {risk_after_col}{h_after.risk_level.upper()}{RESET}")
    print()
    pause(1.5)


if __name__ == "__main__":
    asyncio.run(main())
