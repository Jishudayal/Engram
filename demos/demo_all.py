"""Full demo — runs all 4 scenes in sequence for asciinema recording.

Run:
    python demos/demo_all.py

Records the complete GIF narrative:
  Scene 1 — Naive storage: two contradicting memories, no signal
  Scene 2 — MemNotary:     contradiction detected, enriched search results
  Scene 3 — Health check:  health() before/after consolidate()
  Scene 4 — Benchmark:     Track 1 table + headline numbers

For asciinema + agg recording:
    asciinema rec demo.cast --overwrite --command "python demos/demo_all.py"
    agg --theme solarized-dark --font-size 18 demo.cast demo.gif
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import BOLD, DIM, RESET, pause

# Import scene mains (guard prevents auto-run on import).
from demos.demo_problem   import main as scene1
from demos.demo_memnotary import main as scene2
from demos.demo_health    import main as scene3
from demos.demo_benchmark import main as scene4_sync


def scene4() -> None:
    scene4_sync()


async def main() -> None:
    print(f"\n{BOLD}MemNotary — live demo{RESET}")
    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(1.0)

    await scene1()
    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(1.2)

    await scene2()
    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(1.2)

    await scene3()
    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(1.2)

    scene4()

    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    print(f"  {BOLD}pip install memnotary{RESET}")
    print(f"  {DIM}github.com/Jishudayal/Engram{RESET}")
    print()
    pause(2.0)


if __name__ == "__main__":
    asyncio.run(main())
