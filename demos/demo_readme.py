"""README hero GIF — Scenes 1 + 2 only (~25s).

Run:
    python demos/demo_readme.py

For recording:
    asciinema rec demo-readme.cast --overwrite --command "python demos/demo_readme.py"
    agg --theme solarized-dark --font-size 18 --cols 88 --rows 28 demo-readme.cast demo-readme.gif

Shows the before/after contrast that is the core pitch:
  Scene 1 — Naive storage: two memories returned with no conflict signal
  Scene 2 — MemNotary:     contradiction caught, results enriched with recommended/stale
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import BOLD, DIM, RESET, pause
from demos.demo_problem   import main as scene1
from demos.demo_memnotary import main as scene2


async def main() -> None:
    print(f"\n{BOLD}MemNotary — demo{RESET}")
    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(0.8)

    await scene1()

    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    pause(1.0)

    await scene2()

    print(f"{DIM}─────────────────────────────────────────────{RESET}")
    print(f"  {BOLD}pip install memnotary{RESET}")
    print(f"  {DIM}github.com/Jishudayal/MemNotary{RESET}")
    print()
    pause(2.0)


if __name__ == "__main__":
    asyncio.run(main())
