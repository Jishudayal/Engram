"""
Track 2 — Behavioral Contradiction Test runner.

Runs 5 scenarios across 4 systems (Mem0, NaiveQdrant, EngramDetect,
EngramConsolidated) and saves results to benchmark/results/.

Requirements:
  OPENAI_API_KEY env var
  Docker: engram-qdrant-mem0 running on port 6333

Usage:
    OPENAI_API_KEY=sk-... python benchmark/run_track2.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add benchmark/ to path so scenario/system imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from behavioral_system import build_systems
from scenarios_track2 import ScenarioResult, run_all

_RESULTS_DIR = Path(__file__).parent / "results"
_SYSTEM_ORDER = ["mem0", "naive-qdrant", "engram-detect", "engram-consolidated"]

# Risk band: score ≥ threshold → label
_RISK_BANDS = [(0.85, "LOW"), (0.65, "MEDIUM"), (0.40, "HIGH"), (0.0, "CRITICAL")]


def _risk_label(score: float) -> str:
    for threshold, label in _RISK_BANDS:
        if score >= threshold:
            return label
    return "CRITICAL"


def _print_table(results: list[ScenarioResult]) -> None:
    scenarios = sorted({r.scenario for r in results})
    systems = [s for s in _SYSTEM_ORDER if any(r.system == s for r in results)]

    # Build lookup: (system, scenario) -> score
    lookup: dict[tuple[str, str], float] = {(r.system, r.scenario): r.score for r in results}

    col_w = 22
    scen_w = 20

    header = f"{'System':<{col_w}}"
    for s in scenarios:
        header += f"  {s[:scen_w]:>{scen_w}}"
    header += f"  {'Avg':>6}  {'Risk'}"
    print(header)
    print("─" * len(header))

    for sys_name in systems:
        scores = [lookup.get((sys_name, sc), 0.0) for sc in scenarios]
        avg = sum(scores) / len(scores) if scores else 0.0
        row = f"{sys_name:<{col_w}}"
        for sc in scores:
            row += f"  {sc:>{scen_w}.2f}"
        row += f"  {avg:>6.4f}  {_risk_label(avg)}"
        print(row)


def _to_json(results: list[ScenarioResult]) -> list[dict[str, object]]:
    scenarios = sorted({r.scenario for r in results})
    systems = [s for s in _SYSTEM_ORDER if any(r.system == s for r in results)]
    lookup: dict[tuple[str, str], ScenarioResult] = {(r.system, r.scenario): r for r in results}

    output = []
    for sys_name in systems:
        system_results = [lookup.get((sys_name, sc)) for sc in scenarios]
        scores = [r.score if r else 0.0 for r in system_results]
        avg = sum(scores) / len(scores) if scores else 0.0
        output.append(
            {
                "track": "track2",
                "backend": sys_name,
                "overall_score": round(avg, 4),
                "hallucination_risk": _risk_label(avg),
                "by_scenario": {
                    sc: round(lookup[(sys_name, sc)].score, 4)
                    for sc in scenarios
                    if (sys_name, sc) in lookup
                },
                "scenario_notes": {
                    sc: lookup[(sys_name, sc)].notes for sc in scenarios if (sys_name, sc) in lookup
                },
            }
        )
    return output


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set.")
        sys.exit(1)

    print("Initializing systems...")
    systems = build_systems(openai_api_key=api_key)
    print(f"  {len(systems)} systems: {[s.name for s in systems]}")

    print("\nRunning 5 behavioral scenarios...")
    try:
        results = await run_all(systems)
    finally:
        for s in systems:
            await s.close()
    print(f"  {len(results)} results collected.")

    print()
    _print_table(results)

    _RESULTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = _RESULTS_DIR / f"{date_str}_track2.json"

    payload = _to_json(results)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
