"""
Track 2 — Behavioral Contradiction Test runner.

Runs 5 scenarios across 4 systems (Mem0, NaiveQdrant, EngramDetect,
EngramConsolidated) and saves results to benchmark/results/.

Requirements:
  OPENAI_API_KEY env var
  Docker: memnotary-qdrant-mem0 running on port 6333

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
_SYSTEM_ORDER = ["mem0", "naive-qdrant", "memnotary-detect", "memnotary-consolidated"]

# Risk band: overall score ≥ threshold → label
_RISK_BANDS = [(0.85, "LOW"), (0.65, "MEDIUM"), (0.40, "HIGH"), (0.0, "CRITICAL")]


def _risk_label(score: float) -> str:
    for threshold, label in _RISK_BANDS:
        if score >= threshold:
            return label
    return "CRITICAL"


def _dim_avgs(results: list[ScenarioResult]) -> tuple[float, float, float, float]:
    """Return (correctness, signal, preservation, overall) averages over a list."""
    if not results:
        return 0.0, 0.0, 0.0, 0.0
    n = len(results)
    c = sum(r.correctness for r in results) / n
    s = sum(r.signal for r in results) / n
    p = sum(r.preservation for r in results) / n
    o = sum(r.score for r in results) / n
    return c, s, p, o


def _print_summary_table(results: list[ScenarioResult]) -> None:
    """Per-system averages across all scenarios for each scoring dimension."""
    systems = [s for s in _SYSTEM_ORDER if any(r.system == s for r in results)]

    col_sys = 24
    header = (
        f"{'System':<{col_sys}}"
        f"  {'Correctness':>11}"
        f"  {'Signal':>6}"
        f"  {'Preservation':>12}"
        f"  {'Overall':>7}"
        f"  {'Risk'}"
    )
    print(header)
    print("─" * len(header))

    for sys_name in systems:
        sys_results = [r for r in results if r.system == sys_name]
        c, s, p, o = _dim_avgs(sys_results)
        print(
            f"{sys_name:<{col_sys}}  {c:>11.2f}  {s:>6.2f}  {p:>12.2f}  {o:>7.4f}  {_risk_label(o)}"
        )


def _print_scenario_detail(results: list[ScenarioResult]) -> None:
    """Per-scenario composite scores — useful for spotting which scenario fails."""
    scenarios = sorted({r.scenario for r in results})
    systems = [s for s in _SYSTEM_ORDER if any(r.system == s for r in results)]
    lookup = {(r.system, r.scenario): r for r in results}

    col_sys = 24
    col_sc = 20
    header = f"{'System':<{col_sys}}"
    for sc in scenarios:
        header += f"  {sc[:col_sc]:>{col_sc}}"
    print(header)
    print("─" * len(header))

    for sys_name in systems:
        row = f"{sys_name:<{col_sys}}"
        for sc in scenarios:
            r = lookup.get((sys_name, sc))
            val = round(r.score, 2) if r else 0.0
            row += f"  {val:>{col_sc}.2f}"
        print(row)


def _to_json(results: list[ScenarioResult]) -> list[dict[str, object]]:
    scenarios = sorted({r.scenario for r in results})
    systems = [s for s in _SYSTEM_ORDER if any(r.system == s for r in results)]
    lookup = {(r.system, r.scenario): r for r in results}

    output = []
    for sys_name in systems:
        sys_results = [lookup[k] for k in lookup if k[0] == sys_name]
        c, s, p, o = _dim_avgs(sys_results)
        output.append(
            {
                "track": "track2",
                "backend": sys_name,
                "overall_score": round(o, 4),
                "correctness_avg": round(c, 4),
                "signal_avg": round(s, 4),
                "preservation_avg": round(p, 4),
                "hallucination_risk": _risk_label(o),
                "by_scenario": {
                    sc: round(lookup[(sys_name, sc)].score, 4)
                    for sc in scenarios
                    if (sys_name, sc) in lookup
                },
                "by_dimension": {
                    sc: {
                        "correctness": round(lookup[(sys_name, sc)].correctness, 4),
                        "signal": round(lookup[(sys_name, sc)].signal, 4),
                        "preservation": round(lookup[(sys_name, sc)].preservation, 4),
                    }
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

    print("\n── Summary (avg across all scenarios) ──")
    _print_summary_table(results)

    print("\n── Scenario detail (composite score) ──")
    _print_scenario_detail(results)

    _RESULTS_DIR.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = _RESULTS_DIR / f"{date_str}_track2.json"

    payload = _to_json(results)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
