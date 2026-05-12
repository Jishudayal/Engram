"""Scene 4 — Benchmark: Track 1 results from pre-baked JSON.

Run:
    python demos/demo_benchmark.py

Loads the most-recent track1 results file and prints a compact comparison
table plus headline numbers. No live compute — instant, deterministic output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from demos._common import BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, pause, section

# Reuse report.py helpers (no live benchmark run needed).
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
from report import _headline_numbers, _latest_results_file, _load

_RISK_COLOR = {
    "low":      GREEN,
    "medium":   YELLOW,
    "high":     YELLOW,
    "critical": RED,
}

_ENGRAM_BACKENDS = {
    "memnotary-inmemory-adapter",
    "memnotary-qdrant-adapter",
    "memnotary-chroma-adapter",
    "memnotary-pgvector-adapter",
    "engram-inmemory-adapter",
    "engram-qdrant-adapter",
    "engram-chroma-adapter",
    "engram-pgvector-adapter",
}


def _short_name(backend: str) -> str:
    return (
        backend
        .replace("memnotary-", "memnotary + ")
        .replace("engram-", "memnotary + ")
        .replace("naive-", "naive ")
        .replace("-adapter", "")
    )


def _print_table(results: list[dict]) -> None:
    col_w = 26
    cats = ["temporal", "contrdct", "multihop", "import", "cross"]
    header = f"  {'Backend':<{col_w}} {'Score':>6}  {'Risk':<8}  {'Pass':>5}"
    for c in cats:
        header += f"  {c:>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in results:
        risk = r["hallucination_risk"]
        col = _RISK_COLOR.get(risk, RESET)
        name = _short_name(r["backend"])
        row = (
            f"  {name:<{col_w}}"
            f" {r['overall_score']:>6.4f}"
            f"  {col}{risk.upper():<8}{RESET}"
            f"  {r['passed_cases']:>2}/{r['total_cases']:<2}"
        )
        cat_keys = ["temporal", "contradiction", "multihop", "importance", "cross_type"]
        for k in cat_keys:
            score = r["by_category"].get(k, 0.0)
            row += f"  {score:>7.2f}"
        print(row)
        pause(0.5)


def main() -> None:
    path = _latest_results_file()
    results = _load(path)

    section("Track 1 — Infrastructure Reliability Benchmark")
    pause(0.3)

    print(f"  {DIM}source: {path.name}{RESET}")
    print()
    pause(0.4)

    _print_table(results)
    pause(0.8)

    h = _headline_numbers(results)
    print()
    print(f"  {BOLD}Headline numbers{RESET}")
    pause(0.4)
    print(f"    {GREEN}memnotary avg score  {h['engram_avg_score']:.4f}  LOW risk{RESET}")
    pause(0.3)
    print(f"    {RED}naive-qdrant score   {h['naive_avg_score']:.4f}  CRITICAL risk{RESET}")
    pause(0.3)
    print()
    gap_str = f"+{h['reliability_gap_pct']:.0f} pp reliability gap"
    print(f"    {CYAN}{BOLD}{gap_str}{RESET}")
    pause(0.5)
    print(f"    temporal reliability  "
          f"memnotary {h['engram_temporal_score']:.2f} vs "
          f"naive {h['naive_temporal_score']:.2f}  "
          f"(+{h['temporal_gap_pct']:.0f} pp)")
    print()
    pause(1.5)


if __name__ == "__main__":
    main()
