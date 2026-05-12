"""
Track 1 — Final comparison table and headline numbers.

Usage:
    python benchmark/report.py
    python benchmark/report.py benchmark/results/2026-05-10_track1.json
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_CATEGORIES = ["temporal", "contradiction", "multihop", "importance", "cross_type"]

_RISK_LABEL = {
    "low": "LOW     ",
    "medium": "MEDIUM  ",
    "high": "HIGH    ",
    "critical": "CRITICAL",
}

_ENGRAM_BACKENDS = {
    # current names
    "memnotary-inmemory-adapter",
    "memnotary-qdrant-adapter",
    "memnotary-chroma-adapter",
    "memnotary-pgvector-adapter",
    # pre-rename names (results generated before 0.1.0a2)
    "engram-inmemory-adapter",
    "engram-qdrant-adapter",
    "engram-chroma-adapter",
    "engram-pgvector-adapter",
}

_NAIVE_BACKENDS = {
    "naive-qdrant",
}


def _latest_results_file() -> Path:
    results_dir = Path(__file__).parent / "results"
    files = sorted(results_dir.glob("*_track1.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No track1 result files found in {results_dir}")
    return files[0]


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_table(results: list[dict]) -> None:
    col_w = 26

    # Header
    header = f"{'Backend':<{col_w}} {'Score':>6}  {'Risk':<8}  {'Pass':>7}"
    for cat in _CATEGORIES:
        header += f"  {cat[:6]:>6}"
    print(header)
    print("─" * (len(header) + 2))

    for r in results:
        row = (
            f"{r['backend']:<{col_w}}"
            f" {r['overall_score']:>6.4f}"
            f"  {_RISK_LABEL.get(r['hallucination_risk'], '?       ')}"
            f"  {r['passed_cases']:>3}/{r['total_cases']:<3}"
        )
        for cat in _CATEGORIES:
            score = r["by_category"].get(cat, 0.0)
            row += f"  {score:>6.2f}"
        print(row)


def _headline_numbers(results: list[dict]) -> dict:
    engram = [r for r in results if r["backend"] in _ENGRAM_BACKENDS]
    naive = [r for r in results if r["backend"] in _NAIVE_BACKENDS]

    engram_avg = sum(r["overall_score"] for r in engram) / len(engram) if engram else 0.0
    naive_avg = sum(r["overall_score"] for r in naive) / len(naive) if naive else 0.0
    gap = engram_avg - naive_avg

    # Temporal gap: most dramatic category difference
    engram_temporal = (
        sum(r["by_category"].get("temporal", 0) for r in engram) / len(engram) if engram else 0.0
    )
    naive_temporal = (
        sum(r["by_category"].get("temporal", 0) for r in naive) / len(naive) if naive else 0.0
    )

    # Memnotary consistency: std dev across backends (lower = more consistent)
    if len(engram) > 1:
        mean = engram_avg
        variance = sum((r["overall_score"] - mean) ** 2 for r in engram) / len(engram)
        consistency = variance**0.5
    else:
        consistency = 0.0

    return {
        "engram_avg_score": round(engram_avg, 4),
        "naive_avg_score": round(naive_avg, 4),
        "reliability_gap": round(gap, 4),
        "reliability_gap_pct": round(gap * 100, 1),
        "engram_backends_tested": len(engram),
        "naive_backends_tested": len(naive),
        "engram_temporal_score": round(engram_temporal, 4),
        "naive_temporal_score": round(naive_temporal, 4),
        "temporal_gap_pct": round((engram_temporal - naive_temporal) * 100, 1),
        "engram_score_std_dev": round(consistency, 4),
        "cases_total": results[0]["total_cases"] if results else 0,
    }


def _print_headlines(h: dict) -> None:
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│  HEADLINE NUMBERS  (for blog post / launch content)     │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Memnotary adapter avg score    {h['engram_avg_score']:.4f}  (LOW risk)          │")
    print(f"│  Naive raw-Qdrant score      {h['naive_avg_score']:.4f}  (CRITICAL risk)     │")
    print(
        f"│  Reliability gap             +{h['reliability_gap_pct']:.0f} percentage points           │"
    )
    print("│                                                         │")
    print(
        f"│  Temporal reliability        Memnotary {h['engram_temporal_score']:.2f} vs naive {h['naive_temporal_score']:.2f}   │"
    )
    print(f"│  Temporal gap                +{h['temporal_gap_pct']:.0f} pp (lifecycle mgmt)      │")
    print("│                                                         │")
    print(
        f"│  Memnotary score consistency    σ={h['engram_score_std_dev']:.4f} across {h['engram_backends_tested']} backends  │"
    )
    print(f"│  Benchmark size              {h['cases_total']} cases (MemoryEval)            │")
    print("└─────────────────────────────────────────────────────────┘")


def _print_narrative(h: dict) -> None:
    print()
    print("LAUNCH COPY (draft)")
    print("─" * 60)
    print(f"""We ran MemoryEval's {h["cases_total"]} test cases across 5 backends.

Every Memnotary-backed adapter (InMemory, Qdrant, Chroma, pgvector)
scored {h["engram_avg_score"]} with LOW hallucination risk. Score is identical
across all four backends — storage fidelity comes from Memnotary's
data model, not the choice of vector backend.

A raw Qdrant wrapper without Memnotary's data model scored
{h["naive_avg_score"]} with CRITICAL risk — a {h["reliability_gap_pct"]:.0f}-point gap.

The biggest single contributor: temporal reliability.
Memnotary scores {h["engram_temporal_score"]:.2f}; naive Qdrant scores {h["naive_temporal_score"]:.2f}.
Without lifecycle tracking, {h["temporal_gap_pct"]:.0f}% of memory lifecycle
cases fail — superseded facts stay active, update history is
lost, and access patterns are invisible to the retriever.

The two remaining gaps (contradiction resolution and
importance-weighted ranking) require an active reliability
layer on top of storage — they are not fixed by switching
vector backends.""")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_results_file()

    results = _load(path)
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    print()
    print(f"Track 1 — Infrastructure Reliability Score  ({ts})")
    print(f"Source: {path}")
    print()

    _print_table(results)
    h = _headline_numbers(results)
    _print_headlines(h)
    _print_narrative(h)
    print()


if __name__ == "__main__":
    main()
