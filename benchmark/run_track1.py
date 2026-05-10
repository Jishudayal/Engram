"""
Track 1 — Infrastructure Reliability Score
Runs MemoryEval against each backend adapter and writes results to benchmark/results/.

Usage:
    python benchmark/run_track1.py

Results are saved to benchmark/results/YYYY-MM-DD_track1.json
and a human-readable summary is printed to stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project paths are available when run from the project root
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "memoryeval" / "src"))
sys.path.insert(0, str(_HERE))

import chromadb
from engram.adapters.chroma import ChromaAdapter
from engram.adapters.memory import InMemoryAdapter
from engram.adapters.pgvector import PgVectorAdapter
from engram.adapters.qdrant import QdrantAdapter
from memoryeval.scorer import BenchmarkScorer
from memoryeval.types import BenchmarkReport
from naive_qdrant_adapter import NaiveQdrantAdapter


def _category_scores(report: BenchmarkReport) -> dict[str, float]:
    return {cat.value: round(score, 4) for cat, score in report.scores_by_category.items()}


def _failed_cases(report: BenchmarkReport) -> list[dict]:
    return [
        {"case": r.case_name, "category": r.category.value, "score": r.score, "error": r.error}
        for r in report.all_results
        if not r.passed
    ]


def _pass_rate(report: BenchmarkReport) -> float:
    total = len(report.all_results)
    passed = sum(1 for r in report.all_results if r.passed)
    return round(passed / total, 4) if total else 0.0


async def run_adapter(adapter, backend_name: str) -> dict:
    """Run all 50 MemoryEval cases against a single adapter and return a result dict."""
    scorer = BenchmarkScorer(adapter, backend_name=backend_name)
    report = await scorer.run()

    result = {
        "track": "track1",
        "backend": backend_name,
        "run_at": datetime.now(UTC).isoformat(),
        "overall_score": report.overall_score,
        "hallucination_risk": report.hallucination_risk.value,
        "pass_rate": _pass_rate(report),
        "total_cases": len(report.all_results),
        "passed_cases": sum(1 for r in report.all_results if r.passed),
        "by_category": _category_scores(report),
        "failed_cases": _failed_cases(report),
    }
    return result


def print_result(result: dict) -> None:
    """Print a compact human-readable summary for one backend."""
    risk_emoji = {
        "low": "✓",
        "medium": "~",
        "high": "!",
        "critical": "✗",
    }.get(result["hallucination_risk"], "?")

    print(f"\n{'─' * 60}")
    print(f"  Backend : {result['backend']}")
    print(
        f"  Score   : {result['overall_score']}  {risk_emoji} {result['hallucination_risk'].upper()}"
    )
    print(
        f"  Passed  : {result['passed_cases']}/{result['total_cases']}  ({result['pass_rate'] * 100:.0f}%)"
    )
    print("  By category:")
    for cat, score in result["by_category"].items():
        bar = "█" * int(score * 20)
        print(f"    {cat:<14} {score:.4f}  {bar}")
    if result["failed_cases"]:
        print(f"  Failed cases ({len(result['failed_cases'])}):")
        for f in result["failed_cases"]:
            err = f"  ERROR: {f['error']}" if f["error"] else ""
            print(f"    [{f['category']}] {f['case']}  score={f['score']}{err}")
    else:
        print("  Failed cases: none")


async def main() -> None:
    results: list[dict] = []

    # ── Step 1.1 ── InMemoryAdapter (baseline, no dependencies) ──────────────
    print("\nRunning Step 1.1 — InMemoryAdapter (baseline)...")
    result = await run_adapter(InMemoryAdapter(), backend_name="engram-inmemory-adapter")
    results.append(result)
    print_result(result)

    # ── Step 1.2 ── QdrantAdapter in :memory: mode (no Docker required) ───────
    print("\nRunning Step 1.2 — QdrantAdapter (:memory: mode)...")
    qdrant_adapter = QdrantAdapter(
        ":memory:",
        "memoryeval_benchmark",
        vector_size=16,  # matches MemoryEval synthetic embeddings
    )
    await qdrant_adapter.open()
    try:
        result = await run_adapter(qdrant_adapter, backend_name="engram-qdrant-adapter")
    finally:
        await qdrant_adapter.close()
    results.append(result)
    print_result(result)

    # ── Step 1.3 ── ChromaAdapter (EphemeralClient, no Docker required) ───────
    print("\nRunning Step 1.3 — ChromaAdapter (ephemeral/in-memory)...")
    chroma_client = chromadb.EphemeralClient()
    chroma_adapter = ChromaAdapter(chroma_client, "memoryeval_benchmark", vector_size=16)
    await chroma_adapter.open()
    try:
        result = await run_adapter(chroma_adapter, backend_name="engram-chroma-adapter")
    finally:
        await chroma_adapter.close()
    results.append(result)
    print_result(result)

    # ── Step 1.4 ── PgVectorAdapter (dedicated Docker container, port 5433) ───
    print("\nRunning Step 1.4 — PgVectorAdapter (pgvector/pgvector:pg16)...")
    pg_adapter = PgVectorAdapter(
        "postgresql://engram:engram@localhost:5433/engram_bench",
        vector_size=16,
    )
    await pg_adapter.open()
    try:
        result = await run_adapter(pg_adapter, backend_name="engram-pgvector-adapter")
    finally:
        await pg_adapter.close()
    results.append(result)
    print_result(result)

    # ── Step 1.5 ── NaiveQdrantAdapter (no Engram data model) ────────────────
    print("\nRunning Step 1.5 — NaiveQdrantAdapter (raw Qdrant, no data model)...")
    naive_adapter = NaiveQdrantAdapter(vector_size=16)
    await naive_adapter.open()
    try:
        result = await run_adapter(naive_adapter, backend_name="naive-qdrant")
    finally:
        await naive_adapter.close()
    results.append(result)
    print_result(result)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = out_dir / f"{ts}_track1.json"

    # Merge with existing file if it already has results from today
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        existing_backends = {r["backend"] for r in existing}
        for r in results:
            if r["backend"] not in existing_backends:
                existing.append(r)
            else:
                # Overwrite with latest run
                existing = [r if r["backend"] == e["backend"] else e for e in existing]
        results = existing

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'─' * 60}")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
