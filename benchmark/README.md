# Engram Benchmark Suite

Two-track reliability benchmark for Engram and comparable memory systems.

## Track 1 — Infrastructure Reliability Score

Runs MemoryEval's 50 deterministic test cases against each storage adapter.
No API key required — all embeddings are synthetic 16-dim vectors.

```bash
# Start the pgvector container (required for Step 1.4)
docker run -d --name engram-pgvector-bench \
  -e POSTGRES_USER=engram -e POSTGRES_PASSWORD=engram -e POSTGRES_DB=engram_bench \
  -p 5433:5432 pgvector/pgvector:pg16

docker exec engram-pgvector-bench psql -U engram -d engram_bench \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

python benchmark/run_track1.py   # runs all 5 adapters
python benchmark/report.py       # prints comparison table + headline numbers
```

**Results:** `benchmark/results/YYYY-MM-DD_track1.json`

## Track 2 — Behavioral Contradiction Test

Behavioral comparison across Mem0, raw Qdrant, and Engram+Qdrant using
real LLM extraction pipelines.

**Requirements:**
- `OPENAI_API_KEY` environment variable (used by Mem0 for extraction + embeddings)
- Docker containers:
  - `engram-qdrant-mem0` — Qdrant server for Mem0 (port 6333)

```bash
# Start Qdrant (shared by Mem0, NaiveQdrant, and Engram adapters)
docker run -d --name engram-qdrant-mem0 -p 6333:6333 qdrant/qdrant

# Smoke test — confirms Mem0 is wired correctly before running Track 2
OPENAI_API_KEY=sk-... python benchmark/smoke_mem0.py

# Run Track 2 — all 7 scenarios across 4 systems
OPENAI_API_KEY=sk-... python benchmark/run_track2.py
```

**Results:** `benchmark/results/YYYY-MM-DD_track2.json`

### Systems compared

| System | Description |
|---|---|
| `mem0` | Mem0 2.x — LLM extraction + OpenAI embeddings (black-box dedup) |
| `naive-qdrant` | Raw Qdrant cosine search, no Engram data model |
| `engram-detect` | Engram + ContradictionDetector — flags conflicts, no resolution |
| `engram-consolidated` | Engram + ContradictionDetector + Consolidator — flags + resolves |

### Scenarios

| Scenario | What it tests |
|---|---|
| B1 — Direct contradiction | Old fact superseded by new; only new should be recommended |
| B2 — Retention | Three unrelated facts; all must remain retrievable |
| B3 — Temporal chain | Three versions of the same fact; only the latest is recommended |
| B4 — False positive guard | Two non-contradictory sub-policies; both should be recommended |
| B5 — Temporal language | Rescheduled event; only the new schedule should surface |
| B6 — Explicit temporal language | Varied phrasing with explicit month markers; tests detection across lexically different sentences |
| B7 — Metadata timestamp | Current fact stored first, stale fact stored second with older `created_at`; tests whether structured timestamps override insertion order |

### Results (2026-05-11)

Scoring dimensions: **correctness** (right answer returned, weight 0.4) · **signal** (stale content flagged or suppressed, weight 0.4) · **preservation** (no false deletions, weight 0.2).

| System | Overall | Correctness | Signal | Preservation | Risk |
|---|---|---|---|---|---|
| `engram-detect` | **0.9429** | 1.00 | 0.86 | 1.00 | LOW |
| `engram-consolidated` | **0.9429** | 1.00 | 0.86 | 1.00 | LOW |
| `mem0` | 0.7714 | 1.00 | 0.43 | 1.00 | MEDIUM |
| `naive-qdrant` | 0.7143 | 1.00 | 0.29 | 1.00 | MEDIUM |

Per-scenario composite scores:

| Scenario | engram-detect | engram-consolidated | mem0 | naive-qdrant |
|---|---|---|---|---|
| B1 — Direct contradiction | 1.00 | 1.00 | 0.60 | 0.60 |
| B2 — Retention | 1.00 | 1.00 | 1.00 | 1.00 |
| B3 — Temporal chain | 1.00 | 1.00 | 0.60 | 0.60 |
| B4 — False positive guard | 1.00 | 1.00 | 1.00 | 1.00 |
| B5 — Temporal language | 1.00 | 1.00 | 1.00 | 0.60 |
| B6 — Explicit temporal language | 0.60 | 0.60 | 0.60 | 0.60 |
| B7 — Metadata timestamp | 1.00 | 1.00 | 0.60 | 0.60 |

**B6 note:** All four systems score 0.60 on B6. The two sentences in this scenario are phrased differently enough that their cosine similarity falls below Engram's 0.82 cluster threshold, so the LLM classifier is never invoked and no conflict is recorded. This is a known architectural trade-off: conflict detection requires sufficient semantic overlap at the embedding level before the more expensive LLM step is triggered. Varied real-world phrasing that expresses the same underlying fact can fall below this threshold.

**Dependencies (benchmark-only, not in pyproject.toml extras):**

| Package | Purpose |
|---|---|
| `mem0ai>=2.0.2` | Mem0 memory system |
| `qdrant-client>=1.9` | Qdrant adapter (also used in Track 1) |
| `chromadb>=0.5` | Chroma adapter (also used in Track 1) |
| `asyncpg>=0.29` | pgvector adapter (also used in Track 1) |
| `pgvector>=0.3` | pgvector adapter (also used in Track 1) |

Install all:
```bash
pip install mem0ai qdrant-client chromadb asyncpg pgvector
```

## Docker containers summary

| Container | Image | Port | Used by |
|---|---|---|---|
| `engram-pgvector-bench` | `pgvector/pgvector:pg16` | 5433 | Track 1 Step 1.4 |
| `engram-qdrant-mem0` | `qdrant/qdrant` | 6333 | Track 2 (Mem0, NaiveQdrant, Engram) |
