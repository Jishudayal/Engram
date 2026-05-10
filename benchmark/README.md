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

# Run Track 2 — all 5 scenarios across 4 systems
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
