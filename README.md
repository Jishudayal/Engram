# memnotary

Memory your AI agents can actually trust.

[![Python](https://img.shields.io/pypi/pyversions/memnotary.svg)](https://pypi.org/project/memnotary/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/memnotary.svg)](https://pypi.org/project/memnotary/)

---

AI agents accumulate memories over time, and some will contradict each other. Your agent stored "refund policy is 30 days" in January, then "refund policy is 14 days" in March. Both sit in your vector store. When retrieved together, your LLM picks one — silently, often wrongly, with no flag that a contradiction exists.

memnotary wraps your existing vector backend and adds what's missing: contradiction detection, memory health scoring, automatic consolidation, and an audit trail. You don't replace anything. You just stop trusting your memory blindly.

## What memnotary is not

memnotary is not a vector database and does not replace Qdrant, Chroma, or Postgres. It is the reliability layer on top: health checks, conflict detection, consolidation, and provenance. Your existing storage stays exactly where it is.

## When to use memnotary

Use memnotary if your agent stores long-lived memories and you need to know:

- whether two memories contradict each other
- whether old facts are still being retrieved when they shouldn't be
- whether memory quality is getting worse over time
- why a memory exists and where it came from
- what would happen before automatic consolidation mutates state

## 30-second example

memnotary is provider-agnostic. Bring your own LLM and embedding function:

```python
async def your_llm(prompt: str) -> str:
    # OpenAI, Anthropic, a local model — anything works
    response = await client.chat.completions.create(...)
    return response.choices[0].message.content

def embed(text: str) -> list[float]:
    # any embedding function — OpenAI, sentence-transformers, etc.
    ...
```

```python
from memnotary import memnotary, ContradictionDetector, Consolidator, Memory, InMemoryAdapter

eng = memnotary(
    InMemoryAdapter(),
    detector=ContradictionDetector(llm_fn=your_llm),
    consolidator=Consolidator(llm_fn=your_llm),
)

async with eng:
    await eng.store(Memory(agent_id="bot", text="Refund policy is 30 days", embedding=embed("Refund policy is 30 days")))
    await eng.store(Memory(agent_id="bot", text="Refund policy changed to 14 days", embedding=embed("Refund policy changed to 14 days")))
    # ↑ Conflict detected on the second store. memnotary saved a ConflictRecord.

    results = await eng.search("bot", embed("refund policy"), top_k=5)
    for result in results:
        if result.conflict_flag:
            print(result.conflict_summary)  # one-sentence explanation of the conflict
            print(result.recommended)       # False if a higher-ranked result already covers this

    await eng.consolidate("bot")
    # memnotary supersedes, merges, or flags the conflict depending on type and confidence.
```

## What it does

**Contradiction detection.** Every `store()` call runs a similarity search against existing memories. If potential conflicts are found, your LLM classifies them. Confirmed contradictions become `ConflictRecord` objects you can inspect, act on, or queue for review.

**Health scoring.** `await eng.health(agent_id)` returns a snapshot with signals like `contradiction_score`, `freshness_score`, and `confidence_accuracy_gap`. Useful for dashboards or for deciding when to run consolidation.

**Consolidation.** `await eng.consolidate(agent_id)` reads all pending conflicts and plans a batch of actions: supersede the outdated memory, merge duplicates, or flag uncertain cases for a human. Then it executes them.

**Provenance.** Memories can carry a `ProvenanceRecord` — where it came from, who ingested it, what it was derived from. `await eng.export_provenance_json(agent_id, memory_id)` gives you a compliance-ready audit trail.

## How it compares

|  | memnotary | Mem0 | Zep | Raw vector DB |
|---|---|---|---|---|
| Stores memories | wraps yours | yes | yes | yes |
| Detects contradictions | **yes** | partial | no | no |
| Health scoring | **yes** | no | no | no |
| Provenance / audit trail | **yes** | no | partial | no |
| Bring your own backend | **yes** | no | no | — |
| Bring your own LLM | **yes** | partial | yes | — |

memnotary doesn't replace Mem0 or Zep — you can run it on top of either. It replaces the blind trust in whatever is already storing your memories.

## LangChain bridge

Drop-in `VectorStore` and `BaseChatMessageHistory` backed by memnotary. Bulk adds skip per-document detection — call `scan_contradictions()` after loading to catch conflicts across the batch.

```python
from memnotary.integrations.langchain import MemnotaryVectorStore, MemnotaryChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

store = MemnotaryVectorStore(eng, embeddings=OpenAIEmbeddings(), agent_id="bot")
await store.aadd_texts(["Refund policy is 30 days", "Refund policy changed to 14 days"])
await eng.scan_contradictions("bot")  # detect conflicts across the batch
docs = await store.asimilarity_search("what is the refund policy", k=3)
# docs[0].metadata["_memory_id"] lets you trace back to the original Memory

history = MemnotaryChatMessageHistory(eng, session_id="conv-42")
await history.aadd_messages([
    HumanMessage(content="What's the refund policy?"),
    AIMessage(content="It's 14 days."),
])
msgs = await history.aget_messages()
```

## Backends

| Backend | Best for | Install |
|---|---|---|
| In-memory | tests, local development | built-in |
| Qdrant | production deployments, hybrid search | `pip install "memnotary[qdrant]"` |
| Chroma | local-first apps, prototypes | `pip install "memnotary[chroma]"` |
| pgvector | teams already on Postgres | `pip install "memnotary[pgvector]"` |

Every backend is a subclass of `AbstractAdapter`. Adding your own takes one file.

## Installation

```bash
pip install memnotary                   # core + in-memory adapter
pip install "memnotary[qdrant]"         # + Qdrant
pip install "memnotary[chroma]"         # + Chroma
pip install "memnotary[pgvector]"       # + pgvector (requires asyncpg)
pip install "memnotary[langchain]"      # + LangChain bridge
pip install "memnotary[all]"            # everything
```

Requires Python 3.11+. memnotary is fully async.

## Benchmark

Two tracks. Track 1 is infrastructure correctness. Track 2 is behavioral — how each system handles real contradiction scenarios, scored on three dimensions.

### Track 2 — Behavioral (the headline numbers)

We ran 7 real-world conflict scenarios against memnotary, Mem0, and raw Qdrant. Each scenario was scored on:

- **Correctness** (weight 0.4) — did the right answer come back?
- **Signal** (weight 0.4) — was the stale or conflicting memory flagged or suppressed?
- **Preservation** (weight 0.2) — were unrelated facts left untouched?

> **memnotary flagged 6 of 7 contradiction scenarios. Mem0 flagged 3. Raw Qdrant flagged 2.**
> All three systems returned the right answer — the difference is whether the wrong answer was also surfaced silently alongside it.

| System | Overall | Correctness | Signal | Preservation | Risk |
|---|---|---|---|---|---|
| **engram** | **0.94** | 1.00 | 0.86 | 1.00 | LOW |
| mem0 | 0.77 | 1.00 | 0.43 | 1.00 | MEDIUM |
| naive-qdrant | 0.71 | 1.00 | 0.29 | 1.00 | MEDIUM |

Every system eventually surfaces the right answer in the top-k. But Mem0 and raw Qdrant also return the contradicting wrong answer alongside it, with no flag. Your LLM sees both. It picks one. You don't know which. **Signal is the difference between an agent that knows it's uncertain and one that confidently returns the wrong policy.**

Per-scenario breakdown:

| Scenario | engram | mem0 | naive-qdrant | What it tests |
|---|---|---|---|---|
| B1 — Direct contradiction | **1.00** | 0.60 | 0.60 | Old fact superseded by new |
| B2 — Retention | **1.00** | 1.00 | 1.00 | Three unrelated facts all survive |
| B3 — Temporal chain | **1.00** | 0.60 | 0.60 | Three versions; only the latest surfaces |
| B4 — False positive guard | **1.00** | 1.00 | 1.00 | Two non-contradictory sub-policies both survive |
| B5 — Temporal language | **1.00** | 1.00 | 0.60 | Rescheduled event; old schedule flagged |
| B6 — Lexically varied temporal | 0.60 | 0.60 | 0.60 | Same fact, different phrasing |
| B7 — Metadata timestamp | **1.00** | 0.60 | 0.60 | Structured timestamps override insertion order |

**B6:** All three systems score 0.60 here. The two sentences are phrased differently enough that their cosine similarity falls below memnotary's 0.82 cluster threshold, so the LLM classifier is never invoked. This is a known trade-off: conflict detection requires semantic overlap at the embedding level before the more expensive LLM step fires. Varied real-world phrasing that expresses the same underlying fact can fall below this threshold.

To reproduce (requires `OPENAI_API_KEY` and Docker):

```bash
docker run -d --name engram-qdrant-mem0 -p 6333:6333 qdrant/qdrant
OPENAI_API_KEY=sk-... python benchmark/run_track2.py
```

### Track 1 — Infrastructure Reliability

50 deterministic test cases across five backends — four memnotary-backed adapters and one raw Qdrant wrapper with no memnotary data model. No API key required.

| Backend | Score | Risk | Pass |
|---|---|---|---|
| engram-inmemory | 0.88 | LOW | 44/50 |
| engram-qdrant | 0.88 | LOW | 44/50 |
| engram-chroma | 0.88 | LOW | 44/50 |
| engram-pgvector | 0.88 | LOW | 44/50 |
| naive-qdrant | 0.42 | CRITICAL | 20/50 |

Score is identical across all four memnotary backends — reliability comes from the data model, not the choice of vector backend. The largest gap is in temporal reliability: memnotary scores **1.00**, naive Qdrant scores **0.05**.

To reproduce:

```bash
python benchmark/run_track1.py   # ~2 min, no API key needed
python benchmark/report.py       # prints the table above
```

### Limitations

- Track 1 uses synthetic 16-dim embeddings; production embeddings (768–3072 dim) will produce different absolute scores. The data-model gap should hold but margins may compress.
- Track 2 is 7 hand-crafted scenarios. Small N is intentional — every failure is inspectable — but it is not a stress test.
- B6 reveals a real ceiling: lexically distant phrasings of the same fact fall below the cosine cluster threshold and never reach the LLM classifier. Improving this is on the 0.2 roadmap.
- Mem0 was tested with default settings; advanced configurations may close the signal gap.
- Cost/latency comparison (tokens per stored memory across systems) is coming in 0.2.

See [`benchmark/README.md`](benchmark/README.md) for full setup and Docker requirements.

## Status

`0.1.0-alpha` — the core reliability loop (store → detect → score → consolidate → provenance) is complete and covered by 880+ unit tests. The pgvector adapter and LangChain bridge are included.

Not production-tested yet. The API is stable but may have breaking changes before 1.0.

What's planned for 0.2: LlamaIndex bridge, a sync facade for non-async code, OpenTelemetry instrumentation, and cost/latency benchmarks.

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help build it.

## License

MIT
