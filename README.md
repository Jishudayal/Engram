# Engram

Memory your AI agents can actually trust.

---

AI agents accumulate memories over time. The problem is some of those memories will eventually contradict each other — your agent stored "our refund policy is 30 days" in January, then "the refund policy changed to 14 days" in March. Both facts sit in your vector store. When your agent retrieves them together, it either picks one at random or hallucinates a reconciliation.

Engram wraps your existing vector backend and adds what's missing: contradiction detection, memory health scoring, automatic consolidation, and an audit trail. You don't replace anything. You just stop trusting your memory blindly.

## What Engram is not

Engram is not a vector database and does not replace Qdrant, Chroma, or Postgres. It is the reliability layer on top: health checks, conflict detection, consolidation, and provenance. Your existing storage stays exactly where it is.

## When to use Engram

Use Engram if your agent stores long-lived memories and you need to know:

- whether two memories contradict each other
- whether old facts are still being retrieved when they shouldn't be
- whether memory quality is getting worse over time
- why a memory exists and where it came from
- what would happen before automatic consolidation mutates state

## 30-second example

Engram is provider-agnostic. `your_llm` is any async function that takes a prompt string and returns the model's reply:

```python
async def your_llm(prompt: str) -> str:
    # OpenAI, Anthropic, a local model — anything works
    response = await client.chat.completions.create(...)
    return response.choices[0].message.content
```

```python
from engram import Engram, ContradictionDetector, Consolidator, Memory, InMemoryAdapter

# embed() is whatever embedding function you already use
eng = Engram(
    InMemoryAdapter(),
    detector=ContradictionDetector(llm_fn=your_llm),
    consolidator=Consolidator(llm_fn=your_llm),
)

async with eng:
    await eng.store(Memory(agent_id="bot", text="Refund policy is 30 days", embedding=embed("...")))
    await eng.store(Memory(agent_id="bot", text="Refund policy changed to 14 days", embedding=embed("...")))
    # ↑ Conflict detected on the second store. Engram saved a ConflictRecord.

    results = await eng.search("bot", embed("refund policy"), top_k=5)
    for result in results:
        if result.conflict_flag:
            print(result.conflict_summary)  # one-sentence explanation of the conflict
            print(result.recommended)       # False if a higher-ranked result already covers this

    await eng.consolidate("bot")
    # Engram supersedes, merges, or flags the conflict depending on type and confidence.
```

## What it does

**Contradiction detection.** Every `store()` call runs a similarity search against existing memories. If potential conflicts are found, your LLM classifies them. Confirmed contradictions become `ConflictRecord` objects you can inspect, act on, or queue for review.

**Health scoring.** `await eng.health(agent_id)` returns a snapshot with signals like `contradiction_score`, `freshness_score`, and `confidence_accuracy_gap`. Useful for dashboards or for deciding when to run consolidation.

**Consolidation.** `await eng.consolidate(agent_id)` reads all pending conflicts and plans a batch of actions: supersede the outdated memory, merge duplicates, or flag uncertain cases for a human. Then it executes them.

**Provenance.** Memories can carry a `ProvenanceRecord` — where it came from, who ingested it, what it was derived from. `await eng.export_provenance_json(agent_id, memory_id)` gives you a compliance-ready audit trail.

## LangChain bridge

Drop-in `VectorStore` and `BaseChatMessageHistory` backed by Engram. Bulk adds skip per-document detection — call `scan_contradictions()` after loading to catch conflicts across the batch.

```python
from engram.integrations.langchain import EngramVectorStore, EngramChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

store = EngramVectorStore(eng, embeddings=OpenAIEmbeddings(), agent_id="bot")
await store.aadd_texts(["Refund policy is 30 days", "Refund policy changed to 14 days"])
await eng.scan_contradictions("bot")  # detect conflicts across the batch
docs = await store.asimilarity_search("what is the refund policy", k=3)
# docs[0].metadata["_memory_id"] lets you trace back to the original Memory

history = EngramChatMessageHistory(eng, session_id="conv-42")
await history.aadd_messages([
    HumanMessage(content="What's the refund policy?"),
    AIMessage(content="It's 14 days."),
])
msgs = await history.aget_messages()
```

## Backends

| Backend | Install |
|---|---|
| In-memory (built-in, good for tests) | — |
| Qdrant | `pip install "engram[qdrant]"` |
| Chroma | `pip install "engram[chroma]"` |
| pgvector | `pip install "engram[pgvector]"` |

Every backend is a subclass of `AbstractAdapter`. Adding your own takes one file.

## Installation

```bash
pip install engram                   # core + in-memory adapter
pip install "engram[qdrant]"         # + Qdrant
pip install "engram[chroma]"         # + Chroma
pip install "engram[pgvector]"       # + pgvector (requires asyncpg)
pip install "engram[langchain]"      # + LangChain bridge
pip install "engram[all]"            # everything
```

Requires Python 3.11+. Engram is fully async.

## Benchmark

We ran MemoryEval's 50 deterministic test cases across five backends — four Engram-backed adapters and one raw Qdrant wrapper with no data model.

| Backend | Score | Risk | Pass |
|---|---|---|---|
| engram-inmemory-adapter | 0.88 | LOW | 44/50 |
| engram-qdrant-adapter | 0.88 | LOW | 44/50 |
| engram-chroma-adapter | 0.88 | LOW | 44/50 |
| engram-pgvector-adapter | 0.88 | LOW | 44/50 |
| naive-qdrant (no data model) | 0.42 | CRITICAL | 20/50 |

**+46 percentage points** overall reliability gap. Score is identical across all four Engram backends — reliability comes from the data model, not the choice of vector backend.

The largest single contributor is temporal reliability: Engram scores **1.00**, naive Qdrant scores **0.05**. Without lifecycle tracking, superseded facts stay active, update history is lost, and access patterns are invisible to the retriever.

To reproduce:

```bash
python benchmark/run_track1.py   # runs all 5 adapters (~2 min, no API key needed)
python benchmark/report.py       # prints the table above
```

See [`benchmark/README.md`](benchmark/README.md) for full setup and Docker requirements.

## Status

`0.1.0-alpha` — the core reliability loop (store → detect → score → consolidate → provenance) is complete and covered by 880+ unit tests. The pgvector adapter and LangChain bridge are included.

Not production-tested yet. The API is stable but may have breaking changes before 1.0.

What's planned for 0.2: LlamaIndex bridge, a sync facade for non-async code, and OpenTelemetry instrumentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help build it.

## License

MIT
