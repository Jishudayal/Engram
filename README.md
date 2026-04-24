# Engram

**Memory traces you can trust.**

Engram is the open-source memory reliability layer for production AI agents. It wraps your existing vector memory backend and adds contradiction detection, temporal health scoring, self-healing consolidation, and provenance tracking — without replacing anything you already have.

> **Note:** The API below reflects the target interface being built. See [Status](#status) for what's available today.

```python
from engram import Engram
from engram.adapters import QdrantAdapter
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
memory = Engram(
    backend=QdrantAdapter(client, collection="agent_memories"),
    agent_id="my-agent",
)

memory.add("Our refund policy is 30 days")
results = memory.search("refund policy")

health = memory.health()
# contradiction_score: 0.31 — you have a problem
# confidence_accuracy_gap: 0.28 — your system is confident and wrong

plan = memory.consolidate(dry_run=True)
# would_merge: 12, would_supersede: 8, estimated_improvement: +23%
```

> Engram exposes a sync API by default. Every method has an async counterpart (`add_async`, `search_async`, etc.) for use inside async frameworks.

## Why Engram

Most memory systems tell you *what* is stored. Engram tells you whether you can *trust* it.

| What exists today | What Engram adds |
|---|---|
| Memory storage + retrieval | Contradiction detection before the LLM sees it |
| Agent observability (traces, latency) | Memory health scoring (freshness, accuracy gap) |
| RAG evaluation (answer quality) | Infrastructure-level memory evaluation |
| Manual memory management | Self-healing consolidation engine |

## Installation

```bash
pip install engram                    # core
pip install "engram[qdrant]"          # with Qdrant adapter
pip install "engram[chroma]"          # with Chroma adapter
pip install memoryeval                # standalone benchmark suite
```

## Status

`0.1.0-alpha` — under active development. Not production-ready yet.

See [CONTRIBUTING.md](CONTRIBUTING.md) if you want to help build it.

## License

MIT
