# We measured AI memory reliability across 4 systems. The results were uncomfortable.

*A behavioral benchmark comparing Engram, Mem0, and raw Qdrant on contradiction detection, temporal reasoning, and false positive avoidance.*

---

Here is a scenario that happens silently in production every day.

Your AI agent is helping a customer. It retrieves context from memory. The right answer is there — your refund policy is 14 days, updated in March. But the old answer is also there — 30 days, from January. Both sit in the vector store. Both have high cosine similarity to the query. Your LLM sees them together and picks one. It might pick the right one. It might not. Either way, no flag was raised. No conflict was recorded. The agent just answered, and you have no way to know it was working with contradictory context.

We wanted to know how common this failure mode actually is, and which systems — if any — do something about it. So we built a benchmark and ran it.

---

## What we measured

Most memory benchmarks measure correctness: did the right answer come back? That is a necessary condition but not sufficient. An agent that returns the right answer *alongside* a contradicting wrong answer — with no flag, no signal, no warning — is not reliable. It is lucky.

We designed our benchmark around three dimensions:

**Correctness** (weight 0.4) — Did the right answer surface in the top results? Every system we tested passes this. It is the easy part.

**Signal** (weight 0.4) — Was the stale or contradicting answer flagged or suppressed? This is where systems diverge sharply. A high signal score means the system told the agent *which* answer to trust. A low signal score means both answers came back looking equally authoritative.

**Preservation** (weight 0.2) — Were unrelated facts left untouched? A system that aggressively deletes or suppresses memories to avoid contradictions would score well on signal but fail here. Every system we tested handles this correctly.

We ran two tracks.

**Track 1** is infrastructure reliability: 50 deterministic test cases across five backends, checking whether the data model correctly tracks memory lifecycle, supersession links, provenance, and conflict state. No LLM required.

**Track 2** is behavioral: 7 real-world conflict scenarios, each scored on the three dimensions above, run against three systems using live LLM extraction and real embeddings.

---

## The systems

We tested three configurations:

- **Engram** — our open-source reliability layer for AI memory. Wraps your existing vector backend; adds contradiction detection, health scoring, and consolidation.
- **Mem0** — the most widely deployed AI memory library (52K+ GitHub stars). Used default settings throughout.
- **Naive Qdrant** — raw Qdrant with no memory data model. No lifecycle tracking, no conflict detection, no supersession. The baseline for "just store vectors."

---

## Track 2 results

| System | Overall | Correctness | Signal | Risk |
|---|---|---|---|---|
| **Engram** | **0.94** | 1.00 | 0.86 | LOW |
| Mem0 | 0.77 | 1.00 | 0.43 | MEDIUM |
| Naive Qdrant | 0.71 | 1.00 | 0.29 | MEDIUM |

All three systems return the right answer. The divergence is entirely in signal.

Engram flagged or suppressed the stale/contradicting memory in **6 of 7 scenarios**. Mem0 caught **3 of 7**. Raw Qdrant caught **2 of 7**.

The signal gap between Engram and Mem0 is not about correctness — it is about whether your agent knows it is uncertain. Mem0's signal score of 0.43 means that in the majority of conflict scenarios, the stale answer surfaces alongside the fresh answer with nothing to distinguish them. Your LLM is left to resolve the ambiguity on its own, silently.

---

## Scenario by scenario

**B1 — Direct contradiction** (Engram 1.00, Mem0 0.60, Naive 0.60)

We stored two memories for the same agent: "rate limit is 100 req/s" then "rate limit is 500 req/s." A week later, we retrieved "what is the rate limit?"

All three systems returned the correct current value. But Mem0 and Naive Qdrant also returned the old value with no flag. Engram detected the conflict on store, recorded a ConflictRecord, and marked the older memory as stale at retrieval time. The agent using Engram received one recommendation, not two contradictory ones.

**B2 — Retention** (all 1.00)

Three unrelated facts about the same agent. All systems correctly preserved and retrieved all three. There is no false positive on standard multi-fact retrieval.

**B3 — Temporal chain** (Engram 1.00, Mem0 0.60, Naive 0.60)

We stored three versions of the same fact across three dates: SLA 99.5%, then 99.9%, then 99.7%. Only the latest should be recommended.

Engram's chain-aware consolidation detected all three as a connected temporal supersession cluster and resolved them in a single pass — the newest memory was kept, both predecessors were marked superseded, all pairwise conflicts were resolved. Mem0 and Naive Qdrant returned multiple versions with no ordering signal.

**B4 — False positive guard** (all 1.00)

Two memories about different topics for the same agent. "Recommended stack for machine learning projects: PyTorch" and "Recommended hardware: M2 MacBook Pro." Both are retrievable and neither is flagged as conflicting. All three systems pass this correctly.

This matters as much as the contradiction tests. An overly aggressive conflict detector that flags non-contradictions is not useful in production. All three systems avoid this failure mode.

**B5 — Temporal language** (Engram 1.00, Mem0 1.00, Naive 0.60)

"The team standup is on Wednesdays at 10am" then "Standup rescheduled to Thursdays at 2pm." We retrieved "when is the standup?"

Engram and Mem0 both handled this correctly — the old schedule was flagged, the new one recommended. Naive Qdrant returned both schedules with no ordering signal. The advantage here for both structured systems is semantic detection of the temporal update pattern.

**B6 — Lexically varied temporal** (all 0.60)

This is the uncomfortable one.

We stored "The annual planning meeting is in January" then "The annual strategy session is scheduled for March." Same underlying fact, different phrasing, explicit month markers.

All three systems scored 0.60. None flagged the conflict.

The root cause for Engram: the two sentences are phrased differently enough that their cosine similarity falls below the 0.82 cluster threshold. The LLM classifier is never invoked because the embedding distance suggests these are unrelated facts, not conflicting ones. Engram never sees them as a candidate pair.

This is a real architectural ceiling. Contradiction detection that depends on embedding similarity fails when the same fact is expressed with low lexical overlap. The refund policy scenario at the top of this post — if phrased as "We give 30 days to return items" and "Our return window is two weeks" — would likely fall through the same gap.

We are working on a fuzzy-match layer that operates above the embedding threshold to catch these cases. It is not in the current release.

**B7 — Metadata timestamp** (Engram 1.00, Mem0 0.60, Naive 0.60)

This one catches an easy implementation mistake.

We stored the current policy first (return window is 60 days, created today), then an old policy second (return window was 90 days, with a `created_at` timestamp 30 days in the past). The insertion order is the opposite of the semantic order.

Naive systems that rely on insertion order or recency-of-store would recommend the 90-day policy because it was stored last. Engram uses the `created_at` field from the memory's metadata — not when it was ingested — to determine which is newer. The 60-day current policy wins. Mem0 and Naive Qdrant both got this wrong.

---

## Track 1 results

| Backend | Overall | Temporal | Contradiction | Multihop | Importance | Cross-type |
|---|---|---|---|---|---|---|
| engram-inmemory | 0.88 | **1.00** | 0.90 | 1.00 | 0.50 | 1.00 |
| engram-qdrant | 0.88 | **1.00** | 0.90 | 1.00 | 0.50 | 1.00 |
| engram-chroma | 0.88 | **1.00** | 0.90 | 1.00 | 0.50 | 1.00 |
| engram-pgvector | 0.88 | **1.00** | 0.90 | 1.00 | 0.50 | 1.00 |
| naive-qdrant | 0.42 | 0.05 | 0.50 | 1.00 | 0.22 | 0.33 |

Three things stand out.

First, the score is identical across all four Engram backends. The reliability properties come from the data model, not from which vector store is underneath. Qdrant, Chroma, pgvector, and in-memory all score 0.88. You can migrate backends without touching your reliability guarantees.

Second, temporal reliability for naive Qdrant collapses to 0.05. Without lifecycle tracking — status fields, superseded_by links, created_at metadata — a vector store has no way to know which version of a fact is current. Nineteen of the twenty temporal test cases fail outright. This is not a Qdrant problem; it would happen with any vector store used naively.

Third, Engram's 6 failures are all in the importance category: ranking results by importance score, boosting by access count, surface recently-accessed memories first. These are features the current version explicitly defers. The failures are intentional, documented, and on the roadmap for 0.2.

---

## What this means in practice

The correctness-signal gap is the silent killer in production memory systems. When both the right answer and the wrong answer come back with high similarity scores and no conflict signal, your LLM faces an impossible task: it cannot know which to trust. It picks one. Sometimes it picks well. Over time, as memory grows, the odds get worse.

A few concrete implications:

**Retrieval correctness is table stakes.** Every system we tested returns the right answer. This is expected behavior, not a differentiator. If your evaluation only measures whether the correct memory surfaces, you are measuring the easy part.

**The failure mode is silent.** Mem0's MEDIUM risk rating is not because it returns wrong answers — it returns correct ones. The risk is that it returns correct and incorrect answers together, with equal weight, and your LLM has to sort it out. In most cases it will. In edge cases it will not. You will not know which situation you are in.

**The data model matters more than the backend.** The jump from naive Qdrant (0.42) to any Engram adapter (0.88) comes entirely from the data model: status fields, supersession links, conflict records, created_at ordering. None of this requires a different vector store. It requires a layer between your agent and the vector store that tracks these properties.

**No system solves lexically varied phrasing.** B6 is a hard problem. We are not claiming otherwise. Any system that relies on embedding similarity as the sole gate for conflict detection will miss rephrasing cases. The honest answer is that this is an open research problem.

---

## Run it yourself

Both benchmark tracks are in the repository and fully reproducible. Track 1 requires no API key and runs in about two minutes:

```bash
git clone https://github.com/Jishudayal/Engram
cd Engram
pip install -e ".[all]"
python benchmark/run_track1.py
python benchmark/report.py
```

Track 2 requires an OpenAI API key and a running Qdrant container:

```bash
docker run -d --name engram-qdrant-mem0 -p 6333:6333 qdrant/qdrant
OPENAI_API_KEY=sk-... python benchmark/run_track2.py
```

Full results, scenario definitions, and scoring code are in [`benchmark/`](../benchmark/). If you run this against your own stack and get different numbers, we want to know.

---

## What Engram is

Engram is an open-source reliability layer for AI memory. It wraps your existing vector backend — Qdrant, Chroma, pgvector, or in-memory — and adds contradiction detection, health scoring, automatic consolidation, and an audit trail. You do not replace your storage. You add a layer that makes it trustworthy.

```bash
pip install engram
```

The repository is at [github.com/Jishudayal/Engram](https://github.com/Jishudayal/Engram). The benchmark code, all raw results, and the full scenario definitions are included.

If you are building agents that store long-lived memories and you have not measured your memory reliability, the Track 1 benchmark will tell you where you stand in about two minutes.
