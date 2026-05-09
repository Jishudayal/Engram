"""
LangChain retriever backed by Engram.

Scenario: A documentation bot ingests product FAQs through LangChain's
standard VectorStore API. The pricing FAQ gets updated mid-way through
ingestion. Without Engram, both versions sit in the store and the bot
gives inconsistent answers. Engram catches the conflict and resolves it.

What this shows:
  - EngramVectorStore is a drop-in LangChain VectorStore
  - Bulk ingestion via aadd_texts (standard LangChain pattern)
  - scan_contradictions() catches conflicts across a batch
  - Consolidation resolves them; retrieval becomes consistent

Run:
    pip install "engram[langchain]"
    python examples/langchain_retriever.py

No API key needed — this example uses stub LLM and embedding functions.
"""

import asyncio
import json

from langchain_core.embeddings import Embeddings

from engram import (
    Consolidator,
    ContradictionDetector,
    Engram,
    InMemoryAdapter,
    MemoryStatus,
)
from engram.integrations.langchain import EngramVectorStore


# ---------------------------------------------------------------------------
# Stubs — replace these with your real LLM and embedding model
# ---------------------------------------------------------------------------


async def my_llm(prompt: str) -> str:
    """Replace with OpenAI, Anthropic, Gemini, Ollama — anything async."""
    return json.dumps({
        "verdict": "temporal_supersession",
        "confidence": 0.95,
        "summary": "Newer pricing supersedes the outdated one.",
    })


async def my_consolidate_llm(prompt: str) -> str:
    return json.dumps({
        "action": "flag",
        "confidence": 0.80,
        "reasoning": "Requires human review.",
    })


# Topic-specific unit vectors so only same-topic docs are flagged as candidates.
# In production, a real embedding model (OpenAI, Cohere, etc.) does this automatically.
_VECS: dict[str, list[float]] = {
    "pricing":   [1.0, 0.0, 0.0, 0.0],
    "shipping":  [0.0, 1.0, 0.0, 0.0],
    "returns":   [0.0, 0.0, 1.0, 0.0],
    "support":   [0.0, 0.0, 0.0, 1.0],
}


class StubEmbeddings(Embeddings):
    """Deterministic stub that maps topic keywords to orthogonal unit vectors."""

    def _vec(self, text: str) -> list[float]:
        for topic, vec in _VECS.items():
            if topic in text.lower():
                return vec
        return [0.25, 0.25, 0.25, 0.25]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    eng = Engram(
        InMemoryAdapter(),
        detector=ContradictionDetector(llm_fn=my_llm),
        consolidator=Consolidator(llm_fn=my_consolidate_llm),
    )

    async with eng:
        store = EngramVectorStore(eng, StubEmbeddings(), agent_id="docs-bot")

        # --- Bulk ingest (standard LangChain pattern) ---

        print("Ingesting FAQ documents in two batches...\n")

        batch_1 = [
            "pricing: Pro plan is $49/month. Enterprise starts at $199/month.",
            "shipping: Standard shipping takes 3–5 business days.",
            "returns: We accept returns within 30 days of purchase.",
        ]
        await store.aadd_texts(batch_1)
        print("  Batch 1 (3 docs): shipping, returns, pricing ($49).")

        batch_2 = [
            "pricing: New pricing as of Q2 — Pro plan is now $29/month. Enterprise is $149/month.",
            "support: Contact support at support@example.com or via live chat.",
        ]
        await store.aadd_texts(batch_2)
        print("  Batch 2 (2 docs): support contact, pricing updated ($29).\n")

        # aadd_texts uses store_batch() — per-document detection is skipped.
        # scan_contradictions() detects conflicts across the full batch.
        print("  aadd_texts() skips per-document detection (batch mode).")
        print("  Calling scan_contradictions() to catch conflicts across the batch...\n")
        await eng.scan_contradictions("docs-bot")

        # --- Show detected conflicts ---

        conflicts = await eng.pending_review("docs-bot")
        print(f"Conflicts detected ({len(conflicts)} total):\n")
        for c in conflicts:
            print(f"  ⚠  [{c.conflict_type.value}]  {c.description}")
        print()

        # --- What the bot retrieves right now ---

        print("Retrieving docs for query 'pricing' (before fix):\n")
        results = await eng.search("docs-bot", _VECS["pricing"], top_k=3, score_threshold=0.5)
        for r in results:
            flag = "⚠  conflict" if r.conflict_flag else "✓  ok      "
            print(f"  [{flag}]  {r.memory.text[:70]}...")
        print()
        print("  Two conflicting prices. The bot would give inconsistent answers.\n")

        # --- Fix it ---

        print("Running consolidation...\n")
        plan = await eng.consolidate("docs-bot")
        if plan:
            print(f"  Actions taken: {len(plan.actions)}")
            print(f"  Superseded:    {len(plan.would_supersede)}  (resolved automatically)")
            print(f"  Flagged:       {len(plan.would_flag_for_review)}  (queued for human review)")
        print()

        # --- Health after ---

        health = await eng.health("docs-bot")
        print("Memory health after consolidation:\n")
        print(f"  Active documents:    {health.total_memories}")
        print(f"  Pending conflicts:   {health.conflict_count}")
        print(f"  Contradiction score: {health.contradiction_score:.2f}  (lower is better)")
        print(f"  Overall score:       {health.score:.2f}  (higher is better)")
        print()

        # --- Retrieve again ---

        print("Same query after consolidation:\n")
        results = await eng.search("docs-bot", _VECS["pricing"], top_k=3, score_threshold=0.5)
        active = [r for r in results if r.memory.status == MemoryStatus.ACTIVE]
        for r in active:
            print(f"  ✓  {r.memory.text[:72]}...")
        print()
        print("  One answer. Consistent. Correct.")
        print()
        print("  Note: store.asimilarity_search() returns LangChain Documents with")
        print("  a '_memory_id' metadata key, so you can fetch the full Engram")
        print("  Memory or SearchResult when you need conflict metadata.")


if __name__ == "__main__":
    asyncio.run(main())
