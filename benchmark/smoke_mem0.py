"""
Substep 2.1.2 — Mem0 smoke test / diagnostic script.

Confirms that Mem0 is wired correctly before running Track 2 behavioral tests.
Not part of the scored benchmark.

Usage:
    OPENAI_API_KEY=sk-... python benchmark/smoke_mem0.py
"""

from __future__ import annotations

import os
import sys

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY env var not set.")
    sys.exit(1)

from mem0 import Memory

from mem0_config import build_mem0_config

config = build_mem0_config("mem0_smoke")
print("Initializing Mem0...")
m = Memory.from_config(config)
print("  OK")

print("\nCalling memory.add()...")
messages = [
    {"role": "user", "content": "The refund policy is 30 days for all products."},
    {"role": "assistant", "content": "Got it, I'll remember that."},
    {
        "role": "user",
        "content": "Actually, the refund window was extended to 60 days last quarter.",
    },
]
result = m.add(messages, user_id="smoke_test")
print(f"  add() returned {len(result.get('results', []))} extracted memories:")
for r in result.get("results", []):
    print(f"    [{r.get('event', '?')}] {r.get('memory', '')}")

print("\nCalling memory.search()...")
hits = m.search("refund policy", filters={"user_id": "smoke_test"})
print(f"  search() returned {len(hits.get('results', []))} results:")
for h in hits.get("results", []):
    print(f"    score={h.get('score', 0):.4f}  {h.get('memory', '')}")

print("\nDone — Mem0 is operational.")
