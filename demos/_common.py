"""Shared utilities for MemNotary demo recording scripts."""

from __future__ import annotations

import json
import time

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def pause(seconds: float = 0.5) -> None:
    time.sleep(seconds)

def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    pause(0.4)

def cmd(text: str, pre: float = 0.3, post: float = 0.6) -> None:
    """Print a command line the way a human would type it, then pause."""
    pause(pre)
    print(f"{DIM}>>>{RESET} {text}")
    pause(post)

# ---------------------------------------------------------------------------
# Embeddings
#
# 4-dimensional unit vectors chosen so that:
#   cos_sim(REFUND_30, REFUND_14)  = 0.951  → above 0.82 cluster threshold
#   cos_sim(QUERY,     REFUND_14)  = 0.979  → 14-day policy ranks 1st
#   cos_sim(QUERY,     REFUND_30)  = 0.870  → 30-day policy ranks 2nd
# QUERY is intentionally distinct from REFUND_14 so similarity reads as 0.98
# rather than a suspicious 1.00.
# ---------------------------------------------------------------------------
EMBED_REFUND_30  = [1.000, 0.000, 0.0, 0.0]   # "Refund policy is 30 days"
EMBED_REFUND_14  = [0.951, 0.309, 0.0, 0.0]   # "Refund policy changed to 14 days"
EMBED_QUERY      = [0.870, 0.492, 0.0, 0.0]   # "what is our refund policy?"

# ---------------------------------------------------------------------------
# Mock LLM for ContradictionDetector
#
# Returns the exact JSON format the detector parser expects.
# Always classifies the refund pair as a direct contradiction.
# No API key, no latency, no cost — reliable retakes.
# ---------------------------------------------------------------------------
async def mock_llm(prompt: str) -> str:
    """Drop-in LLMClassifyFn that always returns a direct contradiction verdict."""
    return json.dumps({
        "verdict":    "direct_contradiction",
        "confidence": 0.95,
        "summary":    "Policy changed from 30 days to 14 days",
    })


async def mock_consolidate_llm(prompt: str) -> str:
    """Drop-in LLMConsolidateFn that always returns a high-confidence merge action."""
    return json.dumps({
        "action":     "merge",
        "confidence": 0.95,
        "reasoning":  "Policy changed from 30 to 14 days",
    })
