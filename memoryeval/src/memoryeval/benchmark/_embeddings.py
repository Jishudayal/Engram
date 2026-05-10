"""Synthetic embeddings for MemoryEval test cases.

All test cases share a fixed 16-dimensional vocabulary so fixtures are
compact, deterministic, and require no external API or embedding model.

Usage::

    from memoryeval.benchmark._embeddings import VECTOR_SIZE, cosine, vec

    # same-topic embeddings: high cosine similarity
    policy_query = vec("refund", "policy")
    policy_mem   = vec("refund", "policy", "days")

    # different-topic embeddings: near-zero cosine similarity
    api_mem = vec("api", "rate", "limit")

    # compute similarity
    sim = cosine(policy_query, policy_mem)   # → ~0.82

Known keywords and their dimensions:

    Refund / return cluster  →  refund(0)  policy(1)  return(2)  window(3)
    API / rate cluster       →  api(4)     rate(5)    limit(6)   request(7)
    Security / password      →  password(8) security(9) requirement(10) expire(11)
    Pricing / subscription   →  price(12)  plan(13)   subscription(14) days(15)

Unknown keywords are silently ignored — add to _VOCAB rather than
relying on a zero dimension contributing nothing to similarity.
"""

from __future__ import annotations

import math

__all__ = ["VECTOR_SIZE", "cosine", "vec"]

VECTOR_SIZE: int = 16

# One keyword per dimension. Topics are grouped so related terms sit in
# adjacent dimensions and produce naturally high cosine similarity.
_VOCAB: dict[str, int] = {
    # Refund / return policy cluster
    "refund":       0,
    "policy":       1,
    "return":       2,
    "window":       3,
    # API / rate limiting cluster
    "api":          4,
    "rate":         5,
    "limit":        6,
    "request":      7,
    # Security / password cluster
    "password":     8,
    "security":     9,
    "requirement":  10,
    "expire":       11,
    # Pricing / subscription cluster
    "price":        12,
    "plan":         13,
    "subscription": 14,
    "days":         15,
}


def vec(*keywords: str) -> list[float]:
    """Return a unit-normalized 16D vector for the given keywords.

    Each known keyword sets its dimension to 1.0; the result is then
    L2-normalized. Returns a zero vector if no known keywords are given.
    """
    v = [0.0] * VECTOR_SIZE
    for kw in keywords:
        idx = _VOCAB.get(kw)
        if idx is not None:
            v[idx] = 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0.0 else v


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors, clamped to [0.0, 1.0].

    Returns 0.0 for zero vectors.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    return max(0.0, min(1.0, dot / denom)) if denom > 0.0 else 0.0
