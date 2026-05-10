"""
Track 2 — Behavioral contradiction scenarios.

Five scenarios that expose reliability gaps between raw vector search and
Engram's contradiction-detection layer. Each scenario runs a small sequence
of facts through a system, then scores the resulting retrieval behavior.

Scenario design
---------------
Each scenario uses the ``namespace`` argument as the tenant key — systems use
it as collection name / user_id / agent_id so runs don't cross-contaminate.

Scoring (per scenario, per system):
  0.0  — system fails the reliability check
  0.5  — partial (e.g. one of two sub-checks passed)
  1.0  — system passes the reliability check

B1 — Direct contradiction:   add old fact then new fact; check retrieval.
B2 — Retention (no loss):    add three independent facts; check all retained.
B3 — Three-version chain:    add v1→v2→v3; check only v3 is recommended.
B4 — False positive guard:   add two non-contradictory facts; check no flags.
B5 — Temporal language:      update a scheduled event; check current version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavioral_system import BehavioralHit, BehavioralSystem


@dataclass
class ScenarioResult:
    scenario: str
    system: str
    score: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _contains(hits: list[BehavioralHit], snippet: str) -> BehavioralHit | None:
    """Return the first hit whose text contains ``snippet`` (case-insensitive)."""
    s = snippet.lower()
    return next((h for h in hits if s in h.text.lower()), None)


def _recommended_hits(hits: list[BehavioralHit]) -> list[BehavioralHit]:
    return [h for h in hits if h.recommended]


# ---------------------------------------------------------------------------
# B1 — Direct contradiction
# Facts: an old policy, then a superseding one.
# Check: (a) new fact is in recommended results; (b) old fact is NOT recommended.
# ---------------------------------------------------------------------------


async def scenario_b1(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Refund window: 30 days → 60 days.  Only the 60-day version should surface."""
    ns = "b1_refund"
    results = []

    # Structurally identical sentences; only the value differs → high cosine similarity.
    old_fact = "Company refund policy: all standard purchases have a 30-day return window."
    new_fact = "Company refund policy: all standard purchases have a 60-day return window."
    query = "company refund policy return window"
    new_snippet = "60-day"
    old_snippet = "30-day"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, old_fact)
        await sys.add_fact(ns, new_fact)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)

        new_hit = _contains(hits, new_snippet)
        old_hit = _contains(hits, old_snippet)

        new_recommended = new_hit is not None and new_hit.recommended
        # Credit: old absent, de-ranked, OR flagged as a conflict.
        old_suppressed = old_hit is None or not old_hit.recommended or (old_hit.conflict_flag)

        score = (0.5 if new_recommended else 0.0) + (0.5 if old_suppressed else 0.0)
        notes = [
            f"new_recommended={new_recommended}",
            f"old_suppressed={old_suppressed}",
            f"old_conflict_flag={old_hit.conflict_flag if old_hit else 'absent'}",
            f"total_hits={len(hits)}",
        ]
        results.append(ScenarioResult("B1-contradiction", sys.name, score, notes))

    return results


# ---------------------------------------------------------------------------
# B2 — Retention (no false deduplication)
# Three independent facts on unrelated topics.
# Check: all three are retained and retrievable.
# ---------------------------------------------------------------------------


async def scenario_b2(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Three unrelated facts — all three must remain in storage."""
    ns = "b2_retention"
    results = []

    facts = [
        "Password minimum length policy is 12 characters.",
        "The main conference room seats 20 people.",
        "API rate limit is 100 requests per minute.",
    ]
    snippets = ["12 characters", "20 people", "100 requests"]

    for sys in systems:
        await sys.reset(ns)
        for fact in facts:
            await sys.add_fact(ns, fact)
        await sys.consolidate(ns)

        count = await sys.retained_count(ns)

        # Also verify each fact is individually searchable
        found = 0
        for snippet, fact in zip(snippets, facts, strict=True):
            hits = await sys.search(ns, fact, top_k=3)
            if _contains(hits, snippet) is not None:
                found += 1

        score = min(1.0, (count / len(facts)) * 0.5 + (found / len(facts)) * 0.5)
        notes = [f"retained={count}/{len(facts)}", f"searchable={found}/{len(facts)}"]
        results.append(ScenarioResult("B2-retention", sys.name, score, notes))

    return results


# ---------------------------------------------------------------------------
# B3 — Three-version temporal chain
# v1 → v2 → v3 of the same fact.  Only v3 should be recommended.
# ---------------------------------------------------------------------------


async def scenario_b3(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """SLA: 99.5% → 99.9% → 99.7%.  Only the 99.7% version should be recommended."""
    ns = "b3_sla"
    results = []

    # Identical template; only the percentage differs → high cosine similarity across all pairs.
    v1 = "Service availability SLA: our platform guarantees 99.5% monthly uptime."
    v2 = "Service availability SLA: our platform guarantees 99.9% monthly uptime."
    v3 = "Service availability SLA: our platform guarantees 99.7% monthly uptime."
    query = "service availability SLA platform uptime"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, v1)
        await sys.add_fact(ns, v2)
        await sys.add_fact(ns, v3)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)

        v3_hit = _contains(hits, "99.7")
        v1_hit = _contains(hits, "99.5")
        v2_hit = _contains(hits, "99.9")

        v3_recommended = v3_hit is not None and v3_hit.recommended
        # Credit: stale version absent, de-ranked, OR flagged as a conflict.
        v1_suppressed = v1_hit is None or not v1_hit.recommended or (v1_hit.conflict_flag)
        v2_suppressed = v2_hit is None or not v2_hit.recommended or (v2_hit.conflict_flag)

        score = (
            (0.5 if v3_recommended else 0.0)
            + (0.25 if v1_suppressed else 0.0)
            + (0.25 if v2_suppressed else 0.0)
        )
        notes = [
            f"v3_recommended={v3_recommended}",
            f"v1_suppressed={v1_suppressed}  (conflict_flag={v1_hit.conflict_flag if v1_hit else 'absent'})",
            f"v2_suppressed={v2_suppressed}  (conflict_flag={v2_hit.conflict_flag if v2_hit else 'absent'})",
        ]
        results.append(ScenarioResult("B3-temporal-chain", sys.name, score, notes))

    return results


# ---------------------------------------------------------------------------
# B4 — False positive guard
# Two facts on related topics that are NOT contradictory.
# Check: both are returned as recommended (no over-flagging).
# ---------------------------------------------------------------------------


async def scenario_b4(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Software (30-day refund) vs hardware (14-day return) — separate categories, not contradictory."""
    ns = "b4_falsepositives"
    results = []

    # Explicit category separation reduces ambiguity for the LLM detector.
    fact_a = "Digital software products: the refund eligibility window is 30 days from purchase."
    fact_b = "Physical hardware products: the return inspection window is 14 days from delivery."
    query = "refund and return policy"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, fact_a)
        await sys.add_fact(ns, fact_b)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)

        hit_a = _contains(hits, "software")
        hit_b = _contains(hits, "hardware")

        a_recommended = hit_a is not None and hit_a.recommended
        b_recommended = hit_b is not None and hit_b.recommended

        score = (0.5 if a_recommended else 0.0) + (0.5 if b_recommended else 0.0)
        notes = [
            f"software_recommended={a_recommended}",
            f"hardware_recommended={b_recommended}",
        ]
        results.append(ScenarioResult("B4-false-positive", sys.name, score, notes))

    return results


# ---------------------------------------------------------------------------
# B5 — Temporal language update
# An event is rescheduled.  Only the new schedule should surface.
# ---------------------------------------------------------------------------


async def scenario_b5(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """All-hands moved from Wednesday to Thursday — only Thursday should be recommended."""
    ns = "b5_schedule"
    results = []

    # Identical template; only day and time differ → high cosine similarity.
    old_fact = "All-hands meeting schedule: the weekly company sync is every Wednesday at 3:00 PM."
    new_fact = "All-hands meeting schedule: the weekly company sync is every Thursday at 2:00 PM."
    query = "all-hands meeting schedule weekly company sync"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, old_fact)
        await sys.add_fact(ns, new_fact)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)

        new_hit = _contains(hits, "thursday")
        old_hit = _contains(hits, "wednesday")

        new_recommended = new_hit is not None and new_hit.recommended
        # Credit: old absent, de-ranked, OR flagged as a conflict.
        old_suppressed = old_hit is None or not old_hit.recommended or (old_hit.conflict_flag)

        score = (0.5 if new_recommended else 0.0) + (0.5 if old_suppressed else 0.0)
        notes = [
            f"thursday_recommended={new_recommended}",
            f"wednesday_suppressed={old_suppressed}",
            f"wednesday_conflict_flag={old_hit.conflict_flag if old_hit else 'absent'}",
        ]
        results.append(ScenarioResult("B5-temporal-language", sys.name, score, notes))

    return results


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

ALL_SCENARIOS = [scenario_b1, scenario_b2, scenario_b3, scenario_b4, scenario_b5]


async def run_all(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Run every scenario against every system. Caller owns lifecycle (close)."""
    all_results: list[ScenarioResult] = []
    for scenario_fn in ALL_SCENARIOS:
        results = await scenario_fn(systems)
        all_results.extend(results)
    return all_results
