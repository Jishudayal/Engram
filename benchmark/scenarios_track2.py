"""
Track 2 — Behavioral contradiction scenarios.

Five scenarios that expose reliability gaps between raw vector search and
Engram's contradiction-detection layer. Each scenario runs a small sequence
of facts through a system, then scores the resulting retrieval behavior on
three dimensions:

  Correctness  (weight 0.4) — the latest/correct fact is the top recommended result
  Signal       (weight 0.4) — the system flags or suppresses stale/contradicting facts
  Preservation (weight 0.2) — correct/independent facts are not lost from storage

Composite score = 0.4 * correctness + 0.4 * signal + 0.2 * preservation

Scoring conventions
-------------------
Correctness: 0.0 or 1.0 (binary per scenario; averaged over multi-fact scenarios).
Signal:
  1.0  — stale hit absent from results, OR has conflict_flag=True
  0.5  — stale hit demoted (not recommended) but no explicit conflict_flag
  0.0  — stale hit returned and recommended with no flag
  For non-contradiction scenarios (B2, B4):
    1.0  — no false-positive flags
    0.5  — flagged for review but not suppressed (acceptable for B4)
    0.0  — incorrectly flagged AND suppressed
Preservation: fraction of expected facts that are still retrievable.

Scenario design
---------------
Each scenario uses the ``namespace`` argument as the tenant key so runs
across systems do not cross-contaminate.

B1 — Direct contradiction:        add old fact then new fact; check retrieval.
B2 — Retention (no loss):         add three independent facts; check all retained.
B3 — Three-version chain:         add v1→v2→v3; check only v3 is recommended.
B4 — False positive guard:        add two non-contradictory facts; check no flags.
B5 — Temporal language (implicit): update a scheduled event; check current version.
B6 — Temporal language (explicit): date-cued policy update; check current version.
B7 — Metadata created_at order:   inverted storage order; only created_at reveals truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavioral_system import BehavioralHit, BehavioralSystem


@dataclass
class ScenarioResult:
    scenario: str
    system: str
    correctness: float  # 0.0–1.0: latest/correct fact is recommended
    signal: float  # 0.0–1.0: stale facts flagged or suppressed
    preservation: float  # 0.0–1.0: correct/independent facts not lost
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Weighted composite: 40% correctness + 40% signal + 20% preservation."""
        return 0.4 * self.correctness + 0.4 * self.signal + 0.2 * self.preservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contains(hits: list[BehavioralHit], snippet: str) -> BehavioralHit | None:
    """Return the first hit whose text contains ``snippet`` (case-insensitive)."""
    s = snippet.lower()
    return next((h for h in hits if s in h.text.lower()), None)


def _stale_signal(hit: BehavioralHit | None) -> float:
    """Signal score for a hit that should be flagged or suppressed as stale.

    1.0  — absent from results (best case) OR has conflict_flag=True (explicitly detected)
    0.5  — demoted (not recommended) but no explicit conflict_flag (could be cosine ranking)
    0.0  — returned as recommended with no conflict flag (contradiction missed)
    """
    if hit is None:
        return 1.0
    if hit.conflict_flag:
        return 1.0
    if not hit.recommended:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# B1 — Direct contradiction
# Facts: an old policy, then a superseding one.
# Correctness: new fact is recommended.
# Signal:      old fact is flagged or suppressed.
# Preservation: new fact is present in results.
# ---------------------------------------------------------------------------


async def scenario_b1(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Refund window: 30 days → 60 days.  Only the 60-day version should surface."""
    ns = "b1_refund"
    results = []

    old_fact = "Company refund policy: all standard purchases have a 30-day return window."
    new_fact = "Company refund policy: all standard purchases have a 60-day return window."
    query = "company refund policy return window"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, old_fact)
        await sys.add_fact(ns, new_fact)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)
        new_hit = _contains(hits, "60-day")
        old_hit = _contains(hits, "30-day")

        correctness = 1.0 if (new_hit is not None and new_hit.recommended) else 0.0
        signal = _stale_signal(old_hit)
        preservation = 1.0 if new_hit is not None else 0.0

        notes = [
            f"correctness={correctness}  new_recommended={new_hit is not None and new_hit.recommended}",
            f"signal={signal}  old_conflict_flag={old_hit.conflict_flag if old_hit else 'absent'}",
            f"preservation={preservation}",
        ]
        results.append(
            ScenarioResult("B1-contradiction", sys.name, correctness, signal, preservation, notes)
        )

    return results


# ---------------------------------------------------------------------------
# B2 — Retention (no false deduplication)
# Three independent facts on unrelated topics.
# Correctness: all three are searchable.
# Signal:      no false-positive conflict flags.
# Preservation: all three retained in storage.
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

        found = 0
        all_hits: list[BehavioralHit] = []
        for snippet, fact in zip(snippets, facts, strict=True):
            hits = await sys.search(ns, fact, top_k=3)
            all_hits.extend(hits)
            if _contains(hits, snippet) is not None:
                found += 1

        correctness = found / len(facts)
        preservation = min(count / len(facts), 1.0)

        # Signal: independent facts should not be conflict-flagged.
        # A false-positive suppression (flagged and not recommended) is the worst case.
        any_suppressed = any(h.conflict_flag and not h.recommended for h in all_hits)
        any_flagged = any(h.conflict_flag for h in all_hits)
        if any_suppressed:
            signal = 0.0
        elif any_flagged:
            signal = 0.5  # flagged for review but not suppressed
        else:
            signal = 1.0

        notes = [
            f"correctness={correctness:.2f}  searchable={found}/{len(facts)}",
            f"signal={signal}  any_flagged={any_flagged}  any_suppressed={any_suppressed}",
            f"preservation={preservation:.2f}  retained={count}/{len(facts)}",
        ]
        results.append(
            ScenarioResult("B2-retention", sys.name, correctness, signal, preservation, notes)
        )

    return results


# ---------------------------------------------------------------------------
# B3 — Three-version temporal chain
# v1 → v2 → v3 of the same fact.  Only v3 should be recommended.
# Correctness: v3 is recommended.
# Signal:      v1 and v2 are flagged or suppressed (average).
# Preservation: v3 is present in results.
# ---------------------------------------------------------------------------


async def scenario_b3(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """SLA: 99.5% → 99.9% → 99.7%.  Only the 99.7% version should be recommended."""
    ns = "b3_sla"
    results = []

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

        correctness = 1.0 if (v3_hit is not None and v3_hit.recommended) else 0.0
        signal = (_stale_signal(v1_hit) + _stale_signal(v2_hit)) / 2
        preservation = 1.0 if v3_hit is not None else 0.0

        notes = [
            f"correctness={correctness}  v3_recommended={v3_hit is not None and v3_hit.recommended}",
            f"signal={signal:.2f}  v1={_stale_signal(v1_hit):.1f}  v2={_stale_signal(v2_hit):.1f}",
            f"preservation={preservation}",
        ]
        results.append(
            ScenarioResult("B3-temporal-chain", sys.name, correctness, signal, preservation, notes)
        )

    return results


# ---------------------------------------------------------------------------
# B4 — False positive guard
# Two facts on related topics that are NOT contradictory.
# Correctness: both are recommended (no over-suppression).
# Signal:      no incorrect conflict detection (flagged-for-review is 0.5, not 0).
# Preservation: both facts are retrievable.
# ---------------------------------------------------------------------------


async def scenario_b4(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Software (30-day refund) vs hardware (14-day return) — separate categories, not contradictory."""
    ns = "b4_falsepositives"
    results = []

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
        correctness = 1.0 if (a_recommended and b_recommended) else 0.0

        # Signal for a false-positive scenario: reward systems that do NOT flag.
        # Flagged-but-still-recommended counts as 0.5 (review queue but not suppressed).
        a_flagged = hit_a is not None and hit_a.conflict_flag
        b_flagged = hit_b is not None and hit_b.conflict_flag
        any_suppressed = (hit_a is not None and not hit_a.recommended) or (
            hit_b is not None and not hit_b.recommended
        )
        if any_suppressed:
            signal = 0.0
        elif a_flagged or b_flagged:
            signal = 0.5
        else:
            signal = 1.0

        preservation = ((hit_a is not None) + (hit_b is not None)) / 2.0

        notes = [
            f"correctness={correctness}  software_rec={a_recommended}  hardware_rec={b_recommended}",
            f"signal={signal}  software_flag={a_flagged}  hardware_flag={b_flagged}",
            f"preservation={preservation}",
        ]
        results.append(
            ScenarioResult("B4-false-positive", sys.name, correctness, signal, preservation, notes)
        )

    return results


# ---------------------------------------------------------------------------
# B5 — Temporal language update
# An event is rescheduled.  Only the new schedule should surface.
# Correctness: new schedule is recommended.
# Signal:      old schedule is flagged or suppressed.
# Preservation: new schedule is present in results.
# ---------------------------------------------------------------------------


async def scenario_b5(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """All-hands moved from Wednesday to Thursday — only Thursday should be recommended."""
    ns = "b5_schedule"
    results = []

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

        correctness = 1.0 if (new_hit is not None and new_hit.recommended) else 0.0
        signal = _stale_signal(old_hit)
        preservation = 1.0 if new_hit is not None else 0.0

        notes = [
            f"correctness={correctness}  thursday_recommended={new_hit is not None and new_hit.recommended}",
            f"signal={signal}  wednesday_flag={old_hit.conflict_flag if old_hit else 'absent'}",
            f"preservation={preservation}",
        ]
        results.append(
            ScenarioResult(
                "B5-temporal-language", sys.name, correctness, signal, preservation, notes
            )
        )

    return results


# ---------------------------------------------------------------------------
# B6 — Explicit temporal language in text
# Two facts about the same policy where the text itself carries date cues.
# Correctness: newer (March) version is recommended.
# Signal:      older (January) version is flagged or suppressed.
# Preservation: newer version is present in results.
#
# Unlike B1/B5 (structurally identical sentences, value-only change), B6
# embeds explicit temporal markers ("As of January / As of March, updated").
# This tests whether the LLM classifier picks up text-based temporal cues,
# and whether systems like Mem0 that rely on LLM extraction handle dated
# policy language correctly.
# ---------------------------------------------------------------------------


async def scenario_b6(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Return policy: As of January (30 days) → As of March (14 days)."""
    ns = "b6_temporal_language"
    results = []

    old_fact = "As of January, the standard return window for all purchases is 30 days."
    new_fact = (
        "As of March, the return policy was updated: purchases now have a 14-day return window."
    )
    query = "return policy refund window current"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, old_fact)
        await sys.add_fact(ns, new_fact)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)
        new_hit = _contains(hits, "14-day")
        old_hit = _contains(hits, "30 days")

        correctness = 1.0 if (new_hit is not None and new_hit.recommended) else 0.0
        signal = _stale_signal(old_hit)
        preservation = 1.0 if new_hit is not None else 0.0

        notes = [
            f"correctness={correctness}  march_recommended={new_hit is not None and new_hit.recommended}",
            f"signal={signal}  january_flag={old_hit.conflict_flag if old_hit else 'absent'}",
            f"preservation={preservation}",
        ]
        results.append(
            ScenarioResult(
                "B6-temporal-language-explicit", sys.name, correctness, signal, preservation, notes
            )
        )

    return results


# ---------------------------------------------------------------------------
# B7 — Metadata created_at ordering (no temporal language in text)
# Two near-identical facts where only structured created_at reveals which is current.
# Correctness: newer-by-metadata (60-day) fact is recommended.
# Signal:      older-by-metadata (90-day) fact is flagged or suppressed.
# Preservation: newer fact is present in results.
#
# Key design: the current fact (60-day) is stored FIRST in code but carries
# created_at = today. The outdated fact (90-day) is stored SECOND but carries
# created_at = 30 days ago. Native storage order therefore gives the wrong
# answer — "90-day" appears newer by insertion time. Only a system that reads
# structured created_at metadata (Engram via rule 3) gets this right.
# Systems without timestamp awareness (Naive, Mem0) will likely recommend the
# 90-day fact because it was stored last.
# ---------------------------------------------------------------------------


async def scenario_b7(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Token expiry: 60-day (current, created today) vs 90-day (outdated, created 30 days ago)."""
    from datetime import UTC, datetime, timedelta

    ns = "b7_metadata_timestamp"
    results = []

    today = datetime.now(UTC)
    thirty_days_ago = today - timedelta(days=30)

    # Stored first in code — but created_at = today (the current policy).
    new_fact = "API authentication tokens expire after 60 days of inactivity."
    # Stored second in code — but created_at = 30 days ago (the outdated policy).
    old_fact = "API authentication tokens expire after 90 days of inactivity."
    query = "API token authentication expiry inactivity"

    for sys in systems:
        await sys.reset(ns)
        await sys.add_fact(ns, new_fact, created_at=today)
        await sys.add_fact(ns, old_fact, created_at=thirty_days_ago)
        await sys.consolidate(ns)

        hits = await sys.search(ns, query)
        new_hit = _contains(hits, "60 days")
        old_hit = _contains(hits, "90 days")

        correctness = 1.0 if (new_hit is not None and new_hit.recommended) else 0.0
        signal = _stale_signal(old_hit)
        preservation = 1.0 if new_hit is not None else 0.0

        notes = [
            f"correctness={correctness}  60day_recommended={new_hit is not None and new_hit.recommended}",
            f"signal={signal}  90day_flag={old_hit.conflict_flag if old_hit else 'absent'}",
            f"preservation={preservation}",
            "storage_order=new_first  created_at=new:today  old:30d_ago",
        ]
        results.append(
            ScenarioResult(
                "B7-metadata-timestamp", sys.name, correctness, signal, preservation, notes
            )
        )

    return results


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

ALL_SCENARIOS = [
    scenario_b1,
    scenario_b2,
    scenario_b3,
    scenario_b4,
    scenario_b5,
    scenario_b6,
    scenario_b7,
]


async def run_all(systems: list[BehavioralSystem]) -> list[ScenarioResult]:
    """Run every scenario against every system. Caller owns lifecycle (close)."""
    all_results: list[ScenarioResult] = []
    for scenario_fn in ALL_SCENARIOS:
        results = await scenario_fn(systems)
        all_results.extend(results)
    return all_results
