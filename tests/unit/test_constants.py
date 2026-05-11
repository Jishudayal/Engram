"""Tests for engram.core.constants.

These tests lock the string values of every enum member and the numeric
ranges of threshold constants. They exist to make breaking changes visible:
renaming an enum value or shifting a threshold is a contract change that
affects stored data and downstream integrations.
"""

from engram.core.constants import (
    AUTO_FLAG_THRESHOLD,
    AUTO_MERGE_THRESHOLD,
    CLUSTER_SIMILARITY_THRESHOLD,
    CONTRADICTION_CONFIDENCE_THRESHOLD,
    DEFAULT_EXPIRATION_DAYS,
    DEFAULT_IMPORTANCE,
    ActionType,
    ConflictType,
    ConsolidationTier,
    MemoryStatus,
    MemoryType,
    ResolutionStatus,
    RiskLevel,
)


class TestMemoryType:
    def test_members_frozen(self) -> None:
        assert {m.value for m in MemoryType} == {"conversation", "skill", "custom"}

    def test_f_string_interpolation(self) -> None:
        assert f"{MemoryType.CONVERSATION}" == "conversation"


class TestMemoryStatus:
    def test_members_frozen(self) -> None:
        assert {m.value for m in MemoryStatus} == {
            "active",
            "superseded",
            "flagged",
            "archived",
            "pending_review",
        }

    def test_f_string_interpolation(self) -> None:
        assert f"{MemoryStatus.ACTIVE}" == "active"


class TestConflictType:
    def test_members_frozen(self) -> None:
        assert {m.value for m in ConflictType} == {
            "direct_contradiction",
            "temporal_supersession",
            "partial_conflict",
        }


class TestResolutionStatus:
    def test_members_frozen(self) -> None:
        assert {m.value for m in ResolutionStatus} == {
            "pending",
            "auto_resolved",
            "human_reviewed",
        }


class TestActionType:
    def test_members_frozen(self) -> None:
        assert {m.value for m in ActionType} == {"merge", "supersede", "chain_supersede", "flag", "archive"}


class TestConsolidationTier:
    def test_members_frozen(self) -> None:
        assert {m.value for m in ConsolidationTier} == {
            "auto_merge",
            "auto_flag",
            "human_review",
        }

    def test_from_confidence_auto_merge(self) -> None:
        assert ConsolidationTier.from_confidence(0.95) is ConsolidationTier.AUTO_MERGE
        assert (
            ConsolidationTier.from_confidence(AUTO_MERGE_THRESHOLD) is ConsolidationTier.AUTO_MERGE
        )

    def test_from_confidence_auto_flag(self) -> None:
        assert ConsolidationTier.from_confidence(0.85) is ConsolidationTier.AUTO_FLAG
        assert ConsolidationTier.from_confidence(AUTO_FLAG_THRESHOLD) is ConsolidationTier.AUTO_FLAG

    def test_from_confidence_human_review(self) -> None:
        assert ConsolidationTier.from_confidence(0.50) is ConsolidationTier.HUMAN_REVIEW
        assert ConsolidationTier.from_confidence(0.0) is ConsolidationTier.HUMAN_REVIEW

    def test_from_confidence_boundaries(self) -> None:
        """Values just below each threshold cross into the lower tier."""
        assert (
            ConsolidationTier.from_confidence(AUTO_MERGE_THRESHOLD - 0.01)
            is ConsolidationTier.AUTO_FLAG
        )
        assert (
            ConsolidationTier.from_confidence(AUTO_FLAG_THRESHOLD - 0.01)
            is ConsolidationTier.HUMAN_REVIEW
        )


class TestRiskLevel:
    def test_members_frozen(self) -> None:
        assert {m.value for m in RiskLevel} == {"low", "medium", "high", "critical"}


class TestThresholds:
    def test_threshold_ordering(self) -> None:
        """AUTO_FLAG < AUTO_MERGE, both within (0, 1)."""
        assert 0.0 < AUTO_FLAG_THRESHOLD < AUTO_MERGE_THRESHOLD < 1.0

    def test_cluster_threshold_in_range(self) -> None:
        assert 0.0 < CLUSTER_SIMILARITY_THRESHOLD < 1.0

    def test_cluster_threshold_below_merge(self) -> None:
        """Clustering must be more permissive than auto-merge."""
        assert CLUSTER_SIMILARITY_THRESHOLD < AUTO_MERGE_THRESHOLD

    def test_contradiction_threshold_in_range(self) -> None:
        assert 0.0 < CONTRADICTION_CONFIDENCE_THRESHOLD < 1.0

    def test_default_importance_in_range(self) -> None:
        assert 0.0 <= DEFAULT_IMPORTANCE <= 1.0

    def test_default_expiration_is_none(self) -> None:
        """No expiration applied unless caller sets one explicitly."""
        assert DEFAULT_EXPIRATION_DAYS is None
