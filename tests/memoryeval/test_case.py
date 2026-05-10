"""Tests for memoryeval.case — TestCase ABC."""

import pytest
from memoryeval.case import TestCase
from memoryeval.types import BenchmarkCategory

# ---------------------------------------------------------------------------
# Concrete test case implementations used across tests
# ---------------------------------------------------------------------------


class ValidCase(TestCase):
    category = BenchmarkCategory.TEMPORAL
    name = "valid_temporal_case"
    description = "A valid test case used for testing the ABC"

    async def setup(self, adapter):
        pass

    async def run(self, adapter):
        return []

    def score(self, result):
        return 1.0


class HighThresholdCase(ValidCase):
    name = "high_threshold_case"
    description = "Overrides pass_threshold"
    pass_threshold = 0.9


class AnotherValidCase(TestCase):
    category = BenchmarkCategory.CONTRADICTION
    name = "another_valid_case"
    description = "A second concrete case for isolation tests"

    async def setup(self, adapter):
        pass

    async def run(self, adapter):
        return None

    def score(self, result):
        return 0.5


# ---------------------------------------------------------------------------
# Class attribute enforcement
# ---------------------------------------------------------------------------


class TestClassAttributeEnforcement:
    def test_missing_category_raises_at_definition(self) -> None:
        with pytest.raises(TypeError, match="category"):

            class MissingCategory(TestCase):
                name = "x"
                description = "x"

                async def setup(self, adapter):
                    pass

                async def run(self, adapter):
                    return None

                def score(self, result):
                    return 0.0

    def test_missing_name_raises_at_definition(self) -> None:
        with pytest.raises(TypeError, match="name"):

            class MissingName(TestCase):
                category = BenchmarkCategory.TEMPORAL
                description = "x"

                async def setup(self, adapter):
                    pass

                async def run(self, adapter):
                    return None

                def score(self, result):
                    return 0.0

    def test_missing_description_raises_at_definition(self) -> None:
        with pytest.raises(TypeError, match="description"):

            class MissingDescription(TestCase):
                category = BenchmarkCategory.TEMPORAL
                name = "x"

                async def setup(self, adapter):
                    pass

                async def run(self, adapter):
                    return None

                def score(self, result):
                    return 0.0

    def test_valid_subclass_instantiates(self) -> None:
        case = ValidCase()
        assert case.name == "valid_temporal_case"
        assert case.category == BenchmarkCategory.TEMPORAL

    def test_abstract_methods_enforced_by_python(self) -> None:
        """Python's own ABC mechanism refuses to instantiate an incomplete class."""

        with pytest.raises(TypeError):

            class Incomplete(TestCase):
                category = BenchmarkCategory.TEMPORAL
                name = "x"
                description = "x"
                # Missing: setup, run, score

            Incomplete()  # ABC raises here

    def test_explicit_abstract_intermediary_skips_enforcement(self) -> None:
        """A class that declares abc.ABC in its own bases is an intermediary."""
        import abc

        # Should NOT raise — this is an abstract layer, not a concrete case.
        class AbstractTemporalCase(TestCase, abc.ABC):
            category = BenchmarkCategory.TEMPORAL
            # name and description intentionally missing

        assert True  # no TypeError raised


# ---------------------------------------------------------------------------
# agent_id
# ---------------------------------------------------------------------------


class TestAgentId:
    def test_starts_with_memoryeval_prefix(self) -> None:
        assert ValidCase().agent_id.startswith("memoryeval_")

    def test_contains_class_name(self) -> None:
        assert "validcase" in ValidCase().agent_id

    def test_contains_module_name(self) -> None:
        # Module path is included so same-named classes in different modules
        # don't share a fixture namespace.
        assert "test_case" in ValidCase().agent_id

    def test_different_classes_have_different_agent_ids(self) -> None:
        assert ValidCase().agent_id != AnotherValidCase().agent_id

    def test_same_class_always_same_agent_id(self) -> None:
        assert ValidCase().agent_id == ValidCase().agent_id

    def test_agent_id_is_lowercase(self) -> None:
        agent_id = ValidCase().agent_id
        assert agent_id == agent_id.lower()


# ---------------------------------------------------------------------------
# pass_threshold
# ---------------------------------------------------------------------------


class TestPassThreshold:
    def test_default_threshold(self) -> None:
        assert ValidCase().pass_threshold == 0.7

    def test_custom_threshold_inherited(self) -> None:
        assert HighThresholdCase().pass_threshold == 0.9


# ---------------------------------------------------------------------------
# teardown (default implementation)
# ---------------------------------------------------------------------------


class TestTeardown:
    async def test_teardown_deletes_all_agent_memories(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.models import Memory

        adapter = InMemoryAdapter()
        case = ValidCase()

        for text in ("first memory", "second memory", "third memory"):
            await adapter.store(Memory(agent_id=case.agent_id, text=text))

        assert await adapter.count(case.agent_id) == 3

        await case.teardown(adapter)

        assert await adapter.count(case.agent_id) == 0

    async def test_teardown_is_no_op_when_already_empty(self) -> None:
        from engram.adapters.memory import InMemoryAdapter

        adapter = InMemoryAdapter()
        case = ValidCase()

        # Should not raise even when nothing was stored
        await case.teardown(adapter)
        assert await adapter.count(case.agent_id) == 0

    async def test_teardown_only_removes_own_agent_memories(self) -> None:
        from engram.adapters.memory import InMemoryAdapter
        from engram.core.models import Memory

        adapter = InMemoryAdapter()
        case_a = ValidCase()
        case_b = AnotherValidCase()

        await adapter.store(Memory(agent_id=case_a.agent_id, text="case A memory"))
        await adapter.store(Memory(agent_id=case_b.agent_id, text="case B memory"))

        await case_a.teardown(adapter)

        assert await adapter.count(case_a.agent_id) == 0
        assert await adapter.count(case_b.agent_id) == 1
