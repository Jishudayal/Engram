"""TestCase — abstract base class for all MemoryEval benchmark test cases."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from memoryeval.types import BenchmarkCategory

if TYPE_CHECKING:
    # AbstractAdapter is only needed for type checking; importing it at runtime
    # would make memoryeval hard-depend on engram. At runtime the adapter is
    # whatever duck-typed object the scorer passes in.
    from memnotary.adapters.base import AbstractAdapter


__all__ = ["TestCase"]


class TestCase(abc.ABC):
    """A single benchmark test case.

    Subclass this, declare the three required class attributes, and implement
    the three abstract methods. The scorer calls them in order:

        await case.setup(adapter)          # store test fixtures
        result = await case.run(adapter)   # execute the scenario
        score  = case.score(result)        # compute 0.0–1.0 reliability score
        await case.teardown(adapter)       # clean up (default deletes all fixtures)

    Each case gets a dedicated ``agent_id`` derived from its class name, so
    cases running against the same adapter are fully isolated from each other.

    Minimal example::

        class NewerPolicyRanksFirst(TestCase):
            category    = BenchmarkCategory.TEMPORAL
            name        = "newer_policy_ranks_first"
            description = "A fresher policy memory should outrank a stale one"

            async def setup(self, adapter):
                await adapter.store(Memory(agent_id=self.agent_id, text="30 days", ...))
                await adapter.store(Memory(agent_id=self.agent_id, text="60 days", ...))

            async def run(self, adapter):
                return await adapter.search(self.agent_id, query_vec, top_k=2)

            def score(self, results):
                return 1.0 if "60 days" in results[0].memory.text else 0.0
    """

    # --- Required class attributes (enforced at subclass definition time) ---
    category: BenchmarkCategory
    name: str
    description: str

    # Minimum score to count as passing. Override per case if needed.
    pass_threshold: float = 0.7

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip enforcement for explicit abstract intermediaries — classes that
        # declare abc.ABC in their own bases haven't finished being defined yet.
        if abc.ABC in cls.__bases__:
            return
        for attr in ("category", "name", "description"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")

    @property
    def agent_id(self) -> str:
        """Unique agent namespace for this test case's fixtures.

        Combines ``__module__`` and ``__qualname__`` so two classes with the
        same name in different modules (e.g. ``temporal.FactUpdate`` and
        ``contradiction.FactUpdate``) always get distinct fixture namespaces.
        """
        cls = type(self)
        key = f"{cls.__module__}.{cls.__qualname__}".lower().replace(".", "_")
        return f"memoryeval_{key}"

    # --- Abstract methods ---

    @abc.abstractmethod
    async def setup(self, adapter: AbstractAdapter) -> None:
        """Store test fixtures into the adapter before the scenario runs."""

    @abc.abstractmethod
    async def run(self, adapter: AbstractAdapter) -> Any:
        """Execute the test scenario; return raw result data for score()."""

    @abc.abstractmethod
    def score(self, result: Any) -> float:
        """Compute a reliability score in [0.0, 1.0] from the raw result."""

    # --- Default implementations ---

    async def teardown(self, adapter: AbstractAdapter) -> None:
        """Delete all memories stored under ``self.agent_id``.

        Override this if your test case stores memories under multiple agent
        IDs or requires a custom cleanup strategy.
        """
        memories = await adapter.list_all(self.agent_id, limit=1000)
        if memories:
            await adapter.delete_batch(self.agent_id, [m.memory_id for m in memories])
