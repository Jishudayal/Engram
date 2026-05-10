"""ProvenanceManifest — compliance-ready flat view of a memory and its origin.

Defined in its own module to avoid a Pydantic v2 bug where field validators
with duplicate names leak across sibling model classes defined in the same
module, causing PydanticUserError at class creation time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from engram.core.constants import MemoryStatus, SourceType
from engram.core.models import Memory, UTCDatetime

__all__ = ["ProvenanceManifest"]


class ProvenanceManifest(BaseModel):
    """Compliance-ready flat view of a memory and its provenance.

    A read-only aggregate that combines a Memory and its attached ProvenanceRecord
    into a single serialisable object. Designed for audit trails, GDPR exports,
    and compliance queries.

    Use ProvenanceManifest.from_memory(memory) to build one. Returns None when
    the memory has no attached ProvenanceRecord.

    Fields are grouped by concern:
      Identity — memory and provenance record IDs
      State    — current status, importance, and creation time
      Lineage  — supersession chain and merge provenance
      Source   — origin channel, external reference, and ingestion metadata
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    memory_id: str
    agent_id: str
    record_id: str  # ProvenanceRecord.record_id

    # --- State ---
    text: str
    status: MemoryStatus
    importance: float
    memory_created_at: UTCDatetime

    # --- Lineage ---
    supersedes: tuple[str, ...] = ()  # memories this one replaced (Memory.supersedes)
    superseded_by: str | None = None  # memory that replaced this one (Memory.superseded_by)
    derived_from: tuple[str, ...] = ()  # source memory IDs when produced by consolidation

    # --- Source ---
    source_type: SourceType
    source_id: str | None = None  # e.g. document_id, conversation_id
    ingested_by: str | None = None  # agent or user who triggered ingestion
    ingested_at: UTCDatetime
    raw_content: str | None = None  # original text before chunking/processing
    chunk_index: int | None = None  # position within a chunked document

    # --- Factory ---

    @classmethod
    def from_memory(cls, memory: Memory) -> ProvenanceManifest | None:
        """Build a manifest from a Memory. Returns None if no provenance is attached."""
        prov = memory.provenance
        if prov is None:
            return None
        return cls(
            memory_id=memory.memory_id,
            agent_id=memory.agent_id,
            record_id=prov.record_id,
            text=memory.text,
            status=memory.status,
            importance=memory.importance,
            memory_created_at=memory.created_at,
            supersedes=tuple(memory.supersedes),
            superseded_by=memory.superseded_by,
            derived_from=prov.derived_from,
            source_type=prov.source_type,
            source_id=prov.source_id,
            ingested_by=prov.ingested_by,
            ingested_at=prov.ingested_at,
            raw_content=prov.raw_content,
            chunk_index=prov.chunk_index,
        )
