"""pgvector adapter — async, tenant-isolated via asyncpg + pgvector.

Requires the optional 'pgvector' extra::

    pip install "engram[pgvector]"

Call ``open()`` (or use ``async with``) before any other method.

Vector size is fixed at construction time; all embeddings must match it.
If the table already exists, ``open()`` validates that the stored vector
dimension matches the adapter's configuration, raising AdapterError on
mismatch to catch config drift at startup.

Null embeddings: stored as NULL in the vector column and excluded from vector
search. They appear normally in fetch() and list_all().

Tenant isolation: the primary key is UUID5 derived from (agent_id, memory_id)
— the same scheme used by the Qdrant and Chroma adapters. agent_id is also a
dedicated column for efficient server-side WHERE filtering.

Score conversion: pgvector's ``<=>`` operator returns cosine distance in
[0, 2]. This adapter converts to similarity score via
``score = max(0.0, 1.0 - distance)`` so scores are always in [0, 1].

Metadata filters (search): flat ``{key: value}`` dict matched against the
``metadata`` sub-object inside the payload JSONB column. Applied in Python
after top_k results are returned from the vector index — no server-side
JSONB path filtering.

Table layout (one table per adapter instance)::

    CREATE TABLE engram_memories (
        id          UUID         PRIMARY KEY,
        agent_id    TEXT         NOT NULL,
        memory_id   TEXT         NOT NULL,
        embedding   vector(N),               -- NULL when no embedding
        payload     JSONB        NOT NULL,   -- full Memory via memory_to_payload
        created_at  TIMESTAMPTZ  NOT NULL    -- copied out of payload for ordering
    );

    CREATE INDEX ... ON ... (agent_id, created_at);           -- list_all
    CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) -- search
        WHERE embedding IS NOT NULL;
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from engram.adapters._utils import (
    POINT_NAMESPACE,
    map_adapter_errors,
    memory_to_payload,
    payload_to_memory,
)
from engram.adapters.base import AbstractAdapter
from engram.core.constants import MemoryStatus, ResolutionStatus
from engram.core.exceptions import AdapterError, NotFoundError
from engram.core.models import ConflictRecord, Memory, SearchResult

__all__ = ["PgVectorAdapter"]

_PG_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError)

# pgvector stores the vector dimension in atttypmod as (dim + 4).
_VECTOR_TYPMOD_OFFSET = 4

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table(name: str) -> None:
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"table name {name!r} must start with a letter or underscore "
            "and contain only ASCII letters, digits, and underscores"
        )


def _row_id(agent_id: str, memory_id: str) -> uuid.UUID:
    """Deterministic primary key from (agent_id, memory_id) — shared UUID5 namespace."""
    return uuid.uuid5(POINT_NAMESPACE, f"{agent_id}\x00{memory_id}")


def _conflict_row_id(agent_id: str, conflict_id: str) -> uuid.UUID:
    """Deterministic PK for ConflictRecords — namespaced separately from memory rows."""
    return uuid.uuid5(POINT_NAMESPACE, f"conflict\x00{agent_id}\x00{conflict_id}")


def _to_float_list(v: Any) -> list[float] | None:
    """Normalise a pgvector result (numpy array, array.array, or None) to list[float]."""
    return None if v is None else [float(x) for x in v]


class PgVectorAdapter(AbstractAdapter):
    """Async PostgreSQL + pgvector adapter.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        ``"postgresql://user:pass@localhost/mydb"``.
    table:
        Table name. Must match ``^[A-Za-z_][A-Za-z0-9_]*$``.
        Defaults to ``"engram_memories"``.
    vector_size:
        Dimensionality of the embedding vectors. All embeddings stored via
        this adapter must have exactly this many dimensions.
    min_pool_size / max_pool_size:
        asyncpg connection pool bounds. Defaults: 1 / 10.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "engram_memories",
        vector_size: int,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if vector_size <= 0:
            raise ValueError(f"vector_size must be > 0, got {vector_size}")
        _validate_table(table)
        self._dsn = dsn
        self._table = table.lower()  # PostgreSQL folds unquoted identifiers to lowercase
        self._vector_size = vector_size
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    @property
    def backend_name(self) -> str:
        return "pgvector"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Open the connection pool and create/validate the memory table."""
        if self._pool is not None:
            return  # idempotent

        async def _init(conn: asyncpg.Connection) -> None:
            await register_vector(conn)
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )

        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                init=_init,
            )
        except (asyncpg.PostgresError, OSError) as exc:
            raise AdapterError(f"PgVectorAdapter.open: {exc}") from exc

        try:
            await self._setup()
        except BaseException:
            await self._pool.close()
            self._pool = None
            raise

    async def close(self) -> None:
        """Close the connection pool. Idempotent."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PgVectorAdapter:
        await self.open()
        return self

    @property
    def _p(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "PgVectorAdapter is not open — use 'async with' or call open() first"
            )
        return self._pool

    async def _setup(self) -> None:
        """Enable the vector extension; create the table + indexes on first use."""
        async with self._p.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            exists: bool = await conn.fetchval(
                "SELECT EXISTS("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_schema = current_schema() AND table_name = $1"
                ")",
                self._table,
            )

            if exists:
                # Validate that the stored vector dimension matches.
                row = await conn.fetchrow(
                    "SELECT atttypmod FROM pg_attribute"
                    "  WHERE attrelid = ($1)::regclass"
                    "    AND attname = 'embedding'"
                    "    AND NOT attisdropped",
                    self._table,
                )
                if row is not None and row["atttypmod"] != -1:
                    stored = row["atttypmod"] - _VECTOR_TYPMOD_OFFSET
                    if stored != self._vector_size:
                        raise AdapterError(
                            f"table {self._table!r} has vector_size={stored}; "
                            f"adapter configured with vector_size={self._vector_size}"
                        )
            else:
                await conn.execute(
                    f"CREATE TABLE {self._table} ("
                    f"  id          UUID         PRIMARY KEY,"
                    f"  agent_id    TEXT         NOT NULL,"
                    f"  memory_id   TEXT         NOT NULL,"
                    f"  embedding   vector({self._vector_size}),"
                    f"  payload     JSONB        NOT NULL,"
                    f"  created_at  TIMESTAMPTZ  NOT NULL"
                    f")"
                )
                await conn.execute(
                    f"CREATE INDEX {self._table}_agent_created_idx"
                    f"  ON {self._table} (agent_id, created_at)"
                )
                # HNSW index — partial: only rows that have an embedding.
                await conn.execute(
                    f"CREATE INDEX {self._table}_hnsw_idx"
                    f"  ON {self._table} USING hnsw (embedding vector_cosine_ops)"
                    f"  WHERE embedding IS NOT NULL"
                )

            # Conflicts sidecar table — always created alongside the memory table.
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._conflicts_table} ("
                f"  id           UUID         PRIMARY KEY,"
                f"  agent_id     TEXT         NOT NULL,"
                f"  conflict_id  TEXT         NOT NULL,"
                f"  payload      JSONB        NOT NULL,"
                f"  detected_at  TIMESTAMPTZ  NOT NULL"
                f")"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._conflicts_table}_agent_status_idx"
                f"  ON {self._conflicts_table} (agent_id, detected_at)"
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_PG_ERRORS)
    async def store(self, memory: Memory) -> None:
        await self._p.execute(
            f"INSERT INTO {self._table}"
            f"  (id, agent_id, memory_id, embedding, payload, created_at)"
            f"  VALUES ($1, $2, $3, $4, $5, $6)"
            f"  ON CONFLICT (id) DO UPDATE SET"
            f"    embedding = EXCLUDED.embedding,"
            f"    payload   = EXCLUDED.payload",
            _row_id(memory.agent_id, memory.memory_id),
            memory.agent_id,
            memory.memory_id,
            memory.embedding,
            memory_to_payload(memory),
            memory.created_at,
        )

    @map_adapter_errors(error=_PG_ERRORS)
    async def update(self, memory: Memory) -> None:
        tag = await self._p.execute(
            f"UPDATE {self._table}"
            f"  SET embedding = $3, payload = $4"
            f"  WHERE id = $1 AND agent_id = $2",
            _row_id(memory.agent_id, memory.memory_id),
            memory.agent_id,
            memory.embedding,
            memory_to_payload(memory),
        )
        if tag == "UPDATE 0":
            raise NotFoundError(
                f"memory {memory.memory_id!r} not found for agent {memory.agent_id!r}"
            )

    @map_adapter_errors(error=_PG_ERRORS)
    async def delete(self, agent_id: str, memory_id: str) -> bool:
        tag = await self._p.execute(
            f"DELETE FROM {self._table} WHERE id = $1 AND agent_id = $2",
            _row_id(agent_id, memory_id),
            agent_id,
        )
        return tag != "DELETE 0"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_PG_ERRORS)
    async def fetch(self, agent_id: str, memory_id: str) -> Memory | None:
        row = await self._p.fetchrow(
            f"SELECT payload, embedding"
            f"  FROM {self._table}"
            f"  WHERE id = $1 AND agent_id = $2",
            _row_id(agent_id, memory_id),
            agent_id,
        )
        if row is None:
            return None
        return payload_to_memory(row["payload"], embedding=_to_float_list(row["embedding"]))

    @map_adapter_errors(error=_PG_ERRORS)
    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if top_k == 0:
            return []

        # Push score_threshold into SQL as a distance bound so LIMIT top_k
        # applies after threshold filtering — avoids returning fewer results
        # than exist when filtering is done post-LIMIT in Python.
        sql_params: list[Any] = [query_embedding, agent_id]
        extra_where = ""
        if score_threshold is not None:
            sql_params.append(1.0 - score_threshold)  # score ≥ threshold ↔ distance ≤ 1-threshold
            extra_where = f" AND embedding <=> $1 <= ${len(sql_params)}"
        sql_params.append(top_k)

        rows = await self._p.fetch(
            f"SELECT payload, embedding,"
            f"       1.0 - (embedding <=> $1) AS score"
            f"  FROM {self._table}"
            f"  WHERE agent_id = $2 AND embedding IS NOT NULL{extra_where}"
            f"  ORDER BY embedding <=> $1"
            f"  LIMIT ${len(sql_params)}",
            *sql_params,
        )

        results: list[SearchResult] = []
        rank = 1
        for row in rows:
            score = min(1.0, max(0.0, float(row["score"])))
            memory = payload_to_memory(
                row["payload"], embedding=_to_float_list(row["embedding"])
            )
            if filters:
                meta = memory.metadata or {}
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            results.append(SearchResult(memory=memory, score=score, rank=rank))
            rank += 1
        return results

    @map_adapter_errors(error=_PG_ERRORS)
    async def list_all(
        self,
        agent_id: str,
        *,
        status: MemoryStatus | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        where = "agent_id = $1"
        params: list[Any] = [agent_id]

        if status is not None:
            params.append(status.value)
            where += f" AND payload->>'status' = ${len(params)}"

        sql = (
            f"SELECT payload, embedding"
            f"  FROM {self._table}"
            f"  WHERE {where}"
            f"  ORDER BY created_at"
        )
        if limit is not None:
            params.append(limit)
            sql += f" LIMIT ${len(params)}"
        if offset:
            params.append(offset)
            sql += f" OFFSET ${len(params)}"

        rows = await self._p.fetch(sql, *params)
        return [
            payload_to_memory(r["payload"], embedding=_to_float_list(r["embedding"]))
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Bulk overrides
    # ------------------------------------------------------------------

    @map_adapter_errors(error=_PG_ERRORS)
    async def store_batch(self, memories: list[Memory]) -> None:
        if not memories:
            return
        records = [
            (
                _row_id(m.agent_id, m.memory_id),
                m.agent_id,
                m.memory_id,
                m.embedding,
                memory_to_payload(m),
                m.created_at,
            )
            for m in memories
        ]
        await self._p.executemany(
            f"INSERT INTO {self._table}"
            f"  (id, agent_id, memory_id, embedding, payload, created_at)"
            f"  VALUES ($1, $2, $3, $4, $5, $6)"
            f"  ON CONFLICT (id) DO UPDATE SET"
            f"    embedding = EXCLUDED.embedding,"
            f"    payload   = EXCLUDED.payload",
            records,
        )

    @map_adapter_errors(error=_PG_ERRORS)
    async def fetch_batch(self, agent_id: str, memory_ids: list[str]) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        row_ids = [_row_id(agent_id, mid) for mid in memory_ids]
        rows = await self._p.fetch(
            f"SELECT payload, embedding"
            f"  FROM {self._table}"
            f"  WHERE id = ANY($1::uuid[]) AND agent_id = $2",
            row_ids,
            agent_id,
        )
        return {
            r["payload"]["memory_id"]: payload_to_memory(
                r["payload"], embedding=_to_float_list(r["embedding"])
            )
            for r in rows
        }

    @map_adapter_errors(error=_PG_ERRORS)
    async def delete_batch(self, agent_id: str, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        row_ids = [_row_id(agent_id, mid) for mid in memory_ids]
        tag = await self._p.execute(
            f"DELETE FROM {self._table}"
            f"  WHERE id = ANY($1::uuid[]) AND agent_id = $2",
            row_ids,
            agent_id,
        )
        return int(tag.split()[-1])

    @map_adapter_errors(error=_PG_ERRORS)
    async def count(self, agent_id: str, *, status: MemoryStatus | None = None) -> int:
        if status is None:
            return await self._p.fetchval(
                f"SELECT COUNT(*) FROM {self._table} WHERE agent_id = $1",
                agent_id,
            )
        return await self._p.fetchval(
            f"SELECT COUNT(*) FROM {self._table}"
            f"  WHERE agent_id = $1 AND payload->>'status' = $2",
            agent_id,
            status.value,
        )

    @map_adapter_errors(error=_PG_ERRORS)
    async def exists(self, agent_id: str, memory_id: str) -> bool:
        result = await self._p.fetchval(
            f"SELECT EXISTS("
            f"  SELECT 1 FROM {self._table} WHERE id = $1 AND agent_id = $2"
            f")",
            _row_id(agent_id, memory_id),
            agent_id,
        )
        return bool(result)

    # ------------------------------------------------------------------
    # Conflict storage — sidecar table "{table}_conflicts"
    # ------------------------------------------------------------------

    @property
    def _conflicts_table(self) -> str:
        return f"{self._table}_conflicts"

    @map_adapter_errors(error=_PG_ERRORS)
    async def store_conflict(self, conflict: ConflictRecord) -> None:
        await self._p.execute(
            f"INSERT INTO {self._conflicts_table}"
            f"  (id, agent_id, conflict_id, payload, detected_at)"
            f"  VALUES ($1, $2, $3, $4, $5)"
            f"  ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
            _conflict_row_id(conflict.agent_id, conflict.conflict_id),
            conflict.agent_id,
            conflict.conflict_id,
            conflict.model_dump(mode="json"),
            conflict.detected_at,
        )

    @map_adapter_errors(error=_PG_ERRORS)
    async def fetch_conflict(
        self, agent_id: str, conflict_id: str
    ) -> ConflictRecord | None:
        row = await self._p.fetchrow(
            f"SELECT payload FROM {self._conflicts_table}"
            f"  WHERE id = $1 AND agent_id = $2",
            _conflict_row_id(agent_id, conflict_id),
            agent_id,
        )
        if row is None:
            return None
        return ConflictRecord.model_validate(row["payload"])

    @map_adapter_errors(error=_PG_ERRORS)
    async def list_conflicts(
        self,
        agent_id: str,
        *,
        status: ResolutionStatus | None = None,
    ) -> list[ConflictRecord]:
        where = "agent_id = $1"
        params: list[Any] = [agent_id]

        if status is not None:
            params.append(status.value)
            where += f" AND payload->>'resolution_status' = ${len(params)}"

        rows = await self._p.fetch(
            f"SELECT payload FROM {self._conflicts_table}"
            f"  WHERE {where}"
            f"  ORDER BY detected_at",
            *params,
        )
        return [ConflictRecord.model_validate(r["payload"]) for r in rows]

    @map_adapter_errors(error=_PG_ERRORS)
    async def update_conflict(self, conflict: ConflictRecord) -> None:
        tag = await self._p.execute(
            f"UPDATE {self._conflicts_table}"
            f"  SET payload = $3"
            f"  WHERE id = $1 AND agent_id = $2",
            _conflict_row_id(conflict.agent_id, conflict.conflict_id),
            conflict.agent_id,
            conflict.model_dump(mode="json"),
        )
        if tag == "UPDATE 0":
            raise NotFoundError(
                f"conflict {conflict.conflict_id!r} not found for agent {conflict.agent_id!r}"
            )

    @map_adapter_errors(error=_PG_ERRORS)
    async def delete_conflict(self, agent_id: str, conflict_id: str) -> bool:
        tag = await self._p.execute(
            f"DELETE FROM {self._conflicts_table}"
            f"  WHERE id = $1 AND agent_id = $2",
            _conflict_row_id(agent_id, conflict_id),
            agent_id,
        )
        return tag != "DELETE 0"
