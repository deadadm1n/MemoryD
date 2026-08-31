from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .models import Memory


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class BrainStore:
    """SQLite persistence. The runtime, rather than an agent, owns this layer."""

    def __init__(self, database: str | Path = "brain.db") -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def connection(self) -> sqlite3.Connection:
        """Return one configured connection per thread instead of reopening on every call."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.connection = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    def _initialize(self) -> None:
        conn = self.connection()
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    importance REAL NOT NULL CHECK(importance >= 0 AND importance <= 1),
                    strength REAL NOT NULL CHECK(strength >= 0 AND strength <= 1),
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, content);
                CREATE TABLE IF NOT EXISTS links (
                    id TEXT PRIMARY KEY,
                    from_id TEXT NOT NULL REFERENCES memories(id),
                    to_id TEXT NOT NULL REFERENCES memories(id),
                    relation TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(from_id, to_id, relation)
                );
                CREATE INDEX IF NOT EXISTS links_from_idx ON links(from_id);
                CREATE INDEX IF NOT EXISTS links_to_idx ON links(to_id);
                CREATE INDEX IF NOT EXISTS memories_active_updated_idx ON memories(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    normalized_name TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL DEFAULT 'concept',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id TEXT NOT NULL REFERENCES memories(id),
                    entity_id TEXT NOT NULL REFERENCES entities(id),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, entity_id)
                );
                CREATE INDEX IF NOT EXISTS memory_entities_entity_idx ON memory_entities(entity_id, memory_id);
                CREATE TABLE IF NOT EXISTS embeddings (
                    memory_id TEXT NOT NULL REFERENCES memories(id),
                    provider TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, provider)
                );
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    memory_id TEXT REFERENCES memories(id),
                    payload TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_created_idx ON events(created_at DESC);
                CREATE TABLE IF NOT EXISTS state_facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    memory_id TEXT NOT NULL REFERENCES memories(id),
                    is_current INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS state_facts_current_idx
                    ON state_facts(subject, state_key) WHERE is_current = 1;
                CREATE INDEX IF NOT EXISTS state_facts_history_idx
                    ON state_facts(subject, state_key, created_at DESC);
            """)
        conn.execute("INSERT OR REPLACE INTO schema_metadata(key, value) VALUES ('schema_version', '4')")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()

    @staticmethod
    def _memory(row: sqlite3.Row) -> Memory:
        values = dict(row)
        values.pop("rank", None)
        values.pop("vector", None)
        values["metadata"] = json.loads(values["metadata"])
        return Memory(**values)

    def create(self, content: str, *, kind: str, source: str, confidence: float,
               importance: float, metadata: dict[str, Any] | None = None) -> Memory:
        memory_id, timestamp = str(uuid.uuid4()), now()
        record = (memory_id, content.strip(), kind, source, confidence, importance, importance,
                  "active", timestamp, timestamp, json.dumps(metadata or {}, sort_keys=True))
        with self.connection() as conn:
            conn.execute("""INSERT INTO memories
                (id, content, kind, source, confidence, importance, strength, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", record)
            conn.execute("INSERT INTO memory_fts(id, content) VALUES (?, ?)", (memory_id, content))
            conn.commit()
        return self.get(memory_id)

    def get(self, memory_id: str) -> Memory | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._memory(row) if row else None

    def update_status(self, memory_id: str, status: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE memories SET status = ?, updated_at = ? WHERE id = ?", (status, now(), memory_id))
            conn.commit()

    def reinforce(self, memory_ids: Iterable[str]) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        with self.connection() as conn:
            conn.execute(f"""UPDATE memories SET access_count = access_count + 1,
                    accessed_at = ?, updated_at = ?, strength = MIN(1.0, strength + 0.03)
                    WHERE id IN ({marks})""", (now(), now(), *ids))
            conn.commit()

    def link(self, from_id: str, to_id: str, relation: str, metadata: dict[str, Any] | None = None) -> None:
        with self.connection() as conn:
            conn.execute("""INSERT OR IGNORE INTO links(id, from_id, to_id, relation, created_at, metadata)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                         (str(uuid.uuid4()), from_id, to_id, relation, now(), json.dumps(metadata or {})))
            conn.commit()

    def related(self, memory_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("""SELECT l.relation, l.from_id, l.to_id, m.content, m.kind
                FROM links l JOIN memories m ON m.id = CASE WHEN l.from_id = ? THEN l.to_id ELSE l.from_id END
                WHERE l.from_id = ? OR l.to_id = ? ORDER BY l.created_at DESC""",
                (memory_id, memory_id, memory_id)).fetchall()
        return [dict(row) for row in rows]

    def search_fts(self, query: str, limit: int) -> list[tuple[Memory, float]]:
        # FTS MATCH is its own small query language. Supplying only normalized tokens
        # prevents punctuation in natural-language prompts from becoming syntax.
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        terms = " OR ".join(f'"{token}"' for token in tokens)
        if not tokens:
            return []
        with self.connection() as conn:
            rows = conn.execute("""SELECT m.*, bm25(memory_fts) AS rank FROM memory_fts
                JOIN memories m ON m.id = memory_fts.id WHERE memory_fts MATCH ? AND m.status = 'active'
                ORDER BY rank LIMIT ?""", (terms, limit)).fetchall()
        return [(self._memory(row), float(row["rank"])) for row in rows]

    def recent(self, limit: int = 20, kind: str | None = None) -> list[Memory]:
        sql, params = "SELECT * FROM memories WHERE status = 'active'", []
        if kind:
            sql += " AND kind = ?"; params.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ?"; params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._memory(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute("SELECT kind, COUNT(*) AS count FROM memories WHERE status = 'active' GROUP BY kind").fetchall()
            links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        with self.connection() as conn:
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {**{row["kind"]: row["count"] for row in rows}, "memories": sum(row["count"] for row in rows), "relationships": links, "events": events}

    def find_duplicate(self, content: str) -> Memory | None:
        normalized = " ".join(content.casefold().split())
        with self.connection() as conn:
            row = conn.execute("""SELECT * FROM memories WHERE status = 'active'
                AND lower(trim(content)) = ? ORDER BY created_at DESC LIMIT 1""", (normalized,)).fetchone()
        return self._memory(row) if row else None

    def add_entities(self, memory_id: str, entities: Iterable[tuple[str, str]]) -> None:
        timestamp = now()
        with self.connection() as conn:
            for name, kind in entities:
                normalized = name.casefold()
                entity_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"memoryd:entity:{normalized}"))
                conn.execute("""INSERT OR IGNORE INTO entities(id, name, normalized_name, kind, created_at)
                    VALUES (?, ?, ?, ?, ?)""", (entity_id, name, normalized, kind, timestamp))
                conn.execute("INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, created_at) VALUES (?, ?, ?)",
                             (memory_id, entity_id, timestamp))

    def entity_related(self, memory_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("""SELECT DISTINCT m.id, m.content, m.kind, e.name AS entity
                FROM memory_entities mine
                JOIN memory_entities other ON other.entity_id = mine.entity_id AND other.memory_id != mine.memory_id
                JOIN memories m ON m.id = other.memory_id AND m.status = 'active'
                JOIN entities e ON e.id = mine.entity_id
                WHERE mine.memory_id = ? ORDER BY m.updated_at DESC LIMIT ?""", (memory_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def upsert_embedding(self, memory_id: str, provider: str, vector: list[float]) -> None:
        with self.connection() as conn:
            conn.execute("""INSERT INTO embeddings(memory_id, provider, dimensions, vector, created_at)
                VALUES (?, ?, ?, ?, ?) ON CONFLICT(memory_id, provider) DO UPDATE SET
                dimensions = excluded.dimensions, vector = excluded.vector, created_at = excluded.created_at""",
                (memory_id, provider, len(vector), json.dumps(vector, separators=(",", ":")), now()))

    def embeddings(self, provider: str) -> list[tuple[Memory, list[float]]]:
        with self.connection() as conn:
            rows = conn.execute("""SELECT m.*, e.vector FROM embeddings e JOIN memories m ON m.id = e.memory_id
                WHERE e.provider = ? AND m.status = 'active'""", (provider,)).fetchall()
        return [(self._memory(row), json.loads(row["vector"])) for row in rows]

    def record_event(self, event_type: str, memory_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
        with self.connection() as conn:
            conn.execute("INSERT INTO events(id, event_type, memory_id, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                         (str(uuid.uuid4()), event_type, memory_id, json.dumps(payload or {}, sort_keys=True), now()))

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def set_state(self, subject: str, state_key: str, value: str, memory_id: str) -> str | None:
        """Set one current fact and return the superseded source memory, if any."""
        timestamp = now()
        with self.connection() as conn:
            prior = conn.execute("""SELECT memory_id FROM state_facts
                WHERE subject = ? AND state_key = ? AND is_current = 1""", (subject, state_key)).fetchone()
            conn.execute("""UPDATE state_facts SET is_current = 0, updated_at = ?
                WHERE subject = ? AND state_key = ? AND is_current = 1""", (timestamp, subject, state_key))
            conn.execute("""INSERT INTO state_facts(id, subject, state_key, value, memory_id, is_current, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)""", (str(uuid.uuid4()), subject, state_key, value, memory_id, timestamp, timestamp))
        return prior["memory_id"] if prior else None

    def state(self, subject: str | None = None, state_key: str | None = None, include_history: bool = False) -> list[dict[str, Any]]:
        clauses, params = [], []
        if not include_history:
            clauses.append("f.is_current = 1")
        if subject:
            clauses.append("f.subject = ?"); params.append(subject)
        if state_key:
            clauses.append("f.state_key = ?"); params.append(state_key)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connection() as conn:
            rows = conn.execute("""SELECT f.*, m.content, m.kind, m.status
                FROM state_facts f JOIN memories m ON m.id = f.memory_id""" + where + " ORDER BY f.updated_at DESC", params).fetchall()
        return [dict(row) for row in rows]
