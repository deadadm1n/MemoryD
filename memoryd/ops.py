"""Portable, validated operational helpers for a memoryd database.

These functions deliberately form the only supported interchange boundary.  A
caller never needs to know the SQLite schema (or issue SQLite statements) to
move a brain between machines.
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import BrainStore
from .scopes import validate_scope


FORMAT = "memoryd-export"
FORMAT_VERSION = 3
TABLES = ("memories", "links", "entities", "memory_entities", "embeddings", "events", "state_facts", "prospective_triggers", "branch_metadata")
_REQUIRED: dict[str, set[str]] = {
    "memories": {"id", "content", "kind", "source", "confidence", "importance", "strength", "status", "created_at", "updated_at", "accessed_at", "access_count", "scope", "metadata"},
    "links": {"id", "from_id", "to_id", "relation", "created_at", "metadata"},
    "entities": {"id", "name", "normalized_name", "kind", "created_at"},
    "memory_entities": {"memory_id", "entity_id", "created_at"},
    "embeddings": {"memory_id", "provider", "dimensions", "vector", "created_at"},
    "events": {"id", "event_type", "memory_id", "payload", "created_at", "scope"},
    "state_facts": {"id", "subject", "state_key", "value", "memory_id", "is_current", "created_at", "updated_at", "valid_from", "valid_until", "scope"},
    "prospective_triggers": {"memory_id", "phrase", "category", "weight", "created_at"},
    "branch_metadata": {"id", "kind", "name", "branch_id", "parent_branch_id", "base_memory_ids", "created_at"},
}


class ImportValidationError(ValueError):
    """The supplied export is malformed or cannot safely form a brain."""


def _new_destination(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def backup(database: str | Path, destination: str | Path) -> Path:
    """Create a consistent SQLite backup without copying WAL files.

    The destination must not already exist, avoiding an accidental overwrite.
    SQLite's backup API gives a consistent snapshot even while a daemon writes.
    """
    source = Path(database)
    if not source.is_file():
        raise FileNotFoundError(f"database not found: {source}")
    target = _new_destination(destination)
    source_conn = sqlite3.connect(source)
    target_conn = sqlite3.connect(target)
    try:
        source_conn.backup(target_conn)
    except Exception:
        target_conn.close()
        target.unlink(missing_ok=True)
        raise
    else:
        target_conn.close()
        return target
    finally:
        source_conn.close()


def export_json(database: str | Path, destination: str | Path) -> Path:
    """Write a complete, portable JSON snapshot; never overwrite its target."""
    if not Path(database).is_file():
        raise FileNotFoundError(f"database not found: {database}")
    target = _new_destination(destination)
    store = BrainStore(database)
    try:
        conn = store.connection()
        payload: dict[str, Any] = {"format": FORMAT, "version": FORMAT_VERSION, "tables": {}}
        for table in TABLES:
            payload["tables"][table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return target
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        store.close()


def _require_string(row: dict[str, Any], field: str, table: str, *, nullable: bool = False) -> None:
    value = row[field]
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ImportValidationError(f"{table}.{field} must be a non-empty string")


def _validate(payload: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") not in {1, 2, FORMAT_VERSION}:
        raise ImportValidationError("unsupported memoryd export format")
    tables = payload.get("tables")
    if payload.get("version") in {1, 2} and isinstance(tables, dict):
        for row in tables.get("memories", []): row.setdefault("scope", "shared")
        for row in tables.get("state_facts", []): row.setdefault("scope", "shared")
        for row in tables.get("events", []): row.setdefault("scope", "shared")
    if not isinstance(tables, dict) or set(tables) != set(TABLES):
        raise ImportValidationError("export must contain exactly the supported tables")
    for table in TABLES:
        rows = tables[table]
        if not isinstance(rows, list):
            raise ImportValidationError(f"tables.{table} must be an array")
        for row in rows:
            if not isinstance(row, dict) or set(row) != _REQUIRED[table]:
                raise ImportValidationError(f"{table} row has missing or unsupported fields")
    memories = tables["memories"]
    ids = {row["id"] for row in memories}
    if len(ids) != len(memories) or any(not isinstance(value, str) or not value for value in ids):
        raise ImportValidationError("memory IDs must be unique non-empty strings")
    for row in memories:
        for field in ("id", "content", "kind", "source", "status", "created_at", "updated_at"):
            _require_string(row, field, "memories")
        _require_string(row, "scope", "memories")
        try: validate_scope(row["scope"])
        except ValueError as exc: raise ImportValidationError("memories.scope is invalid") from exc
        _require_string(row, "accessed_at", "memories", nullable=True)
        if not isinstance(row["access_count"], int) or row["access_count"] < 0:
            raise ImportValidationError("memories.access_count must be a non-negative integer")
        for field in ("confidence", "importance", "strength"):
            value = row[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ImportValidationError(f"memories.{field} must be a finite number between 0 and 1")
        if not isinstance(row["metadata"], str):
            raise ImportValidationError("memories.metadata must be JSON text")
        try:
            if not isinstance(json.loads(row["metadata"]), dict): raise ValueError
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ImportValidationError("memories.metadata must encode an object") from exc
    entity_ids = {row["id"] for row in tables["entities"]}
    if len(entity_ids) != len(tables["entities"]): raise ImportValidationError("entity IDs must be unique")
    for row in tables["entities"]:
        for field in _REQUIRED["entities"]: _require_string(row, field, "entities")
    for row in tables["links"]:
        for field in ("id", "from_id", "to_id", "relation", "created_at"): _require_string(row, field, "links")
        if row["from_id"] not in ids or row["to_id"] not in ids: raise ImportValidationError("links reference an unknown memory")
        _json_object(row["metadata"], "links.metadata")
    for row in tables["memory_entities"]:
        if row["memory_id"] not in ids or row["entity_id"] not in entity_ids: raise ImportValidationError("memory_entities reference an unknown record")
        _require_string(row, "created_at", "memory_entities")
    for row in tables["embeddings"]:
        if row["memory_id"] not in ids: raise ImportValidationError("embeddings reference an unknown memory")
        _require_string(row, "provider", "embeddings"); _require_string(row, "created_at", "embeddings")
        if not isinstance(row["dimensions"], int) or row["dimensions"] <= 0: raise ImportValidationError("embeddings.dimensions must be positive")
        try: vector = json.loads(row["vector"])
        except (TypeError, json.JSONDecodeError) as exc: raise ImportValidationError("embeddings.vector must be JSON") from exc
        if not isinstance(vector, list) or len(vector) != row["dimensions"] or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in vector): raise ImportValidationError("embeddings.vector dimensions are invalid")
    for row in tables["events"]:
        for field in ("id", "event_type", "created_at", "scope"): _require_string(row, field, "events")
        try: validate_scope(row["scope"])
        except ValueError as exc: raise ImportValidationError("events.scope is invalid") from exc
        if row["memory_id"] is not None and row["memory_id"] not in ids: raise ImportValidationError("events reference an unknown memory")
        _json_object(row["payload"], "events.payload")
    current: set[tuple[str, str, str]] = set()
    for row in tables["state_facts"]:
        for field in ("id", "subject", "state_key", "value", "memory_id", "created_at", "updated_at", "valid_from", "scope"): _require_string(row, field, "state_facts")
        try: validate_scope(row["scope"])
        except ValueError as exc: raise ImportValidationError("state_facts.scope is invalid") from exc
        _require_string(row, "valid_until", "state_facts", nullable=True)
        if row["memory_id"] not in ids: raise ImportValidationError("state_facts reference an unknown memory")
        if row["is_current"] not in (0, 1): raise ImportValidationError("state_facts.is_current must be 0 or 1")
        pair = (row["scope"], row["subject"], row["state_key"])
        if row["is_current"] and pair in current: raise ImportValidationError("state_facts has duplicate current facts")
        if row["is_current"]: current.add(pair)
    for row in tables["prospective_triggers"]:
        if row["memory_id"] not in ids: raise ImportValidationError("prospective_triggers reference an unknown memory")
        for field in ("phrase", "category", "created_at"): _require_string(row, field, "prospective_triggers")
        if not isinstance(row["weight"], (int, float)) or isinstance(row["weight"], bool) or not math.isfinite(row["weight"]) or not 0 <= row["weight"] <= 1:
            raise ImportValidationError("prospective_triggers.weight must be a finite number between 0 and 1")
    if len(tables["branch_metadata"]) > 1:
        raise ImportValidationError("branch_metadata may contain at most one row")
    for row in tables["branch_metadata"]:
        if row["id"] != 1 or row["kind"] not in {"snapshot", "fork"}:
            raise ImportValidationError("branch_metadata is invalid")
        for field in ("name", "branch_id", "created_at"):
            _require_string(row, field, "branch_metadata")
        _require_string(row, "parent_branch_id", "branch_metadata", nullable=True)
        try:
            base_ids = json.loads(row["base_memory_ids"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ImportValidationError("branch_metadata.base_memory_ids must be JSON") from exc
        if not isinstance(base_ids, list) or any(not isinstance(item, str) for item in base_ids):
            raise ImportValidationError("branch_metadata.base_memory_ids must be a string array")
    return tables


def _json_object(value: Any, label: str) -> None:
    if not isinstance(value, str): raise ImportValidationError(f"{label} must be JSON text")
    try:
        if not isinstance(json.loads(value), dict): raise ValueError
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ImportValidationError(f"{label} must encode an object") from exc


def import_json(source: str | Path, database: str | Path) -> Path:
    """Create an empty brain from a validated JSON export, preserving all IDs."""
    source_path, target = Path(source), Path(database)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportValidationError(f"cannot read JSON export: {source_path}") from exc
    tables = _validate(payload)
    if target.exists():
        # A target that already holds a database must be explicitly moved aside;
        # import should never silently merge or erase a brain.
        raise FileExistsError(f"database destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    store = BrainStore(target)
    try:
        conn = store.connection()
        with conn:
            for table in ("memories", "entities", "links", "memory_entities", "embeddings", "events", "state_facts", "prospective_triggers", "branch_metadata"):
                rows = tables[table]
                if not rows: continue
                columns = sorted(_REQUIRED[table])
                marks = ", ".join("?" for _ in columns)
                conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})", [tuple(row[key] for key in columns) for row in rows])
            conn.execute("DELETE FROM memory_fts")
            conn.execute("INSERT INTO memory_fts(id, content) SELECT id, content FROM memories")
        return target
    except Exception:
        store.close(); target.unlink(missing_ok=True)
        raise
    finally:
        store.close()


def _branch_record(database: str | Path) -> dict[str, Any] | None:
    store = BrainStore(database)
    try:
        row = store.connection().execute("SELECT * FROM branch_metadata WHERE id = 1").fetchone()
        return dict(row) if row else None
    finally:
        store.close()


def _write_branch_record(database: str | Path, *, kind: str, name: str, base_memory_ids: list[str], parent_branch_id: str | None) -> dict[str, Any]:
    if kind not in {"snapshot", "fork"} or not name.strip():
        raise ValueError("branch kind and non-empty name are required")
    store = BrainStore(database)
    try:
        record = {"kind": kind, "name": name.strip(), "branch_id": str(uuid.uuid4()), "parent_branch_id": parent_branch_id,
                  "base_memory_ids": json.dumps(sorted(base_memory_ids)), "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
        with store.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO branch_metadata (id, kind, name, branch_id, parent_branch_id, base_memory_ids, created_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
                         (record["kind"], record["name"], record["branch_id"], record["parent_branch_id"], record["base_memory_ids"], record["created_at"]))
        return {**record, "base_memory_ids": json.loads(record["base_memory_ids"])}
    finally:
        store.close()


def snapshot(database: str | Path, name: str, destination: str | Path) -> dict[str, Any]:
    """Create a named, immutable-in-practice SQLite snapshot without overwriting."""
    if _branch_record(database):
        raise ValueError("snapshot the root brain, not an existing snapshot or fork")
    target = backup(database, destination)
    source = BrainStore(database)
    try:
        base_ids = [row["id"] for row in source.connection().execute("SELECT id FROM memories")]
    finally:
        source.close()
    metadata = _write_branch_record(target, kind="snapshot", name=name, base_memory_ids=base_ids, parent_branch_id=None)
    return {"snapshot": str(target), "metadata": metadata}


def fork(snapshot_database: str | Path, name: str, destination: str | Path) -> dict[str, Any]:
    """Create an isolated writable fork from a named snapshot; never overwrite."""
    parent = _branch_record(snapshot_database)
    if not parent or parent["kind"] not in {"snapshot", "fork"}:
        raise ValueError("fork source must be a MemoryD snapshot or fork")
    target = backup(snapshot_database, destination)
    base_ids = json.loads(parent["base_memory_ids"])
    metadata = _write_branch_record(target, kind="fork", name=name, base_memory_ids=base_ids, parent_branch_id=parent["branch_id"])
    return {"fork": str(target), "metadata": metadata}


def merge(fork_database: str | Path, database: str | Path) -> dict[str, Any]:
    """Import only new fork knowledge; conflicting current state is reported, never overwritten."""
    metadata = _branch_record(fork_database)
    if not metadata or metadata["kind"] != "fork":
        raise ValueError("merge source must be a MemoryD fork")
    if _branch_record(database):
        raise ValueError("merge destination must be the root brain, not a snapshot or fork")
    base_ids = set(json.loads(metadata["base_memory_ids"]))
    source, target = BrainStore(fork_database), BrainStore(database)
    try:
        source_conn, target_conn = source.connection(), target.connection()
        source_memories = {row["id"]: dict(row) for row in source_conn.execute("SELECT * FROM memories")}
        candidate_ids = set(source_memories) - base_ids
        target_ids = {row["id"] for row in target_conn.execute("SELECT id FROM memories")}
        new_ids = sorted(candidate_ids - target_ids)
        if not new_ids:
            return {"merged_memory_ids": [], "state_conflicts": [], "already_present": sorted(candidate_ids), "fork": metadata["branch_id"]}
        marks = ",".join("?" for _ in new_ids)
        state_conflicts: list[dict[str, str]] = []
        with target_conn:
            columns = sorted(_REQUIRED["memories"])
            target_conn.executemany(f"INSERT INTO memories ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                [tuple(source_memories[item][key] for key in columns) for item in new_ids])
            target_conn.executemany("INSERT INTO memory_fts(id, content) VALUES (?, ?)", [(item, source_memories[item]["content"]) for item in new_ids])
            entity_rows = source_conn.execute(f"SELECT DISTINCT e.* FROM entities e JOIN memory_entities me ON me.entity_id=e.id WHERE me.memory_id IN ({marks})", new_ids).fetchall()
            target_conn.executemany("INSERT OR IGNORE INTO entities (id, name, normalized_name, kind, created_at) VALUES (?, ?, ?, ?, ?)", [tuple(row) for row in entity_rows])
            target_conn.executemany("INSERT OR IGNORE INTO memory_entities (memory_id, entity_id, created_at) SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM entities WHERE id = ?)",
                [(row["memory_id"], row["entity_id"], row["created_at"], row["entity_id"]) for row in source_conn.execute(f"SELECT * FROM memory_entities WHERE memory_id IN ({marks})", new_ids)])
            for table, cols in (("embeddings", ("memory_id", "provider", "dimensions", "vector", "created_at")), ("prospective_triggers", ("memory_id", "phrase", "category", "weight", "created_at"))):
                rows = source_conn.execute(f"SELECT * FROM {table} WHERE memory_id IN ({marks})", new_ids).fetchall()
                target_conn.executemany(f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [tuple(row[key] for key in cols) for row in rows])
            all_target_ids = target_ids | set(new_ids)
            for row in source_conn.execute(f"SELECT * FROM links WHERE from_id IN ({marks})", new_ids):
                if row["to_id"] in all_target_ids:
                    target_conn.execute("INSERT OR IGNORE INTO links (id, from_id, to_id, relation, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)", tuple(row))
            for row in source_conn.execute(f"SELECT * FROM state_facts WHERE memory_id IN ({marks})", new_ids):
                current = target_conn.execute("SELECT 1 FROM state_facts WHERE scope=? AND subject=? AND state_key=? AND is_current=1", (row["scope"], row["subject"], row["state_key"])).fetchone()
                if row["is_current"] and current:
                    state_conflicts.append({"subject": row["subject"], "key": row["state_key"], "memory_id": row["memory_id"]})
                    continue
                target_conn.execute("INSERT OR IGNORE INTO state_facts (id, subject, state_key, value, memory_id, is_current, created_at, updated_at, valid_from, valid_until, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(row))
            for row in source_conn.execute(f"SELECT * FROM events WHERE memory_id IN ({marks})", new_ids):
                target_conn.execute("INSERT OR IGNORE INTO events (id, event_type, memory_id, payload, created_at, scope) VALUES (?, ?, ?, ?, ?, ?)", tuple(row))
            target_conn.execute("INSERT INTO events (id, event_type, memory_id, payload, created_at, scope) VALUES (?, 'fork_merged', NULL, ?, datetime('now'), 'shared')",
                (str(uuid.uuid4()), json.dumps({"fork_id": metadata["branch_id"], "fork_name": metadata["name"], "memory_ids": new_ids, "state_conflicts": state_conflicts}, sort_keys=True)))
        return {"merged_memory_ids": new_ids, "state_conflicts": state_conflicts, "already_present": sorted(candidate_ids & target_ids), "fork": metadata["branch_id"]}
    finally:
        source.close(); target.close()
