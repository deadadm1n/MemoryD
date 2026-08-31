"""Portable, validated operational helpers for a memoryd database.

These functions deliberately form the only supported interchange boundary.  A
caller never needs to know the SQLite schema (or issue SQLite statements) to
move a brain between machines.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from .store import BrainStore


FORMAT = "memoryd-export"
FORMAT_VERSION = 1
TABLES = ("memories", "links", "entities", "memory_entities", "embeddings", "events", "state_facts")
_REQUIRED: dict[str, set[str]] = {
    "memories": {"id", "content", "kind", "source", "confidence", "importance", "strength", "status", "created_at", "updated_at", "accessed_at", "access_count", "metadata"},
    "links": {"id", "from_id", "to_id", "relation", "created_at", "metadata"},
    "entities": {"id", "name", "normalized_name", "kind", "created_at"},
    "memory_entities": {"memory_id", "entity_id", "created_at"},
    "embeddings": {"memory_id", "provider", "dimensions", "vector", "created_at"},
    "events": {"id", "event_type", "memory_id", "payload", "created_at"},
    "state_facts": {"id", "subject", "state_key", "value", "memory_id", "is_current", "created_at", "updated_at"},
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
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != FORMAT_VERSION:
        raise ImportValidationError("unsupported memoryd export format")
    tables = payload.get("tables")
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
        for field in ("id", "event_type", "created_at"): _require_string(row, field, "events")
        if row["memory_id"] is not None and row["memory_id"] not in ids: raise ImportValidationError("events reference an unknown memory")
        _json_object(row["payload"], "events.payload")
    current: set[tuple[str, str]] = set()
    for row in tables["state_facts"]:
        for field in ("id", "subject", "state_key", "value", "memory_id", "created_at", "updated_at"): _require_string(row, field, "state_facts")
        if row["memory_id"] not in ids: raise ImportValidationError("state_facts reference an unknown memory")
        if row["is_current"] not in (0, 1): raise ImportValidationError("state_facts.is_current must be 0 or 1")
        pair = (row["subject"], row["state_key"])
        if row["is_current"] and pair in current: raise ImportValidationError("state_facts has duplicate current facts")
        if row["is_current"]: current.add(pair)
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
            for table in ("memories", "entities", "links", "memory_entities", "embeddings", "events", "state_facts"):
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
