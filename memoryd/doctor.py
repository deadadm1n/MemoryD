"""Read-only health checks for a portable memoryd brain."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def diagnose(database: str | Path) -> dict[str, Any]:
    path = Path(database)
    if not path.is_file():
        return {"ok": False, "database": str(path), "checks": [{"name": "database_exists", "ok": False, "detail": "database file not found"}]}
    connection = sqlite3.connect(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        memory_count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        checks = [
            {"name": "integrity", "ok": integrity == "ok", "detail": integrity},
            {"name": "foreign_keys", "ok": not foreign_keys, "detail": f"{len(foreign_keys)} violations"},
            {"name": "schema_version", "ok": schema_version >= 4, "detail": schema_version},
            {"name": "fts_sync", "ok": memory_count == fts_count, "detail": {"memories": memory_count, "fts_rows": fts_count}},
            {"name": "journal_mode", "ok": str(journal_mode).lower() == "wal", "detail": journal_mode},
        ]
        return {"ok": all(check["ok"] for check in checks), "database": str(path), "checks": checks}
    finally:
        connection.close()
