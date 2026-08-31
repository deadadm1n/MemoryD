import json
import sqlite3

import pytest

from memoryd.ops import ImportValidationError, backup, export_json, import_json
from memoryd.runtime import MemoryRuntime


def seeded(tmp_path):
    runtime = MemoryRuntime(tmp_path / "source.db")
    old = runtime.remember("Project: database = SQLite.", kind="state")
    current = runtime.remember("Project: database = PostgreSQL.", kind="state")
    other = runtime.remember("PostgreSQL migration is planned.", kind="decision")
    runtime.link(current.id, other.id, "supports")
    runtime.consolidate()
    return runtime, old, current, other


def test_backup_produces_readable_consistent_database(tmp_path):
    runtime, _, current, _ = seeded(tmp_path)
    target = backup(runtime.store.path, tmp_path / "backup.db")
    copied = MemoryRuntime(target)
    assert copied.get(current.id)["content"] == "Project: database = PostgreSQL."
    assert copied.state(subject="Project", key="database")[0]["value"] == "PostgreSQL"
    with pytest.raises(FileExistsError): backup(runtime.store.path, target)


def test_export_import_round_trip_preserves_graph_state_entities_and_events(tmp_path):
    runtime, _, current, other = seeded(tmp_path)
    exported = export_json(runtime.store.path, tmp_path / "brain.json")
    restored_path = import_json(exported, tmp_path / "restored.db")
    restored = MemoryRuntime(restored_path)
    assert restored.get(current.id)["content"] == "Project: database = PostgreSQL."
    assert any(link["relation"] == "supports" and link["to_id"] == other.id for link in restored.get(current.id)["relationships"])
    assert restored.state(subject="Project", key="database")[0]["memory_id"] == current.id
    assert restored.store.entity_related(current.id)
    assert restored.events()
    assert restored.recall("PostgreSQL", limit=3)


def test_import_rejects_malformed_and_does_not_leave_database(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"format": "memoryd-export", "version": 1, "tables": {}}), encoding="utf-8")
    target = tmp_path / "not-created.db"
    with pytest.raises(ImportValidationError): import_json(source, target)
    assert not target.exists()

    runtime, _, _, _ = seeded(tmp_path)
    exported = export_json(runtime.store.path, tmp_path / "valid.json")
    payload = json.loads(exported.read_text(encoding="utf-8"))
    payload["tables"]["links"][0]["to_id"] = "missing"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImportValidationError, match="unknown memory"):
        import_json(source, target)
    assert not target.exists()


def test_export_never_overwrites_destination(tmp_path):
    runtime, _, _, _ = seeded(tmp_path)
    target = tmp_path / "exists.json"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError): export_json(runtime.store.path, target)
    assert target.read_text(encoding="utf-8") == "keep"
