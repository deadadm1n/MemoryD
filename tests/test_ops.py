import json
import sqlite3

import pytest

from memoryd.ops import ImportValidationError, backup, export_json, fork, import_json, merge, snapshot
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
    history = restored.state(subject="Project", key="database", history=True)
    assert any(fact["value"] == "SQLite" and fact["valid_until"] for fact in history)
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


def test_snapshot_fork_and_merge_preserve_new_knowledge_without_overwriting_state(tmp_path):
    main = MemoryRuntime(tmp_path / "main.db")
    main.remember("Project: database = SQLite.", kind="state")
    snap = snapshot(main.store.path, "before-experiment", tmp_path / "snapshot.db")
    assert snap["metadata"]["kind"] == "snapshot"
    forked = fork(snap["snapshot"], "postgres-experiment", tmp_path / "experiment.db")
    assert forked["metadata"]["kind"] == "fork"

    experiment = MemoryRuntime(forked["fork"])
    decision = experiment.remember("PostgreSQL needs a load test before adoption.", kind="decision")
    experiment.remember("Project: database = PostgreSQL.", kind="state")
    main.remember("Project: database = MySQL.", kind="state")

    outcome = merge(forked["fork"], main.store.path)
    assert decision.id in outcome["merged_memory_ids"]
    assert len(outcome["state_conflicts"]) == 1
    assert outcome["state_conflicts"][0]["subject"] == "Project"
    assert outcome["state_conflicts"][0]["key"] == "database"
    assert main.get(decision.id)["content"] == "PostgreSQL needs a load test before adoption."
    assert main.state(subject="Project", key="database")[0]["value"] == "MySQL"
    assert any(event["event_type"] == "fork_merged" for event in main.events())


def test_branching_refuses_overwrites_and_non_fork_merge_sources(tmp_path):
    main = MemoryRuntime(tmp_path / "main.db")
    main.remember("Project: stage = testing.", kind="state")
    target = tmp_path / "snapshot.db"
    snapshot(main.store.path, "testing", target)
    with pytest.raises(FileExistsError):
        snapshot(main.store.path, "testing", target)
    with pytest.raises(ValueError, match="fork"):
        merge(target, main.store.path)
