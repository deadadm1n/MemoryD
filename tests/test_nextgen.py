import sqlite3

from memoryd.runtime import MemoryRuntime


def test_schema_settings_and_deduplication(tmp_path):
    database = tmp_path / "brain.db"
    runtime = MemoryRuntime(database)
    first = runtime.remember("SQLite is the local database decision.", kind="decision")
    second = runtime.remember("SQLite is the local database decision.", kind="decision")

    assert second.id == first.id
    assert runtime.store.stats()["memories"] == 1
    assert runtime.store.get(first.id).access_count == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        assert connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()[0] == "7"
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_hybrid_recall_and_structured_context(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    database = runtime.remember("SQLite was selected for the local portable brain.", kind="decision", importance=.95)
    runtime.remember("The current implementation exposes an MCP server.", kind="state", importance=.8)
    runtime.remember("Open question: which embedding model should production use?", kind="speculation")

    results = runtime.recall("What database did we choose?", limit=5)
    sqlite = next(result for result in results if result.memory.id == database.id)
    assert "semantic match" in sqlite.reasons

    context = runtime.context("continue the project", budget=1_000)
    assert database.id in {item["id"] for item in context["sections"]["decisions"]}
    assert context["sections"]["current_state"]
    assert context["sections"]["open_questions"]
    assert "DECISIONS" in context["text"]


def test_entity_associations_are_available_without_mutating_metadata(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    first = runtime.remember("SQLite is used by the Memory Runtime.", metadata={"project": "memoryd"})
    second = runtime.remember("Python maintains SQLite migrations in the Memory Runtime.")

    fetched = runtime.get(first.id)
    assert fetched["metadata"] == {"project": "memoryd"}
    assert any(item["id"] == second.id and item["entity"] == "SQLite" for item in fetched["entity_related"])


def test_consolidation_preserves_sources_and_event_provenance(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    decision = runtime.remember("SQLite is selected for version one.", kind="decision")
    state = runtime.remember("The current stage is implementation.", kind="state")

    consolidated = runtime.consolidate()

    assert consolidated["created"] is True
    summary = consolidated["memory"]
    assert runtime.store.get(decision.id).status == "active"
    assert runtime.store.get(state.id).status == "active"
    relationships = runtime.get(summary["id"])["relationships"]
    assert {item["relation"] for item in relationships} == {"derived_from"}
    assert any(event["event_type"] == "memories_consolidated" for event in runtime.events())


def test_state_fact_replaces_current_value_and_keeps_history(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    first = runtime.remember("Memory Runtime: database = SQLite.", kind="state")
    second = runtime.remember("Memory Runtime: database = PostgreSQL.", kind="state")

    current = runtime.state(subject="Memory Runtime", key="database")
    assert [(item["value"], item["memory_id"]) for item in current] == [("PostgreSQL", second.id)]
    history = runtime.state(subject="Memory Runtime", key="database", history=True)
    assert {item["value"] for item in history} == {"SQLite", "PostgreSQL"}
    assert next(item for item in history if item["value"] == "SQLite")["valid_until"]
    assert runtime.store.get(first.id).status == "superseded"
    assert any(item["relation"] == "supersedes" and item["to_id"] == first.id
               for item in runtime.get(second.id)["relationships"])
