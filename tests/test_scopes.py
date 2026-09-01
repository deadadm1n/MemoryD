from __future__ import annotations

import pytest

from memoryd.runtime import MemoryRuntime
from memoryd.scopes import ScopeContext


def test_project_scope_filters_every_runtime_read_path(tmp_path):
    database = tmp_path / "brain.db"
    alpha = MemoryRuntime(database, scope_context=ScopeContext(project_id="alpha"))
    beta = MemoryRuntime(database, scope_context=ScopeContext(project_id="beta"))
    visible = alpha.remember("Alpha Runbook: database = SQLite.", kind="state", scope="project:alpha")
    hidden = beta.remember("Beta Runbook: database = PostgreSQL.", kind="state", scope="project:beta")
    beta.remember("When planning the beta launch, revisit PostgreSQL concurrency.", scope="project:beta")

    assert alpha.get(hidden.id) is None
    assert hidden.id not in {item.memory.id for item in alpha.recall("database beta launch", limit=10)}
    assert hidden.id not in {item["id"] for item in alpha.context("planning beta launch", budget=1000)["memories"]}
    assert hidden.id not in {item["id"] for item in alpha.timeline(limit=10)}
    assert [fact["memory_id"] for fact in alpha.state(subject="Alpha Runbook", key="database")] == [visible.id]
    assert alpha.state(subject="Beta Runbook", key="database") == []
    assert all(hidden.id not in belief["evidence_ids"] for belief in alpha.beliefs()["beliefs"])
    assert all(event.get("memory_id") != hidden.id for event in alpha.events(limit=50))


def test_scopes_keep_duplicate_and_state_histories_independent(tmp_path):
    database = tmp_path / "brain.db"
    alpha = MemoryRuntime(database, scope_context=ScopeContext(project_id="alpha"))
    beta = MemoryRuntime(database, scope_context=ScopeContext(project_id="beta"))
    one = alpha.remember("Runtime: database = SQLite.", kind="state", scope="project:alpha")
    two = beta.remember("Runtime: database = SQLite.", kind="state", scope="project:beta")

    assert one.id != two.id
    assert alpha.state(subject="Runtime", key="database")[0]["value"] == "SQLite"
    assert beta.state(subject="Runtime", key="database")[0]["memory_id"] == two.id


def test_context_cannot_elevate_or_link_across_scopes(tmp_path):
    database = tmp_path / "brain.db"
    alpha = MemoryRuntime(database, scope_context=ScopeContext(project_id="alpha", principal_id="alice"))
    beta = MemoryRuntime(database, scope_context=ScopeContext(project_id="beta", principal_id="bob"))
    alpha_memory = alpha.remember("Alpha private plan.", scope="private:alice")
    beta_memory = beta.remember("Beta private plan.", scope="private:bob")

    with pytest.raises(ValueError, match="not writable"):
        alpha.remember("Attempted scope elevation.", scope="project:beta")
    with pytest.raises(KeyError):
        alpha.link(alpha_memory.id, beta_memory.id, "depends_on")
    assert beta.get(alpha_memory.id) is None
