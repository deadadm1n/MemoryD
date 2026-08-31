from memoryd.runtime import MemoryRuntime


def test_observe_selects_durable_experiences_and_ignores_chatter(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")

    ignored = runtime.observe("Thanks, that looks good.", actor="Doug")
    observed = runtime.observe("MemoryD.database = SQLite.", actor="Doug", context={"project": "MemoryD"})

    assert ignored == {"stored": [], "ignored": True, "candidate_count": 0}
    assert observed["ignored"] is False
    current = runtime.state(subject="MemoryD", key="database")
    assert current[0]["value"] == "SQLite"


def test_context_compiler_surfaces_prospective_relevance_and_beliefs(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    state = runtime.remember("MemoryD: database = SQLite.", kind="state")
    future = runtime.remember("When planning a production release, revisit SQLite concurrency assumptions.", kind="semantic")

    context = runtime.context("We are planning the production release", budget=1_000)
    likely = context["sections"]["likely_relevant_soon"]

    assert any(item["id"] == future.id and item["triggers"] for item in likely)
    belief = runtime.beliefs()
    assert any(item["evidence_ids"] == [state.id] for item in belief["beliefs"])
