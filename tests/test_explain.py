from memoryd.runtime import MemoryRuntime


def test_explain_returns_exact_evidence_for_a_current_state_belief(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    source = runtime.remember("MemoryD: database = SQLite.", kind="state", confidence=.98)

    explanation = runtime.explain(subject="MemoryD", key="database")

    assert explanation["belief"]["evidence_ids"] == [source.id]
    assert explanation["evidence"][0]["id"] == source.id


def test_explain_requires_an_explicit_belief_selector(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    try:
        runtime.explain()
    except ValueError as exc:
        assert "provide subject" in str(exc)
    else:
        raise AssertionError("expected a selector validation error")
