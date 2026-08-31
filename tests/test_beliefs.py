from memoryd.beliefs import build_belief_set, derive_beliefs, explain
from memoryd.runtime import MemoryRuntime


def test_direct_state_and_decision_beliefs_include_inspectable_evidence(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    state = runtime.remember("MemoryD: database = SQLite.", kind="state", confidence=.9)
    decision = runtime.remember("We decided to keep MemoryD local first.", kind="decision", confidence=.8)
    runtime.remember("Maybe use a cloud sync service later.", kind="speculation")

    beliefs, unresolved = derive_beliefs(runtime.store.recent(20))

    assert not unresolved
    database = next(item for item in beliefs if item.key == "database")
    assert database.value == "SQLite"
    assert database.evidence_ids == (state.id,)
    decision_belief = next(item for item in beliefs if item.kind == "decision")
    assert decision_belief.evidence_ids == (decision.id,)
    assert explain(database)["supporting_evidence_ids"] == [state.id]


def test_conflicting_active_state_is_unresolved_not_a_belief(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    first = runtime.remember("MemoryD: database = SQLite.", kind="state")
    # Supply independent records to the pure module: normal runtime writes
    # supersede the old value, whereas this checks its defensive conflict path.
    second = runtime.store.create("MemoryD: database = PostgreSQL.", kind="state", source="test", confidence=.7, importance=.5)

    view = build_belief_set([first, second])

    assert not view["beliefs"]
    assert view["unresolved"] == [{
        "subject": "MemoryD", "key": "database", "values": ["PostgreSQL", "SQLite"],
        "evidence_ids": sorted([first.id, second.id]),
        "reason": "Conflicting active state assertions; no conclusion was emitted.",
    }]


def test_superseded_state_is_excluded_from_beliefs(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    runtime.remember("MemoryD: database = SQLite.", kind="state")
    latest = runtime.remember("MemoryD: database = PostgreSQL.", kind="state")

    view = build_belief_set([runtime.store.get(memory["memory_id"]) for memory in runtime.state(history=True)])

    assert [(item["key"], item["value"], item["evidence_ids"]) for item in view["beliefs"]] == [
        ("database", "PostgreSQL", [latest.id])
    ]
