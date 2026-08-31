from memoryd.runtime import MemoryRuntime


def test_reflection_is_review_only_and_finds_open_questions(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    first = runtime.remember("SQLite was selected for the portable brain.", kind="decision")
    second = runtime.remember("Open question: should the production embedding model run locally?", kind="speculation")

    reflection = runtime.reflect()

    kinds = {proposal["kind"] for proposal in reflection["proposals"]}
    assert {"consolidate", "review_open_questions"} <= kinds
    assert runtime.store.get(first.id).status == "active"
    assert runtime.store.get(second.id).status == "active"
    assert any(event["event_type"] == "reflection_proposed" for event in runtime.events())
