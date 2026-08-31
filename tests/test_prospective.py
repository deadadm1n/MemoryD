from memoryd.prospective import derive_triggers, score_relevance


def test_derive_triggers_is_stable_and_extracts_technology_decision_and_future_cues():
    memory = {
        "content": "We decided to use SQLite with MCP. Before release, run the retrieval benchmark.",
        "kind": "decision",
    }

    triggers = derive_triggers(memory)

    assert [(item.phrase, item.category) for item in triggers] == [
        ("SQLite", "technology"),
        ("Model Context Protocol", "technology"),
        ("decision", "concept"),
        ("release", "concept"),
        ("testing", "concept"),
        ("Before release, run the retrieval benchmark", "future"),
    ]
    assert triggers == derive_triggers(memory)


def test_state_metadata_becomes_atomic_retrieval_cues():
    triggers = derive_triggers({
        "content": "The active data store is local.",
        "kind": "state",
        "metadata": {"state": {"subject": "MemoryD", "key": "database", "value": "SQLite"}},
    })

    assert [(item.phrase, item.category) for item in triggers] == [
        ("MemoryD", "state"),
        ("database", "state"),
        ("SQLite", "state"),
    ]


def test_score_relevance_matches_technology_aliases_and_reports_evidence():
    memory = {"content": "We decided to expose the brain through Model Context Protocol.", "kind": "decision"}

    result = score_relevance(memory, "Configure an MCP client for this agent")

    assert result.score > 0.3
    assert [(item.phrase, item.category) for item in result.matched] == [
        ("Model Context Protocol", "technology"),
    ]


def test_score_relevance_rewards_exact_future_situation_more_than_partial_overlap():
    memory = "Before release, run the retrieval benchmark."

    exact = score_relevance(memory, "Before release, run the retrieval benchmark.")
    partial = score_relevance(memory, "Plan the release notes.")
    unrelated = score_relevance(memory, "Choose a color for the documentation site.")

    assert exact.score > partial.score > unrelated.score
    assert unrelated.score == 0.0


def test_empty_and_malformed_input_is_safe():
    assert derive_triggers({"content": None, "metadata": "not a mapping"}) == ()
    assert score_relevance("SQLite", "").score == 0.0
