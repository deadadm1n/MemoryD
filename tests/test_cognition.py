from memoryd.cognition import (
    DeterministicObservationAnalyzer,
    MemoryCandidate,
    ObservationAnalyzer,
)


def test_observation_analyzer_protocol_and_explicit_decision():
    analyzer = DeterministicObservationAnalyzer()
    assert isinstance(analyzer, ObservationAnalyzer)

    candidates = analyzer.analyze(
        "Decision: use SQLite for the portable brain.",
        actor="maintainer",
        context={"project": "MemoryD"},
    )

    assert candidates == [MemoryCandidate(
        kind="decision", confidence=0.95, importance=0.8,
        content="Decision: use SQLite for the portable brain.",
        actor="maintainer", context={"project": "MemoryD"},
    )]


def test_observation_analyzer_extracts_explicit_current_state():
    candidate = DeterministicObservationAnalyzer().analyze(
        "MemoryD.database = SQLite", actor="maintainer"
    )[0]

    assert candidate.kind == "state"
    assert candidate.state == {"subject": "MemoryD", "key": "database", "value": "SQLite"}
    assert candidate.confidence == 0.96


def test_observation_analyzer_marks_questions_as_speculation_and_preserves_context():
    candidate = DeterministicObservationAnalyzer().analyze(
        "Open question: should embeddings remain local?",
        context={"conversation_id": "42"},
    )[0]

    assert candidate.kind == "speculation"
    assert candidate.confidence < 0.6
    assert candidate.context == {"conversation_id": "42"}


def test_observation_analyzer_ignores_ambiguous_or_ephemeral_text():
    analyzer = DeterministicObservationAnalyzer()

    assert analyzer.analyze("I had coffee and looked at the logs.") == []
    assert analyzer.analyze("We might use SQLite someday.") == [MemoryCandidate(
        kind="speculation", confidence=0.5, importance=0.55,
        content="We might use SQLite someday.", context={},
    )]
