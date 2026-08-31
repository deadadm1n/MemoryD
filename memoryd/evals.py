"""Deterministic, dependency-free regression evaluation for ``memoryd``.

Run with ``python -m memoryd.evals --database eval-brain.db``.  The evaluator
uses a fixed corpus so changes to ranking or context assembly can be compared
meaningfully between releases.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import MemoryRuntime


@dataclass(frozen=True)
class EvaluationResult:
    """A single named check with compact, machine-readable evidence."""

    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


def seed_corpus(runtime: MemoryRuntime) -> dict[str, str]:
    """Load a small, representative project-continuity corpus.

    The previous database decision is intentionally superseded.  This makes the
    corpus useful for detecting a particularly costly failure mode: resurfacing
    stale facts as current guidance.
    """
    old_database = runtime.remember(
        "Atlas Runtime: database = PostgreSQL.", kind="state", importance=.85,
        metadata={"state": {"subject": "Atlas Runtime", "key": "database", "value": "PostgreSQL"}},
    )
    database = runtime.remember(
        "Atlas Runtime: database = SQLite.", kind="state", importance=.98,
        metadata={"state": {"subject": "Atlas Runtime", "key": "database", "value": "SQLite"}},
    )
    mcp = runtime.remember(
        "We decided to expose memory operations through MCP for coding agents.",
        kind="decision", importance=.92,
    )
    question = runtime.remember(
        "Open question: which local embedding model best improves semantic recall?",
        kind="speculation", confidence=.5, importance=.7,
    )
    procedure = runtime.remember(
        "Before a release, run the retrieval evaluation corpus and inspect failed cases.",
        kind="procedural", importance=.75,
    )
    return {"old_database": old_database.id, "database": database.id, "mcp": mcp.id,
            "question": question.id, "procedure": procedure.id}


def _result(name: str, condition: bool, **details: Any) -> EvaluationResult:
    return EvaluationResult(name=name, passed=condition, details=details)


def evaluate(runtime: MemoryRuntime) -> dict[str, Any]:
    """Seed ``runtime`` and run the stable core quality checks.

    The supplied runtime must point at an empty database.  Keeping setup inside
    this function makes the test corpus reusable from pytest and CI runners.
    """
    ids = seed_corpus(runtime)
    checks: list[EvaluationResult] = []

    retrieval_cases = {
        "database_selection": ("Which database does Atlas Runtime use now?", ids["database"]),
        "agent_interface": ("How do coding agents access memory operations?", ids["mcp"]),
        "open_embedding_question": ("What embedding issue remains unresolved?", ids["question"]),
    }
    case_details: dict[str, Any] = {}
    retrieval_ok = True
    for label, (query, expected_id) in retrieval_cases.items():
        result_ids = [item.memory.id for item in runtime.recall(query, limit=5)]
        case_details[label] = {"expected_id": expected_id, "result_ids": result_ids}
        retrieval_ok &= expected_id in result_ids
    checks.append(_result("retrieval_correctness", retrieval_ok, cases=case_details))

    stale_results = [item.memory.id for item in runtime.recall("Atlas PostgreSQL database", limit=10)]
    current_state = runtime.state(subject="Atlas Runtime", key="database")
    checks.append(_result(
        "supersession_exclusion",
        ids["old_database"] not in stale_results
        and len(current_state) == 1
        and current_state[0]["memory_id"] == ids["database"],
        stale_result_ids=stale_results,
        current_state_memory_ids=[item["memory_id"] for item in current_state],
    ))

    budget = 600
    context = runtime.context("continue Atlas Runtime", budget=budget)
    text_characters = len(context["text"])
    # Runtime uses four characters per token and enforces a 400-character floor.
    character_limit = max(400, budget * 4)
    checks.append(_result(
        "context_budget_compliance",
        text_characters <= character_limit,
        budget_tokens=budget,
        text_characters=text_characters,
        character_limit=character_limit,
        selected_memory_ids=[item["id"] for item in context["memories"]],
    ))

    consolidation = runtime.consolidate(limit=20)
    summary_id = consolidation.get("memory", {}).get("id")
    source_ids = consolidation.get("source_memory_ids", [])
    relationships = runtime.get(summary_id)["relationships"] if summary_id else []
    derived_ids = {item["to_id"] for item in relationships if item["relation"] == "derived_from"}
    source_statuses = {memory_id: runtime.store.get(memory_id).status for memory_id in source_ids}
    checks.append(_result(
        "consolidation_provenance",
        consolidation.get("created") is True
        and set(source_ids) == derived_ids
        and all(status == "active" for status in source_statuses.values()),
        summary_id=summary_id,
        source_ids=source_ids,
        derived_ids=sorted(derived_ids),
        source_statuses=source_statuses,
    ))

    passed = all(check.passed for check in checks)
    return {"passed": passed, "checks": [check.to_dict() for check in checks], "stats": runtime.store.stats()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run memoryd's deterministic evaluation corpus.")
    parser.add_argument("--database", type=Path, help="SQLite database to create for this run; defaults to a temporary file.")
    args = parser.parse_args(argv)
    if args.database:
        runtime = MemoryRuntime(args.database)
        report = evaluate(runtime)
        runtime.store.close()
    else:
        with tempfile.TemporaryDirectory(prefix="memoryd-eval-") as directory:
            runtime = MemoryRuntime(Path(directory) / "brain.db")
            report = evaluate(runtime)
            runtime.store.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the module entry point.
    raise SystemExit(main())
