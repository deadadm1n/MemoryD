import json

from memoryd.evals import evaluate, main
from memoryd.runtime import MemoryRuntime


def test_fixed_corpus_passes_all_quality_gates(tmp_path):
    report = evaluate(MemoryRuntime(tmp_path / "eval.db"))

    assert report["passed"] is True
    assert {check["name"] for check in report["checks"]} == {
        "retrieval_correctness",
        "supersession_exclusion",
        "context_budget_compliance",
        "consolidation_provenance",
    }
    assert all(check["passed"] for check in report["checks"])


def test_module_runner_emits_machine_readable_report(tmp_path, capsys):
    exit_code = main(["--database", str(tmp_path / "runner.db")])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["passed"] is True
