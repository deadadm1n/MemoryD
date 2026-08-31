from memoryd.doctor import diagnose
from memoryd.runtime import MemoryRuntime


def test_doctor_reports_healthy_brain(tmp_path):
    database = tmp_path / "brain.db"
    MemoryRuntime(database).remember("A durable memory.")
    report = diagnose(database)
    assert report["ok"] is True
    assert all(check["ok"] for check in report["checks"])


def test_doctor_reports_missing_database(tmp_path):
    report = diagnose(tmp_path / "absent.db")
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "database_exists"
