"""Black-box coverage for memoryd's published REST, MCP, and CLI contracts."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.client import HTTPConnection

from memoryd.mcp import serve_stdio
from memoryd.runtime import MemoryRuntime
from memoryd.server import MemoryHandler, ThreadingHTTPServer


@contextmanager
def rest_server(runtime: MemoryRuntime):
    """Run the public HTTP handler on an ephemeral loopback port for one test."""
    handler = type("TestMemoryHandler", (MemoryHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def http_json(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_rest_public_contract_round_trip(tmp_path):
    """Every documented REST operation works together against one brain."""
    with rest_server(MemoryRuntime(tmp_path / "rest-brain.db")) as port:
        status, first = http_json(port, "POST", "/remember", {
            "content": "We selected SQLite for the portable local brain.",
            "kind": "decision", "importance": 0.95, "metadata": {"project": "memoryd"},
        })
        assert status == 201
        assert first["kind"] == "decision"

        status, second = http_json(port, "POST", "/remember", {
            "content": "The MCP server is the agent-facing integration.", "kind": "state",
        })
        assert status == 201

        status, linked = http_json(port, "POST", "/link", {
            "from_id": second["id"], "to_id": first["id"], "relation": "depends_on",
        })
        assert status == 200 and linked == {"status": "linked"}

        status, fetched = http_json(port, "GET", f"/memories/{second['id']}")
        assert status == 200
        assert fetched["metadata"] == {}
        assert fetched["relationships"][0]["relation"] == "depends_on"

        status, recalled = http_json(port, "POST", "/recall", {"query": "portable SQLite brain"})
        assert status == 200
        assert recalled["results"][0]["memory"]["id"] == first["id"]

        status, context = http_json(port, "POST", "/context", {"query": "SQLite", "budget": 200})
        assert status == 200
        assert first["id"] in {item["id"] for item in context["memories"]}

        status, timeline = http_json(port, "GET", "/timeline?limit=10")
        assert status == 200
        assert {item["id"] for item in timeline["memories"]} == {first["id"], second["id"]}

        status, health = http_json(port, "GET", "/health")
        assert status == 200
        assert health["status"] == "ok"
        assert health["stats"]["memories"] == 2
        assert health["stats"]["relationships"] == 1

        status, consolidated = http_json(port, "POST", "/consolidate", {"limit": 10})
        assert status == 200 and consolidated["created"] is True
        summary = consolidated["memory"]
        assert set(consolidated["source_memory_ids"]) == {first["id"], second["id"]}
        assert summary["metadata"]["consolidation"] is True
        status, fetched_summary = http_json(port, "GET", f"/memories/{summary['id']}")
        assert status == 200
        assert {relationship["relation"] for relationship in fetched_summary["relationships"]} == {"derived_from"}

        status, events = http_json(port, "GET", "/events?limit=50")
        assert status == 200
        event_types = {event["event_type"] for event in events["events"]}
        assert {"memory_created", "memories_linked", "memories_consolidated"} <= event_types
        consolidated_event = next(event for event in events["events"] if event["event_type"] == "memories_consolidated")
        assert consolidated_event["memory_id"] == summary["id"]
        assert set(consolidated_event["payload"]["source_memory_ids"]) == {first["id"], second["id"]}

        status, forgotten = http_json(port, "POST", f"/forget/{first['id']}")
        assert status == 200 and forgotten == {"status": "forgotten"}
        status, recalled = http_json(port, "POST", "/recall", {"query": "portable SQLite brain"})
        assert status == 200
        assert first["id"] not in {item["memory"]["id"] for item in recalled["results"]}


def test_mcp_stdio_lifecycle_and_all_memory_tools(tmp_path):
    """Exercise newline-delimited JSON-RPC exactly as a real stdio MCP client does."""
    database = tmp_path / "mcp-brain.db"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "memory_remember", "arguments": {"content": "SQLite is the selected local database.", "kind": "decision"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "memory_remember", "arguments": {"content": "MCP exposes the agent integration.", "kind": "state"}}},
    ]
    stdin, stdout = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n"), io.StringIO()
    serve_stdio(MemoryRuntime(database), stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3, 4]
    assert {tool["name"] for tool in replies[1]["result"]["tools"]} == {
        "memory_remember", "memory_recall", "memory_context", "memory_get", "memory_link", "memory_timeline", "memory_forget",
        "memory_consolidate", "memory_events", "memory_state", "memory_reflect",
    }
    first_id = replies[2]["result"]["structuredContent"]["id"]
    second_id = replies[3]["result"]["structuredContent"]["id"]

    tool_calls = [
        ("memory_recall", {"query": "selected database"}),
        ("memory_context", {"query": "agent integration", "budget": 200}),
        ("memory_get", {"id": first_id}),
        ("memory_link", {"from_id": second_id, "to_id": first_id, "relation": "depends_on"}),
        ("memory_timeline", {"limit": 10}),
        ("memory_consolidate", {"limit": 10}),
        ("memory_events", {"limit": 50}),
        ("memory_forget", {"id": first_id}),
    ]
    follow_up = [{"jsonrpc": "2.0", "id": index, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
                 for index, (name, arguments) in enumerate(tool_calls, start=5)]
    stdin, stdout = io.StringIO("\n".join(json.dumps(item) for item in follow_up) + "\n"), io.StringIO()
    serve_stdio(MemoryRuntime(database), stdin, stdout)
    # A new stdio session must still negotiate before it can invoke tools.
    assert all(reply["error"]["code"] == -32002 for reply in (json.loads(line) for line in stdout.getvalue().splitlines()))

    negotiated = [requests[0], requests[1], *follow_up]
    stdin, stdout = io.StringIO("\n".join(json.dumps(item) for item in negotiated) + "\n"), io.StringIO()
    serve_stdio(MemoryRuntime(database), stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()][1:]
    assert replies[0]["result"]["structuredContent"]["results"][0]["memory"]["id"] == first_id
    assert second_id in {item["id"] for item in replies[1]["result"]["structuredContent"]["memories"]}
    assert replies[2]["result"]["structuredContent"]["id"] == first_id
    assert replies[3]["result"]["structuredContent"] == {"status": "linked"}
    assert {item["id"] for item in replies[4]["result"]["structuredContent"]["memories"]} == {first_id, second_id}
    consolidation = replies[5]["result"]["structuredContent"]
    assert consolidation["created"] is True
    assert set(consolidation["source_memory_ids"]) == {first_id, second_id}
    assert any(event["event_type"] == "memories_consolidated"
               for event in replies[6]["result"]["structuredContent"]["events"])
    assert replies[7]["result"]["structuredContent"] == {"status": "forgotten"}


def test_cli_remember_then_recall_uses_the_same_database(tmp_path):
    database = tmp_path / "cli-brain.db"
    common = [sys.executable, "-m", "memoryd.cli", "--database", str(database)]
    remembered = subprocess.run(
        [*common, "remember", "The CLI keeps durable memories in its configured brain.", "--kind", "state"],
        check=True, capture_output=True, text=True,
    )
    saved = json.loads(remembered.stdout)
    subprocess.run(
        [*common, "remember", "The CLI exposes explicit consolidation for durable summaries.", "--kind", "semantic"],
        check=True, capture_output=True, text=True,
    )
    recalled = subprocess.run(
        [*common, "recall", "configured durable brain"], check=True, capture_output=True, text=True,
    )
    results = json.loads(recalled.stdout)
    assert results[0]["memory"]["id"] == saved["id"]
    consolidated = subprocess.run(
        [*common, "consolidate", "--limit", "10"], check=True, capture_output=True, text=True,
    )
    summary = json.loads(consolidated.stdout)
    assert summary["created"] is True
    assert saved["id"] in summary["source_memory_ids"]
    events = subprocess.run([*common, "events", "--limit", "50"], check=True, capture_output=True, text=True)
    assert any(event["event_type"] == "memories_consolidated" for event in json.loads(events.stdout))


def test_state_contracts_over_rest_mcp_and_cli(tmp_path):
    """The materialized state view works on every public transport."""
    database = tmp_path / "state-brain.db"
    with rest_server(MemoryRuntime(database)) as port:
        for value in ("SQLite", "PostgreSQL"):
            status, _ = http_json(port, "POST", "/remember", {
                "content": f"Memory Runtime: database = {value}.", "kind": "state",
            })
            assert status == 201
        status, state = http_json(port, "GET", "/state?subject=Memory%20Runtime&key=database")
        assert status == 200 and [fact["value"] for fact in state["state"]] == ["PostgreSQL"]
        status, history = http_json(port, "GET", "/state?subject=Memory%20Runtime&key=database&history=true")
        assert status == 200 and {fact["value"] for fact in history["state"]} == {"SQLite", "PostgreSQL"}
        status, reflection = http_json(port, "POST", "/reflect", {"limit": 10})
        assert status == 200 and reflection["review_required"] is True

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "memory_state", "arguments": {"subject": "Memory Runtime", "key": "database"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_reflect", "arguments": {"limit": 10}}},
    ]
    stdin, stdout = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n"), io.StringIO()
    serve_stdio(MemoryRuntime(database), stdin, stdout)
    mcp_state = [json.loads(line) for line in stdout.getvalue().splitlines()][1]
    assert mcp_state["result"]["structuredContent"]["state"][0]["value"] == "PostgreSQL"
    mcp_reflection = [json.loads(line) for line in stdout.getvalue().splitlines()][2]
    assert mcp_reflection["result"]["structuredContent"]["review_required"] is True

    cli = [sys.executable, "-m", "memoryd.cli", "--database", str(database), "state", "--subject", "Memory Runtime", "--key", "database"]
    result = subprocess.run(cli, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)[0]["value"] == "PostgreSQL"
    reflection = subprocess.run([sys.executable, "-m", "memoryd.cli", "--database", str(database), "reflect"], check=True, capture_output=True, text=True)
    assert json.loads(reflection.stdout)["review_required"] is True


def test_cli_doctor_backup_export_and_import_round_trip(tmp_path):
    source = tmp_path / "source.db"
    common = [sys.executable, "-m", "memoryd.cli", "--database", str(source)]
    subprocess.run([*common, "remember", "Portable brain: database = SQLite.", "--kind", "state"], check=True, capture_output=True, text=True)
    doctor = subprocess.run([*common, "doctor"], check=True, capture_output=True, text=True)
    assert json.loads(doctor.stdout)["ok"] is True

    backup_path, export_path, restored = tmp_path / "backup.db", tmp_path / "brain.json", tmp_path / "restored.db"
    subprocess.run([*common, "backup", str(backup_path)], check=True, capture_output=True, text=True)
    subprocess.run([*common, "export", str(export_path)], check=True, capture_output=True, text=True)
    imported = subprocess.run([sys.executable, "-m", "memoryd.cli", "--database", str(restored), "import", str(export_path)], check=True, capture_output=True, text=True)
    assert json.loads(imported.stdout)["database"] == str(restored)
    restored_doctor = subprocess.run([sys.executable, "-m", "memoryd.cli", "--database", str(restored), "doctor"], check=True, capture_output=True, text=True)
    assert backup_path.is_file() and json.loads(restored_doctor.stdout)["ok"] is True
