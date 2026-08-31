from memoryd.runtime import MemoryRuntime
from memoryd.mcp import MCPServer


def test_decision_recall_and_context(tmp_path):
    brain = MemoryRuntime(tmp_path / "brain.db")
    choice = brain.remember("We're going with SQLite for the first version.", importance=.95)
    old = brain.remember("PostgreSQL might be better later.", confidence=.42)
    updated = brain.remember("PostgreSQL is selected for version 2.", supersedes=old.id, importance=.9)

    assert choice.kind == "decision"
    results = brain.recall("What database did we choose?")
    assert results
    assert any(result.memory.id == choice.id for result in results)
    assert brain.store.get(old.id).status == "superseded"
    assert brain.get(updated.id)["relationships"][0]["relation"] == "supersedes"
    assert "SQLite" in brain.context("database decision", budget=1000)["text"]


def test_forget_hides_a_memory(tmp_path):
    brain = MemoryRuntime(tmp_path / "brain.db")
    memory = brain.remember("A private temporary fact.")
    brain.forget(memory.id)
    assert brain.recall("private temporary fact") == []


def test_mcp_lifecycle_and_memory_tool(tmp_path):
    server = MCPServer(MemoryRuntime(tmp_path / "brain.db"))
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
    assert init["result"]["protocolVersion"] == "2025-11-25"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {tool["name"] for tool in listed["result"]["tools"]} >= {"memory_remember", "memory_context"}
    saved = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_remember", "arguments": {"content": "MCP is the agent interface.", "kind": "decision"}}})
    assert saved["result"]["structuredContent"]["kind"] == "decision"
