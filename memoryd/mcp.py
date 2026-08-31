"""Dependency-free MCP stdio adapter for memoryd.

The transport uses newline-delimited JSON-RPC as required for MCP stdio servers.
Nothing other than protocol messages is written to standard output.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, TextIO

from .runtime import MemoryRuntime

PROTOCOL_VERSION = "2025-11-25"


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


TOOLS = [
    {"name": "memory_remember", "title": "Remember", "description": "Store one durable, atomic memory. Use supersedes instead of overwriting a changed fact.",
     "inputSchema": _schema({"content": {"type": "string"}, "source": {"type": "string", "default": "conversation"},
         "kind": {"type": "string", "enum": ["decision", "state", "procedural", "semantic", "speculation"]},
         "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": .8},
         "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": .5},
         "metadata": {"type": "object"}, "supersedes": {"type": "string"}}, ["content"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}},
    {"name": "memory_recall", "title": "Recall", "description": "Find relevant active memories with hybrid lexical, importance, confidence, recency, and reinforcement scoring.",
     "inputSchema": _schema({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
         "kind": {"type": "string"}}, ["query"]),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_context", "title": "Build context", "description": "Build a compact LLM-ready context for continuing a task or conversation.",
     "inputSchema": _schema({"query": {"type": "string"}, "budget": {"type": "integer", "minimum": 100, "default": 4000}}, ["query"]),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_get", "title": "Get memory", "description": "Fetch a memory and its provenance relationships by ID.",
     "inputSchema": _schema({"id": {"type": "string"}}, ["id"]),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_link", "title": "Link memories", "description": "Create a typed directional relationship between two existing memories.",
     "inputSchema": _schema({"from_id": {"type": "string"}, "to_id": {"type": "string"}, "relation": {"type": "string"}}, ["from_id", "to_id", "relation"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_timeline", "title": "Timeline", "description": "Return the most recently updated active memories.",
     "inputSchema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_forget", "title": "Forget memory", "description": "Soft-forget a memory so it no longer appears in normal recall. Do not use for changed facts; use supersedes.",
     "inputSchema": _schema({"id": {"type": "string"}}, ["id"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_consolidate", "title": "Consolidate memories", "description": "Create a provenance-linked durable summary from recent atomic memories. This does not delete its sources.",
     "inputSchema": _schema({"limit": {"type": "integer", "minimum": 2, "maximum": 1000, "default": 200}}),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}},
    {"name": "memory_events", "title": "Memory events", "description": "Read the append-only runtime event history for auditing and debugging.",
     "inputSchema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_state", "title": "Current state", "description": "Read current materialized state facts, or their history, by subject and key.",
     "inputSchema": _schema({"subject": {"type": "string"}, "key": {"type": "string"}, "history": {"type": "boolean", "default": False}, "at": {"type": "string", "description": "ISO-8601 timestamp for historical lookup"}}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_reflect", "title": "Reflection proposals", "description": "Generate reviewable consolidation, open-question, and duplicate proposals without changing memory.",
     "inputSchema": _schema({"limit": {"type": "integer", "minimum": 2, "maximum": 1000, "default": 200}}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_observe", "title": "Observe experience", "description": "Analyze an experience and store only conservative durable candidates; ordinary chatter is ignored.",
     "inputSchema": _schema({"content": {"type": "string"}, "actor": {"type": "string"}, "context": {"type": "object"}, "source": {"type": "string", "default": "observation"}}, ["content"]),
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}},
    {"name": "memory_beliefs", "title": "Evidence-backed beliefs", "description": "Read conservative direct beliefs and unresolved conflicts with their source memory IDs.",
     "inputSchema": _schema({}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
    {"name": "memory_explain", "title": "Explain belief", "description": "Return one direct belief and the exact active memories that support it.",
     "inputSchema": _schema({"subject": {"type": "string"}, "key": {"type": "string"}, "statement": {"type": "string"}}),
     "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}},
]


class MCPServer:
    def __init__(self, runtime: MemoryRuntime) -> None:
        self.runtime = runtime
        self.initialized = False

    @staticmethod
    def _response(message_id: str | int, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: str | int | None, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_result(value: Any, error: bool = False) -> dict[str, Any]:
        text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": error}
        if not error and isinstance(value, dict):
            result["structuredContent"] = value
        return result

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., Any]] = {
            "memory_remember": lambda **kw: self.runtime.remember(**kw).to_dict(),
            "memory_recall": lambda **kw: {"results": [item.to_dict() for item in self.runtime.recall(**kw)]},
            "memory_context": lambda **kw: self.runtime.context(**kw),
            "memory_get": lambda **kw: self.runtime.get(kw["id"]),
            "memory_link": lambda **kw: (self.runtime.link(**kw) or {"status": "linked"}),
            "memory_timeline": lambda **kw: {"memories": self.runtime.timeline(**kw)},
            "memory_forget": lambda **kw: (self.runtime.forget(kw["id"]) or {"status": "forgotten"}),
            "memory_consolidate": lambda **kw: self.runtime.consolidate(**kw),
            "memory_events": lambda **kw: {"events": self.runtime.events(**kw)},
            "memory_state": lambda **kw: {"state": self.runtime.state(**kw)},
            "memory_reflect": lambda **kw: self.runtime.reflect(**kw),
            "memory_observe": lambda **kw: self.runtime.observe(**kw),
            "memory_beliefs": lambda **kw: self.runtime.beliefs(),
            "memory_explain": lambda **kw: self.runtime.explain(**kw),
        }
        if name not in handlers:
            raise LookupError(f"unknown tool: {name}")
        value = handlers[name](**arguments)
        if value is None:
            raise KeyError("memory not found")
        return self._tool_result(value)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if message.get("jsonrpc") != "2.0":
            return self._error(message.get("id"), -32600, "JSON-RPC 2.0 is required")
        method, message_id = message.get("method"), message.get("id")
        if method == "notifications/initialized":
            self.initialized = True
            return None
        if method == "initialize":
            self.initialized = False
            return self._response(message_id, {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                                                "serverInfo": {"name": "memoryd", "version": "0.5.0"}})
        if method == "ping":
            return self._response(message_id, {})
        if not self.initialized:
            return self._error(message_id, -32002, "server is not initialized")
        if method == "tools/list":
            return self._response(message_id, {"tools": TOOLS})
        if method == "tools/call":
            params = message.get("params", {})
            try:
                return self._response(message_id, self._call(params["name"], params.get("arguments", {})))
            except LookupError as exc:
                return self._error(message_id, -32601, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                return self._response(message_id, self._tool_result(str(exc), error=True))
        return self._error(message_id, -32601, f"method not found: {method}")


def serve_stdio(runtime: MemoryRuntime, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    server = MCPServer(runtime)
    for line in stdin:
        try:
            reply = server.handle(json.loads(line))
        except json.JSONDecodeError:
            reply = server._error(None, -32700, "parse error")
        if reply is not None:
            stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            stdout.flush()
