from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .doctor import diagnose
from .runtime import MemoryRuntime
from .mcp import serve_stdio
from .ops import backup, export_json, import_json
from .server import serve
from .ui import serve as serve_ui


def _metadata(raw: str | None) -> dict[str, Any] | None:
    return json.loads(raw) if raw else None


def main() -> None:
    parser = argparse.ArgumentParser(prog="memoryd", description="Portable model-independent memory runtime")
    parser.add_argument("--database", default="brain.db", help="path to the portable SQLite brain")
    commands = parser.add_subparsers(dest="command", required=True)
    daemon = commands.add_parser("serve", help="run REST daemon")
    daemon.add_argument("--host", default="127.0.0.1")
    daemon.add_argument("--port", type=int, default=7319)
    ui = commands.add_parser("ui", help="run read-only local memory inspector")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7320)
    commands.add_parser("mcp", help="run MCP server over standard input/output")
    remember = commands.add_parser("remember", help="store a memory")
    remember.add_argument("content")
    remember.add_argument("--source", default="conversation")
    remember.add_argument("--kind")
    remember.add_argument("--confidence", type=float, default=.8)
    remember.add_argument("--importance", type=float, default=.5)
    remember.add_argument("--metadata")
    remember.add_argument("--supersedes")
    recall = commands.add_parser("recall", help="retrieve relevant memories")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=10)
    context = commands.add_parser("context", help="build LLM-ready context")
    context.add_argument("query")
    context.add_argument("--budget", type=int, default=4000)
    consolidate = commands.add_parser("consolidate", help="create a provenance-linked consolidated memory")
    consolidate.add_argument("--limit", type=int, default=200)
    events = commands.add_parser("events", help="show runtime event history")
    events.add_argument("--limit", type=int, default=50)
    state = commands.add_parser("state", help="show materialized current state")
    state.add_argument("--subject")
    state.add_argument("--key")
    state.add_argument("--history", action="store_true")
    reflect = commands.add_parser("reflect", help="produce reviewable memory reflection proposals")
    reflect.add_argument("--limit", type=int, default=200)
    backup_command = commands.add_parser("backup", help="create a consistent SQLite backup without overwriting")
    backup_command.add_argument("destination")
    export_command = commands.add_parser("export", help="write a portable JSON brain export without overwriting")
    export_command.add_argument("destination")
    import_command = commands.add_parser("import", help="create a new database from a validated JSON export")
    import_command.add_argument("source")
    commands.add_parser("doctor", help="run read-only brain integrity and consistency checks")
    commands.add_parser("timeline", help="show recent history")
    args = parser.parse_args()
    if args.command == "import":
        print(json.dumps({"database": str(import_json(args.source, args.database))}, indent=2)); return
    if args.command == "backup":
        print(json.dumps({"backup": str(backup(args.database, args.destination))}, indent=2)); return
    if args.command == "export":
        print(json.dumps({"export": str(export_json(args.database, args.destination))}, indent=2)); return
    if args.command == "doctor":
        print(json.dumps(diagnose(args.database), indent=2)); return
    runtime = MemoryRuntime(Path(args.database))
    if args.command == "serve": serve(runtime, args.host, args.port); return
    if args.command == "ui": serve_ui(runtime, args.host, args.port); return
    if args.command == "mcp": serve_stdio(runtime); return
    if args.command == "remember":
        print(json.dumps(runtime.remember(args.content, source=args.source, kind=args.kind,
              confidence=args.confidence, importance=args.importance, metadata=_metadata(args.metadata),
              supersedes=args.supersedes).to_dict(), indent=2)); return
    if args.command == "recall": print(json.dumps([r.to_dict() for r in runtime.recall(args.query, limit=args.limit)], indent=2)); return
    if args.command == "context": print(json.dumps(runtime.context(args.query, budget=args.budget), indent=2)); return
    if args.command == "consolidate": print(json.dumps(runtime.consolidate(limit=args.limit), indent=2)); return
    if args.command == "events": print(json.dumps(runtime.events(limit=args.limit), indent=2)); return
    if args.command == "state": print(json.dumps(runtime.state(subject=args.subject, key=args.key, history=args.history), indent=2)); return
    if args.command == "reflect": print(json.dumps(runtime.reflect(limit=args.limit), indent=2)); return
    print(json.dumps(runtime.timeline(), indent=2))


if __name__ == "__main__":
    main()
