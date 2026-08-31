# memoryd

`memoryd` is a portable, model-independent memory runtime. It owns a SQLite brain file and gives agents a small API: remember, recall, context, link, timeline, and forget. Agents never need direct database access.

## Run it

Requires Python 3.11+. Install from this checkout:

```powershell
python -m pip install -e .
memoryd --database brain.db serve
```

The daemon listens locally at `http://127.0.0.1:7319`.

```powershell
memoryd --database brain.db remember "We're going with SQLite for the first version." --importance .95
memoryd --database brain.db recall "What database did we choose?"
memoryd --database brain.db context "Help continue designing the memory runtime" --budget 4000
```

Check, back up, and move a brain without inspecting its database schema directly:

```powershell
memoryd --database C:\path\to\brain.db doctor
memoryd --database C:\path\to\brain.db backup C:\backups\brain-2026-08-31.db
memoryd --database C:\path\to\brain.db export C:\backups\brain.json
memoryd --database C:\restored\brain.db import C:\backups\brain.json
```

Backups and exports refuse to overwrite a destination. Imports validate every relationship, embedding, state fact, and event before creating a new brain.

To inspect a brain in a browser without exposing any write controls:

```powershell
memoryd --database C:\path\to\brain.db ui
```

The inspector is loopback-only by default at `http://127.0.0.1:7320`.

## MCP

Run an MCP server over standard input/output. The MCP client should start it with an explicit path to the brain file:

```powershell
memoryd --database C:\path\to\brain.db mcp
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "memoryd": {
      "command": "memoryd",
      "args": ["--database", "C:\\path\\to\\brain.db", "mcp"]
    }
  }
}
```

It exposes `memory_remember`, `memory_recall`, `memory_context`, `memory_get`, `memory_link`, `memory_timeline`, `memory_forget`, `memory_consolidate`, `memory_events`, `memory_state`, and `memory_reflect`. Its tool descriptions guide agents to retain durable, atomic knowledge without direct database access.

## REST API

```text
POST /remember  {"content":"...", "source":"conversation", "importance":0.8}
POST /recall    {"query":"database decision", "limit":10}
POST /context   {"query":"continue the project", "budget":4000}
POST /link      {"from_id":"...", "to_id":"...", "relation":"related_to"}
POST /forget/:id
GET  /memories/:id
GET  /timeline
GET  /health
```

`brain.db` is a self-contained SQLite file. It stores atomic memories, typed relationships, entities, deterministic local embeddings, full-text search, confidence, importance, reinforcement, supersession history, and an append-only event log. SQLite runs in WAL mode with a busy timeout for responsive concurrent local clients.

## Context, retrieval, and consolidation

Recall combines full-text and deterministic local-vector results with importance, confidence, recency, and reinforcement. Pass a model-backed `EmbeddingProvider` to `MemoryRuntime` when the application needs richer semantic embeddings; the stored provider name keeps vectors isolated by model.

For a real local embedding model, install `sentence-transformers` and initialize the runtime with `SentenceTransformerEmbeddingProvider`. The built-in deterministic provider remains the portable fallback and is always covered by the evaluation suite.

`context` returns structured `current_state`, `decisions`, `relevant_memories`, and `open_questions` sections, in addition to the prompt-ready `text` field. It avoids highly overlapping results and briefly caches unchanged requests.

Run explicit consolidation only when a compact durable view is useful:

```powershell
memoryd --database C:\path\to\brain.db consolidate
memoryd --database C:\path\to\brain.db events
```

Consolidation produces a new memory and `derived_from` links to every source; it never deletes source memories.

`reflect` is review-only: it proposes consolidation, open-question, and possible-duplicate reviews but never changes the brain. Inspect a proposal, then explicitly call the appropriate write operation.

## Current state

`memoryd` recognizes explicit state assignments and maintains a fast materialized current-state view while preserving history. Use metadata for unambiguous state updates:

```json
{
  "content": "Memory Runtime will use SQLite for V1.",
  "kind": "state",
  "metadata": {"state": {"subject": "Memory Runtime", "key": "database", "value": "SQLite"}}
}
```

Writing a later value for the same subject and key automatically makes the earlier source historical and adds a `supersedes` link. Read current facts with `GET /state`, `memoryd --database C:\path\to\brain.db state`, or the `memory_state` MCP tool. Add `history=true` or `--history` to include prior values.

## Quality gates

Run the deterministic regression corpus before trusting ranking or consolidation changes:

```powershell
python -m memoryd.evals
python -m pytest -q
```

The corpus checks retrieval correctness, exclusion of superseded state, context budget compliance, and consolidation provenance.

## Development

```powershell
python -m pytest
```
