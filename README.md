# MemoryD

**A portable, model-independent memory runtime for AI agents.**

MemoryD gives agents a durable brain they can share across models, applications, and machines. Point Codex, Claude, local agents, game NPCs, or your own agent system at the same `brain.db`; they retrieve the current project state, decisions, open questions, and evidence behind them without receiving an entire conversation history.

It is local-first, dependency-light, and built around a simple rule: **agents ask the memory runtime to remember and recall; they never manipulate the database directly.**

> Status: early alpha. The current release is designed for local, single-user use and is covered by deterministic unit, integration, and retrieval-quality checks.

## Why MemoryD?

Long-running agent work loses continuity. Raw transcripts become too large, summaries drift, and switching models resets context.

MemoryD maintains layers of durable knowledge instead:

```text
Raw conversations and sources
            ↓
Atomic memories with confidence and provenance
            ↓
Current state, relationships, and consolidated views
            ↓
Small, relevant context sent to the next agent
```

The brain is one portable SQLite file. Move it to another machine, connect a different model, and the ongoing work does not have to start over.

## What it does today

- Stores typed memories: decisions, state, procedures, semantic facts, and speculation.
- Retrieves with full-text search, local deterministic vectors, recency, importance, confidence, and reinforcement.
- Builds compact, structured agent context: current state, decisions, relevant memories, and open questions.
- Preserves history with `supersedes` and `derived_from` provenance links.
- Materializes current state while keeping prior values available as history.
- Extracts entities and surfaces related memories automatically.
- Supports review-first reflection and explicit, non-destructive consolidation.
- Runs as a CLI, REST daemon, MCP server, and small read-only local inspector.
- Includes health checks, online SQLite backup, validated JSON export/import, and a fixed evaluation corpus.

## Quick start

Requires Python 3.11 or newer.

```powershell
git clone https://github.com/deadadm1n/MemoryD.git
cd MemoryD
python -m pip install -e .
```

Create a portable brain and add a first decision:

```powershell
memoryd --database brain.db remember "Version 1 will use SQLite." --kind decision --confidence .99 --importance .95
memoryd --database brain.db context "Continue the MemoryD project" --budget 3000
```

Run the local REST daemon:

```powershell
memoryd --database brain.db serve
```

It listens on `http://127.0.0.1:7319` by default.

## Connect an AI agent through MCP

MemoryD exposes a local stdio MCP server. Start it with an explicit brain path:

```powershell
memoryd --database C:\path\to\brain.db mcp
```

An MCP client can use this configuration shape:

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

The server provides:

```text
memory_remember     memory_recall       memory_context
memory_get          memory_link         memory_timeline
memory_forget       memory_state        memory_events
memory_consolidate  memory_reflect
```

Write tools are explicit. `memory_reflect` only proposes possible consolidation, open-question, or duplicate reviews; it never mutates the brain.

## Core workflow

### Remember durable facts

Keep memories atomic and self-contained. Explicit decisions deserve high confidence; possibilities should remain speculation.

```powershell
memoryd --database brain.db remember "MCP is the primary agent interface." --kind decision --confidence .99 --importance .95
memoryd --database brain.db remember "Open question: benchmark a production local embedding model." --kind speculation --confidence .60
```

### Maintain current state

Use a state memory when an agent needs a reliable answer to "what is true now?" A later value for the same subject and key automatically retains the old source as historical evidence.

```powershell
memoryd --database brain.db remember "MemoryD: stage = hardening." --kind state
memoryd --database brain.db state --subject MemoryD --key stage
```

For arbitrary keys, make the state assignment explicit in metadata:

```json
{
  "content": "MemoryD uses SQLite for V1.",
  "kind": "state",
  "metadata": {
    "state": {
      "subject": "MemoryD",
      "key": "database",
      "value": "SQLite"
    }
  }
}
```

### Retrieve the right context

```powershell
memoryd --database brain.db recall "Why did we choose a local database?"
memoryd --database brain.db context "Help continue the MemoryD project" --budget 4000
```

`context` returns structured JSON plus prompt-ready text. It is intended to give an agent the relevant working set, not an ocean of history.

## REST API

```text
POST /remember       POST /recall        POST /context
POST /link           POST /consolidate   POST /reflect
POST /forget/:id
GET  /memories/:id   GET  /timeline      GET  /state
GET  /events         GET  /health
```

## Inspect, back up, and move a brain

Open a read-only local inspector with search, timeline, state, provenance, and entity relationships:

```powershell
memoryd --database brain.db ui
```

It is loopback-only by default at `http://127.0.0.1:7320`.

Use the operational tools instead of handling SQLite tables directly:

```powershell
memoryd --database brain.db doctor
memoryd --database brain.db backup C:\backups\brain.db
memoryd --database brain.db export C:\backups\brain.json
memoryd --database C:\restored\brain.db import C:\backups\brain.json
```

Backup and export refuse to overwrite destinations. Import validates memory IDs, relationships, vectors, state facts, and event history before creating a new database.

## Quality gates

MemoryD includes a repeatable evaluation corpus. Run it before trusting retrieval or consolidation changes:

```powershell
python -m memoryd.evals
python -m pytest -q
```

The evaluator checks retrieval correctness, exclusion of superseded state, context-budget compliance, and consolidation provenance.

## Embeddings

The built-in `HashEmbeddingProvider` is deterministic, local, and dependency-free. For richer semantic similarity, applications can pass `SentenceTransformerEmbeddingProvider` to `MemoryRuntime` after installing `sentence-transformers`.

Provider names are stored with their vectors, so embeddings from different models remain isolated in the same brain.

## Design principles

- **Portable:** the brain is a self-contained SQLite database.
- **Model-independent:** no memory is owned by a particular LLM.
- **Provenance first:** derived knowledge links back to its sources.
- **Local-first:** no cloud account or API key is required for the core runtime.
- **Review before autonomy:** reflection proposes; explicit tools make changes.
- **Small interface, deep internals:** agents get `remember`, `recall`, `context`, and a few supporting operations.

## Roadmap

- Benchmark and document a recommended real local embedding provider.
- Add configurable background scheduling for reviewable reflection cycles.
- Expand retrieval evaluation with real-world project corpora.
- Add secure multi-user and remote deployment modes without weakening the local-first default.

## Development

```powershell
python -m pytest -q
python -m memoryd.evals
```
