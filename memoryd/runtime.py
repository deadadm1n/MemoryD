from __future__ import annotations

import math
import re
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider, HashEmbeddingProvider, cosine_similarity
from .beliefs import build_belief_set
from .cognition import DeterministicObservationAnalyzer, ObservationAnalyzer
from .entities import extract_entities
from .models import Memory, RecallResult
from .prospective import derive_triggers, score_relevance
from .reflection import propose as reflection_proposals
from .store import BrainStore


class MemoryRuntime:
    def __init__(self, database: str | Path = "brain.db", *, embedding_provider: EmbeddingProvider | None = None,
                 observation_analyzer: ObservationAnalyzer | None = None) -> None:
        self.store = BrainStore(database)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.observation_analyzer = observation_analyzer or DeterministicObservationAnalyzer()
        self._cache: dict[tuple[str, int], tuple[float, int, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()
        self._revision = 0

    def _invalidate_cache(self) -> None:
        with self._cache_lock:
            self._revision += 1
            self._cache.clear()

    @staticmethod
    def classify(content: str) -> str:
        text = content.lower()
        if any(p in text for p in ("decided", "will use", "we're going with", "we are going with", "selected", "rejected")):
            return "decision"
        if any(p in text for p in ("current ", "status", "stage", "is now", "currently")):
            return "state"
        if any(p in text for p in ("how to", "first ", "then ", "procedure")):
            return "procedural"
        if any(p in text for p in ("might", "may ", "perhaps", "considering")):
            return "speculation"
        return "semantic"

    @staticmethod
    def extract_state(content: str, metadata: dict[str, Any]) -> tuple[str, str, str] | None:
        """Recognize explicit state assignments without guessing from prose.

        Callers can use ``metadata.state`` for arbitrary state names. The text
        fallback intentionally accepts only ``subject: key = value`` or a small
        set of unambiguous global assignments, preserving correctness over recall.
        """
        supplied = metadata.get("state")
        if isinstance(supplied, dict):
            try:
                subject, key, value = (str(supplied[name]).strip() for name in ("subject", "key", "value"))
            except KeyError as exc:
                raise ValueError("metadata.state requires subject, key, and value") from exc
            if subject and key and value:
                return subject, key.casefold().replace(" ", "_"), value
            raise ValueError("metadata.state values must not be empty")
        match = re.match(r"^\s*(?:(?P<subject>[A-Za-z][\w .-]{1,64})\s*:\s*)?"
                         r"(?P<key>database|stage|status|interface|owner|priority|version)\s*(?:=|is)\s*"
                         r"(?P<value>.+?)[.?!]?\s*$", content, flags=re.IGNORECASE)
        if not match:
            return None
        subject = (match.group("subject") or str(metadata.get("project", "global"))).strip()
        return subject, match.group("key").casefold(), match.group("value").strip()

    def remember(self, content: str, *, source: str = "conversation", kind: str | None = None,
                 confidence: float = 0.8, importance: float = 0.5,
                 metadata: dict[str, Any] | None = None, supersedes: str | None = None) -> Memory:
        if not content or not content.strip():
            raise ValueError("content is required")
        if not 0 <= confidence <= 1 or not 0 <= importance <= 1:
            raise ValueError("confidence and importance must be between 0 and 1")
        duplicate = self.store.find_duplicate(content)
        if duplicate:
            self.store.reinforce([duplicate.id])
            self.store.record_event("memory_reinforced", duplicate.id, {"reason": "duplicate"})
            return self.store.get(duplicate.id) or duplicate
        kind = kind or self.classify(content)
        entities = extract_entities(content)
        record_metadata = dict(metadata or {})
        memory = self.store.create(content, kind=kind, source=source, confidence=float(confidence),
                                   importance=float(importance), metadata=record_metadata)
        self.store.add_entities(memory.id, ((name, "concept") for name in entities))
        self.store.upsert_embedding(memory.id, self.embedding_provider.name, self.embedding_provider.embed(memory.content))
        triggers = derive_triggers(memory)
        self.store.add_prospective_triggers(memory.id, ((item.phrase, item.category, item.weight) for item in triggers))
        self.store.record_event("memory_created", memory.id, {"kind": kind, "source": source, "entities": entities})
        state = self.extract_state(content, record_metadata) if kind == "state" or isinstance(record_metadata.get("state"), dict) else None
        replaced_by_state = None
        if state:
            subject, state_key, value = state
            replaced_by_state = self.store.set_state(subject, state_key, value, memory.id)
            self.store.record_event("state_set", memory.id, {"subject": subject, "key": state_key, "value": value})
        if supersedes:
            if not self.store.get(supersedes):
                raise KeyError(f"unknown memory: {supersedes}")
            self.store.link(memory.id, supersedes, "supersedes")
            self.store.update_status(supersedes, "superseded")
            self.store.record_event("memory_superseded", supersedes, {"by": memory.id})
        if replaced_by_state and replaced_by_state != memory.id and replaced_by_state != supersedes:
            self.store.link(memory.id, replaced_by_state, "supersedes")
            self.store.update_status(replaced_by_state, "superseded")
            self.store.record_event("state_replaced", replaced_by_state, {"by": memory.id})
        self._invalidate_cache()
        return memory

    def observe(self, content: str, *, actor: str | None = None, context: dict[str, Any] | None = None,
                source: str = "observation") -> dict[str, Any]:
        """Let a pluggable cognition layer decide what an experience is worth keeping.

        The default analyzer is intentionally conservative: ordinary chatter is
        ignored, while explicit decisions, state changes, and questions become
        small durable memories with observation provenance.
        """
        candidates = self.observation_analyzer.analyze(content, actor=actor, context=context)
        stored = []
        for candidate in candidates:
            metadata: dict[str, Any] = {"observation": {"actor": candidate.actor, "context": candidate.context}}
            if candidate.state:
                metadata["state"] = candidate.state
            memory = self.remember(candidate.content, source=source, kind=candidate.kind,
                                   confidence=candidate.confidence, importance=candidate.importance, metadata=metadata)
            stored.append(memory.to_dict())
        self.store.record_event("observation_processed", payload={"candidate_count": len(candidates), "stored_ids": [item["id"] for item in stored]})
        return {"stored": stored, "ignored": not stored, "candidate_count": len(candidates)}

    def recall(self, query: str, *, limit: int = 10, kind: str | None = None) -> list[RecallResult]:
        candidate_limit = max(limit * 4, 20)
        lexical = self.store.search_fts(query, candidate_limit)
        lexical_ranks = {memory.id: index for index, (memory, _) in enumerate(lexical, start=1)}
        candidates = {memory.id: memory for memory, _ in lexical}
        query_embedding = self.embedding_provider.embed(query)
        similarities = sorted(((memory, cosine_similarity(query_embedding, vector))
                               for memory, vector in self.store.embeddings(self.embedding_provider.name)),
                              key=lambda item: item[1], reverse=True)
        semantic_ranks = {memory.id: index for index, (memory, score) in enumerate(similarities[:candidate_limit], start=1)
                          if score > 0}
        candidates.update({memory.id: memory for memory, score in similarities[:candidate_limit] if score > 0})
        # If a query has no lexical or vector signal, retain a small working set.
        if not candidates:
            candidates = {memory.id: memory for memory in self.store.recent(max(limit * 2, 10), kind=kind)}
        results: list[RecallResult] = []
        current = datetime.now(UTC)
        for memory in candidates.values():
            if kind and memory.kind != kind:
                continue
            age_days = max(0, (current - datetime.fromisoformat(memory.updated_at)).total_seconds() / 86400)
            lexical_score = 1 / (60 + lexical_ranks[memory.id]) if memory.id in lexical_ranks else 0
            semantic_score = 1 / (60 + semantic_ranks[memory.id]) if memory.id in semantic_ranks else 0
            recency = math.exp(-age_days / 180)
            score = (lexical_score + semantic_score) * 10 + memory.importance * .20 + memory.strength * .15 + memory.confidence * .08 + recency * .05
            reasons = [f"{memory.kind} memory"]
            if memory.id in lexical_ranks: reasons.append("keyword match")
            if memory.id in semantic_ranks: reasons.append("semantic match")
            if memory.importance >= .75: reasons.append("high importance")
            if age_days < 14: reasons.append("recent")
            results.append(RecallResult(memory, score, reasons))
        results.sort(key=lambda item: item.score, reverse=True)
        results = results[:limit]
        self.store.reinforce(item.memory.id for item in results)
        self.store.record_event("memory_recalled", payload={"count": len(results), "memory_ids": [item.memory.id for item in results]})
        return results

    def context(self, query: str, *, budget: int = 4000) -> dict[str, Any]:
        cache_key = (query, budget)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > time.monotonic() and cached[1] == self._revision:
                return deepcopy(cached[2])
        # A conservative four-characters-per-token estimate keeps output below the requested budget.
        available, selected, seen_tokens = max(400, budget * 4), [], []
        sections: dict[str, list[dict[str, Any]]] = {"current_state": [], "decisions": [], "relevant_memories": [], "open_questions": [], "likely_relevant_soon": []}
        for result in self.recall(query, limit=40):
            item = result.memory
            tokens = set(re.findall(r"[a-z0-9_]+", item.content.casefold()))
            overlap = max((len(tokens & earlier) / max(1, len(tokens | earlier)) for earlier in seen_tokens), default=0.0)
            rendered = f"- [{item.kind}, confidence {item.confidence:.2f}] {item.content}"
            if overlap > .82 or len(rendered) > available:
                continue
            packed = {"id": item.id, "content": item.content, "kind": item.kind, "confidence": item.confidence}
            selected.append(packed); seen_tokens.append(tokens); available -= len(rendered)
            if item.kind == "state": section = "current_state"
            elif item.kind == "decision": section = "decisions"
            elif item.kind == "speculation" or "open question" in item.content.casefold() or "?" in item.content: section = "open_questions"
            else: section = "relevant_memories"
            sections[section].append(packed)
        selected_ids = {item["id"] for item in selected}
        prospective = sorted(((memory, score_relevance(memory, query)) for memory in self.store.prospective_memories()),
                             key=lambda item: item[1].score, reverse=True)
        for memory, match in prospective[:8]:
            if match.score < .15:
                continue
            if memory.id in selected_ids:
                existing = next(item for item in selected if item["id"] == memory.id)
                existing["prospective_score"] = match.score
                existing["triggers"] = [item.phrase for item in match.matched]
                for memories in sections.values():
                    if existing in memories:
                        memories.remove(existing)
                sections["likely_relevant_soon"].append(existing)
                continue
            rendered = f"- [{memory.kind}, anticipated relevance {match.score:.2f}] {memory.content}"
            if len(rendered) > available:
                continue
            packed = {"id": memory.id, "content": memory.content, "kind": memory.kind, "confidence": memory.confidence,
                      "prospective_score": match.score, "triggers": [item.phrase for item in match.matched]}
            selected.append(packed); selected_ids.add(memory.id); available -= len(rendered)
            sections["likely_relevant_soon"].append(packed)
        lines = []
        for title, memories in sections.items():
            if memories:
                lines.append(title.replace("_", " ").upper())
                lines.extend(f"- [{memory['kind']}] {memory['content']}" for memory in memories)
        result = {"query": query, "budget": budget, "memories": selected, "sections": sections,
                  "text": "\n".join(lines), "stats": self.store.stats()}
        with self._cache_lock:
            self._cache[cache_key] = (time.monotonic() + 5, self._revision, deepcopy(result))
        return result

    def get(self, memory_id: str) -> dict[str, Any] | None:
        memory = self.store.get(memory_id)
        if not memory:
            return None
        return {**memory.to_dict(), "relationships": self.store.related(memory_id),
                "entity_related": self.store.entity_related(memory_id)}

    def timeline(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return [memory.to_dict() for memory in self.store.recent(limit)]

    def link(self, from_id: str, to_id: str, relation: str) -> None:
        if not self.store.get(from_id) or not self.store.get(to_id):
            raise KeyError("both memory IDs must exist")
        self.store.link(from_id, to_id, relation)
        self.store.record_event("memories_linked", from_id, {"to": to_id, "relation": relation})
        self._invalidate_cache()

    def forget(self, memory_id: str) -> None:
        if not self.store.get(memory_id):
            raise KeyError(f"unknown memory: {memory_id}")
        self.store.update_status(memory_id, "forgotten")
        self.store.record_event("memory_forgotten", memory_id)
        self._invalidate_cache()

    def consolidate(self, *, limit: int = 200) -> dict[str, Any]:
        """Create one traceable, deterministic summary from recent atomic memories.

        This is deliberately explicit rather than an automatic background mutation.
        An LLM reflection provider can later replace the renderer while retaining the
        same `derived_from` relationship contract.
        """
        source_memories = [memory for memory in self.store.recent(limit)
                           if not memory.metadata.get("consolidation")]
        if len(source_memories) < 2:
            return {"created": False, "reason": "at least two non-consolidated active memories are required"}
        groups = {"decisions": [], "current state": [], "open questions": [], "other": []}
        for memory in source_memories:
            key = "decisions" if memory.kind == "decision" else "current state" if memory.kind == "state" else "open questions" if memory.kind == "speculation" else "other"
            groups[key].append(memory.content)
        lines = [f"Consolidated view of {len(source_memories)} recent memories."]
        for title, contents in groups.items():
            if contents:
                lines.append(f"{title.title()}: " + "; ".join(contents[:8]))
        summary = self.remember("\n".join(lines), source="consolidation", kind="semantic", confidence=.75,
                                importance=max(memory.importance for memory in source_memories),
                                metadata={"consolidation": True, "derived_from": [memory.id for memory in source_memories]})
        for memory in source_memories:
            self.store.link(summary.id, memory.id, "derived_from")
        self.store.record_event("memories_consolidated", summary.id, {"source_memory_ids": [memory.id for memory in source_memories]})
        self._invalidate_cache()
        return {"created": True, "memory": summary.to_dict(), "source_memory_ids": [memory.id for memory in source_memories]}

    def events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.events(limit)

    def beliefs(self) -> dict[str, Any]:
        return build_belief_set(self.store.recent(10_000))

    def explain(self, *, subject: str | None = None, key: str | None = None, statement: str | None = None) -> dict[str, Any]:
        """Explain one direct belief with its exact source memories.

        Callers must name a state subject/key or an exact decision statement;
        MemoryD intentionally refuses to invent a causal explanation from a
        fuzzy query.
        """
        if not ((subject and key) or statement):
            raise ValueError("provide subject and key, or an exact statement")
        belief_set = self.beliefs()
        target = next((belief for belief in belief_set["beliefs"]
                       if (subject and key and belief.get("subject") == subject and belief.get("key") == key)
                       or (statement and belief.get("statement") == statement)), None)
        if not target:
            raise KeyError("no direct active belief matches the requested explanation")
        evidence = [self.get(memory_id) for memory_id in target["evidence_ids"]]
        return {"belief": target, "explanation": "This is a direct active assertion; inspect the evidence records below.",
                "evidence": [item for item in evidence if item]}

    def reflect(self, *, limit: int = 200) -> dict[str, Any]:
        """Produce reviewable reflection proposals without changing the brain."""
        result = reflection_proposals(self.store.recent(limit), limit)
        self.store.record_event("reflection_proposed", payload={"proposal_count": len(result["proposals"]), "memory_count": result["memory_count"]})
        return result

    def state(self, *, subject: str | None = None, key: str | None = None, history: bool = False, at: str | None = None) -> list[dict[str, Any]]:
        return self.store.state(subject, key, history, at)
