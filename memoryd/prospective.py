"""Deterministic prospective-memory cues.

Prospective memory is a reminder to retrieve a durable fact *when a later
situation makes it useful*.  This module deliberately does not predict the
future or call a model.  It extracts a small, explainable set of cues from a
memory and scores a new situation using only lexical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from .models import Memory


@dataclass(frozen=True)
class ProspectiveTrigger:
    """A situation phrase that may warrant retrieving a memory."""

    phrase: str
    category: str
    weight: float


@dataclass(frozen=True)
class ProspectiveMatch:
    """An explainable relevance score for one situation."""

    score: float
    matched: tuple[ProspectiveTrigger, ...]


_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+#/-]*", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_TECHNOLOGIES: tuple[tuple[str, str], ...] = (
    (r"\bsqlite\b", "SQLite"),
    (r"\bpostgre(?:sql|s)\b", "PostgreSQL"),
    (r"\bmysql\b", "MySQL"),
    (r"\bmongodb\b", "MongoDB"),
    (r"\bredis\b", "Redis"),
    (r"\bpython\b", "Python"),
    (r"\btypescript\b", "TypeScript"),
    (r"\bjavascript\b", "JavaScript"),
    (r"\bnode(?:\.js)?\b", "Node.js"),
    (r"\bfastapi\b", "FastAPI"),
    (r"\bdjango\b", "Django"),
    (r"\breact\b", "React"),
    (r"\bdocker\b", "Docker"),
    (r"\bkubernetes\b|\bk8s\b", "Kubernetes"),
    (r"\bgit(?:hub)?\b", "GitHub"),
    (r"\bmcp\b|\bmodel context protocol\b", "Model Context Protocol"),
    (r"\bjson-rpc\b", "JSON-RPC"),
    (r"\brest(?:ful)?(?: api)?\b", "REST API"),
)
_CONCEPTS: tuple[tuple[str, str], ...] = (
    (r"\b(?:decide|decided|decision|choose|chosen|select(?:ed|ion)?|adopt(?:ed|ion)?)\b", "decision"),
    (r"\b(?:migrat(?:e|ed|ion)|upgrade|replace|supersed(?:e|ed))\b", "migration"),
    (r"\b(?:deploy(?:ment|ed)?|release|ship(?:ped|ping)?)\b", "release"),
    (r"\b(?:test(?:ing)?|benchmark|evaluat(?:e|ion))\b", "testing"),
    (r"\b(?:security|credential|secret|auth(?:entication|orization)?)\b", "security"),
    (r"\b(?:performance|latency|throughput|fast(?:er)?)\b", "performance"),
)
_FUTURE_CUE = re.compile(
    r"\b(?:when|before|after|during|next|later|once|upon|until|prior to|follow(?: |-)?up|revisit)\b"
    r"[^.!?;]{0,80}",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset({"a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "our", "that", "the", "this", "to", "we", "with", "will"})


def derive_triggers(memory: Memory | Mapping[str, Any] | str) -> tuple[ProspectiveTrigger, ...]:
    """Derive stable technology, concept, and future-situation cues.

    ``memory`` may be a :class:`Memory`, a mapping with ``content``, ``kind``
    and optional ``metadata``, or plain content.  Equal input always yields
    the same ordered, duplicate-free tuple.
    """

    content, kind, metadata = _memory_parts(memory)
    candidates: list[ProspectiveTrigger] = []
    for pattern, name in _TECHNOLOGIES:
        if re.search(pattern, content, flags=re.IGNORECASE):
            candidates.append(ProspectiveTrigger(name, "technology", 1.0))

    if kind.casefold() == "decision":
        candidates.append(ProspectiveTrigger("decision", "concept", 0.8))
    for pattern, name in _CONCEPTS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            candidates.append(ProspectiveTrigger(name, "concept", 0.75))

    state = metadata.get("state") if isinstance(metadata, Mapping) else None
    if isinstance(state, Mapping):
        for field in ("subject", "key", "value"):
            value = state.get(field)
            if isinstance(value, str) and value.strip():
                candidates.append(ProspectiveTrigger(_clean_phrase(value), "state", 0.9))

    for match in _FUTURE_CUE.finditer(content):
        phrase = _clean_phrase(match.group())
        if len(_meaningful_tokens(phrase)) >= 2:
            candidates.append(ProspectiveTrigger(phrase, "future", 0.85))

    result: list[ProspectiveTrigger] = []
    seen: set[tuple[str, str]] = set()
    for trigger in candidates:
        key = (trigger.category, trigger.phrase.casefold())
        if key not in seen:
            seen.add(key)
            result.append(trigger)
    return tuple(result)


def score_relevance(memory: Memory | Mapping[str, Any] | str, situation: str) -> ProspectiveMatch:
    """Score how strongly a situation suggests retrieving ``memory``.

    The score is in ``[0, 1]``.  It is a weighted mean of cues with at least
    one lexical match, so unrelated text scores exactly zero.  ``matched``
    retains the evidence a caller can show to a user or use for ranking.
    """

    if not isinstance(situation, str) or not situation.strip():
        return ProspectiveMatch(0.0, ())
    situation_tokens = set(_meaningful_tokens(_canonicalize_technology(situation)))
    matched: list[ProspectiveTrigger] = []
    earned = 0.0
    available = 0.0
    for trigger in derive_triggers(memory):
        tokens = set(_meaningful_tokens(_canonicalize_technology(trigger.phrase)))
        if not tokens:
            continue
        overlap = len(tokens & situation_tokens) / len(tokens)
        if overlap:
            # Single-token cues need an exact match.  Multi-token cues receive
            # modest credit for partial overlap, but reserve the full score for
            # an exact phrase-level match.
            phrase = _clean_phrase(_canonicalize_technology(trigger.phrase))
            exact = phrase in _clean_phrase(_canonicalize_technology(situation))
            evidence = 1.0 if exact else overlap * 0.65
            earned += trigger.weight * evidence
            matched.append(trigger)
        available += trigger.weight
    score = earned / available if available else 0.0
    return ProspectiveMatch(round(min(1.0, score), 4), tuple(matched))


def _memory_parts(memory: Memory | Mapping[str, Any] | str) -> tuple[str, str, Mapping[str, Any]]:
    if isinstance(memory, Memory):
        return memory.content, memory.kind, memory.metadata
    if isinstance(memory, Mapping):
        content = memory.get("content", "")
        kind = memory.get("kind", "")
        metadata = memory.get("metadata", {})
        return (content if isinstance(content, str) else "", kind if isinstance(kind, str) else "", metadata if isinstance(metadata, Mapping) else {})
    return (memory if isinstance(memory, str) else "", "", {})


def _canonicalize_technology(value: str) -> str:
    for pattern, canonical in _TECHNOLOGIES:
        value = re.sub(pattern, canonical, value, flags=re.IGNORECASE)
    return value


def _clean_phrase(value: str) -> str:
    return _SPACE.sub(" ", value.strip(" \t\n,;:.!?"))


def _meaningful_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN.findall(value) if token.casefold() not in _STOP_WORDS)
