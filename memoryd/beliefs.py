"""Conservative, evidence-backed views over durable memories.

This module is deliberately read-only.  It does not attempt natural-language
inference: a belief is only emitted when it is stated directly by an active
``state`` or ``decision`` memory.  Consumers can always show the source memory
IDs alongside a belief and retrieve the underlying records from the runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from .models import Memory


@dataclass(frozen=True)
class Belief:
    """A directly asserted, non-speculative conclusion.

    ``evidence_ids`` contains every active memory with the same direct
    assertion.  ``confidence`` is the *lowest* source confidence; agreement
    never manufactures confidence that the evidence did not provide.
    """

    kind: str
    statement: str
    confidence: float
    evidence_ids: tuple[str, ...]
    subject: str | None = None
    key: str | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


@dataclass(frozen=True)
class UnresolvedBelief:
    """Conflicting direct state assertions that must not become a belief."""

    subject: str
    key: str
    values: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason: str = "Conflicting active state assertions; no conclusion was emitted."

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["values"] = list(self.values)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


_STATE_ASSIGNMENT = re.compile(
    r"^\s*(?:(?P<subject>[A-Za-z][\w .-]{1,64})\s*:\s*)?"
    r"(?P<key>[A-Za-z][\w -]{0,63})\s*(?:=|is)\s*(?P<value>.+?)[.?!]?\s*$",
    re.IGNORECASE,
)


def _state_assertion(memory: Memory) -> tuple[str, str, str] | None:
    supplied = memory.metadata.get("state")
    if isinstance(supplied, dict) and all(name in supplied for name in ("subject", "key", "value")):
        subject, key, value = (str(supplied[name]).strip() for name in ("subject", "key", "value"))
        if subject and key and value:
            return subject, key.casefold().replace(" ", "_"), value
        return None
    match = _STATE_ASSIGNMENT.match(memory.content)
    if not match:
        return None
    subject = (match.group("subject") or str(memory.metadata.get("project", "global"))).strip()
    return subject, match.group("key").strip().casefold().replace(" ", "_"), match.group("value").strip()


def derive_beliefs(memories: Iterable[Memory]) -> tuple[list[Belief], list[UnresolvedBelief]]:
    """Return direct beliefs and unresolved state conflicts from active memories.

    Superseded, forgotten, speculative, and semantic memories are intentionally
    ignored.  A state key with more than one distinct active value is reported
    as unresolved rather than choosing an arbitrary winner.
    """
    active = [memory for memory in memories if memory.status == "active" and memory.kind in {"state", "decision"}]
    state_groups: dict[tuple[str, str], dict[str, list[Memory]]] = {}
    decision_groups: dict[str, list[Memory]] = {}
    unstructured_states: dict[str, list[Memory]] = {}
    for memory in active:
        if memory.kind == "decision":
            decision_groups.setdefault(" ".join(memory.content.split()).casefold(), []).append(memory)
            continue
        assertion = _state_assertion(memory)
        if assertion is None:
            unstructured_states.setdefault(" ".join(memory.content.split()).casefold(), []).append(memory)
            continue
        subject, key, value = assertion
        state_groups.setdefault((subject, key), {}).setdefault(value, []).append(memory)

    beliefs: list[Belief] = []
    unresolved: list[UnresolvedBelief] = []
    for (subject, key), values in state_groups.items():
        if len(values) != 1:
            evidence = tuple(sorted(memory.id for group in values.values() for memory in group))
            unresolved.append(UnresolvedBelief(subject, key, tuple(sorted(values)), evidence))
            continue
        value, evidence = next(iter(values.items()))
        beliefs.append(Belief("state", f"{subject}: {key} = {value}", min(item.confidence for item in evidence),
                              tuple(sorted(item.id for item in evidence)), subject, key, value))
    for group in (decision_groups, unstructured_states):
        for evidence in group.values():
            exemplar = evidence[0]
            beliefs.append(Belief(exemplar.kind, exemplar.content.strip(), min(item.confidence for item in evidence),
                                  tuple(sorted(item.id for item in evidence))))
    beliefs.sort(key=lambda item: (item.kind, item.subject or "", item.key or "", item.statement.casefold()))
    unresolved.sort(key=lambda item: (item.subject.casefold(), item.key))
    return beliefs, unresolved


def explain(belief: Belief) -> dict[str, object]:
    """Return an inspectable explanation without inventing causal reasoning."""
    return {
        "belief": belief.to_dict(),
        "explanation": "This belief is a direct active memory assertion; inspect evidence_ids for its sources.",
        "supporting_evidence_ids": list(belief.evidence_ids),
    }


def build_belief_set(memories: Iterable[Memory]) -> dict[str, object]:
    """Build a serializable belief view suitable for an API or UI."""
    beliefs, unresolved = derive_beliefs(memories)
    return {
        "beliefs": [belief.to_dict() for belief in beliefs],
        "unresolved": [item.to_dict() for item in unresolved],
        "policy": "direct active state and decision assertions only",
    }
