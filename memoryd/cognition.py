"""Precision-first observation analysis for durable memory candidates.

This module deliberately only *proposes* memories.  Callers remain responsible
for reviewing a candidate and passing it to :class:`MemoryRuntime`; an ordinary
conversation should not silently become durable storage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class MemoryCandidate:
    """A reviewable suggestion for one atomic durable memory."""

    kind: str
    confidence: float
    importance: float
    content: str
    state: dict[str, str] | None = None
    actor: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ObservationAnalyzer(Protocol):
    """Pluggable boundary for an LLM or deterministic observation analyzer."""

    def analyze(
        self,
        content: str,
        *,
        actor: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> list[MemoryCandidate]:
        """Return zero or more proposals without persisting anything."""


class DeterministicObservationAnalyzer:
    """Conservative fallback that recognizes only unambiguous durable claims."""

    _state_patterns = (
        re.compile(
            r"^(?P<subject>[A-Za-z][A-Za-z0-9 _.-]{0,79})['’]s "
            r"(?P<key>[A-Za-z][A-Za-z0-9 _-]{0,49}) is now "
            r"(?P<value>[^.?!]{1,160})[.!]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:set|update) (?P<subject>[A-Za-z][A-Za-z0-9 _.-]{0,79}) "
            r"(?P<key>[A-Za-z][A-Za-z0-9 _-]{0,49}) to "
            r"(?P<value>[^.?!]{1,160})[.!]?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<subject>[A-Za-z][A-Za-z0-9 _.-]{0,79})\.(?P<key>[A-Za-z][A-Za-z0-9_-]{0,49})\s*=\s*"
            r"(?P<value>[^.?!]{1,160})[.!]?$",
        ),
    )
    _decision = re.compile(
        r"^(?:decision\s*:\s*|we\s+(?:decided|agreed|chose|will)\b|"
        r"(?:approved|rejected)\b)",
        re.IGNORECASE,
    )
    _speculation = re.compile(
        r"^(?:open question\s*:\s*|question\s*:\s*|(?:we\s+)?(?:might|could|may|should)\b|"
        r"proposal\s*:\s*)",
        re.IGNORECASE,
    )

    def analyze(
        self,
        content: str,
        *,
        actor: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> list[MemoryCandidate]:
        text = " ".join(content.split())
        if not text:
            return []
        safe_context = dict(context or {})

        for pattern in self._state_patterns:
            match = pattern.fullmatch(text)
            if match:
                state = {key: value.strip() for key, value in match.groupdict().items()}
                return [MemoryCandidate(
                    kind="state", confidence=0.96, importance=0.78, content=text,
                    state=state, actor=actor, context=safe_context,
                )]

        if self._decision.match(text):
            return [MemoryCandidate(
                kind="decision", confidence=0.95, importance=0.8, content=text,
                actor=actor, context=safe_context,
            )]

        if text.endswith("?") or self._speculation.match(text):
            return [MemoryCandidate(
                kind="speculation", confidence=0.5, importance=0.55, content=text,
                actor=actor, context=safe_context,
            )]
        return []
