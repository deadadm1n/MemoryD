from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    kind: str
    source: str
    confidence: float
    importance: float
    strength: float
    status: str
    created_at: str
    updated_at: str
    accessed_at: str | None
    access_count: int
    scope: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecallResult:
    memory: Memory
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"memory": self.memory.to_dict(), "score": round(self.score, 4), "reasons": self.reasons}
