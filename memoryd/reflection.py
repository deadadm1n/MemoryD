"""Deterministic, review-first reflection proposals.

Reflection is advisory: this module never mutates the brain. It identifies
consolidation candidates, unresolved questions, and suspiciously overlapping
memories so a caller can inspect the source IDs before taking action.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Iterable

from .models import Memory


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.casefold()))


def _overlap(left: Memory, right: Memory) -> float:
    left_tokens, right_tokens = _tokens(left.content), _tokens(right.content)
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def propose(memories: Iterable[Memory], limit: int = 200) -> dict[str, Any]:
    items = [memory for memory in list(memories)[:limit] if not memory.metadata.get("consolidation")]
    proposals: list[dict[str, Any]] = []
    if len(items) >= 2:
        proposals.append({"kind": "consolidate", "review_required": True,
                          "reason": f"{len(items)} recent atomic memories can be summarized without deleting sources.",
                          "source_memory_ids": [memory.id for memory in items]})
    questions = [memory for memory in items if memory.kind == "speculation" or "open question" in memory.content.casefold() or "?" in memory.content]
    if questions:
        proposals.append({"kind": "review_open_questions", "review_required": True,
                          "reason": f"{len(questions)} unresolved memories may need an answer or a state update.",
                          "source_memory_ids": [memory.id for memory in questions]})
    pairs = []
    for left, right in combinations(items[:60], 2):
        score = _overlap(left, right)
        if score >= .72:
            pairs.append({"memory_ids": [left.id, right.id], "overlap": round(score, 3)})
    if pairs:
        proposals.append({"kind": "review_possible_duplicates", "review_required": True,
                          "reason": "Highly overlapping memories may be duplicates or candidates for consolidation.",
                          "pairs": pairs[:20]})
    return {"review_required": True, "memory_count": len(items), "proposals": proposals}
