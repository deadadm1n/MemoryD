"""Embedding providers used by the recall engine.

The built-in provider is deterministic, offline, and dependency-free. It is a
useful baseline and keeps `brain.db` portable. Applications can inject a model
backed provider that implements the same protocol when they need richer
semantic similarity.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Stable feature-hash embeddings for small, local-first brain files."""

    name = "hash-v1"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9_]+", text.casefold())
        # A small synonym bridge helps common memory questions such as
        # "what database did we choose?" reach SQLite decisions.
        synonyms = {
            "choose": ("selected", "decision", "choice"), "chosen": ("selected", "decision", "choice"),
            "database": ("sqlite", "postgresql", "storage"), "project": ("runtime", "system"),
            "continue": ("current", "state", "recent"),
        }
        expanded = [item for word in words for item in (word, *synonyms.get(word, ()))]
        return expanded + [f"{first}:{second}" for first, second in zip(words, words[1:])]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += 1.0 if value & 1 else -1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class SentenceTransformerEmbeddingProvider:
    """Optional adapter for real local sentence-transformer models.

    The import is intentionally lazy so a portable `memoryd` installation stays
    dependency-free until the user chooses a local model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise RuntimeError("Install sentence-transformers to use a local model-backed embedding provider") from exc
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return [float(value) for value in self._model.encode(text, normalize_embeddings=True).tolist()]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(first * second for first, second in zip(left, right))
