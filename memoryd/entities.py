"""Small, deterministic entity extraction for durable memory metadata.

The extractor deliberately favours precision over recall.  It has no model or
third-party dependency, so the same memory always produces the same metadata
on every machine.  Callers may add the returned names to a memory's metadata
and use the associations as lightweight graph edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EntityAssociation:
    """An unordered co-occurrence between two entities in one sentence."""

    source: str
    target: str
    relation: str = "co_occurs"


# Canonical spellings for technologies and commonly used project terms.  The
# patterns intentionally contain word boundaries so, for example, ``rustic``
# never produces the Rust programming language.
_KNOWN_ENTITIES: tuple[tuple[str, str], ...] = (
    (r"\bSQLite\b", "SQLite"),
    (r"\bPostgreSQL\b|\bPostgres\b", "PostgreSQL"),
    (r"\bMySQL\b", "MySQL"),
    (r"\bMongoDB\b", "MongoDB"),
    (r"\bRedis\b", "Redis"),
    (r"\bPython\b", "Python"),
    (r"\bTypeScript\b", "TypeScript"),
    (r"\bJavaScript\b", "JavaScript"),
    (r"\bNode(?:\.js)?\b", "Node.js"),
    (r"\bFastAPI\b", "FastAPI"),
    (r"\bDjango\b", "Django"),
    (r"\bReact\b", "React"),
    (r"\bDocker\b", "Docker"),
    (r"\bKubernetes\b|\bK8s\b", "Kubernetes"),
    (r"\bGitHub\b", "GitHub"),
    (r"\bOpenAI\b", "OpenAI"),
    (r"\bMCP\b|\bModel Context Protocol\b", "Model Context Protocol"),
    (r"\bJSON-RPC\b", "JSON-RPC"),
    (r"\bREST(?:ful)?(?: API)?\b", "REST API"),
)

_GENERIC_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "how", "i", "if", "in", "is", "it", "its", "of", "on",
        "or", "our", "that", "the", "their", "this", "to", "we", "with",
        "you", "your", "version", "project", "team", "system", "database",
        "api", "server", "client", "memory", "code", "first", "next", "new",
    }
)

# Proper-noun phrases catch project names such as "Atlas Gateway" without
# treating ordinary sentence-leading words as entities.
_PROPER_NOUN_PHRASE = re.compile(r"(?<!\w)([A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*){0,3})(?!\w)")
_SENTENCE = re.compile(r"[^.!?]+")


def extract_entities(text: str) -> list[str]:
    """Return unique, canonical entity names in first-appearance order.

    Recognized technology names are canonicalized (``Postgres`` becomes
    ``PostgreSQL``).  Capitalized project/name phrases are included when they
    contain a non-generic word.  Empty or non-string input yields no entities.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    matches: list[tuple[int, str]] = []
    known_spans: list[tuple[int, int]] = []
    for pattern, canonical in _KNOWN_ENTITIES:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), canonical))
            known_spans.append(match.span())

    for match in _PROPER_NOUN_PHRASE.finditer(text):
        # Do not emit fragments of a known technology (for instance "Model"
        # from "Model Context Protocol").
        if any(match.start() < end and start < match.end() for start, end in known_spans):
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -")
        words = candidate.casefold().split()
        if candidate and any(word not in _GENERIC_WORDS for word in words):
            matches.append((match.start(), candidate))

    result: list[str] = []
    seen: set[str] = set()
    for _, entity in sorted(matches, key=lambda item: item[0]):
        key = entity.casefold()
        if key not in seen:
            seen.add(key)
            result.append(entity)
    return result


def extract_entity_associations(text: str) -> list[EntityAssociation]:
    """Return unique pairwise ``co_occurs`` associations per sentence.

    Associations are directional only to make their serialized representation
    stable: each pair follows the order in which the entities appear.
    """

    if not isinstance(text, str):
        return []
    associations: list[EntityAssociation] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _SENTENCE.findall(text):
        entities = extract_entities(sentence)
        for index, source in enumerate(entities):
            for target in entities[index + 1 :]:
                key = (source.casefold(), target.casefold())
                if key not in seen:
                    seen.add(key)
                    associations.append(EntityAssociation(source, target))
    return associations
