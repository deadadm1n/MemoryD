"""Native memory visibility scopes and the trusted runtime context."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SCOPED = {"project", "person", "agent", "private"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_scope(scope: str) -> str:
    value = scope.strip()
    if value in {"shared", "world"}:
        return value
    prefix, separator, identifier = value.partition(":")
    if separator != ":" or prefix not in _SCOPED or not _IDENTIFIER.fullmatch(identifier):
        raise ValueError("scope must be shared, world, or type:identifier for project, person, agent, or private")
    return value


@dataclass(frozen=True)
class ScopeContext:
    """The identity granted to one runtime instance; this is not authentication."""
    principal_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None
    include_shared: bool = True
    include_world: bool = True

    def visible(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.include_world: values.append("world")
        if self.include_shared: values.append("shared")
        if self.project_id: values.append(validate_scope(f"project:{self.project_id}"))
        if self.principal_id:
            values.extend((validate_scope(f"person:{self.principal_id}"), validate_scope(f"private:{self.principal_id}")))
        if self.agent_id: values.append(validate_scope(f"agent:{self.agent_id}"))
        return tuple(values)

    def can_write(self, scope: str) -> bool:
        return validate_scope(scope) in self.visible()
