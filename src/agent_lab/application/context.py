"""Non-serializable dependencies and identity for one invocation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunContext:
    user_id: str | None = None
    permissions: frozenset[str] = frozenset({"fixture.search"})
    tenant_id: str | None = None
    secrets: object | None = None
