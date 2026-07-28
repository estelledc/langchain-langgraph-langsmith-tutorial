"""Select a runtime without changing the domain request/result boundary."""

from __future__ import annotations

from typing import Protocol

from agent_lab.application.context import RunContext
from agent_lab.domain.models import RunRequest, RunResult


class ResearchRuntime(Protocol):
    name: str

    def run(self, request: RunRequest, *, context: RunContext | None = None) -> RunResult: ...


class ResearchService:
    def __init__(self, runtimes: dict[str, ResearchRuntime]) -> None:
        self._runtimes = dict(runtimes)

    @property
    def runtimes(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtimes))

    def run(
        self,
        request: RunRequest,
        *,
        runtime: str = "workflow",
        context: RunContext | None = None,
    ) -> RunResult:
        selected = self._runtimes.get(runtime)
        if selected is None:
            available = ", ".join(self.runtimes)
            raise ValueError(f"unknown runtime {runtime!r}; available: {available}")
        return selected.run(request, context=context)
