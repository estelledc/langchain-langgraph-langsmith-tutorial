"""Record actions and state transitions without chain-of-thought."""

from __future__ import annotations

from uuid import uuid4

from agent_lab.domain.models import TraceEvent


class TraceRecorder:
    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or str(uuid4())
        self._events: list[TraceEvent] = []

    def record(
        self,
        event: str,
        node: str,
        **attributes: str | int | float | bool | None,
    ) -> None:
        self._events.append(
            TraceEvent(
                sequence=len(self._events),
                event=event,
                node=node,
                attributes=attributes,
            )
        )

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)
