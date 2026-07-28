"""Minimal LangGraph interrupt/resume example for side-effect approval."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class ApprovalState(TypedDict):
    action: str
    payload: dict[str, Any]
    approved: bool | None
    outcome: str | None


def request_approval(state: ApprovalState) -> dict[str, bool]:
    decision = interrupt(
        {
            "kind": "approval_required",
            "action": state["action"],
            "payload": state["payload"],
        }
    )
    return {"approved": bool(decision)}


def apply_decision(state: ApprovalState) -> dict[str, str]:
    if state["approved"]:
        return {"outcome": "approved; caller may execute the idempotent side effect"}
    return {"outcome": "rejected; no side effect executed"}


def build_approval_graph(*, checkpointer: Any) -> Any:
    """Compile with an injected checkpointer so pause/resume state can survive."""

    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("apply_decision", apply_decision)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "apply_decision")
    builder.add_edge("apply_decision", END)
    return builder.compile(checkpointer=checkpointer)
