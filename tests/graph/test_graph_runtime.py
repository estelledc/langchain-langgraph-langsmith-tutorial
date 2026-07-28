from __future__ import annotations

from langgraph.types import Command

from agent_lab.domain.models import RunRequest, RunStatus
from agent_lab.runtimes.approval import build_approval_graph
from agent_lab.runtimes.graph import LangGraphResearchRuntime
from agent_lab.runtimes.workflow import TrustedResearchWorkflow


def test_graph_and_workflow_share_outcome_contract(
    workflow: TrustedResearchWorkflow,
    graph_runtime: LangGraphResearchRuntime,
) -> None:
    request = RunRequest(goal="LangSmith 的评测闭环")
    workflow_result = workflow.run(request)
    graph_result = graph_runtime.run(request)
    assert workflow_result.status is graph_result.status is RunStatus.COMPLETED
    assert {item.evidence_id for item in workflow_result.evidence} == {
        item.evidence_id for item in graph_result.evidence
    }
    assert workflow_result.answer == graph_result.answer
    assert [event.node for event in graph_result.trajectory] == [
        "authorize",
        "retrieve",
        "validate_evidence",
        "synthesize",
        "verify",
        "finalize",
    ]


def test_injected_checkpointer_retains_thread_state(
    graph_runtime: LangGraphResearchRuntime,
) -> None:
    request = RunRequest(thread_id="persistent-thread", goal="Context Engineering")
    graph_runtime.run(request)
    snapshot = graph_runtime.get_thread_state("persistent-thread")
    assert snapshot.values["termination_reason"] == "evidence_verified"
    assert snapshot.values["request"]["thread_id"] == "persistent-thread"


def test_interrupt_requires_resume_with_same_thread() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_approval_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-thread"}}
    first = graph.invoke(
        {
            "action": "publish",
            "payload": {"idempotency_key": "publish-1"},
            "approved": None,
            "outcome": None,
        },
        config=config,
    )
    assert first["__interrupt__"][0].value["kind"] == "approval_required"
    resumed = graph.invoke(Command(resume=True), config=config)
    assert resumed["approved"] is True
    assert resumed["outcome"].startswith("approved")
