from __future__ import annotations

from agent_lab.application.context import RunContext
from agent_lab.application.policies import ToolPolicy
from agent_lab.capabilities.retrieval.models import FIXTURE_SEARCH_SPEC
from agent_lab.domain.models import Budget, RunRequest, RunStatus
from agent_lab.runtimes.workflow import TrustedResearchWorkflow


def test_completed_answer_is_attributed(workflow: TrustedResearchWorkflow) -> None:
    result = workflow.run(RunRequest(goal="LangGraph checkpointer 和 Store 的区别"))
    assert result.status is RunStatus.COMPLETED
    assert "fx-langgraph-memory-v1" in result.answer
    assert result.citations
    assert result.artifacts[0].evidence_ids
    assert result.metrics.model_calls == 0
    assert result.metrics.tool_calls == 1


def test_unknown_answer_abstains(workflow: TrustedResearchWorkflow) -> None:
    result = workflow.run(RunRequest(goal="量子香蕉编译器 2039"))
    assert result.status is RunStatus.UNKNOWN
    assert result.citations == ()
    assert result.termination_reason == "evidence_not_found"


def test_injection_only_result_is_blocked(workflow: TrustedResearchWorkflow) -> None:
    result = workflow.run(RunRequest(goal="injection-test"))
    assert result.status is RunStatus.BLOCKED
    assert result.answer is None
    assert result.termination_reason == "all_evidence_quarantined"


def test_permission_and_budget_are_checked_before_tool_call(
    workflow: TrustedResearchWorkflow,
) -> None:
    denied = workflow.run(
        RunRequest(goal="LangGraph"),
        context=RunContext(permissions=frozenset()),
    )
    exhausted = workflow.run(
        RunRequest(goal="LangGraph", budget=Budget(max_tool_calls=0)),
    )
    assert denied.status is RunStatus.BLOCKED
    assert exhausted.status is RunStatus.BLOCKED
    assert denied.tool_calls == exhausted.tool_calls == ()


def test_policy_normalizes_json_permission_lists() -> None:
    request = RunRequest(goal="LangGraph")
    decision = ToolPolicy().authorize(
        request=request,
        spec=FIXTURE_SEARCH_SPEC,
        permissions=["fixture.search"],
    )
    assert decision.allowed is True
