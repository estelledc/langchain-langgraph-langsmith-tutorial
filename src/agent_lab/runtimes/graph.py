"""LangGraph implementation of the same offline Trusted Research contract."""

from __future__ import annotations

import hashlib
import operator
import time
from typing import Annotated, Any, Literal, TypedDict, cast
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agent_lab.application.context import RunContext
from agent_lab.application.policies import ToolPolicy, safe_evidence, validate_citations
from agent_lab.capabilities.retrieval.models import SearchPort, SearchQuery, SearchStatus
from agent_lab.capabilities.tools.contracts import ToolErrorCode
from agent_lab.domain.models import (
    AgentError,
    Artifact,
    Citation,
    ErrorLayer,
    Evidence,
    RunMetrics,
    RunRequest,
    RunResult,
    RunStatus,
    ToolCallRecord,
    TraceEvent,
    utc_now,
)
from agent_lab.runtimes.workflow import synthesize_evidence_answer


class ResearchGraphState(TypedDict):
    request: dict[str, Any]
    status: str
    termination_reason: str
    raw_evidence: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str | None
    citations: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    visited: Annotated[list[str], operator.add]


class LangGraphResearchRuntime:
    """Explicit graph topology with an injected checkpointer."""

    name = "langgraph"

    def __init__(
        self,
        search: SearchPort,
        *,
        checkpointer: Any | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.search = search
        self.policy = policy or ToolPolicy()
        self.graph = self._build_graph(checkpointer=checkpointer)

    def _build_graph(self, *, checkpointer: Any | None) -> Any:
        builder = StateGraph(ResearchGraphState, context_schema=RunContext)
        builder.add_node("authorize", self._authorize)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("validate_evidence", self._validate_evidence)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("verify", self._verify)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "authorize")
        builder.add_conditional_edges(
            "authorize",
            self._route_running,
            {"continue": "retrieve", "stop": "finalize"},
        )
        builder.add_conditional_edges(
            "retrieve",
            self._route_running,
            {"continue": "validate_evidence", "stop": "finalize"},
        )
        builder.add_conditional_edges(
            "validate_evidence",
            self._route_running,
            {"continue": "synthesize", "stop": "finalize"},
        )
        builder.add_edge("synthesize", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=checkpointer)

    def run(self, request: RunRequest, *, context: RunContext | None = None) -> RunResult:
        started = time.perf_counter()
        invocation_context = context or RunContext(user_id=request.user_id)
        initial: ResearchGraphState = {
            "request": request.model_dump(mode="json"),
            "status": "running",
            "termination_reason": "running",
            "raw_evidence": [],
            "evidence": [],
            "answer": None,
            "citations": [],
            "errors": [],
            "tool_calls": [],
            "visited": [],
        }
        config = {
            "configurable": {"thread_id": request.thread_id},
            "recursion_limit": request.budget.max_steps + 2,
        }
        final = cast(
            ResearchGraphState,
            self.graph.invoke(initial, config=config, context=invocation_context),
        )
        return self._to_result(final, started=started)

    def get_thread_state(self, thread_id: str) -> Any:
        return self.graph.get_state({"configurable": {"thread_id": thread_id}})

    def _authorize(
        self,
        state: ResearchGraphState,
        runtime: Runtime[RunContext],
    ) -> dict[str, Any]:
        request = RunRequest.model_validate(state["request"])
        decision = self.policy.authorize(
            request=request,
            spec=self.search.spec,
            permissions=runtime.context.permissions,
        )
        if not decision.allowed:
            status = (
                RunStatus.APPROVAL_REQUIRED if decision.approval_required else RunStatus.BLOCKED
            )
            error_code = (
                ToolErrorCode.APPROVAL_REQUIRED
                if decision.approval_required
                else ToolErrorCode.UNAUTHORIZED
            )
            return {
                "status": status,
                "termination_reason": (
                    "approval_required" if decision.approval_required else "policy_denied"
                ),
                "errors": [
                    AgentError(
                        code=error_code,
                        message=decision.reason,
                        layer=ErrorLayer.POLICY,
                    ).model_dump(mode="json")
                ],
                "visited": ["authorize:denied"],
            }
        if request.budget.max_tool_calls < 1:
            return {
                "status": RunStatus.BLOCKED,
                "termination_reason": "tool_budget_exhausted",
                "errors": [
                    AgentError(
                        code=ToolErrorCode.BUDGET_EXCEEDED,
                        message="max_tool_calls 必须至少为 1 才能执行检索",
                        layer=ErrorLayer.POLICY,
                    ).model_dump(mode="json")
                ],
                "visited": ["authorize:budget_exhausted"],
            }
        return {"visited": ["authorize:allowed"]}

    def _retrieve(self, state: ResearchGraphState) -> dict[str, Any]:
        request = RunRequest.model_validate(state["request"])
        call_id = f"search-{request.task_id}"
        started_at = utc_now()
        result = self.search.search(
            SearchQuery(
                query=request.goal,
                limit=min(request.budget.max_evidence_items, 20),
            ),
            call_id=call_id,
        )
        finished_at = utc_now()
        tool_call = ToolCallRecord(
            call_id=call_id,
            tool_name=self.search.spec.name,
            capability=self.search.spec.capability,
            arguments={"query": request.goal, "limit": request.budget.max_evidence_items},
            status=result.status,
            error_code=result.error_code,
            started_at=started_at,
            finished_at=finished_at,
        )
        update: dict[str, Any] = {
            "raw_evidence": [item.model_dump(mode="json") for item in result.evidence],
            "tool_calls": [tool_call.model_dump(mode="json")],
            "visited": [f"retrieve:{result.status}"],
        }
        if result.status is SearchStatus.NOT_FOUND:
            update.update(
                {
                    "status": RunStatus.UNKNOWN,
                    "termination_reason": "evidence_not_found",
                    "answer": "现有离线 fixture 没有支持这个问题的证据，因此不能给出事实性结论。",
                    "errors": [
                        AgentError(
                            code=ToolErrorCode.NOT_FOUND,
                            message=result.message,
                            layer=ErrorLayer.TOOL,
                        ).model_dump(mode="json")
                    ],
                }
            )
        return update

    def _validate_evidence(self, state: ResearchGraphState) -> dict[str, Any]:
        raw = tuple(Evidence.model_validate(item) for item in state["raw_evidence"])
        filtered = safe_evidence(raw)
        if not filtered:
            return {
                "status": RunStatus.BLOCKED,
                "termination_reason": "all_evidence_quarantined",
                "errors": [
                    AgentError(
                        code="UNTRUSTED_EVIDENCE",
                        message="候选证据包含明显的提示注入模式，已隔离",
                        layer=ErrorLayer.POLICY,
                    ).model_dump(mode="json")
                ],
                "visited": ["validate_evidence:quarantined"],
            }
        return {
            "evidence": [item.model_dump(mode="json") for item in filtered],
            "visited": ["validate_evidence:accepted"],
        }

    def _synthesize(self, state: ResearchGraphState) -> dict[str, Any]:
        request = RunRequest.model_validate(state["request"])
        evidence = tuple(Evidence.model_validate(item) for item in state["evidence"])
        answer, citations = synthesize_evidence_answer(request.goal, evidence)
        return {
            "answer": answer,
            "citations": [item.model_dump(mode="json") for item in citations],
            "visited": ["synthesize:completed"],
        }

    def _verify(self, state: ResearchGraphState) -> dict[str, Any]:
        evidence = tuple(Evidence.model_validate(item) for item in state["evidence"])
        citations = tuple(Citation.model_validate(item) for item in state["citations"])
        if not validate_citations(citations, evidence):
            return {
                "status": RunStatus.FAILED,
                "termination_reason": "citation_contract_failed",
                "errors": [
                    AgentError(
                        code="INVALID_CITATION",
                        message="答案引用了不存在的 Evidence ID",
                        layer=ErrorLayer.RUNTIME,
                    ).model_dump(mode="json")
                ],
                "visited": ["verify:failed"],
            }
        return {
            "status": RunStatus.COMPLETED,
            "termination_reason": "evidence_verified",
            "visited": ["verify:passed"],
        }

    def _finalize(self, state: ResearchGraphState) -> dict[str, Any]:
        return {"visited": [f"finalize:{state['status']}"]}

    @staticmethod
    def _route_running(state: ResearchGraphState) -> Literal["continue", "stop"]:
        return "continue" if state["status"] == "running" else "stop"

    def _to_result(self, state: ResearchGraphState, *, started: float) -> RunResult:
        request = RunRequest.model_validate(state["request"])
        evidence = tuple(Evidence.model_validate(item) for item in state["evidence"])
        citations = tuple(Citation.model_validate(item) for item in state["citations"])
        errors = tuple(AgentError.model_validate(item) for item in state["errors"])
        tool_calls = tuple(ToolCallRecord.model_validate(item) for item in state["tool_calls"])
        trace_id = str(uuid4())
        trajectory = tuple(
            TraceEvent(
                sequence=index,
                event=value.split(":", maxsplit=1)[-1],
                node=value.split(":", maxsplit=1)[0],
            )
            for index, value in enumerate(state["visited"])
        )
        artifacts: tuple[Artifact, ...] = ()
        if state["answer"] and state["status"] == RunStatus.COMPLETED:
            answer = state["answer"]
            artifacts = (
                Artifact(
                    artifact_type="research_report",
                    content=answer,
                    content_hash=(f"sha256:{hashlib.sha256(answer.encode('utf-8')).hexdigest()}"),
                    evidence_ids=tuple(item.evidence_id for item in evidence),
                ),
            )
        return RunResult(
            task_id=request.task_id,
            thread_id=request.thread_id,
            status=RunStatus(state["status"]),
            answer=state["answer"],
            evidence=evidence,
            citations=citations,
            artifacts=artifacts,
            errors=errors,
            tool_calls=tool_calls,
            trajectory=trajectory,
            metrics=RunMetrics(
                tool_calls=len(tool_calls),
                steps=len(trajectory),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            ),
            trace_id=trace_id,
            termination_reason=state["termination_reason"],
            runtime=self.name,
            versions={"fixture": getattr(self.search, "version", "unknown"), "runtime": "v1"},
        )
