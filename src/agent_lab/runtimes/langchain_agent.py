"""LangChain create_agent adapter behind the shared research contract."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from agent_lab.application.context import RunContext
from agent_lab.application.policies import ToolPolicy, safe_evidence, validate_citations
from agent_lab.capabilities.retrieval.models import SearchPort, SearchQuery
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
    utc_now,
)
from agent_lab.observability.tracing import TraceRecorder
from agent_lab.prompts import TRUSTED_RESEARCH_PROMPT_VERSION, trusted_research_system_prompt


class ModelCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ModelResearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[ModelCitation] = Field(min_length=1)


class LangChainResearchRuntime:
    """Use a model only inside a deterministic permission and evidence shell."""

    name = "langchain"

    def __init__(
        self,
        model: BaseChatModel,
        search: SearchPort,
        *,
        policy: ToolPolicy | None = None,
    ) -> None:
        self.model = model
        self.search = search
        self.policy = policy or ToolPolicy()

    def run(self, request: RunRequest, *, context: RunContext | None = None) -> RunResult:
        started = time.perf_counter()
        invocation_context = context or RunContext(user_id=request.user_id)
        trace = TraceRecorder()
        trace.record("request_received", "validate", task_id=request.task_id)

        decision = self.policy.authorize(
            request=request,
            spec=self.search.spec,
            permissions=invocation_context.permissions,
        )
        if not decision.allowed:
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=(
                    RunStatus.APPROVAL_REQUIRED if decision.approval_required else RunStatus.BLOCKED
                ),
                reason="policy_denied",
                error=AgentError(
                    code=ToolErrorCode.UNAUTHORIZED,
                    message=decision.reason,
                    layer=ErrorLayer.POLICY,
                ),
            )
        if request.budget.max_model_calls < 1:
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=RunStatus.BLOCKED,
                reason="model_budget_exhausted",
                error=AgentError(
                    code=ToolErrorCode.BUDGET_EXCEEDED,
                    message="LangChain runtime 需要 max_model_calls >= 1",
                    layer=ErrorLayer.POLICY,
                ),
            )

        captured: list[Evidence] = []
        calls: list[ToolCallRecord] = []

        @tool("search_fixture_documents")
        def search_fixture_documents(query: str, limit: int = 3) -> str:
            """Search versioned offline fixtures; never accesses the live web."""

            call_id = f"search-{request.task_id}-{len(calls) + 1}"
            call_started = utc_now()
            result = self.search.search(
                SearchQuery(
                    query=query,
                    limit=min(limit, request.budget.max_evidence_items),
                ),
                call_id=call_id,
            )
            captured.extend(result.evidence)
            calls.append(
                ToolCallRecord(
                    call_id=call_id,
                    tool_name=self.search.spec.name,
                    capability=self.search.spec.capability,
                    arguments={"query": query, "limit": limit},
                    status=result.status,
                    error_code=result.error_code,
                    started_at=call_started,
                    finished_at=utc_now(),
                )
            )
            return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

        middleware: list[Any] = [
            ModelCallLimitMiddleware(
                run_limit=request.budget.max_model_calls,
                exit_behavior="error",
            ),
            ToolCallLimitMiddleware(
                run_limit=request.budget.max_tool_calls,
                exit_behavior="error",
            ),
        ]
        agent = create_agent(
            model=self.model,
            tools=[search_fixture_documents],
            system_prompt=trusted_research_system_prompt(),
            middleware=middleware,
            response_format=ToolStrategy(ModelResearchAnswer),
        )

        trace.record("agent_started", "agent", runtime=self.name)
        try:
            state = agent.invoke(
                {"messages": [{"role": "user", "content": request.goal}]},
                config={"recursion_limit": request.budget.max_steps},
            )
        except Exception as exc:  # provider errors must remain system errors
            trace.record("agent_error", "agent", error_type=type(exc).__name__)
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=RunStatus.FAILED,
                reason="agent_runtime_error",
                error=AgentError(
                    code="AGENT_RUNTIME_ERROR",
                    message=str(exc),
                    layer=ErrorLayer.PROVIDER,
                    retryable=True,
                    details={"type": type(exc).__name__},
                ),
                tool_calls=tuple(calls),
            )

        safe = safe_evidence(tuple(captured))
        structured = state.get("structured_response")
        if not safe:
            search_not_found = calls and calls[-1].error_code == ToolErrorCode.NOT_FOUND
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=RunStatus.UNKNOWN if search_not_found else RunStatus.BLOCKED,
                reason="evidence_not_found" if search_not_found else "no_safe_evidence",
                error=AgentError(
                    code=ToolErrorCode.NOT_FOUND if search_not_found else "UNTRUSTED_EVIDENCE",
                    message="模型运行没有获得可用于结论的安全证据",
                    layer=ErrorLayer.TOOL if search_not_found else ErrorLayer.POLICY,
                ),
                tool_calls=tuple(calls),
            )
        if not isinstance(structured, ModelResearchAnswer):
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=RunStatus.FAILED,
                reason="structured_output_missing",
                error=AgentError(
                    code="INVALID_MODEL_OUTPUT",
                    message="Agent 未返回 ModelResearchAnswer",
                    layer=ErrorLayer.PROVIDER,
                ),
                evidence=safe,
                tool_calls=tuple(calls),
            )

        citations = tuple(
            Citation(claim=item.claim, evidence_ids=tuple(item.evidence_ids))
            for item in structured.citations
        )
        if not validate_citations(citations, safe):
            return self._error_result(
                request,
                trace=trace,
                started=started,
                status=RunStatus.FAILED,
                reason="citation_contract_failed",
                error=AgentError(
                    code="INVALID_CITATION",
                    message="模型输出引用了不存在的 Evidence ID",
                    layer=ErrorLayer.RUNTIME,
                ),
                evidence=safe,
                tool_calls=tuple(calls),
            )

        trace.record("structured_output_received", "agent", citations=len(citations))
        trace.record("citation_validation_passed", "verify")
        trajectory = trace.snapshot()
        answer = structured.answer
        artifact = Artifact(
            artifact_type="research_report",
            content=answer,
            content_hash=f"sha256:{hashlib.sha256(answer.encode('utf-8')).hexdigest()}",
            evidence_ids=tuple(item.evidence_id for item in safe),
        )
        model_calls = sum(
            1 for message in state.get("messages", []) if getattr(message, "type", None) == "ai"
        )
        return RunResult(
            task_id=request.task_id,
            thread_id=request.thread_id,
            status=RunStatus.COMPLETED,
            answer=answer,
            evidence=safe,
            citations=citations,
            artifacts=(artifact,),
            tool_calls=tuple(calls),
            trajectory=trajectory,
            metrics=RunMetrics(
                model_calls=model_calls,
                tool_calls=len(calls),
                steps=len(trajectory),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            ),
            trace_id=trace.trace_id,
            termination_reason="evidence_verified",
            runtime=self.name,
            versions={
                "fixture": getattr(self.search, "version", "unknown"),
                "prompt": TRUSTED_RESEARCH_PROMPT_VERSION,
                "runtime": "v1",
            },
        )

    def _error_result(
        self,
        request: RunRequest,
        *,
        trace: TraceRecorder,
        started: float,
        status: RunStatus,
        reason: str,
        error: AgentError,
        evidence: tuple[Evidence, ...] = (),
        tool_calls: tuple[ToolCallRecord, ...] = (),
    ) -> RunResult:
        trajectory = trace.snapshot()
        return RunResult(
            task_id=request.task_id,
            thread_id=request.thread_id,
            status=status,
            evidence=evidence,
            errors=(error,),
            tool_calls=tool_calls,
            trajectory=trajectory,
            metrics=RunMetrics(
                tool_calls=len(tool_calls),
                steps=len(trajectory),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            ),
            trace_id=trace.trace_id,
            termination_reason=reason,
            runtime=self.name,
            versions={
                "fixture": getattr(self.search, "version", "unknown"),
                "prompt": TRUSTED_RESEARCH_PROMPT_VERSION,
                "runtime": "v1",
            },
        )
