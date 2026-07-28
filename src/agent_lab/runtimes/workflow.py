"""Deterministic baseline for Trusted Research."""

from __future__ import annotations

import hashlib
import time

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
    utc_now,
)
from agent_lab.observability.tracing import TraceRecorder


def _artifact_hash(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def synthesize_evidence_answer(
    goal: str,
    evidence: tuple[Evidence, ...],
) -> tuple[str, tuple[Citation, ...]]:
    """Build a transparent offline answer without pretending to use a model."""

    lines = [f"# {goal}", "", "以下结论只来自版本化离线 fixture：", ""]
    citations: list[Citation] = []
    for item in evidence:
        claim = f"{item.title or item.evidence_id}：{item.content}"
        lines.append(f"- {claim} [{item.evidence_id}]")
        citations.append(Citation(claim=claim, evidence_ids=(item.evidence_id,)))
    lines.extend(
        [
            "",
            "> 证据类型为 fixture；它可用于离线教学和回归，不代表刚刚访问了互联网。",
        ]
    )
    return "\n".join(lines), tuple(citations)


class TrustedResearchWorkflow:
    """The simplest correct architecture against which agents are compared."""

    name = "workflow"

    def __init__(self, search: SearchPort, *, policy: ToolPolicy | None = None) -> None:
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
            status = (
                RunStatus.APPROVAL_REQUIRED if decision.approval_required else RunStatus.BLOCKED
            )
            trace.record("policy_denied", "authorize", reason=decision.reason)
            return self._terminal_result(
                request=request,
                status=status,
                reason="approval_required" if decision.approval_required else "policy_denied",
                trace=trace,
                started=started,
                errors=(
                    AgentError(
                        code=(
                            ToolErrorCode.APPROVAL_REQUIRED
                            if decision.approval_required
                            else ToolErrorCode.UNAUTHORIZED
                        ),
                        message=decision.reason,
                        layer=ErrorLayer.POLICY,
                    ),
                ),
            )

        if request.budget.max_tool_calls < 1:
            trace.record("budget_exhausted", "authorize", budget="max_tool_calls")
            return self._terminal_result(
                request=request,
                status=RunStatus.BLOCKED,
                reason="tool_budget_exhausted",
                trace=trace,
                started=started,
                errors=(
                    AgentError(
                        code=ToolErrorCode.BUDGET_EXCEEDED,
                        message="max_tool_calls 必须至少为 1 才能执行检索",
                        layer=ErrorLayer.POLICY,
                    ),
                ),
            )

        trace.record("policy_allowed", "authorize", capability=self.search.spec.capability)
        call_id = f"search-{request.task_id}"
        tool_started = utc_now()
        trace.record("tool_started", "retrieve", tool=self.search.spec.name)
        search_result = self.search.search(
            SearchQuery(
                query=request.goal,
                limit=min(request.budget.max_evidence_items, 20),
            ),
            call_id=call_id,
        )
        tool_finished = utc_now()
        tool_call = ToolCallRecord(
            call_id=call_id,
            tool_name=self.search.spec.name,
            capability=self.search.spec.capability,
            arguments={
                "query": request.goal,
                "limit": request.budget.max_evidence_items,
            },
            status=search_result.status,
            error_code=search_result.error_code,
            started_at=tool_started,
            finished_at=tool_finished,
        )
        trace.record(
            "tool_finished",
            "retrieve",
            status=search_result.status,
            evidence_count=len(search_result.evidence),
        )

        if search_result.status is SearchStatus.NOT_FOUND:
            trace.record("evidence_missing", "validate_evidence")
            return self._terminal_result(
                request=request,
                status=RunStatus.UNKNOWN,
                reason="evidence_not_found",
                trace=trace,
                started=started,
                answer="现有离线 fixture 没有支持这个问题的证据，因此不能给出事实性结论。",
                errors=(
                    AgentError(
                        code=ToolErrorCode.NOT_FOUND,
                        message=search_result.message,
                        layer=ErrorLayer.TOOL,
                    ),
                ),
                tool_calls=(tool_call,),
            )

        filtered = safe_evidence(search_result.evidence)
        quarantined = len(search_result.evidence) - len(filtered)
        trace.record(
            "evidence_validated",
            "validate_evidence",
            accepted=len(filtered),
            quarantined=quarantined,
        )
        if not filtered:
            return self._terminal_result(
                request=request,
                status=RunStatus.BLOCKED,
                reason="all_evidence_quarantined",
                trace=trace,
                started=started,
                errors=(
                    AgentError(
                        code="UNTRUSTED_EVIDENCE",
                        message="候选证据包含明显的提示注入模式，已隔离",
                        layer=ErrorLayer.POLICY,
                    ),
                ),
                tool_calls=(tool_call,),
            )

        answer, citations = synthesize_evidence_answer(request.goal, filtered)
        trace.record("answer_synthesized", "synthesize", citation_count=len(citations))
        if not validate_citations(citations, filtered):
            trace.record("citation_validation_failed", "verify")
            return self._terminal_result(
                request=request,
                status=RunStatus.FAILED,
                reason="citation_contract_failed",
                trace=trace,
                started=started,
                evidence=filtered,
                errors=(
                    AgentError(
                        code="INVALID_CITATION",
                        message="答案引用了不存在的 Evidence ID",
                        layer=ErrorLayer.RUNTIME,
                    ),
                ),
                tool_calls=(tool_call,),
            )

        trace.record("citation_validation_passed", "verify")
        trace.record("run_completed", "finalize", status=RunStatus.COMPLETED)
        elapsed_ms = (time.perf_counter() - started) * 1000
        artifact = Artifact(
            artifact_type="research_report",
            content=answer,
            content_hash=_artifact_hash(answer),
            evidence_ids=tuple(item.evidence_id for item in filtered),
        )
        trajectory = trace.snapshot()
        return RunResult(
            task_id=request.task_id,
            thread_id=request.thread_id,
            status=RunStatus.COMPLETED,
            answer=answer,
            evidence=filtered,
            citations=citations,
            artifacts=(artifact,),
            tool_calls=(tool_call,),
            trajectory=trajectory,
            metrics=RunMetrics(
                tool_calls=1,
                steps=len(trajectory),
                elapsed_ms=elapsed_ms,
            ),
            trace_id=trace.trace_id,
            termination_reason="evidence_verified",
            runtime=self.name,
            versions={"fixture": getattr(self.search, "version", "unknown"), "runtime": "v1"},
        )

    def _terminal_result(
        self,
        *,
        request: RunRequest,
        status: RunStatus,
        reason: str,
        trace: TraceRecorder,
        started: float,
        answer: str | None = None,
        evidence: tuple[Evidence, ...] = (),
        errors: tuple[AgentError, ...] = (),
        tool_calls: tuple[ToolCallRecord, ...] = (),
    ) -> RunResult:
        trajectory = trace.snapshot()
        return RunResult(
            task_id=request.task_id,
            thread_id=request.thread_id,
            status=status,
            answer=answer,
            evidence=evidence,
            errors=errors,
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
            versions={"fixture": getattr(self.search, "version", "unknown"), "runtime": "v1"},
        )
