"""Stable contracts shared by every runtime and evaluator."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware timestamp."""

    return datetime.now(UTC)


class FrozenModel(BaseModel):
    """Immutable value object used at trust boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunStatus(StrEnum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    APPROVAL_REQUIRED = "approval_required"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    FIXTURE = "fixture"
    DOCUMENT = "document"
    WEB = "web"
    DATABASE = "database"
    TOOL = "tool"
    USER = "user"


class TrustLevel(StrEnum):
    FIRST_PARTY = "first_party"
    VERIFIED_THIRD_PARTY = "verified_third_party"
    UNVERIFIED = "unverified"
    SYNTHETIC = "synthetic"


class ErrorLayer(StrEnum):
    INPUT = "input"
    POLICY = "policy"
    TOOL = "tool"
    RUNTIME = "runtime"
    PROVIDER = "provider"
    EVALUATOR = "evaluator"


class Budget(FrozenModel):
    max_model_calls: int = Field(default=0, ge=0, le=100)
    max_tool_calls: int = Field(default=4, ge=0, le=100)
    max_steps: int = Field(default=8, ge=1, le=200)
    max_evidence_items: int = Field(default=3, ge=1, le=100)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    timeout_seconds: float = Field(default=10.0, gt=0, le=3600)


class RunRequest(FrozenModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    thread_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    user_id: str | None = None
    goal: str = Field(min_length=1, max_length=4000)
    constraints: tuple[str, ...] = ()
    allowed_capabilities: frozenset[str] = frozenset({"fixture.search"})
    budget: Budget = Field(default_factory=Budget)
    metadata: dict[str, str] = Field(default_factory=dict)


class Evidence(FrozenModel):
    evidence_id: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str | None = None
    title: str | None = None
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    observed_at: datetime
    valid_at: datetime | None = None
    tool_call_id: str | None = None
    trust_level: TrustLevel
    untrusted: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_content(
        cls,
        *,
        evidence_id: str,
        source_type: SourceType,
        content: str,
        trust_level: TrustLevel,
        source_uri: str | None = None,
        title: str | None = None,
        observed_at: datetime | None = None,
        valid_at: datetime | None = None,
        tool_call_id: str | None = None,
        untrusted: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> Self:
        normalized = content.strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return cls(
            evidence_id=evidence_id,
            source_type=source_type,
            source_uri=source_uri,
            title=title,
            content=normalized,
            content_hash=f"sha256:{digest}",
            observed_at=observed_at or utc_now(),
            valid_at=valid_at,
            tool_call_id=tool_call_id,
            trust_level=trust_level,
            untrusted=untrusted,
            metadata=metadata or {},
        )


class Citation(FrozenModel):
    claim: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class Artifact(FrozenModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: str = Field(min_length=1)
    content: str | None = None
    uri: str | None = None
    content_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    evidence_ids: tuple[str, ...] = ()


class AgentError(FrozenModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    layer: ErrorLayer
    retryable: bool = False
    details: dict[str, str] = Field(default_factory=dict)


class ToolCallRecord(FrozenModel):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    arguments: dict[str, Any]
    status: str
    error_code: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class TraceEvent(FrozenModel):
    sequence: int = Field(ge=0)
    event: str = Field(min_length=1)
    node: str = Field(min_length=1)
    at: datetime = Field(default_factory=utc_now)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RunMetrics(FrozenModel):
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    steps: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class RunResult(FrozenModel):
    task_id: str
    thread_id: str
    status: RunStatus
    answer: str | None = None
    evidence: tuple[Evidence, ...] = ()
    citations: tuple[Citation, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    errors: tuple[AgentError, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    trajectory: tuple[TraceEvent, ...] = ()
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    termination_reason: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        if self.status is RunStatus.COMPLETED and not (self.answer or "").strip():
            raise ValueError("completed result requires a non-empty answer")

        evidence_ids = {item.evidence_id for item in self.evidence}
        duplicate_count = len(self.evidence) - len(evidence_ids)
        if duplicate_count:
            raise ValueError("evidence_id values must be unique")

        for citation in self.citations:
            missing = set(citation.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(f"citation references missing evidence: {sorted(missing)}")
        return self
