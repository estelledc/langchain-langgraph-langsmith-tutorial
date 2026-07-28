"""Run result contracts for a repair attempt."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.domain.models import AgentError, Artifact, Evidence, FrozenModel, RunMetrics
from agent_lab.repair.domain.task import VerificationCheck, validate_relative_path


class RepairStage(StrEnum):
    PREPARED = "prepared"
    REPRODUCING = "reproducing"
    DIAGNOSING = "diagnosing"
    PLAN_READY = "plan_ready"
    APPROVAL_REQUIRED = "approval_required"
    PATCHING = "patching"
    BUILDING = "building"
    TESTING = "testing"
    VERIFYING_RUNTIME = "verifying_runtime"
    DELIVERED = "delivered"
    ROLLED_BACK = "rolled_back"


class RepairStatus(StrEnum):
    APPROVAL_REQUIRED = "approval_required"
    POLICY_BLOCKED = "policy_blocked"
    BUILD_FAILED = "build_failed"
    TEST_FAILED = "test_failed"
    RUNTIME_VERIFICATION_FAILED = "runtime_verification_failed"
    DELIVERED = "delivered"
    UNKNOWN = "unknown"
    INFRA_ERROR = "infra_error"
    ROLLED_BACK = "rolled_back"


class VerificationRecord(FrozenModel):
    check: VerificationCheck
    status: EvalStatus
    message: str = Field(min_length=1)
    artifact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    duration_ms: float = Field(default=0, ge=0)
    error: str | None = None

    @field_validator("artifact_ids", "evidence_ids")
    @classmethod
    def validate_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("verification references must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("verification references must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_error_semantics(self) -> Self:
        if self.status is EvalStatus.ERROR and not self.error:
            raise ValueError("ERROR verification records require error details")
        if self.status is not EvalStatus.ERROR and self.error:
            raise ValueError("only ERROR verification records may carry error details")
        return self


class PatchRecord(FrozenModel):
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    changed_files: tuple[str, ...] = Field(min_length=1)
    diff_artifact_id: str = Field(min_length=1)
    additions: int = Field(default=0, ge=0)
    deletions: int = Field(default=0, ge=0)

    @field_validator("changed_files")
    @classmethod
    def validate_changed_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_relative_path(item, label="changed file") for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("changed files must be unique")
        return normalized


class RepairResult(FrozenModel):
    task_id: str = Field(min_length=1)
    run_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    status: RepairStatus
    stage: RepairStage
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    patch: PatchRecord | None = None
    approval_receipt_id: str | None = None
    checks: tuple[VerificationRecord, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    errors: tuple[AgentError, ...] = ()
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    trace_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    termination_reason: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_contract(self) -> Self:
        check_names = [item.check for item in self.checks]
        if len(set(check_names)) != len(check_names):
            raise ValueError("verification check names must be unique")

        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("repair evidence IDs must be unique")

        artifact_ids = {item.artifact_id for item in self.artifacts}
        if len(artifact_ids) != len(self.artifacts):
            raise ValueError("repair artifact IDs must be unique")

        for check in self.checks:
            missing_artifacts = set(check.artifact_ids) - artifact_ids
            missing_evidence = set(check.evidence_ids) - evidence_ids
            if missing_artifacts:
                raise ValueError(
                    f"verification references missing artifacts: {sorted(missing_artifacts)}"
                )
            if missing_evidence:
                raise ValueError(
                    f"verification references missing evidence: {sorted(missing_evidence)}"
                )

        if self.patch is not None:
            if not self.approval_receipt_id:
                raise ValueError("an applied patch requires an approval receipt")
            if self.patch.base_commit != self.source_commit:
                raise ValueError("patch base commit must match the repair source commit")
            artifacts_by_id = {item.artifact_id: item for item in self.artifacts}
            diff_artifact = artifacts_by_id.get(self.patch.diff_artifact_id)
            if diff_artifact is None:
                raise ValueError("patch references a missing diff artifact")
            if diff_artifact.content_hash != self.patch.content_hash:
                raise ValueError("patch hash must match its diff artifact")

        if self.status is RepairStatus.DELIVERED:
            if self.stage is not RepairStage.DELIVERED:
                raise ValueError("delivered status requires the delivered stage")
            if self.patch is None:
                raise ValueError("delivered repair requires a patch")
            if not self.checks or any(item.status is not EvalStatus.PASS for item in self.checks):
                raise ValueError("delivered repair requires every recorded check to PASS")
            if self.errors:
                raise ValueError("delivered repair must not carry unresolved errors")

        if self.status is RepairStatus.APPROVAL_REQUIRED and self.patch is not None:
            raise ValueError("approval-required result cannot contain an applied patch")
        if self.status is RepairStatus.INFRA_ERROR and not self.errors:
            raise ValueError("infra_error requires a structured error")

        fixed_stages = {
            RepairStatus.APPROVAL_REQUIRED: RepairStage.APPROVAL_REQUIRED,
            RepairStatus.BUILD_FAILED: RepairStage.BUILDING,
            RepairStatus.TEST_FAILED: RepairStage.TESTING,
            RepairStatus.RUNTIME_VERIFICATION_FAILED: RepairStage.VERIFYING_RUNTIME,
            RepairStatus.DELIVERED: RepairStage.DELIVERED,
            RepairStatus.ROLLED_BACK: RepairStage.ROLLED_BACK,
        }
        expected_stage = fixed_stages.get(self.status)
        if expected_stage is not None and self.stage is not expected_stage:
            raise ValueError(f"{self.status} status requires the {expected_stage} stage")
        return self
