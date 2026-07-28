"""Machine-readable proof bundle for a successfully verified repair."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.domain.models import FrozenModel, RunMetrics, utc_now
from agent_lab.repair.domain.result import (
    PatchRecord,
    RepairResult,
    RepairStatus,
    VerificationRecord,
)
from agent_lab.repair.domain.task import (
    IOSExecutionEnvironment,
    ProofArtifactKind,
    RepairTask,
    VerificationCheck,
    scope_contains,
    validate_relative_path,
)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def task_contract_hash(task: RepairTask) -> str:
    """Hash the complete task contract using deterministic JSON."""

    payload = _canonicalize(task.model_dump(mode="json"))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class AgentIdentity(FrozenModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model: str | None = None
    provider: str | None = None
    prompt_version: str = Field(min_length=1)
    tool_versions: dict[str, str] = Field(default_factory=dict)


class ProofArtifact(FrozenModel):
    artifact_id: str = Field(min_length=1)
    kind: ProofArtifactKind
    relative_path: str
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return validate_relative_path(value, label="proof artifact path")


class PatchPassport(FrozenModel):
    schema_version: Literal["patch-passport-v1"] = "patch-passport-v1"
    generated_at: datetime = Field(default_factory=utc_now)
    task_id: str = Field(min_length=1)
    task_contract_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    agent: AgentIdentity
    environment: IOSExecutionEnvironment
    patch: PatchRecord
    approval_receipt_id: str = Field(min_length=1)
    required_checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    verification: tuple[VerificationRecord, ...] = Field(min_length=1)
    required_artifacts: tuple[ProofArtifactKind, ...] = Field(min_length=1)
    artifacts: tuple[ProofArtifact, ...] = Field(min_length=1)
    root_cause_evidence_ids: tuple[str, ...] = Field(min_length=1)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    trace_id: str = Field(min_length=1)
    limitations: tuple[str, ...] = ()

    @field_validator("required_checks", "required_artifacts", "root_cause_evidence_ids")
    @classmethod
    def validate_unique_requirements(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(set(value)) != len(value):
            raise ValueError("passport requirements and evidence IDs must be unique")
        if any(isinstance(item, str) and not item.strip() for item in value):
            raise ValueError("passport requirements and evidence IDs must not be empty")
        return value

    @model_validator(mode="after")
    def validate_passport_contract(self) -> Self:
        if self.source_commit != self.patch.base_commit:
            raise ValueError("passport source commit must match patch base commit")

        check_names = [item.check for item in self.verification]
        if len(set(check_names)) != len(check_names):
            raise ValueError("passport verification checks must be unique")
        checks_by_name = {item.check: item for item in self.verification}
        missing_checks = set(self.required_checks) - set(checks_by_name)
        if missing_checks:
            raise ValueError(f"passport is missing required checks: {sorted(missing_checks)}")
        non_passing = [
            item.check for item in self.verification if item.status is not EvalStatus.PASS
        ]
        if non_passing:
            raise ValueError(f"patch passport cannot contain non-passing checks: {non_passing}")

        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("passport artifact IDs must be unique")
        artifact_kinds = {item.kind for item in self.artifacts}
        missing_artifacts = set(self.required_artifacts) - artifact_kinds
        if missing_artifacts:
            raise ValueError(f"passport is missing required artifacts: {sorted(missing_artifacts)}")

        artifacts_by_id = {item.artifact_id: item for item in self.artifacts}
        for verification_record in self.verification:
            missing_references = set(verification_record.artifact_ids) - set(artifacts_by_id)
            if missing_references:
                raise ValueError(
                    "passport verification references missing artifacts: "
                    f"{sorted(missing_references)}"
                )
        expected_kinds = {
            VerificationCheck.REPRODUCTION: {ProofArtifactKind.BEFORE_SCREENSHOT},
            VerificationCheck.PATCH_SCOPE: {ProofArtifactKind.PATCH_DIFF},
            VerificationCheck.BUILD: {ProofArtifactKind.BUILD_XCRESULT},
            VerificationCheck.EXISTING_TESTS: {ProofArtifactKind.TEST_XCRESULT},
            VerificationCheck.TASK_ORACLE: {ProofArtifactKind.TEST_XCRESULT},
            VerificationCheck.SIMULATOR_REPLAY: {ProofArtifactKind.AFTER_SCREENSHOT},
            VerificationCheck.FORBIDDEN_CHANGES: {ProofArtifactKind.PATCH_DIFF},
            VerificationCheck.REGRESSION: {ProofArtifactKind.TEST_XCRESULT},
        }
        for check_name, kinds in expected_kinds.items():
            record = checks_by_name.get(check_name)
            if record is None:
                continue
            observed = {artifacts_by_id[artifact_id].kind for artifact_id in record.artifact_ids}
            missing_kinds = kinds - observed
            if missing_kinds:
                raise ValueError(
                    f"{check_name} lacks proof artifact kinds: {sorted(missing_kinds)}"
                )
        diff = artifacts_by_id.get(self.patch.diff_artifact_id)
        if diff is None or diff.kind is not ProofArtifactKind.PATCH_DIFF:
            raise ValueError("passport patch must reference a patch.diff artifact")
        if diff.content_hash != self.patch.content_hash:
            raise ValueError("passport patch hash must match the patch.diff artifact")
        return self


def build_patch_passport(
    task: RepairTask,
    result: RepairResult,
    *,
    agent: AgentIdentity,
    artifacts: tuple[ProofArtifact, ...],
    limitations: tuple[str, ...] = (),
) -> PatchPassport:
    """Build a success passport from one task and its verified run result."""

    if result.task_id != task.task_id:
        raise ValueError("repair result does not belong to the task")
    if result.status is not RepairStatus.DELIVERED:
        raise ValueError("only a delivered repair can produce a patch passport")
    if result.source_commit != task.repository.base_commit:
        raise ValueError("repair ran against a different source commit")
    if result.patch is None or result.approval_receipt_id is None:
        raise ValueError("delivered repair is missing patch or approval evidence")
    if len(result.patch.changed_files) > task.verification.changed_file_limit:
        raise ValueError("patch exceeds the task changed-file limit")
    for changed_file in result.patch.changed_files:
        if not any(
            scope_contains(pattern, changed_file) for pattern in task.permissions.writable_paths
        ):
            raise ValueError(f"patch changed a path outside writable scope: {changed_file}")
        if any(
            scope_contains(pattern, changed_file) for pattern in task.permissions.forbidden_paths
        ):
            raise ValueError(f"patch changed a forbidden path: {changed_file}")

    result_artifacts = {item.artifact_id: item for item in result.artifacts}
    for artifact in artifacts:
        source = result_artifacts.get(artifact.artifact_id)
        if source is None:
            raise ValueError(f"proof artifact is absent from repair result: {artifact.artifact_id}")
        if source.artifact_type != artifact.kind or source.content_hash != artifact.content_hash:
            raise ValueError(f"proof artifact metadata drift: {artifact.artifact_id}")

    proof_artifacts = {item.artifact_id: item for item in artifacts}
    diff_artifact = proof_artifacts.get(result.patch.diff_artifact_id)
    if diff_artifact is None:
        raise ValueError("proof bundle is missing the patch diff")
    if diff_artifact.size_bytes > task.verification.max_patch_bytes:
        raise ValueError("patch exceeds the task byte limit")

    checks_by_name = {item.check: item for item in result.checks}
    root_cause = checks_by_name.get(VerificationCheck.ROOT_CAUSE_EVIDENCE)
    if root_cause is None or not root_cause.evidence_ids:
        raise ValueError("patch passport requires root-cause evidence IDs")

    required_artifacts = tuple(
        dict.fromkeys(
            (*task.reproduction.required_artifacts, *task.verification.required_artifacts)
        )
    )
    return PatchPassport(
        task_id=task.task_id,
        task_contract_hash=task_contract_hash(task),
        run_id=result.run_id,
        source_commit=result.source_commit,
        agent=agent,
        environment=task.environment,
        patch=result.patch,
        approval_receipt_id=result.approval_receipt_id,
        required_checks=task.verification.required_checks,
        verification=result.checks,
        required_artifacts=required_artifacts,
        artifacts=artifacts,
        root_cause_evidence_ids=root_cause.evidence_ids,
        metrics=result.metrics,
        trace_id=result.trace_id,
        limitations=limitations,
    )
