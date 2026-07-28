from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.domain.models import Artifact, Budget, Evidence, SourceType, TrustLevel
from agent_lab.repair.domain import (
    AgentIdentity,
    IOSExecutionEnvironment,
    PatchRecord,
    PermissionSpec,
    ProofArtifact,
    ProofArtifactKind,
    RepairResult,
    RepairStage,
    RepairStatus,
    RepairTask,
    RepositoryKind,
    RepositorySpec,
    ReproductionSpec,
    TaskOrigin,
    VerificationCheck,
    VerificationRecord,
    VerificationSpec,
    build_patch_passport,
    task_contract_hash,
)

BASE_COMMIT = "a" * 40
PATCH_HASH = f"sha256:{'b' * 64}"
OTHER_HASH = f"sha256:{'c' * 64}"


def task() -> RepairTask:
    return RepairTask(
        task_id="keyboard-layout-001",
        title="Composer jumps when the keyboard changes state",
        origin=TaskOrigin.SYNTHETIC_SEEDED,
        repository=RepositorySpec(
            url="https://github.com/example/keyboard-fixture.git",
            base_commit=BASE_COMMIT,
            license_spdx="MIT",
        ),
        environment=IOSExecutionEnvironment(
            macos_version="15.5",
            xcode_version="16.4",
            ios_runtime="18.5",
            device="iPhone 16 Pro",
            swift_version="6.1",
        ),
        problem_statement="The composer jumps after the software keyboard changes state.",
        expected_invariants=(
            "The composer remains attached to the keyboard layout guide.",
            "The implementation does not hard-code keyboard height.",
        ),
        reproduction=ReproductionSpec(
            scheme="KeyboardFixture",
            steps=("Launch the fixture.", "Focus the composer.", "Switch keyboard state."),
            required_artifacts=(
                ProofArtifactKind.BEFORE_SCREENSHOT,
                ProofArtifactKind.APPLICATION_LOG,
            ),
            seed_state="fixtures/keyboard-layout-001/seed.json",
        ),
        permissions=PermissionSpec(
            readable_paths=("KeyboardFixture/**",),
            writable_paths=("KeyboardFixture/Sources/**", "KeyboardFixture/Tests/**"),
            forbidden_paths=("KeyboardFixture.xcodeproj/project.pbxproj", "Package.resolved"),
        ),
        budget=Budget(
            max_model_calls=20,
            max_tool_calls=80,
            max_steps=80,
            max_evidence_items=20,
            max_tokens=100_000,
            max_cost_usd=10,
            timeout_seconds=1800,
        ),
        verification=VerificationSpec(
            build_scheme="KeyboardFixture",
            public_test_plans=("KeyboardFixtureTests",),
            task_oracle_ids=("keyboard-layout-oracle-v1",),
            replay_script="fixtures/keyboard-layout-001/verify.yaml",
            required_checks=(
                VerificationCheck.REPRODUCTION,
                VerificationCheck.ROOT_CAUSE_EVIDENCE,
                VerificationCheck.PATCH_SCOPE,
                VerificationCheck.BUILD,
                VerificationCheck.EXISTING_TESTS,
                VerificationCheck.TASK_ORACLE,
                VerificationCheck.SIMULATOR_REPLAY,
                VerificationCheck.FORBIDDEN_CHANGES,
                VerificationCheck.REGRESSION,
            ),
            required_artifacts=(
                ProofArtifactKind.PATCH_DIFF,
                ProofArtifactKind.BUILD_XCRESULT,
                ProofArtifactKind.TEST_XCRESULT,
                ProofArtifactKind.AFTER_SCREENSHOT,
                ProofArtifactKind.AGENT_TRACE,
            ),
        ),
    )


def root_cause_evidence() -> Evidence:
    return Evidence.from_content(
        evidence_id="root-cause-1",
        source_type=SourceType.DOCUMENT,
        content="The bottom constraint follows a stale notification-derived height.",
        trust_level=TrustLevel.FIRST_PARTY,
        observed_at=datetime(2026, 7, 28, tzinfo=UTC),
        untrusted=False,
    )


def result_artifacts() -> tuple[Artifact, ...]:
    return (
        Artifact(
            artifact_id="patch",
            artifact_type=ProofArtifactKind.PATCH_DIFF,
            uri="proof/patch.diff",
            content_hash=PATCH_HASH,
        ),
        Artifact(
            artifact_id="build",
            artifact_type=ProofArtifactKind.BUILD_XCRESULT,
            uri="proof/build.xcresult",
            content_hash=OTHER_HASH,
        ),
        Artifact(
            artifact_id="tests",
            artifact_type=ProofArtifactKind.TEST_XCRESULT,
            uri="proof/test.xcresult",
            content_hash=OTHER_HASH,
        ),
        Artifact(
            artifact_id="before",
            artifact_type=ProofArtifactKind.BEFORE_SCREENSHOT,
            uri="proof/before.png",
            content_hash=OTHER_HASH,
        ),
        Artifact(
            artifact_id="after",
            artifact_type=ProofArtifactKind.AFTER_SCREENSHOT,
            uri="proof/after.png",
            content_hash=OTHER_HASH,
        ),
        Artifact(
            artifact_id="log",
            artifact_type=ProofArtifactKind.APPLICATION_LOG,
            uri="proof/application.log",
            content_hash=OTHER_HASH,
        ),
        Artifact(
            artifact_id="trace",
            artifact_type=ProofArtifactKind.AGENT_TRACE,
            uri="proof/agent-trace.jsonl",
            content_hash=OTHER_HASH,
        ),
    )


def checks(*, failing: VerificationCheck | None = None) -> tuple[VerificationRecord, ...]:
    task_checks = task().verification.required_checks
    records = []
    for check in task_checks:
        status = EvalStatus.FAIL if check is failing else EvalStatus.PASS
        artifact_ids: tuple[str, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        if check is VerificationCheck.REPRODUCTION:
            artifact_ids = ("before", "log")
        elif check is VerificationCheck.ROOT_CAUSE_EVIDENCE:
            evidence_ids = ("root-cause-1",)
        elif check in {VerificationCheck.PATCH_SCOPE, VerificationCheck.FORBIDDEN_CHANGES}:
            artifact_ids = ("patch",)
        elif check is VerificationCheck.BUILD:
            artifact_ids = ("build",)
        elif (
            check in {VerificationCheck.EXISTING_TESTS, VerificationCheck.REGRESSION}
            or check is VerificationCheck.TASK_ORACLE
        ):
            artifact_ids = ("tests",)
        elif check is VerificationCheck.SIMULATOR_REPLAY:
            artifact_ids = ("after",)
        records.append(
            VerificationRecord(
                check=check,
                status=status,
                message=f"{check} {'passed' if status is EvalStatus.PASS else 'failed'}",
                artifact_ids=artifact_ids,
                evidence_ids=evidence_ids,
            )
        )
    return tuple(records)


def delivered_result(*, failing: VerificationCheck | None = None) -> RepairResult:
    return RepairResult(
        task_id="keyboard-layout-001",
        run_id="run-001",
        status=RepairStatus.DELIVERED,
        stage=RepairStage.DELIVERED,
        source_commit=BASE_COMMIT,
        patch=PatchRecord(
            base_commit=BASE_COMMIT,
            content_hash=PATCH_HASH,
            changed_files=("KeyboardFixture/Sources/ComposerView.swift",),
            diff_artifact_id="patch",
            additions=8,
            deletions=4,
        ),
        approval_receipt_id="approval-001",
        checks=checks(failing=failing),
        evidence=(root_cause_evidence(),),
        artifacts=result_artifacts(),
        termination_reason="all mandatory checks passed",
        runtime="deterministic-gold",
    )


def proof_artifacts() -> tuple[ProofArtifact, ...]:
    return tuple(
        ProofArtifact(
            artifact_id=artifact.artifact_id,
            kind=ProofArtifactKind(artifact.artifact_type),
            relative_path=artifact.uri or "proof/missing",
            content_hash=artifact.content_hash or OTHER_HASH,
            size_bytes=1,
        )
        for artifact in result_artifacts()
    )


def agent() -> AgentIdentity:
    return AgentIdentity(
        name="deterministic-gold",
        version="0.1.0",
        prompt_version="none",
        tool_versions={"native-xcodebuild": "planned"},
    )


def test_task_contract_is_stable_across_json_round_trip() -> None:
    original = task()
    restored = RepairTask.model_validate_json(original.model_dump_json())
    assert task_contract_hash(original) == task_contract_hash(restored)
    assert original.repository.base_commit == BASE_COMMIT


def test_permission_contract_rejects_traversal_and_unapproved_writes() -> None:
    with pytest.raises(ValidationError, match="traversal"):
        PermissionSpec(
            readable_paths=("../Secrets",),
            writable_paths=("Sources/**",),
            forbidden_paths=("Package.resolved",),
        )

    with pytest.raises(ValidationError, match="must require approval"):
        PermissionSpec(
            readable_paths=("Sources/**",),
            writable_paths=("Sources/**",),
            forbidden_paths=("Package.resolved",),
            allowed_capabilities=("fs.write",),
            approval_required_capabilities=(),
        )


def test_permission_contract_rejects_ambiguous_glob_semantics() -> None:
    with pytest.raises(ValidationError, match="only supports exact paths"):
        PermissionSpec(
            readable_paths=("Sources/**/*.swift",),
            writable_paths=("Sources/**",),
            forbidden_paths=("Package.resolved",),
        )


def test_task_contract_rejects_partial_commit_hashes() -> None:
    with pytest.raises(ValidationError, match="full 40- or 64-character"):
        RepositorySpec(
            url="https://github.com/example/repo",
            base_commit="abc123",
            license_spdx="MIT",
        )


def test_embedded_fixture_requires_explicit_path_and_synthetic_origin() -> None:
    original = task()
    payload = original.model_dump()
    payload["repository"] = {
        "kind": RepositoryKind.EMBEDDED_FIXTURE,
        "url": "embedded://keyboard-layout-001",
        "base_commit": BASE_COMMIT,
        "license_spdx": "MIT",
        "fixture_path": "benchmarks/ios-repair/fixtures/keyboard-layout-001",
    }
    embedded = RepairTask.model_validate(payload)
    assert embedded.repository.kind is RepositoryKind.EMBEDDED_FIXTURE

    payload["origin"] = TaskOrigin.OPEN_SOURCE_HISTORY
    with pytest.raises(ValidationError, match="must be labeled synthetic-seeded"):
        RepairTask.model_validate(payload)


def test_delivered_result_cannot_hide_a_failed_check() -> None:
    with pytest.raises(ValidationError, match="every recorded check to PASS"):
        delivered_result(failing=VerificationCheck.BUILD)


def test_applied_patch_requires_approval_bound_receipt() -> None:
    good = delivered_result()
    payload = good.model_dump()
    payload["approval_receipt_id"] = None
    with pytest.raises(ValidationError, match="requires an approval receipt"):
        RepairResult.model_validate(payload)


def test_terminal_status_requires_its_matching_stage() -> None:
    good = delivered_result()
    payload = good.model_dump()
    payload["status"] = RepairStatus.BUILD_FAILED
    with pytest.raises(ValidationError, match="requires the building stage"):
        RepairResult.model_validate(payload)


def test_patch_passport_is_built_from_task_and_verified_result() -> None:
    passport = build_patch_passport(
        task(),
        delivered_result(),
        agent=agent(),
        artifacts=proof_artifacts(),
    )
    assert passport.task_contract_hash == task_contract_hash(task())
    assert passport.patch.content_hash == PATCH_HASH
    assert set(passport.required_checks) == set(task().verification.required_checks)
    assert passport.root_cause_evidence_ids == ("root-cause-1",)


def test_patch_passport_rejects_missing_required_proof_artifact() -> None:
    without_trace = tuple(
        artifact
        for artifact in proof_artifacts()
        if artifact.kind is not ProofArtifactKind.AGENT_TRACE
    )
    with pytest.raises(ValidationError, match="missing required artifacts"):
        build_patch_passport(
            task(),
            delivered_result(),
            agent=agent(),
            artifacts=without_trace,
        )


@pytest.mark.parametrize(
    "changed_file",
    (
        "README.md",
        "KeyboardFixture.xcodeproj/project.pbxproj",
    ),
)
def test_patch_passport_rejects_out_of_scope_or_forbidden_patch(changed_file: str) -> None:
    result = delivered_result()
    payload = result.model_dump()
    payload["patch"]["changed_files"] = (changed_file,)
    changed = RepairResult.model_validate(payload)
    with pytest.raises(ValueError, match=r"outside writable scope|forbidden path"):
        build_patch_passport(
            task(),
            changed,
            agent=agent(),
            artifacts=proof_artifacts(),
        )


def test_patch_passport_rejects_oversized_patch() -> None:
    oversized = tuple(
        artifact.model_copy(update={"size_bytes": 200_001})
        if artifact.kind is ProofArtifactKind.PATCH_DIFF
        else artifact
        for artifact in proof_artifacts()
    )
    with pytest.raises(ValueError, match="byte limit"):
        build_patch_passport(
            task(),
            delivered_result(),
            agent=agent(),
            artifacts=oversized,
        )


def test_patch_passport_rejects_source_commit_drift() -> None:
    task_payload = task().model_dump()
    task_payload["repository"]["base_commit"] = "d" * 40
    drifted_task = RepairTask.model_validate(task_payload)
    with pytest.raises(ValueError, match="different source commit"):
        build_patch_passport(
            drifted_task,
            delivered_result(),
            agent=agent(),
            artifacts=proof_artifacts(),
        )
