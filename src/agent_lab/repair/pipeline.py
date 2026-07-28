"""End-to-end deterministic repair pipeline for the first XcodeFixBench task."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Never
from uuid import uuid4

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.domain.models import (
    AgentError,
    Artifact,
    ErrorLayer,
    Evidence,
    RunMetrics,
    SourceType,
    TrustLevel,
)
from agent_lab.repair.artifacts import path_size, sha256_bytes, sha256_path, write_json
from agent_lab.repair.command import CommandExecutionError, SafeCommandRunner
from agent_lab.repair.domain.approval import ApprovalAction, ApprovalReceipt
from agent_lab.repair.domain.proof import (
    AgentIdentity,
    PatchPassport,
    ProofArtifact,
    build_patch_passport,
)
from agent_lab.repair.domain.replay import ReplaySpec
from agent_lab.repair.domain.result import (
    PatchRecord,
    RepairResult,
    RepairStage,
    RepairStatus,
    VerificationRecord,
)
from agent_lab.repair.domain.task import ProofArtifactKind, RepairTask, VerificationCheck
from agent_lab.repair.simulator import NativeSimulatorAdapter, SimulatorDevice
from agent_lab.repair.task_loader import load_replay_spec
from agent_lab.repair.workspace import (
    MaterializedWorkspace,
    SourceSnapshotMismatch,
    materialize_embedded_workspace,
)
from agent_lab.repair.xcode import NativeXcodeAdapter, XcodeBuildOutcome, XcodeTestOutcome


@dataclass(frozen=True)
class RepairPipelineOutcome:
    result: RepairResult
    passport: PatchPassport | None
    output_dir: Path


class PipelineFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: RepairStatus,
        stage: RepairStage,
        layer: ErrorLayer,
        code: str,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.layer = layer
        self.code = code


@dataclass
class _Context:
    task: RepairTask
    output_dir: Path
    runner: SafeCommandRunner
    started: float
    run_id: str
    stage: RepairStage = RepairStage.PREPARED
    checks: list[VerificationRecord] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    proof_artifacts: list[ProofArtifact] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    patch: PatchRecord | None = None
    approval_receipt_id: str | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def event(self, name: str, **attributes: object) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "event": name,
                "attributes": attributes,
            }
        )

    def check(
        self,
        check: VerificationCheck,
        status: EvalStatus,
        message: str,
        *,
        artifact_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        error: str | None = None,
    ) -> None:
        if any(item.check is check for item in self.checks):
            raise ValueError(f"duplicate pipeline check: {check}")
        self.checks.append(
            VerificationRecord(
                check=check,
                status=status,
                message=message,
                artifact_ids=artifact_ids,
                evidence_ids=evidence_ids,
                error=error,
            )
        )


class DeterministicRepairPipeline:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def run(
        self,
        *,
        task: RepairTask,
        patch_path: Path,
        output_dir: Path,
        approval_actor: str | None,
        device_id: str | None = None,
    ) -> RepairPipelineOutcome:
        resolved_output = output_dir.resolve()
        if resolved_output.exists() and any(resolved_output.iterdir()):
            raise FileExistsError(f"run output must be empty: {resolved_output}")
        resolved_output.mkdir(parents=True, exist_ok=True)
        runner = SafeCommandRunner(
            allowed_root=resolved_output,
            receipt_dir=resolved_output / "receipts",
        )
        context = _Context(
            task=task,
            output_dir=resolved_output,
            runner=runner,
            started=monotonic(),
            run_id=str(uuid4()),
        )
        passport: PatchPassport | None = None
        try:
            completed = self._run_success_path(
                context,
                patch_path=patch_path,
                approval_actor=approval_actor,
                requested_device_id=device_id,
            )
            if completed is None:
                result = self._result(
                    context,
                    status=RepairStatus.APPROVAL_REQUIRED,
                    stage=RepairStage.APPROVAL_REQUIRED,
                    termination_reason="candidate patch is inspected and awaits explicit approval",
                )
            else:
                result, passport = completed
        except PipelineFailure as exc:
            result = self._result(
                context,
                status=exc.status,
                stage=exc.stage,
                termination_reason=str(exc),
                errors=(
                    AgentError(
                        code=exc.code,
                        message=str(exc),
                        layer=exc.layer,
                    ),
                ),
            )
        except Exception as exc:
            result = self._result(
                context,
                status=RepairStatus.INFRA_ERROR,
                stage=context.stage,
                termination_reason=f"unexpected pipeline failure: {type(exc).__name__}",
                errors=(
                    AgentError(
                        code="unexpected_pipeline_failure",
                        message=f"{type(exc).__name__}: {exc}",
                        layer=ErrorLayer.RUNTIME,
                    ),
                ),
            )

        write_json(resolved_output / "repair-result.json", result)
        if passport is not None and result.status is RepairStatus.DELIVERED:
            write_json(resolved_output / "patch-passport.json", passport)
        else:
            passport = None
        return RepairPipelineOutcome(result=result, passport=passport, output_dir=resolved_output)

    def _run_success_path(
        self,
        context: _Context,
        *,
        patch_path: Path,
        approval_actor: str | None,
        requested_device_id: str | None,
    ) -> tuple[RepairResult, PatchPassport] | None:
        task = context.task
        candidate = context.output_dir / "input" / "candidate.patch"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        resolved_patch = patch_path.resolve()
        if resolved_patch.is_symlink() or not resolved_patch.is_file():
            self._fail(
                "candidate patch is missing or is a symlink",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.INPUT,
                code="invalid_patch_input",
            )
        if resolved_patch.stat().st_size > task.verification.max_patch_bytes:
            self._fail(
                "candidate patch exceeds the task byte limit",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.POLICY,
                code="patch_budget_exceeded",
            )
        shutil.copyfile(resolved_patch, candidate)

        try:
            workspace = materialize_embedded_workspace(
                project_root=self.project_root,
                task=task,
                destination=context.output_dir / "workspace",
                runner=context.runner,
            )
        except (SourceSnapshotMismatch, CommandExecutionError, ValueError) as exc:
            self._fail(
                f"source snapshot could not be reproduced: {exc}",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.POLICY,
                code="source_snapshot_mismatch",
            )
        context.event("source.materialized", commit=task.repository.base_commit)

        xcode = NativeXcodeAdapter(runner=context.runner)
        simulator = NativeSimulatorAdapter(runner=context.runner)
        self._verify_host_environment(context, xcode=xcode, workspace=workspace)
        try:
            device = simulator.resolve_device(
                cwd=workspace.root,
                device_name=task.environment.device,
                runtime_version=task.environment.ios_runtime,
                requested_udid=requested_device_id,
            )
        except Exception as exc:
            self._fail(
                f"pinned simulator is unavailable: {exc}",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.POLICY,
                code="simulator_environment_mismatch",
            )
        context.versions["simulator_device"] = device.name
        context.versions["simulator_runtime"] = f"{device.runtime_version} ({device.runtime_build})"
        context.versions["simulator_runtime_identifier"] = device.runtime_identifier
        context.event("environment.verified", device=device.name, runtime=device.runtime_identifier)

        replay = load_replay_spec(self.project_root, task)
        self._run_baseline(
            context,
            workspace=workspace,
            xcode=xcode,
            simulator=simulator,
            device=device,
            replay=replay,
        )

        context.stage = RepairStage.PLAN_READY
        try:
            inspection = workspace.inspect_patch(candidate)
        except (ValueError, CommandExecutionError) as exc:
            self._fail(
                f"candidate patch violates the task policy: {exc}",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PLAN_READY,
                layer=ErrorLayer.POLICY,
                code="patch_policy_rejected",
            )
        context.event(
            "patch.inspected",
            payload_hash=inspection.payload_hash,
            changed_files=list(inspection.changed_files),
        )

        if approval_actor is None:
            context.stage = RepairStage.APPROVAL_REQUIRED
            return None

        approval = ApprovalReceipt.issue(
            actor_id=approval_actor,
            action=ApprovalAction.APPLY_PATCH,
            payload=inspection.payload,
            scope=inspection.changed_files,
        )
        context.event(
            "approval.issued",
            approval_id=approval.approval_id,
            payload_hash=approval.payload_hash,
        )
        context.stage = RepairStage.PATCHING
        try:
            consumed = workspace.apply_patch(
                candidate,
                inspection=inspection,
                approval=approval,
            )
        except (ValueError, CommandExecutionError) as exc:
            self._fail(
                f"approved patch could not be applied: {exc}",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PATCHING,
                layer=ErrorLayer.POLICY,
                code="approved_patch_rejected",
            )

        approval_path = context.output_dir / "proof" / "approval-receipt.json"
        write_json(approval_path, consumed)
        self._artifact(
            context,
            artifact_id="approval-receipt",
            kind=ProofArtifactKind.APPROVAL_RECEIPT,
            path=approval_path,
            media_type="application/json",
        )
        context.approval_receipt_id = consumed.approval_id

        diff_path = context.output_dir / "proof" / "patch.diff"
        diff_payload = workspace.write_diff(diff_path)
        if len(diff_payload) > task.verification.max_patch_bytes:
            self._fail(
                "applied diff exceeds the task byte limit",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PATCHING,
                layer=ErrorLayer.POLICY,
                code="applied_patch_budget_exceeded",
            )
        diff_artifact = self._artifact(
            context,
            artifact_id="patch-diff",
            kind=ProofArtifactKind.PATCH_DIFF,
            path=diff_path,
            media_type="text/x-diff",
        )
        context.patch = PatchRecord(
            base_commit=task.repository.base_commit,
            content_hash=diff_artifact.content_hash or sha256_bytes(diff_payload),
            changed_files=inspection.changed_files,
            diff_artifact_id=diff_artifact.artifact_id,
            additions=inspection.additions,
            deletions=inspection.deletions,
        )
        context.check(
            VerificationCheck.PATCH_SCOPE,
            EvalStatus.PASS,
            "applied diff stays within the one-file writable scope",
            artifact_ids=(diff_artifact.artifact_id,),
        )
        context.check(
            VerificationCheck.FORBIDDEN_CHANGES,
            EvalStatus.PASS,
            "oracle, tests, project and runner files were not modified",
            artifact_ids=(diff_artifact.artifact_id,),
        )
        context.event("patch.applied", diff_hash=context.patch.content_hash)

        self._run_patched_verification(
            context,
            workspace=workspace,
            xcode=xcode,
            simulator=simulator,
            device=device,
            replay=replay,
        )
        self._write_trace(context)
        result = self._result(
            context,
            status=RepairStatus.DELIVERED,
            stage=RepairStage.DELIVERED,
            termination_reason="all mandatory repair checks passed",
        )
        passport = build_patch_passport(
            task,
            result,
            agent=AgentIdentity(
                name="deterministic-repair-harness",
                version="1",
                prompt_version="none",
                tool_versions={
                    "xcodebuild": context.versions.get("xcode", "unknown"),
                    "simulator": context.versions.get("simulator_runtime", "unknown"),
                },
            ),
            artifacts=tuple(context.proof_artifacts),
            limitations=(
                "synthetic-seeded task; no production source is included",
                "iOS Simulator replay does not prove real-device third-party keyboard behavior",
            ),
        )
        return result, passport

    def _run_baseline(
        self,
        context: _Context,
        *,
        workspace: MaterializedWorkspace,
        xcode: NativeXcodeAdapter,
        simulator: NativeSimulatorAdapter,
        device: SimulatorDevice,
        replay: ReplaySpec,
    ) -> None:
        context.stage = RepairStage.REPRODUCING
        build = xcode.build(
            name="baseline-build",
            workspace=workspace.root,
            scheme=context.task.verification.build_scheme,
            device_id=device.udid,
            output_dir=context.output_dir / "baseline" / "build",
            timeout=300,
        )
        baseline_build_artifact = self._optional_artifact(
            context,
            artifact_id="baseline-build-xcresult",
            kind=ProofArtifactKind.BUILD_XCRESULT,
            path=build.result_bundle,
        )
        if not build.passed or not build.app_path.is_dir():
            self._fail(
                "broken fixture does not build; task baseline is invalid",
                status=RepairStatus.INFRA_ERROR,
                stage=RepairStage.REPRODUCING,
                layer=ErrorLayer.TOOL,
                code="baseline_build_failed",
            )

        public = self._test_target(
            xcode=xcode,
            context=context,
            workspace=workspace,
            build=build,
            target=context.task.verification.public_test_plans[0],
            name="baseline-public-tests",
            output_dir=context.output_dir / "baseline" / "public-tests",
            device=device,
        )
        public_artifact = self._optional_artifact(
            context,
            artifact_id="baseline-public-tests-xcresult",
            kind=ProofArtifactKind.TEST_XCRESULT,
            path=public.result_bundle,
        )
        if not public.passed:
            self._fail(
                "existing tests fail before the candidate patch; task baseline is invalid",
                status=RepairStatus.INFRA_ERROR,
                stage=RepairStage.REPRODUCING,
                layer=ErrorLayer.TOOL,
                code="baseline_regression_failed",
            )

        oracle = self._test_target(
            xcode=xcode,
            context=context,
            workspace=workspace,
            build=build,
            target=context.task.verification.task_oracle_ids[0],
            name="baseline-task-oracle",
            output_dir=context.output_dir / "baseline" / "task-oracle",
            device=device,
        )
        oracle_artifact = self._optional_artifact(
            context,
            artifact_id="baseline-task-oracle-xcresult",
            kind=ProofArtifactKind.TEST_XCRESULT,
            path=oracle.result_bundle,
        )
        failed_tests = oracle.summary.get("failedTests", 0) if oracle.summary else 0
        if oracle.passed or not isinstance(failed_tests, int) or failed_tests < 1:
            self._fail(
                "task oracle does not fail on the broken fixture",
                status=RepairStatus.UNKNOWN,
                stage=RepairStage.REPRODUCING,
                layer=ErrorLayer.EVALUATOR,
                code="baseline_oracle_not_red",
            )

        replay_outcome = simulator.replay(
            name="baseline-replay",
            cwd=workspace.root,
            device=device,
            app_path=build.app_path,
            spec=replay,
            expectation=replay.baseline,
            output_dir=context.output_dir / "baseline" / "runtime",
        )
        before_runtime = self._artifact(
            context,
            artifact_id="before-runtime-result",
            kind=ProofArtifactKind.RUNTIME_RESULT,
            path=replay_outcome.result_artifact,
            media_type="application/json",
        )
        before_screenshot = self._artifact(
            context,
            artifact_id="before-screenshot",
            kind=ProofArtifactKind.BEFORE_SCREENSHOT,
            path=replay_outcome.screenshot,
            media_type="image/png",
        )
        if not replay_outcome.passed:
            self._fail(
                "broken runtime does not reproduce the task oracle: "
                + "; ".join(replay_outcome.failures),
                status=RepairStatus.UNKNOWN,
                stage=RepairStage.REPRODUCING,
                layer=ErrorLayer.EVALUATOR,
                code="baseline_runtime_not_reproduced",
            )

        source_path = workspace.root / "KeyboardFixture" / "KeyboardHeightResolver.swift"
        source_evidence = Evidence.from_content(
            evidence_id="root-cause-source",
            source_type=SourceType.DOCUMENT,
            source_uri=(
                f"{context.task.repository.url}@{context.task.repository.base_commit}/"
                "KeyboardFixture/KeyboardHeightResolver.swift"
            ),
            title="Broken height-source implementation",
            content=source_path.read_text(encoding="utf-8"),
            trust_level=TrustLevel.SYNTHETIC,
            untrusted=False,
        )
        runtime_evidence = Evidence.from_content(
            evidence_id="root-cause-runtime",
            source_type=SourceType.TOOL,
            source_uri="artifact://before-runtime-result",
            title="Broken runtime geometry",
            content=replay_outcome.result_artifact.read_text(encoding="utf-8"),
            trust_level=TrustLevel.SYNTHETIC,
            untrusted=False,
        )
        context.evidence.extend((source_evidence, runtime_evidence))
        reproduction_artifacts = tuple(
            item.artifact_id
            for item in (
                baseline_build_artifact,
                public_artifact,
                oracle_artifact,
                before_runtime,
                before_screenshot,
            )
            if item is not None
        )
        context.check(
            VerificationCheck.REPRODUCTION,
            EvalStatus.PASS,
            "broken fixture builds, existing tests pass, task oracle fails and runtime mismatch is 196",
            artifact_ids=reproduction_artifacts,
            evidence_ids=(runtime_evidence.evidence_id,),
        )
        context.check(
            VerificationCheck.ROOT_CAUSE_EVIDENCE,
            EvalStatus.PASS,
            "source reads the stale notification while runtime proves the guide reached 521",
            evidence_ids=(source_evidence.evidence_id, runtime_evidence.evidence_id),
        )
        context.event("reproduction.completed", mismatch=replay_outcome.record.mismatch)

    def _run_patched_verification(
        self,
        context: _Context,
        *,
        workspace: MaterializedWorkspace,
        xcode: NativeXcodeAdapter,
        simulator: NativeSimulatorAdapter,
        device: SimulatorDevice,
        replay: ReplaySpec,
    ) -> None:
        context.stage = RepairStage.BUILDING
        build = xcode.build(
            name="patched-build",
            workspace=workspace.root,
            scheme=context.task.verification.build_scheme,
            device_id=device.udid,
            output_dir=context.output_dir / "patched" / "build",
            timeout=300,
        )
        build_artifact = self._optional_artifact(
            context,
            artifact_id="patched-build-xcresult",
            kind=ProofArtifactKind.BUILD_XCRESULT,
            path=build.result_bundle,
        )
        if not build.passed or not build.app_path.is_dir() or build_artifact is None:
            ids = (build_artifact.artifact_id,) if build_artifact else ()
            context.check(
                VerificationCheck.BUILD,
                EvalStatus.FAIL,
                "candidate patch does not build",
                artifact_ids=ids,
            )
            self._fail(
                "candidate patch does not build",
                status=RepairStatus.BUILD_FAILED,
                stage=RepairStage.BUILDING,
                layer=ErrorLayer.TOOL,
                code="patched_build_failed",
            )
        context.check(
            VerificationCheck.BUILD,
            EvalStatus.PASS,
            "patched application builds for the pinned simulator",
            artifact_ids=(build_artifact.artifact_id,),
        )

        context.stage = RepairStage.TESTING
        public = self._test_target(
            xcode=xcode,
            context=context,
            workspace=workspace,
            build=build,
            target=context.task.verification.public_test_plans[0],
            name="patched-public-tests",
            output_dir=context.output_dir / "patched" / "public-tests",
            device=device,
        )
        public_artifact = self._optional_artifact(
            context,
            artifact_id="patched-public-tests-xcresult",
            kind=ProofArtifactKind.TEST_XCRESULT,
            path=public.result_bundle,
        )
        if not public.passed or public_artifact is None:
            ids = (public_artifact.artifact_id,) if public_artifact else ()
            context.check(
                VerificationCheck.EXISTING_TESTS,
                EvalStatus.FAIL,
                "candidate patch regresses existing behavior tests",
                artifact_ids=ids,
            )
            self._fail(
                "candidate patch regresses existing tests",
                status=RepairStatus.TEST_FAILED,
                stage=RepairStage.TESTING,
                layer=ErrorLayer.TOOL,
                code="existing_tests_failed",
            )
        context.check(
            VerificationCheck.EXISTING_TESTS,
            EvalStatus.PASS,
            "existing behavior tests pass after the patch",
            artifact_ids=(public_artifact.artifact_id,),
        )

        oracle = self._test_target(
            xcode=xcode,
            context=context,
            workspace=workspace,
            build=build,
            target=context.task.verification.task_oracle_ids[0],
            name="patched-task-oracle",
            output_dir=context.output_dir / "patched" / "task-oracle",
            device=device,
        )
        oracle_artifact = self._optional_artifact(
            context,
            artifact_id="patched-task-oracle-xcresult",
            kind=ProofArtifactKind.TEST_XCRESULT,
            path=oracle.result_bundle,
        )
        if not oracle.passed or oracle_artifact is None:
            ids = (oracle_artifact.artifact_id,) if oracle_artifact else ()
            context.check(
                VerificationCheck.TASK_ORACLE,
                EvalStatus.FAIL,
                "candidate patch does not satisfy the independent task oracle",
                artifact_ids=ids,
            )
            self._fail(
                "candidate patch does not satisfy the task oracle",
                status=RepairStatus.TEST_FAILED,
                stage=RepairStage.TESTING,
                layer=ErrorLayer.EVALUATOR,
                code="task_oracle_failed",
            )
        context.check(
            VerificationCheck.TASK_ORACLE,
            EvalStatus.PASS,
            "delayed 521 and non-hard-coded 384 guide oracles pass",
            artifact_ids=(oracle_artifact.artifact_id,),
        )

        context.stage = RepairStage.VERIFYING_RUNTIME
        runtime = simulator.replay(
            name="patched-replay",
            cwd=workspace.root,
            device=device,
            app_path=build.app_path,
            spec=replay,
            expectation=replay.patched,
            output_dir=context.output_dir / "patched" / "runtime",
        )
        after_runtime = self._artifact(
            context,
            artifact_id="after-runtime-result",
            kind=ProofArtifactKind.RUNTIME_RESULT,
            path=runtime.result_artifact,
            media_type="application/json",
        )
        after_screenshot = self._artifact(
            context,
            artifact_id="after-screenshot",
            kind=ProofArtifactKind.AFTER_SCREENSHOT,
            path=runtime.screenshot,
            media_type="image/png",
        )
        if not runtime.passed:
            context.check(
                VerificationCheck.SIMULATOR_REPLAY,
                EvalStatus.FAIL,
                "patched runtime does not satisfy the geometry oracle",
                artifact_ids=(after_runtime.artifact_id, after_screenshot.artifact_id),
            )
            self._fail(
                "patched runtime oracle failed: " + "; ".join(runtime.failures),
                status=RepairStatus.RUNTIME_VERIFICATION_FAILED,
                stage=RepairStage.VERIFYING_RUNTIME,
                layer=ErrorLayer.EVALUATOR,
                code="runtime_oracle_failed",
            )
        context.check(
            VerificationCheck.SIMULATOR_REPLAY,
            EvalStatus.PASS,
            "simulator reports one notification, guide 521, panel 521 and mismatch 0",
            artifact_ids=(after_runtime.artifact_id, after_screenshot.artifact_id),
        )
        context.check(
            VerificationCheck.REGRESSION,
            EvalStatus.PASS,
            "existing tests and the variable-height task oracle pass together",
            artifact_ids=(public_artifact.artifact_id, oracle_artifact.artifact_id),
        )
        context.event("verification.completed", mismatch=runtime.record.mismatch)

    def _test_target(
        self,
        *,
        xcode: NativeXcodeAdapter,
        context: _Context,
        workspace: MaterializedWorkspace,
        build: XcodeBuildOutcome,
        target: str,
        name: str,
        output_dir: Path,
        device: SimulatorDevice,
    ) -> XcodeTestOutcome:
        return xcode.test(
            name=name,
            workspace=workspace.root,
            scheme=context.task.verification.build_scheme,
            test_target=target,
            device_id=device.udid,
            derived_data=build.derived_data,
            output_dir=output_dir,
            timeout=300,
        )

    def _verify_host_environment(
        self,
        context: _Context,
        *,
        xcode: NativeXcodeAdapter,
        workspace: MaterializedWorkspace,
    ) -> None:
        actual_xcode = xcode.version(cwd=workspace.root)
        expected_xcode = context.task.environment.xcode_version
        expected_version = expected_xcode.split()[0]
        expected_build = expected_xcode.partition("(")[2].rstrip(")")
        if expected_version not in actual_xcode or (
            expected_build and expected_build not in actual_xcode
        ):
            self._fail(
                f"Xcode mismatch: expected {expected_xcode}, observed {actual_xcode!r}",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.POLICY,
                code="xcode_environment_mismatch",
            )
        context.versions["xcode"] = actual_xcode.replace("\n", " / ")

        macos_version = context.runner.run(
            "macos-version",
            ("/usr/bin/sw_vers", "-productVersion"),
            cwd=workspace.root,
            timeout=30,
        ).stdout.strip()
        macos_build = context.runner.run(
            "macos-build-version",
            ("/usr/bin/sw_vers", "-buildVersion"),
            cwd=workspace.root,
            timeout=30,
        ).stdout.strip()
        expected_macos = context.task.environment.macos_version
        if macos_version not in expected_macos or macos_build not in expected_macos:
            self._fail(
                f"macOS mismatch: expected {expected_macos}, observed {macos_version} ({macos_build})",
                status=RepairStatus.POLICY_BLOCKED,
                stage=RepairStage.PREPARED,
                layer=ErrorLayer.POLICY,
                code="macos_environment_mismatch",
            )
        context.versions["macos"] = f"{macos_version} ({macos_build})"

    def _artifact(
        self,
        context: _Context,
        *,
        artifact_id: str,
        kind: ProofArtifactKind,
        path: Path,
        media_type: str | None = None,
    ) -> Artifact:
        resolved = path.resolve()
        if not resolved.is_relative_to(context.output_dir):
            raise ValueError("proof artifact escapes the run output")
        content_hash = sha256_path(resolved)
        relative = resolved.relative_to(context.output_dir).as_posix()
        artifact = Artifact(
            artifact_id=artifact_id,
            artifact_type=kind,
            uri=relative,
            content_hash=content_hash,
        )
        context.artifacts.append(artifact)
        context.proof_artifacts.append(
            ProofArtifact(
                artifact_id=artifact_id,
                kind=kind,
                relative_path=relative,
                content_hash=content_hash,
                size_bytes=path_size(resolved),
                media_type=media_type,
            )
        )
        return artifact

    def _optional_artifact(
        self,
        context: _Context,
        *,
        artifact_id: str,
        kind: ProofArtifactKind,
        path: Path,
    ) -> Artifact | None:
        if not path.exists():
            return None
        return self._artifact(
            context,
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            media_type="application/vnd.apple.xcresult",
        )

    def _write_trace(self, context: _Context) -> None:
        if any(item.artifact_id == "agent-trace" for item in context.artifacts):
            return
        path = context.output_dir / "proof" / "agent-trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(event, ensure_ascii=False, sort_keys=True) for event in context.events]
        for receipt in context.runner.receipts:
            lines.append(
                json.dumps(
                    {
                        "event": "command.completed",
                        "receipt": receipt.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._artifact(
            context,
            artifact_id="agent-trace",
            kind=ProofArtifactKind.AGENT_TRACE,
            path=path,
            media_type="application/x-ndjson",
        )

    def _result(
        self,
        context: _Context,
        *,
        status: RepairStatus,
        stage: RepairStage,
        termination_reason: str,
        errors: tuple[AgentError, ...] = (),
    ) -> RepairResult:
        self._write_trace(context)
        return RepairResult(
            task_id=context.task.task_id,
            run_id=context.run_id,
            status=status,
            stage=stage,
            source_commit=context.task.repository.base_commit,
            patch=context.patch,
            approval_receipt_id=context.approval_receipt_id,
            checks=tuple(context.checks),
            evidence=tuple(context.evidence),
            artifacts=tuple(context.artifacts),
            errors=errors,
            metrics=RunMetrics(
                model_calls=0,
                tool_calls=len(context.runner.receipts),
                steps=len(context.events),
                elapsed_ms=(monotonic() - context.started) * 1000,
            ),
            termination_reason=termination_reason,
            runtime="deterministic-xcodefix-v1",
            versions=context.versions,
        )

    @staticmethod
    def _fail(
        message: str,
        *,
        status: RepairStatus,
        stage: RepairStage,
        layer: ErrorLayer,
        code: str,
    ) -> Never:
        raise PipelineFailure(
            message,
            status=status,
            stage=stage,
            layer=layer,
            code=code,
        )
