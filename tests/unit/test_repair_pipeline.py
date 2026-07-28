from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import agent_lab.repair.pipeline as pipeline_module
from agent_lab.repair.command import CommandOutcome, CommandReceipt
from agent_lab.repair.domain.replay import RuntimeRecord
from agent_lab.repair.domain.result import RepairStatus
from agent_lab.repair.pipeline import DeterministicRepairPipeline
from agent_lab.repair.simulator import SimulatorDevice, SimulatorReplayOutcome
from agent_lab.repair.task_loader import load_repair_task
from agent_lab.repair.xcode import XcodeBuildOutcome, XcodeTestOutcome


def command(name: str, *, exit_code: int = 0) -> CommandOutcome:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    return CommandOutcome(
        receipt=CommandReceipt(
            name=name,
            argv=(name,),
            cwd="$RUN/workspace",
            started_at=now,
            finished_at=now,
            elapsed_ms=1,
            exit_code=exit_code,
            stdout_path=f"$RUN/receipts/{name}.stdout.log",
            stderr_path=f"$RUN/receipts/{name}.stderr.log",
            stdout_hash=f"sha256:{'0' * 64}",
            stderr_hash=f"sha256:{'0' * 64}",
        ),
        stdout="",
        stderr="",
    )


class FakeXcodeAdapter:
    def __init__(self, *, runner: object) -> None:
        self.runner = runner

    def version(self, *, cwd: Path) -> str:
        return "Xcode 26.0\nBuild version 17A324"

    def build(
        self,
        *,
        name: str,
        workspace: Path,
        scheme: str,
        device_id: str,
        output_dir: Path,
        timeout: float,
    ) -> XcodeBuildOutcome:
        del device_id, timeout
        result_bundle = output_dir / "build.xcresult"
        result_bundle.mkdir(parents=True)
        (result_bundle / "result.json").write_text('{"result":"Passed"}')
        derived_data = output_dir / "DerivedData"
        app_path = derived_data / "Build/Products/Debug-iphonesimulator" / f"{scheme}.app"
        app_path.mkdir(parents=True)
        (app_path / scheme).write_text("fake app")
        return XcodeBuildOutcome(
            command=command(name),
            result_bundle=result_bundle,
            derived_data=derived_data,
            app_path=app_path,
        )

    def test(
        self,
        *,
        name: str,
        workspace: Path,
        scheme: str,
        test_target: str,
        device_id: str,
        derived_data: Path,
        output_dir: Path,
        timeout: float,
    ) -> XcodeTestOutcome:
        del scheme, device_id, derived_data, timeout
        source = (workspace / "KeyboardFixture" / "KeyboardHeightResolver.swift").read_text()
        is_broken = "XCODEFIX_BUG" in source
        is_gold = "return max(0, guideOverlap)" in source
        passed = True if test_target == "KeyboardFixtureTests" else is_gold and not is_broken
        result_bundle = output_dir / f"{test_target}.xcresult"
        result_bundle.mkdir(parents=True)
        summary = {
            "result": "Passed" if passed else "Failed",
            "failedTests": 0 if passed else 1,
            "passedTests": 2 if passed else 1,
        }
        (result_bundle / "summary.json").write_text(json.dumps(summary))
        return XcodeTestOutcome(
            command=command(name, exit_code=0 if passed else 65),
            result_bundle=result_bundle,
            summary=summary,
        )


class FakeSimulatorAdapter:
    def __init__(self, *, runner: object) -> None:
        self.runner = runner

    def resolve_device(
        self,
        *,
        cwd: Path,
        device_name: str,
        runtime_version: str,
        requested_udid: str | None = None,
    ) -> SimulatorDevice:
        del cwd, runtime_version
        return SimulatorDevice(
            udid=requested_udid or "FAKE-DEVICE",
            name=device_name,
            state="Booted",
            runtime_identifier="com.apple.CoreSimulator.SimRuntime.iOS-26-0",
            runtime_version="26.0.1",
            runtime_build="23A8464",
        )

    def replay(
        self,
        *,
        name: str,
        cwd: Path,
        device: SimulatorDevice,
        app_path: Path,
        spec: object,
        expectation: object,
        output_dir: Path,
    ) -> SimulatorReplayOutcome:
        del name, cwd, device, app_path, spec
        output_dir.mkdir(parents=True)
        values = expectation.model_dump()
        record = RuntimeRecord.model_validate(
            {
                "phase": "complete",
                "outcome": values["outcome"],
                "notificationCount": values["notification_count"],
                "notificationHeight": values["notification_height"],
                "guideMinY": 353,
                "guideOverlap": values["guide_overlap"],
                "panelHeight": values["panel_height"],
                "mismatch": values["mismatch"],
            }
        )
        result = output_dir / "runtime-result.json"
        result.write_text(record.model_dump_json(by_alias=True))
        screenshot = output_dir / "screenshot.png"
        screenshot.write_bytes(b"fake png")
        return SimulatorReplayOutcome(
            record=record,
            failures=(),
            result_artifact=result,
            screenshot=screenshot,
        )


@pytest.fixture
def fake_native_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "NativeXcodeAdapter", FakeXcodeAdapter)
    monkeypatch.setattr(pipeline_module, "NativeSimulatorAdapter", FakeSimulatorAdapter)

    def verified_environment(_: object, context: object, **__: object) -> None:
        context.versions.update(
            {
                "xcode": "Xcode 26.0 / Build version 17A324",
                "macos": "26.5.2 (25F84)",
            }
        )

    monkeypatch.setattr(
        DeterministicRepairPipeline,
        "_verify_host_environment",
        verified_environment,
    )


def test_fake_port_pipeline_delivers_gold_passport(
    root: Path, tmp_path: Path, fake_native_ports: None
) -> None:
    del fake_native_ports
    task = load_repair_task(root, "keyboard-layout-001")
    outcome = DeterministicRepairPipeline(project_root=root).run(
        task=task,
        patch_path=root / task.metadata["gold_patch"],
        output_dir=tmp_path / "gold",
        approval_actor="owner",
    )
    assert outcome.result.status is RepairStatus.DELIVERED
    assert outcome.passport is not None
    assert (outcome.output_dir / "patch-passport.json").is_file()
    assert {item.status for item in outcome.result.checks} == {"PASS"}


def test_fake_port_pipeline_pauses_before_write_without_approval(
    root: Path, tmp_path: Path, fake_native_ports: None
) -> None:
    del fake_native_ports
    task = load_repair_task(root, "keyboard-layout-001")
    outcome = DeterministicRepairPipeline(project_root=root).run(
        task=task,
        patch_path=root / task.metadata["gold_patch"],
        output_dir=tmp_path / "approval",
        approval_actor=None,
    )
    assert outcome.result.status is RepairStatus.APPROVAL_REQUIRED
    assert outcome.passport is None
    assert not (outcome.output_dir / "proof" / "patch.diff").exists()
    status = command_output(outcome.output_dir / "workspace", "git", "status", "--porcelain")
    assert status == ""


def test_fake_port_pipeline_rejects_hardcoded_negative_patch(
    root: Path, tmp_path: Path, fake_native_ports: None
) -> None:
    del fake_native_ports
    task = load_repair_task(root, "keyboard-layout-001")
    outcome = DeterministicRepairPipeline(project_root=root).run(
        task=task,
        patch_path=root / task.metadata["negative_patch"],
        output_dir=tmp_path / "negative",
        approval_actor="owner",
    )
    assert outcome.result.status is RepairStatus.TEST_FAILED
    assert outcome.passport is None
    assert outcome.result.checks[-1].check == "task_oracle"
    assert outcome.result.checks[-1].status == "FAIL"


def test_pipeline_blocks_oversized_candidate_before_materialization(
    root: Path, tmp_path: Path
) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    patch = tmp_path / "oversized.patch"
    patch.write_bytes(b"x" * (task.verification.max_patch_bytes + 1))
    outcome = DeterministicRepairPipeline(project_root=root).run(
        task=task,
        patch_path=patch,
        output_dir=tmp_path / "oversized",
        approval_actor="owner",
    )
    assert outcome.result.status is RepairStatus.POLICY_BLOCKED
    assert outcome.result.errors[0].code == "patch_budget_exceeded"


def command_output(cwd: Path, *argv: str) -> str:
    import subprocess

    return subprocess.run(
        list(argv), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()
