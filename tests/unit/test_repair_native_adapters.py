from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_lab.repair.command import CommandOutcome, CommandReceipt
from agent_lab.repair.simulator import NativeSimulatorAdapter, SimulatorDevice
from agent_lab.repair.task_loader import load_repair_task, load_replay_spec
from agent_lab.repair.xcode import NativeXcodeAdapter


def outcome(
    name: str, argv: tuple[str, ...], *, exit_code: int = 0, stdout: str = ""
) -> CommandOutcome:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    return CommandOutcome(
        receipt=CommandReceipt(
            name=name,
            argv=argv,
            cwd="$RUN",
            started_at=now,
            finished_at=now,
            elapsed_ms=1,
            exit_code=exit_code,
            stdout_path="$RUN/stdout.log",
            stderr_path="$RUN/stderr.log",
            stdout_hash=f"sha256:{'0' * 64}",
            stderr_hash=f"sha256:{'0' * 64}",
        ),
        stdout=stdout,
        stderr="failure" if exit_code else "",
    )


class XcodeRecordingRunner:
    def __init__(self, root: Path, *, test_passed: bool = True) -> None:
        self.allowed_root = root
        self.receipts: list[CommandReceipt] = []
        self.test_passed = test_passed
        self.create_result_bundle = True

    def run(
        self,
        name: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: float,
        allowed_exit_codes: object = None,
        env_overrides: object = None,
    ) -> CommandOutcome:
        del cwd, timeout, allowed_exit_codes, env_overrides
        stdout = ""
        exit_code = 0
        if argv[1:] == ("-version",):
            stdout = "Xcode 26.0\nBuild version 17A324\n"
        elif "-resultBundlePath" in argv:
            result = Path(argv[argv.index("-resultBundlePath") + 1])
            if self.create_result_bundle:
                result.mkdir(parents=True)
                (result / "marker").write_text("result")
            if "test" in argv:
                exit_code = 0 if self.test_passed else 65
            else:
                derived = Path(argv[argv.index("-derivedDataPath") + 1])
                scheme = argv[argv.index("-scheme") + 1]
                app = derived / "Build/Products/Debug-iphonesimulator" / f"{scheme}.app"
                app.mkdir(parents=True)
        elif "xcresulttool" in argv:
            stdout = json.dumps(
                {
                    "result": "Passed" if self.test_passed else "Failed",
                    "failedTests": 0 if self.test_passed else 1,
                }
            )
        result = outcome(name, argv, exit_code=exit_code, stdout=stdout)
        self.receipts.append(result.receipt)
        return result


def test_native_xcode_adapter_builds_tests_and_parses_summary(tmp_path: Path) -> None:
    runner = XcodeRecordingRunner(tmp_path)
    adapter = NativeXcodeAdapter(runner=runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert "Build version 17A324" in adapter.version(cwd=workspace)

    build = adapter.build(
        name="build",
        workspace=workspace,
        scheme="KeyboardFixture",
        device_id="DEVICE",
        output_dir=tmp_path / "build",
        timeout=30,
    )
    assert build.passed
    assert build.app_path.is_dir()

    test = adapter.test(
        name="tests",
        workspace=workspace,
        scheme="KeyboardFixture",
        test_target="KeyboardFixtureTests",
        device_id="DEVICE",
        derived_data=build.derived_data,
        output_dir=tmp_path / "tests",
        timeout=30,
    )
    assert test.passed
    assert test.summary == {"result": "Passed", "failedTests": 0}


def test_native_xcode_adapter_preserves_failed_test_and_missing_bundle(tmp_path: Path) -> None:
    runner = XcodeRecordingRunner(tmp_path, test_passed=False)
    adapter = NativeXcodeAdapter(runner=runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test = adapter.test(
        name="tests",
        workspace=workspace,
        scheme="KeyboardFixture",
        test_target="KeyboardTaskOracleTests",
        device_id="DEVICE",
        derived_data=tmp_path / "derived",
        output_dir=tmp_path / "tests",
        timeout=30,
    )
    assert not test.passed
    assert test.command.receipt.exit_code == 65

    runner.create_result_bundle = False
    missing = adapter.build(
        name="missing-build",
        workspace=workspace,
        scheme="KeyboardFixture",
        device_id="DEVICE",
        output_dir=tmp_path / "missing",
        timeout=30,
    )
    assert not missing.passed


class SimulatorRecordingRunner:
    def __init__(self, root: Path, container: Path) -> None:
        self.allowed_root = root
        self.receipts: list[CommandReceipt] = []
        self.container = container
        self.install_exit_code = 0

    def run(
        self,
        name: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: float,
        allowed_exit_codes: object = None,
        env_overrides: object = None,
    ) -> CommandOutcome:
        del cwd, timeout, allowed_exit_codes, env_overrides
        stdout = ""
        exit_code = 0
        if argv[2:6] == ("list", "runtimes", "available", "-j"):
            stdout = json.dumps(
                {
                    "runtimes": [
                        {
                            "isAvailable": True,
                            "version": "26.0.1",
                            "buildversion": "23A8464",
                            "identifier": "com.apple.CoreSimulator.SimRuntime.iOS-26-0",
                        }
                    ]
                }
            )
        elif argv[2:6] == ("list", "devices", "available", "-j"):
            stdout = json.dumps(
                {
                    "devices": {
                        "com.apple.CoreSimulator.SimRuntime.iOS-26-0": [
                            {"name": "iPhone 17 Pro", "udid": "BOOTED", "state": "Booted"},
                            {"name": "iPhone 17 Pro", "udid": "SHUTDOWN", "state": "Shutdown"},
                        ]
                    }
                }
            )
        elif "get_app_container" in argv:
            stdout = f"{self.container}\n"
        elif "screenshot" in argv:
            Path(argv[-1]).write_bytes(b"png")
        elif "install" in argv:
            exit_code = self.install_exit_code
        result = outcome(name, argv, exit_code=exit_code, stdout=stdout)
        self.receipts.append(result.receipt)
        return result


def test_native_simulator_resolves_boots_and_replays_structured_result(
    root: Path, tmp_path: Path
) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    replay = load_replay_spec(root, task)
    container = tmp_path / "container"
    result = container / replay.result_relative_path
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "phase": "complete",
                "outcome": "bug_reproduced",
                "notificationCount": 1,
                "notificationHeight": 325,
                "guideMinY": 353,
                "guideOverlap": 521,
                "panelHeight": 325,
                "mismatch": 196,
            }
        )
    )
    runner = SimulatorRecordingRunner(tmp_path, container)
    adapter = NativeSimulatorAdapter(runner=runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    booted = adapter.resolve_device(
        cwd=workspace,
        device_name="iPhone 17 Pro",
        runtime_version="26.0.1 (23A8464)",
    )
    assert booted.udid == "BOOTED"
    assert booted.runtime_version == "26.0.1"
    assert booted.runtime_build == "23A8464"
    shutdown = adapter.resolve_device(
        cwd=workspace,
        device_name="iPhone 17 Pro",
        runtime_version="26.0.1 (23A8464)",
        requested_udid="SHUTDOWN",
    )
    assert shutdown.state == "Booted"

    app = tmp_path / "KeyboardFixture.app"
    app.mkdir()
    observed = adapter.replay(
        name="baseline",
        cwd=workspace,
        device=booted,
        app_path=app,
        spec=replay,
        expectation=replay.baseline,
        output_dir=tmp_path / "replay",
    )
    assert observed.passed
    assert observed.record.mismatch == 196
    assert observed.screenshot.is_file()


def test_native_simulator_rejects_missing_device_and_install_failure(
    root: Path, tmp_path: Path
) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    replay = load_replay_spec(root, task)
    container = tmp_path / "container"
    container.mkdir()
    runner = SimulatorRecordingRunner(tmp_path, container)
    adapter = NativeSimulatorAdapter(runner=runner)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="no simulator"):
        adapter.resolve_device(
            cwd=workspace,
            device_name="iPad that does not exist",
            runtime_version="26.0.1",
        )

    with pytest.raises(ValueError, match="exactly one available simulator runtime"):
        adapter.resolve_device(
            cwd=workspace,
            device_name="iPhone 17 Pro",
            runtime_version="26.0.1 (WRONG)",
        )

    runner.install_exit_code = 1
    app = tmp_path / "KeyboardFixture.app"
    app.mkdir()
    with pytest.raises(RuntimeError, match="install failed"):
        adapter.replay(
            name="failure",
            cwd=workspace,
            device=SimulatorDevice(
                udid="BOOTED",
                name="iPhone 17 Pro",
                state="Booted",
                runtime_identifier="com.apple.CoreSimulator.SimRuntime.iOS-26-0",
                runtime_version="26.0.1",
                runtime_build="23A8464",
            ),
            app_path=app,
            spec=replay,
            expectation=replay.baseline,
            output_dir=tmp_path / "failed-replay",
        )
