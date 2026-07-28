"""Native Xcode build and XCTest adapter with structured result bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_lab.repair.artifacts import write_json
from agent_lab.repair.command import CommandOutcome, CommandRunner


@dataclass(frozen=True)
class XcodeBuildOutcome:
    command: CommandOutcome
    result_bundle: Path
    derived_data: Path
    app_path: Path

    @property
    def passed(self) -> bool:
        return self.command.receipt.exit_code == 0 and self.result_bundle.is_dir()


@dataclass(frozen=True)
class XcodeTestOutcome:
    command: CommandOutcome
    result_bundle: Path
    summary: dict[str, Any] | None

    @property
    def passed(self) -> bool:
        return (
            self.command.receipt.exit_code == 0
            and self.summary is not None
            and self.summary.get("result") == "Passed"
            and self.summary.get("failedTests") == 0
        )


class NativeXcodeAdapter:
    def __init__(
        self,
        *,
        runner: CommandRunner,
        xcodebuild: str = "xcodebuild",
        xcrun: str = "xcrun",
    ) -> None:
        self.runner = runner
        self.xcodebuild = xcodebuild
        self.xcrun = xcrun

    def version(self, *, cwd: Path) -> str:
        return self.runner.run(
            "xcode-version",
            (self.xcodebuild, "-version"),
            cwd=cwd,
            timeout=30,
        ).stdout.strip()

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
        output_dir.mkdir(parents=True, exist_ok=True)
        project = workspace / f"{scheme}.xcodeproj"
        derived_data = output_dir / "DerivedData"
        result_bundle = output_dir / "build.xcresult"
        command = self.runner.run(
            name,
            (
                self.xcodebuild,
                "-project",
                str(project),
                "-scheme",
                scheme,
                "-configuration",
                "Debug",
                "-destination",
                f"platform=iOS Simulator,id={device_id}",
                "-derivedDataPath",
                str(derived_data),
                "-resultBundlePath",
                str(result_bundle),
                "CODE_SIGNING_ALLOWED=NO",
                "build",
                "-quiet",
            ),
            cwd=workspace,
            timeout=timeout,
            allowed_exit_codes=None,
        )
        return XcodeBuildOutcome(
            command=command,
            result_bundle=result_bundle,
            derived_data=derived_data,
            app_path=derived_data / "Build/Products/Debug-iphonesimulator" / f"{scheme}.app",
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
        output_dir.mkdir(parents=True, exist_ok=True)
        project = workspace / f"{scheme}.xcodeproj"
        result_bundle = output_dir / f"{test_target}.xcresult"
        command = self.runner.run(
            name,
            (
                self.xcodebuild,
                "-project",
                str(project),
                "-scheme",
                scheme,
                "-configuration",
                "Debug",
                "-destination",
                f"platform=iOS Simulator,id={device_id}",
                "-derivedDataPath",
                str(derived_data),
                "-resultBundlePath",
                str(result_bundle),
                f"-only-testing:{test_target}",
                "CODE_SIGNING_ALLOWED=NO",
                "test",
                "-quiet",
            ),
            cwd=workspace,
            timeout=timeout,
            allowed_exit_codes=None,
        )
        summary = self._summary(
            name=f"{name}-summary",
            workspace=workspace,
            result_bundle=result_bundle,
            output_path=output_dir / f"{test_target}-summary.json",
        )
        return XcodeTestOutcome(command=command, result_bundle=result_bundle, summary=summary)

    def _summary(
        self,
        *,
        name: str,
        workspace: Path,
        result_bundle: Path,
        output_path: Path,
    ) -> dict[str, Any] | None:
        if not result_bundle.is_dir():
            return None
        outcome = self.runner.run(
            name,
            (
                self.xcrun,
                "xcresulttool",
                "get",
                "test-results",
                "summary",
                "--path",
                str(result_bundle),
                "--compact",
            ),
            cwd=workspace,
            timeout=60,
            allowed_exit_codes=None,
        )
        if outcome.receipt.exit_code != 0:
            return None
        payload = json.loads(outcome.stdout)
        if not isinstance(payload, dict):
            raise ValueError("xcresult summary must be a JSON object")
        write_json(output_path, payload)
        return payload
