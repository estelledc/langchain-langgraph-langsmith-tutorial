"""iOS Simulator replay adapter that saves structured runtime evidence."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from agent_lab.repair.command import CommandRunner
from agent_lab.repair.domain.replay import ReplaySpec, RuntimeExpectation, RuntimeRecord
from agent_lab.repair.domain.task import validate_relative_path


@dataclass(frozen=True)
class SimulatorDevice:
    udid: str
    name: str
    state: str
    runtime_identifier: str
    runtime_version: str
    runtime_build: str


@dataclass(frozen=True)
class SimulatorReplayOutcome:
    record: RuntimeRecord
    failures: tuple[str, ...]
    result_artifact: Path
    screenshot: Path

    @property
    def passed(self) -> bool:
        return not self.failures


class NativeSimulatorAdapter:
    def __init__(self, *, runner: CommandRunner, xcrun: str = "xcrun") -> None:
        self.runner = runner
        self.xcrun = xcrun

    def resolve_device(
        self,
        *,
        cwd: Path,
        device_name: str,
        runtime_version: str,
        requested_udid: str | None = None,
    ) -> SimulatorDevice:
        expected_version = runtime_version.split()[0]
        build_match = re.search(r"\(([^()]+)\)\s*$", runtime_version)
        expected_build = build_match.group(1) if build_match else None
        runtimes_outcome = self.runner.run(
            "simctl-list-runtimes",
            (self.xcrun, "simctl", "list", "runtimes", "available", "-j"),
            cwd=cwd,
            timeout=30,
        )
        runtimes_payload = json.loads(runtimes_outcome.stdout)
        runtimes = runtimes_payload.get("runtimes") if isinstance(runtimes_payload, dict) else None
        if not isinstance(runtimes, list):
            raise ValueError("simctl runtime list has an unexpected schema")
        matching_runtimes = [
            item
            for item in runtimes
            if isinstance(item, dict)
            and item.get("isAvailable") is True
            and item.get("version") == expected_version
            and (expected_build is None or item.get("buildversion") == expected_build)
            and isinstance(item.get("identifier"), str)
            and isinstance(item.get("buildversion"), str)
        ]
        if len(matching_runtimes) != 1:
            raise ValueError(
                "expected exactly one available simulator runtime matching "
                f"{runtime_version}, found {len(matching_runtimes)}"
            )
        runtime_record = matching_runtimes[0]
        runtime_identifier = str(runtime_record["identifier"])
        observed_build = str(runtime_record["buildversion"])

        outcome = self.runner.run(
            "simctl-list-devices",
            (self.xcrun, "simctl", "list", "devices", "available", "-j"),
            cwd=cwd,
            timeout=30,
        )
        payload = json.loads(outcome.stdout)
        devices = payload.get("devices") if isinstance(payload, dict) else None
        if not isinstance(devices, dict):
            raise ValueError("simctl device list has an unexpected schema")

        candidates: list[SimulatorDevice] = []
        for device_runtime_identifier, items in devices.items():
            if device_runtime_identifier != runtime_identifier:
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or item.get("name") != device_name:
                    continue
                udid = item.get("udid")
                state = item.get("state")
                if isinstance(udid, str) and isinstance(state, str):
                    candidates.append(
                        SimulatorDevice(
                            udid=udid,
                            name=device_name,
                            state=state,
                            runtime_identifier=device_runtime_identifier,
                            runtime_version=expected_version,
                            runtime_build=observed_build,
                        )
                    )
        if requested_udid is not None:
            candidates = [item for item in candidates if item.udid == requested_udid]
        if not candidates:
            raise ValueError("no simulator matches the task device and runtime")
        candidates.sort(key=lambda item: item.state != "Booted")
        selected = candidates[0]
        if selected.state != "Booted":
            boot = self.runner.run(
                "simctl-boot-device",
                (self.xcrun, "simctl", "boot", selected.udid),
                cwd=cwd,
                timeout=60,
                allowed_exit_codes=None,
            )
            if boot.receipt.exit_code != 0:
                raise RuntimeError(f"simulator boot failed: {boot.stderr.strip()}")
            self.runner.run(
                "simctl-boot-status",
                (self.xcrun, "simctl", "bootstatus", selected.udid, "-b"),
                cwd=cwd,
                timeout=120,
            )
            selected = SimulatorDevice(
                udid=selected.udid,
                name=selected.name,
                state="Booted",
                runtime_identifier=selected.runtime_identifier,
                runtime_version=selected.runtime_version,
                runtime_build=selected.runtime_build,
            )
        return selected

    def replay(
        self,
        *,
        name: str,
        cwd: Path,
        device: SimulatorDevice,
        app_path: Path,
        spec: ReplaySpec,
        expectation: RuntimeExpectation,
        output_dir: Path,
    ) -> SimulatorReplayOutcome:
        if not app_path.is_dir():
            raise FileNotFoundError(app_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.runner.run(
            f"{name}-terminate",
            (self.xcrun, "simctl", "terminate", device.udid, spec.bundle_id),
            cwd=cwd,
            timeout=30,
            allowed_exit_codes=None,
        )
        self.runner.run(
            f"{name}-uninstall",
            (self.xcrun, "simctl", "uninstall", device.udid, spec.bundle_id),
            cwd=cwd,
            timeout=30,
            allowed_exit_codes=None,
        )
        install = self.runner.run(
            f"{name}-install",
            (self.xcrun, "simctl", "install", device.udid, str(app_path)),
            cwd=cwd,
            timeout=60,
            allowed_exit_codes=None,
        )
        if install.receipt.exit_code != 0:
            raise RuntimeError(f"simulator install failed: {install.stderr.strip()}")
        launch = self.runner.run(
            f"{name}-launch",
            (
                self.xcrun,
                "simctl",
                "launch",
                "--terminate-running-process",
                device.udid,
                spec.bundle_id,
                *spec.launch_arguments,
            ),
            cwd=cwd,
            timeout=60,
            allowed_exit_codes=None,
        )
        if launch.receipt.exit_code != 0:
            raise RuntimeError(f"simulator launch failed: {launch.stderr.strip()}")
        container = self.runner.run(
            f"{name}-container",
            (self.xcrun, "simctl", "get_app_container", device.udid, spec.bundle_id, "data"),
            cwd=cwd,
            timeout=30,
        ).stdout.strip()
        container_path = Path(container).resolve()
        relative_result = validate_relative_path(
            spec.result_relative_path, label="runtime result path"
        )
        source_result = container_path / relative_result
        deadline = time.monotonic() + spec.timeout_seconds
        while time.monotonic() < deadline and not source_result.is_file():
            time.sleep(0.1)
        if not source_result.is_file():
            raise TimeoutError("runtime result did not appear before the replay deadline")
        if source_result.stat().st_size > 1_000_000:
            raise ValueError("runtime result exceeds the 1 MB evidence limit")

        result_artifact = output_dir / "runtime-result.json"
        shutil.copyfile(source_result, result_artifact)
        record = RuntimeRecord.model_validate_json(result_artifact.read_text(encoding="utf-8"))
        screenshot = output_dir / "screenshot.png"
        shot = self.runner.run(
            f"{name}-screenshot",
            (self.xcrun, "simctl", "io", device.udid, "screenshot", str(screenshot)),
            cwd=cwd,
            timeout=30,
            allowed_exit_codes=None,
        )
        if shot.receipt.exit_code != 0 or not screenshot.is_file():
            raise RuntimeError("simulator screenshot failed")
        return SimulatorReplayOutcome(
            record=record,
            failures=expectation.failures(record, tolerance=spec.tolerance),
            result_artifact=result_artifact,
            screenshot=screenshot,
        )
