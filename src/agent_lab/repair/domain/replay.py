"""Structured runtime replay contract and deterministic oracle."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_lab.domain.models import FrozenModel


class RuntimeRecord(FrozenModel):
    phase: str
    outcome: str
    notification_count: int = Field(alias="notificationCount", ge=0)
    notification_height: float = Field(alias="notificationHeight", ge=0)
    guide_min_y: float = Field(alias="guideMinY")
    guide_overlap: float = Field(alias="guideOverlap", ge=0)
    panel_height: float = Field(alias="panelHeight", ge=0)
    mismatch: float = Field(ge=0)


class RuntimeExpectation(FrozenModel):
    outcome: str = Field(min_length=1)
    notification_count: int = Field(ge=0)
    notification_height: float = Field(ge=0)
    guide_overlap: float = Field(ge=0)
    panel_height: float = Field(ge=0)
    mismatch: float = Field(ge=0)

    def failures(self, record: RuntimeRecord, *, tolerance: float) -> tuple[str, ...]:
        failures: list[str] = []
        if record.phase != "complete":
            failures.append(f"phase={record.phase!r}, expected 'complete'")
        if record.outcome != self.outcome:
            failures.append(f"outcome={record.outcome!r}, expected {self.outcome!r}")
        if record.notification_count != self.notification_count:
            failures.append(
                f"notification_count={record.notification_count}, expected {self.notification_count}"
            )
        numeric = (
            ("notification_height", record.notification_height, self.notification_height),
            ("guide_overlap", record.guide_overlap, self.guide_overlap),
            ("panel_height", record.panel_height, self.panel_height),
            ("mismatch", record.mismatch, self.mismatch),
        )
        for name, observed, expected in numeric:
            if abs(observed - expected) > tolerance:
                failures.append(f"{name}={observed}, expected {expected}±{tolerance}")
        return tuple(failures)


class ReplaySpec(FrozenModel):
    schema_version: Literal["xcodefix-replay-v1"] = "xcodefix-replay-v1"
    bundle_id: str = Field(min_length=1)
    launch_arguments: tuple[str, ...]
    result_relative_path: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=120)
    baseline: RuntimeExpectation
    patched: RuntimeExpectation
    tolerance: float = Field(default=0.5, gt=0, le=10)
