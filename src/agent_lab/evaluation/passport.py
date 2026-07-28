"""Build a verification passport only from completed local gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from agent_lab.domain.evaluation import SuiteReport
from agent_lab.domain.models import FrozenModel, utc_now

REQUIRED_QUALITY_GATES = (
    "ruff-format",
    "ruff-lint",
    "mypy",
    "pytest",
    "curriculum",
    "site-contract",
    "git-diff-check",
)
REQUIRED_SUITES = ("fast", "security", "contracts")


class SuiteEvidence(FrozenModel):
    suite: str
    dataset_version: str
    pass_rate: float = Field(ge=0, le=1)
    unknown_rate: float = Field(ge=0, le=1)
    evaluator_error_rate: float = Field(ge=0, le=1)
    runtime_error_rate: float = Field(ge=0, le=1)
    failed_cases: tuple[str, ...]


class VerificationPassport(FrozenModel):
    schema_version: Literal["verification-passport-v2"] = "verification-passport-v2"
    generated_at: datetime = Field(default_factory=utc_now)
    source_commit: str
    worktree_dirty: bool
    environment_lock_hash: str
    python_requires: str = ">=3.11,<3.14"
    python_version: str
    platform: str
    package_versions: dict[str, str]
    dataset_hashes: dict[str, str]
    suite_config_hashes: dict[str, str]
    suite_report_hashes: dict[str, str]
    gate_statuses: dict[str, Literal["PASS"]]
    suites: dict[str, SuiteEvidence]
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    unknowns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_complete_gate_evidence(self) -> Self:
        required_gates = set(REQUIRED_QUALITY_GATES) | {f"suite:{name}" for name in REQUIRED_SUITES}
        missing_gates = required_gates - set(self.gate_statuses)
        if missing_gates:
            raise ValueError(f"passport is missing required gates: {sorted(missing_gates)}")
        missing_suites = set(REQUIRED_SUITES) - set(self.suites)
        if missing_suites:
            raise ValueError(f"passport is missing required suites: {sorted(missing_suites)}")
        return self


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _suite_evidence(report: SuiteReport) -> SuiteEvidence:
    return SuiteEvidence(
        suite=report.suite,
        dataset_version=report.dataset_version,
        pass_rate=report.pass_rate,
        unknown_rate=report.unknown_rate,
        evaluator_error_rate=report.evaluator_error_rate,
        runtime_error_rate=report.runtime_error_rate,
        failed_cases=report.failed_cases,
    )


def build_passport(
    root: Path,
    reports: dict[str, SuiteReport],
    *,
    gate_statuses: dict[str, Literal["PASS"]],
    artifact_dir: Path | None = None,
) -> VerificationPassport:
    dataset_paths = sorted((root / "datasets").glob("**/*.jsonl"))
    suite_paths = sorted((root / "evals" / "suites").glob("*.yaml"))
    versions = {}
    for package in ("langchain", "langgraph", "langsmith", "pydantic"):
        versions[package] = importlib.metadata.version(package)
    artifacts = (
        sorted(
            path
            for path in artifact_dir.iterdir()
            if path.is_file() and path.name.endswith((".whl", ".tar.gz", ".zip"))
        )
        if artifact_dir is not None and artifact_dir.is_dir()
        else []
    )
    return VerificationPassport(
        source_commit=_git(root, "rev-parse", "HEAD"),
        worktree_dirty=bool(_git(root, "status", "--porcelain")),
        environment_lock_hash=sha256_file(root / "uv.lock"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_versions=versions,
        dataset_hashes={str(path.relative_to(root)): sha256_file(path) for path in dataset_paths},
        suite_config_hashes={
            str(path.relative_to(root)): sha256_file(path) for path in suite_paths
        },
        suite_report_hashes={
            name: sha256_text(report.model_dump_json()) for name, report in reports.items()
        },
        gate_statuses=gate_statuses,
        suites={name: _suite_evidence(report) for name, report in reports.items()},
        artifact_hashes={path.name: sha256_file(path) for path in artifacts},
        unknowns=(
            "live provider smoke 未在离线门禁中执行",
            "LangSmith online evaluation 未在本地门禁中执行",
            "Xcode/Simulator live canary 不属于跨平台离线门禁",
        ),
    )


def write_passport(passport: VerificationPassport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(passport.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
