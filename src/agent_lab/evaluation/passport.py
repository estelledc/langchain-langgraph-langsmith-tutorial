"""Build a reproducible verification passport from local evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import Field

from agent_lab.domain.evaluation import SuiteReport
from agent_lab.domain.models import FrozenModel, utc_now


class VerificationPassport(FrozenModel):
    schema_version: str = "verification-passport-v1"
    generated_at: datetime = Field(default_factory=utc_now)
    source_commit: str
    worktree_dirty: bool
    environment_lock_hash: str
    python_requires: str = ">=3.11,<3.14"
    package_versions: dict[str, str]
    dataset_hashes: dict[str, str]
    suite: str
    dataset_version: str
    test_status: str
    eval_status: str
    pass_rate: float
    unknown_rate: float
    evaluator_error_rate: float
    failed_cases: tuple[str, ...]
    unknowns: tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_passport(
    root: Path,
    report: SuiteReport,
    *,
    test_status: str,
    eval_status: str,
) -> VerificationPassport:
    dataset_paths = sorted((root / "datasets").glob("**/*.jsonl"))
    versions = {}
    for package in ("langchain", "langgraph", "langsmith", "pydantic"):
        versions[package] = importlib.metadata.version(package)
    return VerificationPassport(
        source_commit=_git(root, "rev-parse", "HEAD"),
        worktree_dirty=bool(_git(root, "status", "--porcelain")),
        environment_lock_hash=sha256_file(root / "uv.lock"),
        package_versions=versions,
        dataset_hashes={str(path.relative_to(root)): sha256_file(path) for path in dataset_paths},
        suite=report.suite,
        dataset_version=report.dataset_version,
        test_status=test_status,
        eval_status=eval_status,
        pass_rate=report.pass_rate,
        unknown_rate=report.unknown_rate,
        evaluator_error_rate=report.evaluator_error_rate,
        failed_cases=report.failed_cases,
        unknowns=(
            "live provider smoke 未在离线门禁中执行",
            "LangSmith online evaluation 未在本地门禁中执行",
        ),
    )


def write_passport(passport: VerificationPassport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(passport.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
