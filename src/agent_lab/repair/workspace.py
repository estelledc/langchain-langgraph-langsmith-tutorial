"""Materialize and mutate one isolated, reproducible Git workspace."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from agent_lab.repair.artifacts import sha256_bytes
from agent_lab.repair.command import CommandRunner
from agent_lab.repair.domain.approval import ApprovalAction, ApprovalReceipt
from agent_lab.repair.domain.task import (
    RepairTask,
    RepositoryKind,
    scope_contains,
    validate_relative_path,
)


class SourceSnapshotMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchInspection:
    payload: bytes
    payload_hash: str
    changed_files: tuple[str, ...]
    additions: int
    deletions: int


class MaterializedWorkspace:
    def __init__(
        self,
        *,
        root: Path,
        task: RepairTask,
        runner: CommandRunner,
    ) -> None:
        self.root = root.resolve()
        self.task = task
        self.runner = runner

    def inspect_patch(self, patch_path: Path) -> PatchInspection:
        resolved = patch_path.resolve()
        if not resolved.is_relative_to(self.runner.allowed_root):
            raise ValueError("candidate patch must be copied into the run root")
        payload = resolved.read_bytes()
        outcome = self.runner.run(
            "git-apply-numstat",
            ("git", "apply", "--numstat", str(resolved)),
            cwd=self.root,
            timeout=30,
        )
        changed: list[str] = []
        additions = 0
        deletions = 0
        for line in outcome.stdout.splitlines():
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"unexpected git apply --numstat output: {line!r}")
            if not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError("binary patches are not supported by the v1 repair workspace")
            additions += int(parts[0])
            deletions += int(parts[1])
            changed.append(validate_relative_path(parts[2], label="patch path"))
        if not changed or len(set(changed)) != len(changed):
            raise ValueError("patch must change at least one unique path")
        self.assert_scope(tuple(changed))
        return PatchInspection(
            payload=payload,
            payload_hash=sha256_bytes(payload),
            changed_files=tuple(sorted(changed)),
            additions=additions,
            deletions=deletions,
        )

    def apply_patch(
        self,
        patch_path: Path,
        *,
        inspection: PatchInspection,
        approval: ApprovalReceipt,
    ) -> ApprovalReceipt:
        observed = self.inspect_patch(patch_path)
        if observed != inspection:
            raise ValueError("patch changed after inspection")
        approval.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=inspection.payload,
            paths=inspection.changed_files,
        )
        self.runner.run(
            "git-apply-check",
            ("git", "apply", "--check", "--whitespace=error-all", str(patch_path.resolve())),
            cwd=self.root,
            timeout=30,
        )
        self.runner.run(
            "git-apply",
            ("git", "apply", "--whitespace=error-all", str(patch_path.resolve())),
            cwd=self.root,
            timeout=30,
        )
        changed_files = self.changed_files()
        if changed_files != inspection.changed_files:
            raise ValueError("applied diff changed a different path set")
        self.assert_scope(changed_files)
        return approval.consume()

    def changed_files(self) -> tuple[str, ...]:
        outcome = self.runner.run(
            "git-diff-names",
            ("git", "diff", "--name-only", "--diff-filter=ACMRTUXB"),
            cwd=self.root,
            timeout=30,
        )
        return tuple(sorted(item for item in outcome.stdout.splitlines() if item))

    def write_diff(self, destination: Path) -> bytes:
        outcome = self.runner.run(
            "git-diff-binary",
            ("git", "diff", "--binary", "--no-ext-diff"),
            cwd=self.root,
            timeout=30,
        )
        payload = outcome.stdout.encode("utf-8")
        if not payload:
            raise ValueError("applied patch produced an empty diff")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return payload

    def assert_scope(self, paths: tuple[str, ...]) -> None:
        if len(paths) > self.task.verification.changed_file_limit:
            raise ValueError("patch exceeds changed-file limit")
        for path in paths:
            if not any(
                scope_contains(pattern, path) for pattern in self.task.permissions.writable_paths
            ):
                raise ValueError(f"patch path is outside writable scope: {path}")
            if any(
                scope_contains(pattern, path) for pattern in self.task.permissions.forbidden_paths
            ):
                raise ValueError(f"patch path is forbidden: {path}")


def materialize_embedded_workspace(
    *,
    project_root: Path,
    task: RepairTask,
    destination: Path,
    runner: CommandRunner,
) -> MaterializedWorkspace:
    if task.repository.kind is not RepositoryKind.EMBEDDED_FIXTURE:
        raise ValueError("only embedded fixtures are supported by this materializer")
    if task.repository.fixture_path is None:
        raise ValueError("embedded fixture has no fixture_path")
    source = (project_root / task.repository.fixture_path).resolve()
    if not source.is_relative_to(project_root.resolve()) or not source.is_dir():
        raise ValueError("fixture source is outside the project root or missing")
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("embedded fixtures must not contain symlinks")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)

    workspace = MaterializedWorkspace(root=destination, task=task, runner=runner)
    commands = (
        ("git-init", ("git", "init", "-q", "-b", "main")),
        ("git-config-name", ("git", "config", "user.name", "XcodeFixBench")),
        (
            "git-config-email",
            ("git", "config", "user.email", "xcodefixbench@example.invalid"),
        ),
        ("git-config-autocrlf", ("git", "config", "core.autocrlf", "false")),
        ("git-add-fixture", ("git", "add", "--all")),
    )
    for name, argv in commands:
        runner.run(name, argv, cwd=destination, timeout=30)
    runner.run(
        "git-commit-fixture",
        (
            "git",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "Seed keyboard-layout-001 broken fixture",
        ),
        cwd=destination,
        timeout=30,
        env_overrides={
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        },
    )
    commit = runner.run(
        "git-rev-parse",
        ("git", "rev-parse", "HEAD"),
        cwd=destination,
        timeout=30,
    ).stdout.strip()
    if commit != task.repository.base_commit:
        raise SourceSnapshotMismatch(
            f"fixture commit {commit} != task base {task.repository.base_commit}"
        )
    status = runner.run(
        "git-status-fixture",
        ("git", "status", "--porcelain"),
        cwd=destination,
        timeout=30,
    ).stdout
    if status:
        raise SourceSnapshotMismatch("materialized fixture is dirty before repair")
    return workspace
