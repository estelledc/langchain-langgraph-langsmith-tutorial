"""Non-shell command execution with phase-specific receipts."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from agent_lab.domain.models import FrozenModel, utc_now
from agent_lab.repair.artifacts import sha256_bytes, write_json

_RECEIPT_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ENV_ALLOWLIST = (
    "DEVELOPER_DIR",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "PATH",
    "TMPDIR",
    "USER",
)
_ENV_OVERRIDE_ALLOWLIST = {"GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"}


class CommandReceipt(FrozenModel):
    schema_version: str = "command-receipt-v1"
    command_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    argv: tuple[str, ...]
    cwd: str
    started_at: datetime
    finished_at: datetime
    elapsed_ms: float = Field(ge=0)
    exit_code: int
    timed_out: bool = False
    stdout_path: str
    stderr_path: str
    stdout_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    stderr_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class CommandOutcome:
    receipt: CommandReceipt
    stdout: str
    stderr: str


class CommandExecutionError(RuntimeError):
    def __init__(self, message: str, outcome: CommandOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class CommandRunner(Protocol):
    allowed_root: Path
    receipts: list[CommandReceipt]

    def run(
        self,
        name: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: float,
        allowed_exit_codes: frozenset[int] | None = frozenset({0}),
        env_overrides: Mapping[str, str] | None = None,
    ) -> CommandOutcome: ...


class SafeCommandRunner:
    def __init__(self, *, allowed_root: Path, receipt_dir: Path) -> None:
        self.allowed_root = allowed_root.resolve()
        self.receipt_dir = receipt_dir.resolve()
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        if not self.receipt_dir.is_relative_to(self.allowed_root):
            raise ValueError("receipt directory must be inside the allowed root")
        self.receipts: list[CommandReceipt] = []

    def run(
        self,
        name: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: float,
        allowed_exit_codes: frozenset[int] | None = frozenset({0}),
        env_overrides: Mapping[str, str] | None = None,
    ) -> CommandOutcome:
        if not _RECEIPT_NAME.fullmatch(name):
            raise ValueError("command receipt name must be lowercase kebab-case")
        if not argv or any(not item or "\x00" in item for item in argv):
            raise ValueError("argv must contain non-empty, NUL-free values")
        resolved_cwd = cwd.resolve()
        if not resolved_cwd.is_relative_to(self.allowed_root):
            raise ValueError("command cwd escapes the allowed root")
        if timeout <= 0:
            raise ValueError("command timeout must be positive")

        environment = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
        if env_overrides:
            unexpected = set(env_overrides) - _ENV_OVERRIDE_ALLOWLIST
            if unexpected:
                raise ValueError(f"unsupported environment overrides: {sorted(unexpected)}")
            environment.update(env_overrides)

        started_at = utc_now()
        started = monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                list(argv),
                cwd=resolved_cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

        finished_at = utc_now()
        elapsed_ms = (monotonic() - started) * 1000
        stdout_bytes = stdout.encode("utf-8")
        stderr_bytes = stderr.encode("utf-8")
        receipt_prefix = f"{len(self.receipts):03d}-{name}"
        stdout_path = self.receipt_dir / f"{receipt_prefix}.stdout.log"
        stderr_path = self.receipt_dir / f"{receipt_prefix}.stderr.log"
        stdout_path.write_bytes(stdout_bytes)
        stderr_path.write_bytes(stderr_bytes)

        receipt = CommandReceipt(
            name=name,
            argv=tuple(self._portable(item) for item in argv),
            cwd=self._portable(str(resolved_cwd)),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_ms=elapsed_ms,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout_path=self._portable(str(stdout_path)),
            stderr_path=self._portable(str(stderr_path)),
            stdout_hash=sha256_bytes(stdout_bytes),
            stderr_hash=sha256_bytes(stderr_bytes),
        )
        write_json(self.receipt_dir / f"{receipt_prefix}.receipt.json", receipt)
        self.receipts.append(receipt)
        outcome = CommandOutcome(receipt=receipt, stdout=stdout, stderr=stderr)

        if timed_out:
            raise CommandExecutionError(f"command timed out: {name}", outcome)
        if allowed_exit_codes is not None and exit_code not in allowed_exit_codes:
            raise CommandExecutionError(f"command failed with exit {exit_code}: {name}", outcome)
        return outcome

    def _portable(self, value: str) -> str:
        return value.replace(str(self.allowed_root), "$RUN")
