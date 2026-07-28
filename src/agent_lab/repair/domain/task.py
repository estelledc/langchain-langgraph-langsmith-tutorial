"""Stable task contract for a reproducible iOS repair benchmark."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from agent_lab.domain.models import Budget, FrozenModel

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_TASK_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_relative_path(value: str, *, label: str) -> str:
    """Reject absolute, host-specific and traversal-capable task paths."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if "\x00" in normalized or "\\" in normalized:
        raise ValueError(f"{label} must be a portable POSIX path")
    if normalized.startswith(("/", "~")) or "://" in normalized:
        raise ValueError(f"{label} must be repository-relative")
    if "//" in normalized:
        raise ValueError(f"{label} must not contain empty path segments")

    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not contain traversal segments")
    return normalized


def validate_scope_pattern(value: str) -> str:
    """Accept only exact paths, a full-tree wildcard, or one explicit subtree."""

    normalized = validate_relative_path(value, label="path policy")
    wildcard_count = normalized.count("*")
    if (
        wildcard_count
        and normalized != "**"
        and not (wildcard_count == 2 and normalized.endswith("/**"))
    ):
        raise ValueError("path policy only supports exact paths, **, or a trailing /**")
    if any(character in normalized for character in ("?", "[", "]")):
        raise ValueError("path policy contains an unsupported wildcard")
    return normalized


def scope_contains(pattern: str, path: str) -> bool:
    """Match a validated v1 path policy without shell or platform glob semantics."""

    if pattern == "**":
        return True
    if pattern.endswith("/**"):
        prefix = pattern.removesuffix("/**")
        return path == prefix or path.startswith(f"{prefix}/")
    return path == pattern


def _normalize_unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(item.strip() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"{label} must not contain empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} values must be unique")
    return normalized


class TaskOrigin(StrEnum):
    SYNTHETIC_SEEDED = "synthetic-seeded"
    OPEN_SOURCE_HISTORY = "open-source-history"
    REAL_WORLD_ANONYMIZED = "real-world-anonymized"


class RepositoryKind(StrEnum):
    REMOTE = "remote"
    EMBEDDED_FIXTURE = "embedded-fixture"


class ProofArtifactKind(StrEnum):
    PATCH_DIFF = "patch.diff"
    APPROVAL_RECEIPT = "approval.receipt"
    BUILD_XCRESULT = "build.xcresult"
    TEST_XCRESULT = "test.xcresult"
    BEFORE_SCREENSHOT = "screenshot.before"
    AFTER_SCREENSHOT = "screenshot.after"
    ACCESSIBILITY_SNAPSHOT = "accessibility.snapshot"
    APPLICATION_LOG = "application.log"
    REPRODUCTION_VIDEO = "video.reproduction"
    VERIFICATION_VIDEO = "video.verification"
    GEOMETRY_SNAPSHOT = "geometry.snapshot"
    RUNTIME_RESULT = "runtime.result"
    AGENT_TRACE = "agent.trace"


class VerificationCheck(StrEnum):
    REPRODUCTION = "reproduction"
    ROOT_CAUSE_EVIDENCE = "root_cause_evidence"
    PATCH_SCOPE = "patch_scope"
    BUILD = "build"
    EXISTING_TESTS = "existing_tests"
    TASK_ORACLE = "task_oracle"
    SIMULATOR_REPLAY = "simulator_replay"
    FORBIDDEN_CHANGES = "forbidden_changes"
    REGRESSION = "regression"


CORE_REQUIRED_CHECKS = (
    VerificationCheck.REPRODUCTION,
    VerificationCheck.ROOT_CAUSE_EVIDENCE,
    VerificationCheck.PATCH_SCOPE,
    VerificationCheck.BUILD,
    VerificationCheck.EXISTING_TESTS,
    VerificationCheck.TASK_ORACLE,
    VerificationCheck.FORBIDDEN_CHANGES,
    VerificationCheck.REGRESSION,
)


class RepositorySpec(FrozenModel):
    kind: RepositoryKind = RepositoryKind.REMOTE
    url: str = Field(min_length=1, max_length=500)
    base_commit: str
    license_spdx: str = Field(min_length=1, max_length=100)
    fixture_path: str | None = None

    @field_validator("base_commit")
    @classmethod
    def validate_base_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _COMMIT_RE.fullmatch(normalized):
            raise ValueError("base_commit must be a full 40- or 64-character commit hash")
        return normalized

    @field_validator("fixture_path")
    @classmethod
    def validate_fixture_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_relative_path(value, label="fixture_path")

    @model_validator(mode="after")
    def validate_source_location(self) -> Self:
        parsed = urlsplit(self.url.strip())
        if self.kind is RepositoryKind.REMOTE:
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("remote repository url must be credential-free HTTPS")
            if parsed.query or parsed.fragment or parsed.path in {"", "/"}:
                raise ValueError(
                    "remote repository url must identify one repository without query or fragment"
                )
            if self.fixture_path is not None:
                raise ValueError("remote repositories cannot declare fixture_path")
        else:
            if (
                parsed.scheme != "embedded"
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("embedded fixture url must use embedded://<fixture-id>")
            if self.fixture_path is None:
                raise ValueError("embedded fixtures require fixture_path")
        return self


class IOSExecutionEnvironment(FrozenModel):
    macos_version: str = Field(min_length=1, max_length=100)
    xcode_version: str = Field(min_length=1, max_length=100)
    ios_runtime: str = Field(min_length=1, max_length=100)
    device: str = Field(min_length=1, max_length=100)
    swift_version: str | None = Field(default=None, max_length=100)


class ReproductionSpec(FrozenModel):
    scheme: str = Field(min_length=1, max_length=200)
    steps: tuple[str, ...] = Field(min_length=1, max_length=50)
    required_artifacts: tuple[ProofArtifactKind, ...] = Field(min_length=1)
    seed_state: str | None = Field(default=None, max_length=200)

    @field_validator("scheme")
    @classmethod
    def validate_scheme(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character in normalized for character in ("\n", "\r", "\x00")):
            raise ValueError("scheme must be a single non-empty value")
        return normalized

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_unique(value, label="reproduction steps")

    @field_validator("required_artifacts")
    @classmethod
    def validate_required_artifacts(
        cls, value: tuple[ProofArtifactKind, ...]
    ) -> tuple[ProofArtifactKind, ...]:
        if len(set(value)) != len(value):
            raise ValueError("required reproduction artifacts must be unique")
        return value


class PermissionSpec(FrozenModel):
    readable_paths: tuple[str, ...] = Field(min_length=1)
    writable_paths: tuple[str, ...] = Field(min_length=1)
    forbidden_paths: tuple[str, ...] = Field(min_length=1)
    allowed_capabilities: tuple[str, ...] = (
        "repo.read",
        "repo.search",
        "repo.list",
        "git.diff",
        "test.run",
        "build.run",
        "fs.write",
        "git.commit",
    )
    approval_required_capabilities: tuple[str, ...] = ("fs.write", "git.commit")
    forbidden_capabilities: tuple[str, ...] = (
        "git.push",
        "repo.remote.write",
        "host.unrestricted",
    )

    @field_validator("readable_paths", "writable_paths", "forbidden_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_scope_pattern(item) for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("path policy values must be unique")
        return normalized

    @field_validator(
        "allowed_capabilities",
        "approval_required_capabilities",
        "forbidden_capabilities",
    )
    @classmethod
    def validate_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_unique(value, label="capabilities")

    @model_validator(mode="after")
    def validate_permission_boundaries(self) -> Self:
        allowed = set(self.allowed_capabilities)
        approval_required = set(self.approval_required_capabilities)
        forbidden = set(self.forbidden_capabilities)

        if approval_required - allowed:
            raise ValueError("approval-required capabilities must also be allowed")
        if allowed & forbidden:
            raise ValueError("allowed and forbidden capabilities must be disjoint")
        if {"fs.write", "git.commit"} & allowed - approval_required:
            raise ValueError("source writes and commits must require approval")
        if not {"git.push", "repo.remote.write"}.issubset(forbidden):
            raise ValueError("benchmark tasks must forbid remote repository writes")

        exact_overlap = set(self.writable_paths) & set(self.forbidden_paths)
        if exact_overlap:
            raise ValueError(f"writable and forbidden paths overlap: {sorted(exact_overlap)}")
        return self


class VerificationSpec(FrozenModel):
    build_scheme: str = Field(min_length=1, max_length=200)
    public_test_plans: tuple[str, ...] = Field(min_length=1)
    task_oracle_ids: tuple[str, ...] = Field(min_length=1)
    replay_script: str | None = None
    required_checks: tuple[VerificationCheck, ...] = CORE_REQUIRED_CHECKS
    required_artifacts: tuple[ProofArtifactKind, ...] = Field(min_length=1)
    changed_file_limit: int = Field(default=8, ge=1, le=100)
    max_patch_bytes: int = Field(default=200_000, ge=1, le=10_000_000)
    repetitions: int = Field(default=1, ge=1, le=20)

    @field_validator("build_scheme")
    @classmethod
    def validate_build_scheme(cls, value: str) -> str:
        return ReproductionSpec.validate_scheme(value)

    @field_validator("public_test_plans", "task_oracle_ids")
    @classmethod
    def validate_identifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = _normalize_unique(value, label="verification identifiers")
        if any(any(character in item for character in ("\n", "\r", "\x00")) for item in normalized):
            raise ValueError("verification identifiers must be single values, not commands")
        return normalized

    @field_validator("replay_script")
    @classmethod
    def validate_replay_script(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_relative_path(value, label="replay_script")

    @field_validator("required_checks")
    @classmethod
    def validate_required_checks(
        cls, value: tuple[VerificationCheck, ...]
    ) -> tuple[VerificationCheck, ...]:
        if len(set(value)) != len(value):
            raise ValueError("required checks must be unique")
        missing = set(CORE_REQUIRED_CHECKS) - set(value)
        if missing:
            raise ValueError(f"required checks omit core gates: {sorted(missing)}")
        return value

    @field_validator("required_artifacts")
    @classmethod
    def validate_verification_artifacts(
        cls, value: tuple[ProofArtifactKind, ...]
    ) -> tuple[ProofArtifactKind, ...]:
        if len(set(value)) != len(value):
            raise ValueError("required proof artifacts must be unique")
        if ProofArtifactKind.PATCH_DIFF not in value:
            raise ValueError("required proof artifacts must include patch.diff")
        return value

    @model_validator(mode="after")
    def validate_replay_contract(self) -> Self:
        if VerificationCheck.SIMULATOR_REPLAY in self.required_checks and not self.replay_script:
            raise ValueError("simulator replay requires a repository-relative replay_script")
        return self


class RepairTask(FrozenModel):
    schema_version: Literal["xcodefix-task-v1"] = "xcodefix-task-v1"
    task_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    origin: TaskOrigin
    repository: RepositorySpec
    environment: IOSExecutionEnvironment
    problem_statement: str = Field(min_length=1, max_length=20_000)
    expected_invariants: tuple[str, ...] = Field(min_length=1, max_length=50)
    reproduction: ReproductionSpec
    permissions: PermissionSpec
    budget: Budget
    verification: VerificationSpec
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        normalized = value.strip()
        if not _TASK_ID_RE.fullmatch(normalized):
            raise ValueError("task_id must be lowercase kebab-case")
        return normalized

    @field_validator("title", "problem_statement")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must not be blank")
        return normalized

    @field_validator("expected_invariants")
    @classmethod
    def validate_expected_invariants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_unique(value, label="expected invariants")

    @model_validator(mode="after")
    def validate_task_provenance(self) -> Self:
        if (
            self.repository.kind is RepositoryKind.EMBEDDED_FIXTURE
            and self.origin is not TaskOrigin.SYNTHETIC_SEEDED
        ):
            raise ValueError("embedded fixtures must be labeled synthetic-seeded")
        return self
