"""Versioned case and suite configuration models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from agent_lab.domain.models import Budget, FrozenModel, RunStatus


class EvalCase(FrozenModel):
    case_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    expected_status: RunStatus
    required_evidence_ids: tuple[str, ...] = ()
    forbidden_evidence_ids: tuple[str, ...] = ()
    forbidden_answer_patterns: tuple[str, ...] = ()
    expected_nodes: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    forbidden_nodes: tuple[str, ...] = ()
    expected_tool_calls: int | None = Field(default=None, ge=0)
    expected_tool_names: tuple[str, ...] = ()
    allowed_capabilities: frozenset[str] = frozenset({"fixture.search"})
    permissions: frozenset[str] = frozenset({"fixture.search"})
    budget: Budget = Field(default_factory=Budget)


class SuiteConfig(FrozenModel):
    id: str = Field(min_length=1)
    runner: Literal["runtime", "security", "contracts"] = "runtime"
    dataset_version: str = Field(min_length=1)
    datasets: tuple[str, ...] = Field(min_length=1)
    runtimes: tuple[str, ...] = Field(min_length=1)
    trials: int = Field(default=1, ge=1, le=20)
    graders: tuple[str, ...] = Field(min_length=1)
    min_pass_rate: float = Field(default=1, ge=0, le=1)
    max_unknown_rate: float = Field(default=0, ge=0, le=1)
    max_evaluator_error_rate: float = Field(default=0, ge=0, le=1)
    max_runtime_error_rate: float = Field(default=0, ge=0, le=1)


def load_cases(root: Path, relative_paths: tuple[str, ...]) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for relative in relative_paths:
        path = root / relative
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            case = EvalCase.model_validate(payload)
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id {case.case_id!r} in {path}:{line_number}")
            seen.add(case.case_id)
            cases.append(case)
    return tuple(cases)


def load_suite(path: Path) -> SuiteConfig:
    return SuiteConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
