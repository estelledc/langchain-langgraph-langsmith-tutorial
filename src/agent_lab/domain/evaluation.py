"""Evaluation result semantics that never hide infrastructure failures."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from agent_lab.domain.models import FrozenModel


class EvalStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class EvalResult(FrozenModel):
    grader: str = Field(min_length=1)
    status: EvalStatus
    score: float | None = Field(default=None, ge=0, le=1)
    evidence: tuple[str, ...] = ()
    message: str = Field(min_length=1)
    error: str | None = None
    grader_version: str = "1"

    @model_validator(mode="after")
    def validate_score_semantics(self) -> Self:
        if self.status in {EvalStatus.UNKNOWN, EvalStatus.ERROR} and self.score is not None:
            raise ValueError("UNKNOWN and ERROR must not carry a quality score")
        if self.status in {EvalStatus.PASS, EvalStatus.FAIL} and self.score is None:
            raise ValueError("PASS and FAIL require a score")
        if self.status is EvalStatus.ERROR and not self.error:
            raise ValueError("ERROR requires error details")
        return self


class CaseReport(FrozenModel):
    case_id: str
    trial: int = Field(ge=1)
    runtime: str
    result_status: str
    graders: tuple[EvalResult, ...]
    error_phase: Literal["dataset", "runtime", "evaluator"] | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_error_semantics(self) -> Self:
        error_fields = (self.error_phase, self.error_type, self.error_message)
        if self.result_status == EvalStatus.ERROR:
            if any(value is None for value in error_fields):
                raise ValueError("ERROR requires phase, type and message")
        elif any(value is not None for value in error_fields):
            raise ValueError("runtime error details require result_status=ERROR")
        return self


class SuiteReport(FrozenModel):
    suite: str
    dataset_version: str
    cases: tuple[CaseReport, ...]
    pass_rate: float = Field(ge=0, le=1)
    unknown_rate: float = Field(ge=0, le=1)
    evaluator_error_rate: float = Field(ge=0, le=1)
    runtime_error_rate: float = Field(ge=0, le=1)
    failed_cases: tuple[str, ...] = ()
