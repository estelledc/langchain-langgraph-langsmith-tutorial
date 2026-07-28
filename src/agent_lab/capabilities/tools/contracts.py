"""Common declarations for tools treated as production APIs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from agent_lab.domain.models import FrozenModel


class SideEffect(StrEnum):
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXTERNAL_WRITE = "external_write"


class ToolErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UNAUTHORIZED = "UNAUTHORIZED"
    INVALID_INPUT = "INVALID_INPUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    INTERNAL = "INTERNAL"


class ToolSpec(FrozenModel):
    name: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    side_effect: SideEffect
    idempotent: bool
    required_permissions: frozenset[str]
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=10)
    errors: frozenset[ToolErrorCode]
    output_is_untrusted: bool = True
    may_contain_prompt_injection: bool = True
