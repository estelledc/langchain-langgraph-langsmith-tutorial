"""Typed retrieval request and response contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import Field

from agent_lab.capabilities.tools.contracts import SideEffect, ToolErrorCode, ToolSpec
from agent_lab.domain.models import Evidence, FrozenModel


class SearchStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    ERROR = "error"


class SearchQuery(FrozenModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(FrozenModel):
    status: SearchStatus
    evidence: tuple[Evidence, ...] = ()
    error_code: ToolErrorCode | None = None
    message: str
    fixture_version: str | None = None


class SearchPort(Protocol):
    spec: ToolSpec

    def search(self, request: SearchQuery, *, call_id: str) -> SearchResult: ...


FIXTURE_SEARCH_SPEC = ToolSpec(
    name="search_fixture_documents",
    capability="fixture.search",
    description="在版本化离线教学 fixture 中检索；不会访问互联网，也不保证信息最新。",
    input_schema="SearchQuery",
    output_schema="SearchResult",
    side_effect=SideEffect.READ,
    idempotent=True,
    required_permissions=frozenset({"fixture.search"}),
    timeout_seconds=2,
    max_retries=0,
    errors=frozenset({ToolErrorCode.NOT_FOUND, ToolErrorCode.INVALID_INPUT}),
    output_is_untrusted=True,
    may_contain_prompt_injection=True,
)
