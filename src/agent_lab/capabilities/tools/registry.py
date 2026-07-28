"""Deterministic tool contracts shared by graders and contract suites."""

from __future__ import annotations

from pydantic import BaseModel

from agent_lab.capabilities.retrieval.models import FIXTURE_SEARCH_SPEC, SearchQuery
from agent_lab.capabilities.tools.calculator import CalculatorInput, SafeCalculator
from agent_lab.capabilities.tools.contracts import ToolSpec

TOOL_SPECS: dict[str, ToolSpec] = {
    FIXTURE_SEARCH_SPEC.name: FIXTURE_SEARCH_SPEC,
    SafeCalculator.spec.name: SafeCalculator.spec,
}

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    FIXTURE_SEARCH_SPEC.name: SearchQuery,
    SafeCalculator.spec.name: CalculatorInput,
}
