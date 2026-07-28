from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent_lab.capabilities.retrieval import FixtureSearchAdapter
from agent_lab.runtimes.graph import LangGraphResearchRuntime
from agent_lab.runtimes.workflow import TrustedResearchWorkflow


@pytest.fixture
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def search() -> FixtureSearchAdapter:
    return FixtureSearchAdapter()


@pytest.fixture
def workflow(search: FixtureSearchAdapter) -> TrustedResearchWorkflow:
    return TrustedResearchWorkflow(search)


@pytest.fixture
def graph_runtime(search: FixtureSearchAdapter) -> LangGraphResearchRuntime:
    return LangGraphResearchRuntime(search, checkpointer=InMemorySaver())
