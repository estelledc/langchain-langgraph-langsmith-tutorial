from __future__ import annotations

from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel

from agent_lab.capabilities.retrieval import FixtureSearchAdapter
from agent_lab.domain.models import RunRequest, RunStatus
from agent_lab.runtimes.langchain_agent import LangChainResearchRuntime


def test_model_runtime_cannot_run_without_model_budget() -> None:
    unused_model = cast(BaseChatModel, object())
    runtime = LangChainResearchRuntime(unused_model, FixtureSearchAdapter())
    result = runtime.run(RunRequest(goal="LangGraph"))
    assert result.status is RunStatus.BLOCKED
    assert result.termination_reason == "model_budget_exhausted"
    assert result.tool_calls == ()


def test_package_contains_no_eval_or_exec_calls(root) -> None:
    offenders = []
    for path in (root / "src/agent_lab").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        if "eval(" in text or "exec(" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
