from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_lab.domain.evaluation import EvalResult, EvalStatus
from agent_lab.domain.models import (
    Citation,
    Evidence,
    RunRequest,
    RunResult,
    RunStatus,
    SourceType,
    TrustLevel,
)


def evidence(identifier: str = "ev-1") -> Evidence:
    return Evidence.from_content(
        evidence_id=identifier,
        source_type=SourceType.FIXTURE,
        content="  stable content  ",
        trust_level=TrustLevel.SYNTHETIC,
        observed_at=datetime(2026, 7, 28, tzinfo=UTC),
    )


def test_evidence_factory_normalizes_and_hashes() -> None:
    first = evidence()
    second = evidence()
    assert first.content == "stable content"
    assert first.content_hash == second.content_hash
    assert first.content_hash.startswith("sha256:")


def test_request_defaults_are_bounded_and_unique() -> None:
    first = RunRequest(goal="test")
    second = RunRequest(goal="test")
    assert first.task_id != second.task_id
    assert first.budget.max_tool_calls == 4
    assert first.allowed_capabilities == {"fixture.search"}


def test_completed_result_requires_answer() -> None:
    with pytest.raises(ValidationError, match="non-empty answer"):
        RunResult(
            task_id="task",
            thread_id="thread",
            status=RunStatus.COMPLETED,
            termination_reason="done",
            runtime="test",
        )


def test_citation_must_reference_existing_evidence() -> None:
    with pytest.raises(ValidationError, match="missing evidence"):
        RunResult(
            task_id="task",
            thread_id="thread",
            status=RunStatus.COMPLETED,
            answer="answer",
            evidence=(evidence(),),
            citations=(Citation(claim="claim", evidence_ids=("missing",)),),
            termination_reason="done",
            runtime="test",
        )


def test_eval_unknown_and_error_cannot_hide_a_score() -> None:
    with pytest.raises(ValidationError, match="must not carry"):
        EvalResult(grader="g", status=EvalStatus.UNKNOWN, score=0.5, message="unknown")
    with pytest.raises(ValidationError, match="requires error"):
        EvalResult(grader="g", status=EvalStatus.ERROR, message="broken")
