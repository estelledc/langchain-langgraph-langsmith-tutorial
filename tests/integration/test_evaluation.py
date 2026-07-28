from __future__ import annotations

from pathlib import Path

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.evaluation.cases import load_cases, load_suite
from agent_lab.evaluation.runner import SuiteRunner


def test_fast_suite_is_a_strict_gate(root: Path) -> None:
    suite_path = root / "evals/suites/fast.yaml"
    config = load_suite(suite_path)
    runner = SuiteRunner(root)
    report = runner.run(suite_path)
    passed, failures = runner.gate(report, config)
    assert passed, failures
    assert report.pass_rate == 1
    assert report.unknown_rate == 0
    assert report.evaluator_error_rate == 0
    assert len(report.cases) == 18
    assert all(grader.status is EvalStatus.PASS for case in report.cases for grader in case.graders)


def test_case_ids_are_unique_across_suite(root: Path) -> None:
    config = load_suite(root / "evals/suites/fast.yaml")
    cases = load_cases(root, config.datasets)
    assert len({case.case_id for case in cases}) == len(cases)
