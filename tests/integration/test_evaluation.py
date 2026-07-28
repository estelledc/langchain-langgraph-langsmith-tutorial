from __future__ import annotations

from pathlib import Path

import yaml

from agent_lab.domain.evaluation import EvalStatus
from agent_lab.domain.models import RunResult, RunStatus
from agent_lab.evaluation.cases import load_cases, load_suite
from agent_lab.evaluation.runner import SuiteRunner, build_suite_runner


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
    assert report.runtime_error_rate == 0
    assert len(report.cases) == 18
    assert all(grader.status is EvalStatus.PASS for case in report.cases for grader in case.graders)


def test_case_ids_are_unique_across_suite(root: Path) -> None:
    config = load_suite(root / "evals/suites/fast.yaml")
    cases = load_cases(root, config.datasets)
    assert len({case.case_id for case in cases}) == len(cases)


def test_security_and_contract_suites_are_strict_gates(root: Path) -> None:
    expected_cases = {"security": 4, "contracts": 2}
    for name, count in expected_cases.items():
        suite_path = root / "evals" / "suites" / f"{name}.yaml"
        config = load_suite(suite_path)
        runner = build_suite_runner(root, config)
        report = runner.run(suite_path)
        passed, failures = runner.gate(report, config)

        assert passed, failures
        assert len(report.cases) == count
        assert report.pass_rate == 1
        assert report.unknown_rate == 0
        assert report.evaluator_error_rate == 0
        assert report.runtime_error_rate == 0
        assert all(
            grader.status is EvalStatus.PASS for case in report.cases for grader in case.graders
        )


def test_runtime_exception_is_case_error_and_suite_continues(tmp_path: Path) -> None:
    dataset = tmp_path / "datasets" / "runtime-errors.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        "\n".join(
            [
                '{"case_id":"raises","dataset_version":"v1","goal":"first",'
                '"expected_status":"completed"}',
                '{"case_id":"continues","dataset_version":"v1","goal":"second",'
                '"expected_status":"completed"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "evals" / "suites" / "runtime-errors.yaml"
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text(
        yaml.safe_dump(
            {
                "id": "runtime-errors",
                "dataset_version": "v1",
                "datasets": ["datasets/runtime-errors.jsonl"],
                "runtimes": ["synthetic"],
                "graders": ["status"],
                "max_runtime_error_rate": 0,
            }
        ),
        encoding="utf-8",
    )

    class SequenceRunner(SuiteRunner):
        calls = 0

        def _run_case(self, case, *, runtime: str, trial: int) -> RunResult:  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic provider failure\nwith details")
            return RunResult(
                task_id=case.case_id,
                thread_id=f"{case.case_id}-{trial}",
                status=RunStatus.COMPLETED,
                answer="completed after previous failure",
                termination_reason="synthetic success",
                runtime=runtime,
            )

    runner = SequenceRunner(tmp_path)
    config = load_suite(suite_path)
    report = runner.run(suite_path)
    passed, failures = runner.gate(report, config)

    assert len(report.cases) == 2
    assert report.cases[0].result_status == EvalStatus.ERROR
    assert report.cases[0].error_phase == "runtime"
    assert report.cases[0].error_type == "RuntimeError"
    assert report.cases[0].error_message == "synthetic provider failure with details"
    assert report.cases[1].result_status == RunStatus.COMPLETED
    assert report.runtime_error_rate == 0.5
    assert report.evaluator_error_rate == 0
    assert report.pass_rate == 0.5
    assert not passed
    assert any("runtime_error_rate" in failure for failure in failures)
