"""Run versioned offline suites across interchangeable runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from langgraph.checkpoint.memory import InMemorySaver

from agent_lab.application.context import RunContext
from agent_lab.application.service import ResearchService
from agent_lab.capabilities.retrieval import FixtureSearchAdapter
from agent_lab.domain.evaluation import CaseReport, EvalResult, EvalStatus, SuiteReport
from agent_lab.domain.models import RunRequest, RunResult
from agent_lab.evaluation.cases import EvalCase, SuiteConfig, load_cases, load_suite
from agent_lab.evaluation.graders import GRADERS
from agent_lab.runtimes.graph import LangGraphResearchRuntime
from agent_lab.runtimes.workflow import TrustedResearchWorkflow


def _safe_error_message(exc: Exception, *, limit: int = 500) -> str:
    message = " ".join(str(exc).split()) or "runtime raised without a message"
    return message[:limit]


def build_default_service() -> ResearchService:
    search = FixtureSearchAdapter()
    return ResearchService(
        {
            "workflow": TrustedResearchWorkflow(search),
            "langgraph": LangGraphResearchRuntime(search, checkpointer=InMemorySaver()),
        }
    )


class SuiteExecutor(Protocol):
    def run(self, suite_path: Path) -> SuiteReport: ...

    def gate(self, report: SuiteReport, config: SuiteConfig) -> tuple[bool, tuple[str, ...]]: ...

    @staticmethod
    def write_report(report: SuiteReport, path: Path) -> None: ...


class BaseSuiteRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    def gate(self, report: SuiteReport, config: SuiteConfig) -> tuple[bool, tuple[str, ...]]:
        failures: list[str] = []
        if report.pass_rate < config.min_pass_rate:
            failures.append(
                f"pass_rate {report.pass_rate:.3f} < required {config.min_pass_rate:.3f}"
            )
        if report.unknown_rate > config.max_unknown_rate:
            failures.append(
                f"unknown_rate {report.unknown_rate:.3f} > allowed {config.max_unknown_rate:.3f}"
            )
        if report.evaluator_error_rate > config.max_evaluator_error_rate:
            failures.append(
                "evaluator_error_rate "
                f"{report.evaluator_error_rate:.3f} > allowed {config.max_evaluator_error_rate:.3f}"
            )
        if report.runtime_error_rate > config.max_runtime_error_rate:
            failures.append(
                "runtime_error_rate "
                f"{report.runtime_error_rate:.3f} > allowed {config.max_runtime_error_rate:.3f}"
            )
        return not failures, tuple(failures)

    @staticmethod
    def write_report(report: SuiteReport, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class SuiteRunner(BaseSuiteRunner):
    def __init__(self, root: Path, service: ResearchService | None = None) -> None:
        super().__init__(root)
        self.service = service or build_default_service()

    def run(self, suite_path: Path) -> SuiteReport:
        config = load_suite(suite_path)
        cases = load_cases(self.root, config.datasets)
        reports: list[CaseReport] = []
        failed: list[str] = []
        grader_count = 0
        unknown_count = 0
        error_count = 0
        runtime_error_count = 0
        passed_runs = 0
        total_runs = 0

        for case in cases:
            for runtime in config.runtimes:
                for trial in range(1, config.trials + 1):
                    total_runs += 1
                    run_id = f"{case.case_id}:{runtime}:trial-{trial}"
                    try:
                        result = self._run_case(case, runtime=runtime, trial=trial)
                    except Exception as exc:  # one runtime failure must not abort the suite
                        runtime_error_count += 1
                        failed.append(run_id)
                        safe_message = _safe_error_message(exc)
                        reports.append(
                            CaseReport(
                                case_id=case.case_id,
                                trial=trial,
                                runtime=runtime,
                                result_status=EvalStatus.ERROR,
                                graders=(
                                    EvalResult(
                                        grader="runtime",
                                        status=EvalStatus.ERROR,
                                        message="runtime 执行失败",
                                        error=f"{type(exc).__name__}: {safe_message}",
                                    ),
                                ),
                                error_phase="runtime",
                                error_type=type(exc).__name__,
                                error_message=safe_message,
                            )
                        )
                        continue
                    grader_results: list[EvalResult] = []
                    for name in config.graders:
                        grader = GRADERS.get(name)
                        if grader is None:
                            evaluation = EvalResult(
                                grader=name,
                                status=EvalStatus.ERROR,
                                message="grader 未注册",
                                error=f"unknown grader: {name}",
                            )
                        else:
                            try:
                                evaluation = grader(result, case)
                            except Exception as exc:  # evaluator errors stay ERROR
                                evaluation = EvalResult(
                                    grader=name,
                                    status=EvalStatus.ERROR,
                                    message="grader 执行失败",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                        grader_results.append(evaluation)
                        grader_count += 1
                        unknown_count += evaluation.status is EvalStatus.UNKNOWN
                        error_count += evaluation.status is EvalStatus.ERROR

                    run_passed = all(item.status is EvalStatus.PASS for item in grader_results)
                    if run_passed:
                        passed_runs += 1
                    else:
                        failed.append(run_id)
                    reports.append(
                        CaseReport(
                            case_id=case.case_id,
                            trial=trial,
                            runtime=runtime,
                            result_status=result.status,
                            graders=tuple(grader_results),
                        )
                    )

        denominator = grader_count or 1
        return SuiteReport(
            suite=config.id,
            dataset_version=config.dataset_version,
            cases=tuple(reports),
            pass_rate=passed_runs / total_runs if total_runs else 0,
            unknown_rate=unknown_count / denominator,
            evaluator_error_rate=error_count / denominator,
            runtime_error_rate=runtime_error_count / total_runs if total_runs else 0,
            failed_cases=tuple(failed),
        )

    def _run_case(self, case: EvalCase, *, runtime: str, trial: int) -> RunResult:
        request = RunRequest(
            task_id=f"{case.case_id}-{runtime}-{trial}",
            thread_id=f"{case.case_id}-{runtime}-{trial}",
            goal=case.goal,
            allowed_capabilities=case.allowed_capabilities,
            budget=case.budget,
        )
        return self.service.run(
            request,
            runtime=runtime,
            context=RunContext(permissions=case.permissions),
        )


def run_named_suite(root: Path, name: str) -> tuple[SuiteConfig, SuiteReport]:
    suite_path = root / "evals" / "suites" / f"{name}.yaml"
    config = load_suite(suite_path)
    report = build_suite_runner(root, config).run(suite_path)
    return config, report


def build_suite_runner(root: Path, config: SuiteConfig) -> SuiteExecutor:
    if config.runner == "runtime":
        return SuiteRunner(root)

    from agent_lab.evaluation.deterministic_runner import DeterministicSuiteRunner

    return DeterministicSuiteRunner(root)
