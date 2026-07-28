"""Run versioned security and tool-contract datasets as first-class suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, model_validator

from agent_lab.capabilities.tools.calculator import CalculatorInput, SafeCalculator
from agent_lab.capabilities.tools.contracts import SideEffect, ToolErrorCode
from agent_lab.capabilities.tools.registry import TOOL_SPECS
from agent_lab.domain.evaluation import CaseReport, EvalResult, EvalStatus, SuiteReport
from agent_lab.domain.models import FrozenModel
from agent_lab.evaluation.cases import SuiteConfig, load_suite
from agent_lab.evaluation.runner import BaseSuiteRunner, _safe_error_message


class SecurityCase(FrozenModel):
    case_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    input: dict[str, Any]
    expected_status: Literal["ok", "error"]
    expected_error: ToolErrorCode | None = None
    expected_value: int | float | None = None

    @model_validator(mode="after")
    def validate_expectation(self) -> Self:
        if self.expected_status == "ok" and self.expected_value is None:
            raise ValueError("successful security case requires expected_value")
        if self.expected_status == "error" and self.expected_error is None:
            raise ValueError("error security case requires expected_error")
        return self


class ContractCase(FrozenModel):
    case_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    side_effect: SideEffect
    idempotent: bool
    output_is_untrusted: bool
    errors: frozenset[ToolErrorCode]


def _pass(grader: str, message: str, evidence: tuple[str, ...] = ()) -> EvalResult:
    return EvalResult(
        grader=grader,
        status=EvalStatus.PASS,
        score=1,
        message=message,
        evidence=evidence,
    )


def _fail(grader: str, message: str, evidence: tuple[str, ...] = ()) -> EvalResult:
    return EvalResult(
        grader=grader,
        status=EvalStatus.FAIL,
        score=0,
        message=message,
        evidence=evidence,
    )


class DeterministicSuiteRunner(BaseSuiteRunner):
    """Evaluate non-RunRequest datasets without hiding their distinct contracts."""

    def run(self, suite_path: Path) -> SuiteReport:
        config = load_suite(suite_path)
        if config.runner == "runtime":
            raise ValueError("runtime suite must use SuiteRunner")

        reports: list[CaseReport] = []
        failed: list[str] = []
        unknown_count = 0
        evaluator_error_count = 0
        runtime_error_count = 0
        passed_cases = 0

        for relative_path in config.datasets:
            path = self.root / relative_path
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fallback_id = f"{path.name}:line-{line_number}"
                case_id = fallback_id
                try:
                    payload = json.loads(line)
                    case_id = str(payload.get("case_id", fallback_id))
                    if config.runner == "security":
                        report = self._run_security_case(payload, config)
                    else:
                        report = self._run_contract_case(payload, config)
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    evaluator_error_count += 1
                    safe_message = _safe_error_message(exc)
                    report = CaseReport(
                        case_id=case_id,
                        trial=1,
                        runtime=config.runner,
                        result_status=EvalStatus.ERROR,
                        graders=(
                            EvalResult(
                                grader=config.runner,
                                status=EvalStatus.ERROR,
                                message="dataset/evaluator 合同失败",
                                error=f"{type(exc).__name__}: {safe_message}",
                            ),
                        ),
                        error_phase="dataset",
                        error_type=type(exc).__name__,
                        error_message=safe_message,
                    )
                except Exception as exc:  # tool execution is isolated per case
                    runtime_error_count += 1
                    safe_message = _safe_error_message(exc)
                    report = CaseReport(
                        case_id=case_id,
                        trial=1,
                        runtime=config.runner,
                        result_status=EvalStatus.ERROR,
                        graders=(
                            EvalResult(
                                grader=config.runner,
                                status=EvalStatus.ERROR,
                                message="deterministic tool 执行失败",
                                error=f"{type(exc).__name__}: {safe_message}",
                            ),
                        ),
                        error_phase="runtime",
                        error_type=type(exc).__name__,
                        error_message=safe_message,
                    )

                reports.append(report)
                evaluation = report.graders[0]
                unknown_count += evaluation.status is EvalStatus.UNKNOWN
                if evaluation.status is EvalStatus.PASS:
                    passed_cases += 1
                else:
                    failed.append(f"{report.case_id}:{config.runner}:trial-1")

        total = len(reports)
        denominator = total or 1
        return SuiteReport(
            suite=config.id,
            dataset_version=config.dataset_version,
            cases=tuple(reports),
            pass_rate=passed_cases / total if total else 0,
            unknown_rate=unknown_count / denominator,
            evaluator_error_rate=evaluator_error_count / denominator,
            runtime_error_rate=runtime_error_count / denominator,
            failed_cases=tuple(failed),
        )

    @staticmethod
    def _run_security_case(payload: dict[str, Any], config: SuiteConfig) -> CaseReport:
        case = SecurityCase.model_validate(payload)
        if case.dataset_version != config.dataset_version:
            raise ValueError(
                f"case dataset_version={case.dataset_version} does not match "
                f"suite={config.dataset_version}"
            )
        if case.tool != SafeCalculator.spec.name:
            evaluation = _fail("tool_security", f"unsupported security tool: {case.tool}")
            return CaseReport(
                case_id=case.case_id,
                trial=1,
                runtime=f"tool:{case.tool}",
                result_status="unsupported_tool",
                graders=(evaluation,),
            )

        result = SafeCalculator().calculate(CalculatorInput.model_validate(case.input))
        mismatches: list[str] = []
        if result.status != case.expected_status:
            mismatches.append(f"status expected={case.expected_status} actual={result.status}")
        if case.expected_value is not None and result.value != case.expected_value:
            mismatches.append(f"value expected={case.expected_value} actual={result.value}")
        if case.expected_error is not None and result.error_code != case.expected_error:
            mismatches.append(f"error expected={case.expected_error} actual={result.error_code}")
        evidence = (result.message,)
        evaluation = (
            _fail("tool_security", "; ".join(mismatches), evidence)
            if mismatches
            else _pass("tool_security", "adversarial expectation satisfied", evidence)
        )
        return CaseReport(
            case_id=case.case_id,
            trial=1,
            runtime=f"tool:{case.tool}",
            result_status=result.status,
            graders=(evaluation,),
        )

    @staticmethod
    def _run_contract_case(payload: dict[str, Any], config: SuiteConfig) -> CaseReport:
        case = ContractCase.model_validate(payload)
        if case.dataset_version != config.dataset_version:
            raise ValueError(
                f"case dataset_version={case.dataset_version} does not match "
                f"suite={config.dataset_version}"
            )
        spec = TOOL_SPECS.get(case.tool)
        if spec is None:
            evaluation = _fail("tool_spec_contract", f"tool implementation missing: {case.tool}")
        else:
            expected = {
                "capability": case.capability,
                "side_effect": case.side_effect,
                "idempotent": case.idempotent,
                "output_is_untrusted": case.output_is_untrusted,
                "errors": case.errors,
            }
            actual = {
                "capability": spec.capability,
                "side_effect": spec.side_effect,
                "idempotent": spec.idempotent,
                "output_is_untrusted": spec.output_is_untrusted,
                "errors": spec.errors,
            }
            mismatches = [
                f"{field} expected={expected[field]} actual={actual[field]}"
                for field in expected
                if expected[field] != actual[field]
            ]
            evaluation = (
                _fail("tool_spec_contract", "; ".join(mismatches))
                if mismatches
                else _pass("tool_spec_contract", "versioned ToolSpec contract satisfied")
            )
        return CaseReport(
            case_id=case.case_id,
            trial=1,
            runtime="tool-registry",
            result_status="checked",
            graders=(evaluation,),
        )
