"""Local-first CLI for running, evaluating and verifying the lab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from agent_lab.domain.evaluation import SuiteReport
from agent_lab.domain.models import RunRequest
from agent_lab.evaluation.cases import load_suite
from agent_lab.evaluation.passport import (
    REQUIRED_QUALITY_GATES,
    REQUIRED_SUITES,
    build_passport,
    write_passport,
)
from agent_lab.evaluation.runner import build_default_service, build_suite_runner


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "evals").is_dir():
            return candidate
    packaged = Path(__file__).resolve().parent / "package_root"
    if (packaged / "datasets").is_dir() and (packaged / "evals" / "suites").is_dir():
        return packaged
    raise RuntimeError("未找到 Agent Engineering Lab 项目根目录")


def run_command(args: argparse.Namespace) -> int:
    service = build_default_service()
    request = RunRequest(goal=args.goal)
    result = service.run(request, runtime=args.runtime)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.answer or f"[{result.status}] {result.termination_reason}")
    return 0 if result.status in {"completed", "unknown"} else 1


def eval_command(args: argparse.Namespace) -> int:
    root = project_root()
    suite_path = root / "evals" / "suites" / f"{args.suite}.yaml"
    config = load_suite(suite_path)
    runner = build_suite_runner(root, config)
    report = runner.run(suite_path)
    if args.report:
        runner.write_report(report, root / args.report)
    passed, failures = runner.gate(report, config)
    print(
        json.dumps(
            {
                "suite": report.suite,
                "case_runs": len(report.cases),
                "pass_rate": report.pass_rate,
                "unknown_rate": report.unknown_rate,
                "evaluator_error_rate": report.evaluator_error_rate,
                "runtime_error_rate": report.runtime_error_rate,
                "failed_cases": report.failed_cases,
                "gate": "PASS" if passed else "FAIL",
                "gate_failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def passport_command(args: argparse.Namespace) -> int:
    root = project_root()
    return _execute_verification(
        root,
        report_dir=Path(args.report_dir),
        passport_output=Path(args.output),
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
    )


def _checked(root: Path, command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def _quality_gate_commands() -> tuple[tuple[str, list[str]], ...]:
    return (
        ("ruff-format", ["ruff", "format", "--check", "src", "tests", "scripts"]),
        ("ruff-lint", ["ruff", "check", "src", "tests", "scripts"]),
        ("mypy", ["mypy", "src/agent_lab"]),
        (
            "pytest",
            [
                "pytest",
                "--cov=agent_lab",
                "--cov-report=term-missing",
                "--cov-fail-under=85",
            ],
        ),
        ("curriculum", [sys.executable, "scripts/check_curriculum.py"]),
        ("site-contract", [sys.executable, "scripts/check_site.py"]),
        ("git-diff-check", ["git", "diff", "--check"]),
    )


def _run_quality_gates(root: Path) -> tuple[dict[str, Literal["PASS"]], int]:
    statuses: dict[str, Literal["PASS"]] = {}
    try:
        for name, command in _quality_gate_commands():
            _checked(root, command)
            statuses[name] = "PASS"
    except subprocess.CalledProcessError as exc:
        return statuses, exc.returncode or 1
    return statuses, 0


def _run_required_suites(
    root: Path,
    report_dir: Path,
) -> tuple[dict[str, SuiteReport], dict[str, Literal["PASS"]], bool]:
    reports: dict[str, SuiteReport] = {}
    statuses: dict[str, Literal["PASS"]] = {}
    all_passed = True
    for name in REQUIRED_SUITES:
        suite_path = root / "evals" / "suites" / f"{name}.yaml"
        config = load_suite(suite_path)
        runner = build_suite_runner(root, config)
        report = runner.run(suite_path)
        report_path = report_dir / f"{name}.json"
        runner.write_report(report, report_path)
        passed, failures = runner.gate(report, config)
        print(
            json.dumps(
                {
                    "suite": name,
                    "case_runs": len(report.cases),
                    "pass_rate": report.pass_rate,
                    "unknown_rate": report.unknown_rate,
                    "evaluator_error_rate": report.evaluator_error_rate,
                    "runtime_error_rate": report.runtime_error_rate,
                    "gate": "PASS" if passed else "FAIL",
                    "gate_failures": failures,
                },
                ensure_ascii=False,
            )
        )
        reports[name] = report
        if passed:
            statuses[f"suite:{name}"] = "PASS"
        else:
            all_passed = False
    return reports, statuses, all_passed


def _resolve_from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _execute_verification(
    root: Path,
    *,
    report_dir: Path,
    passport_output: Path | None,
    artifact_dir: Path | None,
) -> int:
    gate_statuses, quality_exit = _run_quality_gates(root)
    if quality_exit:
        return quality_exit

    resolved_report_dir = _resolve_from_root(root, report_dir)
    reports, suite_statuses, suites_passed = _run_required_suites(root, resolved_report_dir)
    gate_statuses.update(suite_statuses)
    if not suites_passed:
        return 1

    if set(gate_statuses) != set(REQUIRED_QUALITY_GATES) | {
        f"suite:{name}" for name in REQUIRED_SUITES
    }:
        raise RuntimeError("verification completed without the required gate set")

    if passport_output is not None:
        resolved_artifacts = (
            _resolve_from_root(root, artifact_dir) if artifact_dir is not None else None
        )
        passport = build_passport(
            root,
            reports,
            gate_statuses=gate_statuses,
            artifact_dir=resolved_artifacts,
        )
        output_path = _resolve_from_root(root, passport_output)
        write_passport(passport, output_path)
        print(output_path)
    return 0


def verify_command(args: argparse.Namespace) -> int:
    root = project_root()
    return _execute_verification(
        root,
        report_dir=Path(args.report_dir),
        passport_output=Path(args.passport_output) if args.passport_output else None,
        artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-lab",
        description="Agent Engineering Lab 的离线运行、评测与验证入口",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run", help="运行离线 Trusted Research")
    run_parser.add_argument("--goal", required=True)
    run_parser.add_argument("--runtime", choices=("workflow", "langgraph"), default="workflow")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(handler=run_command)

    eval_parser = subcommands.add_parser("eval", help="运行版本化离线评测")
    eval_parser.add_argument("--suite", default="fast")
    eval_parser.add_argument("--report", default="evals/reports/fast.json")
    eval_parser.set_defaults(handler=eval_command)

    passport_parser = subcommands.add_parser("passport", help="生成验证护照")
    passport_parser.add_argument("--output", default="verification-passport.json")
    passport_parser.add_argument("--report-dir", default="evals/reports")
    passport_parser.add_argument("--artifact-dir")
    passport_parser.set_defaults(handler=passport_command)

    verify_parser = subcommands.add_parser("verify", help="运行本地完整离线门禁")
    verify_parser.add_argument("--report-dir", default="evals/reports")
    verify_parser.add_argument("--passport-output")
    verify_parser.add_argument("--artifact-dir")
    verify_parser.set_defaults(handler=verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
