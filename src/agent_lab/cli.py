"""Local-first CLI for running, evaluating and verifying the lab."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent_lab.domain.models import RunRequest
from agent_lab.evaluation.cases import load_suite
from agent_lab.evaluation.passport import build_passport, write_passport
from agent_lab.evaluation.runner import SuiteRunner, build_default_service


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "evals").is_dir():
            return candidate
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
    runner = SuiteRunner(root)
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
    suite_path = root / "evals" / "suites" / f"{args.suite}.yaml"
    config = load_suite(suite_path)
    runner = SuiteRunner(root)
    report = runner.run(suite_path)
    passed, _ = runner.gate(report, config)
    passport = build_passport(
        root,
        report,
        test_status=args.test_status,
        eval_status="PASS" if passed else "FAIL",
    )
    write_passport(passport, root / args.output)
    print(root / args.output)
    return 0 if passed else 1


def _checked(root: Path, command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=root, check=True)


def verify_command(args: argparse.Namespace) -> int:
    root = project_root()
    commands = [
        ["ruff", "format", "--check", "src", "tests", "scripts"],
        ["ruff", "check", "src", "tests", "scripts"],
        ["mypy", "src/agent_lab"],
        [
            "pytest",
            "--cov=agent_lab",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
        ],
        [sys.executable, "scripts/check_curriculum.py"],
        [sys.executable, "scripts/check_site.py"],
        ["git", "diff", "--check"],
    ]
    try:
        for command in commands:
            _checked(root, command)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1

    eval_args = argparse.Namespace(suite=args.suite, report=args.report)
    return eval_command(eval_args)


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
    passport_parser.add_argument("--suite", default="fast")
    passport_parser.add_argument("--test-status", choices=("PASS", "FAIL"), default="PASS")
    passport_parser.add_argument("--output", default="verification-passport.json")
    passport_parser.set_defaults(handler=passport_command)

    verify_parser = subcommands.add_parser("verify", help="运行本地完整离线门禁")
    verify_parser.add_argument("--suite", default="fast")
    verify_parser.add_argument("--report", default="evals/reports/fast.json")
    verify_parser.set_defaults(handler=verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
