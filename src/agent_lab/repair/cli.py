"""Command-line entrypoint for deterministic XcodeFixBench task runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from agent_lab.repair.domain.proof import task_contract_hash
from agent_lab.repair.domain.result import RepairStatus
from agent_lab.repair.pipeline import DeterministicRepairPipeline
from agent_lab.repair.task_loader import load_repair_task


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "benchmarks").is_dir():
            return candidate
    packaged = Path(__file__).resolve().parent / "package_root"
    if (packaged / "benchmarks" / "ios-repair").is_dir():
        return packaged
    raise RuntimeError("未找到 XcodeFixBench 项目根目录")


def _patch_path(root: Path, args: argparse.Namespace, task_metadata: dict[str, str]) -> Path:
    if args.patch is not None:
        return Path(args.patch).expanduser().resolve()
    metadata_key = "gold_patch" if args.candidate == "gold" else "negative_patch"
    value = task_metadata.get(metadata_key)
    if value is None:
        raise ValueError(f"task metadata does not define {metadata_key}")
    return (root / value).resolve()


def run_command(args: argparse.Namespace) -> int:
    root = project_root()
    task = load_repair_task(root, args.task)
    patch = _patch_path(root, args, task.metadata)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "runs" / "xcodefix" / task.task_id / str(uuid4())
    )
    outcome = DeterministicRepairPipeline(project_root=root).run(
        task=task,
        patch_path=patch,
        output_dir=output,
        approval_actor="local-cli-user" if args.approve_patch else None,
        device_id=args.device_id,
    )
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "status": outcome.result.status,
                "stage": outcome.result.stage,
                "termination_reason": outcome.result.termination_reason,
                "output": str(outcome.output_dir),
                "patch_passport": (
                    str(outcome.output_dir / "patch-passport.json")
                    if outcome.passport is not None
                    else None
                ),
                "checks": {record.check: record.status for record in outcome.result.checks},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if outcome.result.status is RepairStatus.DELIVERED:
        return 0
    if outcome.result.status is RepairStatus.APPROVAL_REQUIRED:
        return 2
    return 1


def task_command(args: argparse.Namespace) -> int:
    root = project_root()
    task = load_repair_task(root, args.task)
    print(
        json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "contract_hash": task_contract_hash(task),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xcodefix",
        description="XcodeFixBench 的可重放 iOS 修复验证入口",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    task_parser = commands.add_parser("task", help="校验并输出版本化任务合同")
    task_parser.add_argument("--task", default="keyboard-layout-001")
    task_parser.set_defaults(handler=task_command)

    run_parser = commands.add_parser("run", help="运行完整 deterministic repair pipeline")
    run_parser.add_argument("--task", default="keyboard-layout-001")
    source = run_parser.add_mutually_exclusive_group()
    source.add_argument("--patch", help="候选 unified diff 路径")
    source.add_argument(
        "--candidate",
        choices=("gold", "negative-hardcoded"),
        default="gold",
        help="开发集内置候选；默认 gold",
    )
    run_parser.add_argument("--approve-patch", action="store_true")
    run_parser.add_argument("--device-id")
    run_parser.add_argument("--output")
    run_parser.set_defaults(handler=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
