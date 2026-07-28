"""Load versioned XcodeFixBench task and replay contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_lab.repair.domain.replay import ReplaySpec
from agent_lab.repair.domain.task import RepairTask


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def resolve_task_path(root: Path, task: str | Path) -> Path:
    candidate = Path(task)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_absolute():
        path = candidate if candidate.is_absolute() else root / candidate
    else:
        path = root / "benchmarks" / "ios-repair" / "dev" / str(task) / "task.yaml"
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("task path escapes the project root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_repair_task(root: Path, task: str | Path) -> RepairTask:
    path = resolve_task_path(root, task)
    loaded = RepairTask.model_validate(_yaml_mapping(path))
    if path.parent.name != loaded.task_id:
        raise ValueError("task directory name must match task_id")
    return loaded


def load_replay_spec(root: Path, task: RepairTask) -> ReplaySpec:
    if task.verification.replay_script is None:
        raise ValueError("task does not define a replay script")
    path = (root / task.verification.replay_script).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("replay path escapes the project root")
    return ReplaySpec.model_validate(_yaml_mapping(path))
