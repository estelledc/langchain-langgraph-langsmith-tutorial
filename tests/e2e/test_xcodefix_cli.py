from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_lab.repair.cli as cli
from agent_lab.repair.domain.result import RepairStage, RepairStatus


class FakePipeline:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root

    def run(self, **kwargs: object) -> SimpleNamespace:
        approved = kwargs["approval_actor"] is not None
        status = RepairStatus.DELIVERED if approved else RepairStatus.APPROVAL_REQUIRED
        stage = RepairStage.DELIVERED if approved else RepairStage.APPROVAL_REQUIRED
        result = SimpleNamespace(
            status=status,
            stage=stage,
            termination_reason="done" if approved else "approval required",
            checks=(),
        )
        return SimpleNamespace(
            result=result,
            passport=object() if approved else None,
            output_dir=Path(kwargs["output_dir"]),
        )


def test_xcodefix_task_command_outputs_contract_hash(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(root)
    assert cli.main(["task", "--task", "keyboard-layout-001"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task"]["task_id"] == "keyboard-layout-001"
    assert payload["contract_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("approved", "expected_exit", "expected_status"),
    (
        (False, 2, "approval_required"),
        (True, 0, "delivered"),
    ),
)
def test_xcodefix_run_exit_codes_reflect_approval_and_delivery(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    approved: bool,
    expected_exit: int,
    expected_status: str,
) -> None:
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli, "DeterministicRepairPipeline", FakePipeline)
    argv = [
        "run",
        "--task",
        "keyboard-layout-001",
        "--candidate",
        "gold",
        "--output",
        str(tmp_path / "run"),
    ]
    if approved:
        argv.append("--approve-patch")
    assert cli.main(argv) == expected_exit
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
    assert (payload["patch_passport"] is not None) is approved


def test_xcodefix_project_root_rejects_unrelated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="未找到"):
        cli.project_root()


def test_xcodefix_project_root_falls_back_to_packaged_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "agent_lab" / "repair"
    (package / "package_root" / "benchmarks" / "ios-repair").mkdir(parents=True)
    fake_module = package / "cli.py"
    fake_module.write_text("# packaged")
    monkeypatch.setattr(cli, "__file__", str(fake_module))
    monkeypatch.chdir(tmp_path / "agent_lab")
    assert cli.project_root() == package / "package_root"
