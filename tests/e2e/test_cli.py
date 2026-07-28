from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_lab import cli as cli_module
from agent_lab.cli import main
from agent_lab.evaluation import passport as passport_module


def test_run_command_returns_offline_answer(capsys) -> None:
    exit_code = main(["run", "--goal", "LangGraph checkpointer", "--runtime", "workflow"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "fixture" in output
    assert "fx-langgraph-memory-v1" in output


def test_eval_command_writes_report(root: Path, capsys) -> None:
    relative = "evals/reports/test-fast.json"
    exit_code = main(["eval", "--suite", "fast", "--report", relative])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["gate"] == "PASS"
    report_path = root / relative
    assert report_path.is_file()
    report_path.unlink()


def test_eval_command_runs_security_and_contract_suites(root: Path, tmp_path: Path, capsys) -> None:
    for suite in ("security", "contracts"):
        report_path = tmp_path / f"{suite}.json"
        exit_code = main(["eval", "--suite", suite, "--report", str(report_path)])
        output = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert output["gate"] == "PASS"
        assert output["runtime_error_rate"] == 0
        assert report_path.is_file()


def test_passport_records_dirty_and_unknown_live_checks(
    root: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    real_git = passport_module._git

    def fake_git(root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return " M synthetic-change"
        return real_git(root, *args)

    monkeypatch.setattr(passport_module, "_git", fake_git)
    monkeypatch.setattr(cli_module, "_checked", lambda root, command: None)
    output_path = tmp_path / "passport.json"
    artifact_dir = tmp_path / "dist"
    artifact_dir.mkdir()
    (artifact_dir / ".gitignore").write_text("*\n")
    (artifact_dir / "agent_lab.whl").write_bytes(b"wheel")
    (artifact_dir / "agent_lab.tar.gz").write_bytes(b"sdist")
    exit_code = main(
        [
            "passport",
            "--output",
            str(output_path),
            "--report-dir",
            str(tmp_path / "reports"),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    capsys.readouterr()
    passport = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert passport["schema_version"] == "verification-passport-v2"
    assert passport["worktree_dirty"] is True
    assert passport["unknowns"]
    assert passport["dataset_hashes"]
    assert set(passport["suites"]) == {"fast", "security", "contracts"}
    assert all(status == "PASS" for status in passport["gate_statuses"].values())
    assert set(passport["artifact_hashes"]) == {"agent_lab.whl", "agent_lab.tar.gz"}


def test_passport_rejects_caller_supplied_test_status() -> None:
    with pytest.raises(SystemExit):
        main(["passport", "--test-status", "PASS"])


def test_agent_lab_project_root_falls_back_to_packaged_eval_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "agent_lab"
    (package / "package_root" / "datasets").mkdir(parents=True)
    (package / "package_root" / "evals" / "suites").mkdir(parents=True)
    fake_module = package / "cli.py"
    fake_module.write_text("# packaged")
    monkeypatch.setattr(cli_module, "__file__", str(fake_module))
    monkeypatch.chdir(tmp_path)
    assert cli_module.project_root() == package / "package_root"
