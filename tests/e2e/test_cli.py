from __future__ import annotations

import json
from pathlib import Path

from agent_lab.cli import main


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


def test_passport_records_dirty_and_unknown_live_checks(root: Path, tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "passport.json"
    exit_code = main(
        [
            "passport",
            "--suite",
            "fast",
            "--output",
            str(output_path),
        ]
    )
    capsys.readouterr()
    passport = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert passport["eval_status"] == "PASS"
    assert passport["worktree_dirty"] is True
    assert passport["unknowns"]
    assert passport["dataset_hashes"]
