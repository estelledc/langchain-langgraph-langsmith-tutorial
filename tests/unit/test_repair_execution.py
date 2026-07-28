from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_lab.repair.artifacts import sha256_path, write_json
from agent_lab.repair.command import CommandExecutionError, SafeCommandRunner
from agent_lab.repair.domain.approval import ApprovalAction, ApprovalReceipt
from agent_lab.repair.domain.replay import RuntimeRecord
from agent_lab.repair.task_loader import load_repair_task, load_replay_spec
from agent_lab.repair.workspace import materialize_embedded_workspace


def runner(tmp_path: Path) -> SafeCommandRunner:
    return SafeCommandRunner(allowed_root=tmp_path, receipt_dir=tmp_path / "receipts")


def test_task_and_replay_contracts_load_from_versioned_yaml(root: Path) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    replay = load_replay_spec(root, task)
    assert task.repository.base_commit == "ba73f5bb1a942d80dd806257cfa6a9e9a11656e6"
    assert task.verification.changed_file_limit == 1
    assert replay.baseline.outcome == "bug_reproduced"
    assert replay.patched.outcome == "fix_verified"


def test_replay_oracle_reports_specific_runtime_drift(root: Path) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    replay = load_replay_spec(root, task)
    record = RuntimeRecord.model_validate(
        {
            "phase": "complete",
            "outcome": "bug_reproduced",
            "notificationCount": 1,
            "notificationHeight": 325,
            "guideMinY": 353,
            "guideOverlap": 521,
            "panelHeight": 325,
            "mismatch": 196,
        }
    )
    assert replay.baseline.failures(record, tolerance=replay.tolerance) == ()
    failures = replay.patched.failures(record, tolerance=replay.tolerance)
    assert any("outcome" in failure for failure in failures)
    assert any("panel_height" in failure for failure in failures)


def test_approval_binds_payload_scope_expiry_and_single_consumption() -> None:
    now = datetime(2026, 7, 28, tzinfo=UTC)
    payload = b"diff --git a/file b/file"
    receipt = ApprovalReceipt.issue(
        actor_id="benchmark-preapproval",
        action=ApprovalAction.APPLY_PATCH,
        payload=payload,
        scope=("KeyboardFixture/KeyboardHeightResolver.swift",),
        ttl=timedelta(minutes=5),
        now=now,
    )
    receipt.authorize(
        action=ApprovalAction.APPLY_PATCH,
        payload=payload,
        paths=("KeyboardFixture/KeyboardHeightResolver.swift",),
        now=now + timedelta(minutes=1),
    )
    consumed = receipt.consume(now=now + timedelta(minutes=1))
    with pytest.raises(ValueError, match="already been consumed"):
        consumed.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=payload,
            paths=("KeyboardFixture/KeyboardHeightResolver.swift",),
            now=now + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="payload hash"):
        receipt.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=payload + b"changed",
            paths=("KeyboardFixture/KeyboardHeightResolver.swift",),
            now=now + timedelta(minutes=1),
        )
    with pytest.raises(ValueError, match="outside approval scope"):
        receipt.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=payload,
            paths=("KeyboardTaskOracleTests/KeyboardTaskOracleTests.swift",),
            now=now + timedelta(minutes=1),
        )
    with pytest.raises(ValueError, match="expired"):
        receipt.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=payload,
            paths=("KeyboardFixture/KeyboardHeightResolver.swift",),
            now=now + timedelta(minutes=6),
        )


def test_command_runner_stops_on_nonzero_and_keeps_portable_receipt(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    with pytest.raises(CommandExecutionError) as caught:
        command_runner.run(
            "intentional-failure",
            ("python3", "-c", "raise SystemExit(7)"),
            cwd=tmp_path,
            timeout=10,
        )
    assert caught.value.outcome.receipt.exit_code == 7
    assert caught.value.outcome.receipt.cwd == "$RUN"
    assert len(command_runner.receipts) == 1
    receipt_files = sorted((tmp_path / "receipts").glob("*.receipt.json"))
    assert len(receipt_files) == 1
    assert json.loads(receipt_files[0].read_text())["exit_code"] == 7


def test_materialized_fixture_reproduces_commit_and_applies_approved_gold_patch(
    root: Path, tmp_path: Path
) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    command_runner = runner(tmp_path)
    workspace = materialize_embedded_workspace(
        project_root=root,
        task=task,
        destination=tmp_path / "workspace",
        runner=command_runner,
    )
    patch_path = tmp_path / "candidate.patch"
    shutil.copyfile(root / task.metadata["gold_patch"], patch_path)
    inspection = workspace.inspect_patch(patch_path)
    assert inspection.changed_files == ("KeyboardFixture/KeyboardHeightResolver.swift",)

    approval = ApprovalReceipt.issue(
        actor_id="benchmark-preapproval",
        action=ApprovalAction.APPLY_PATCH,
        payload=inspection.payload,
        scope=inspection.changed_files,
    )
    consumed = workspace.apply_patch(
        patch_path,
        inspection=inspection,
        approval=approval,
    )
    assert consumed.consumed_at is not None
    assert workspace.changed_files() == inspection.changed_files
    diff = workspace.write_diff(tmp_path / "proof" / "patch.diff")
    assert b"guideOverlap" in diff
    with pytest.raises(ValueError, match="already been consumed"):
        consumed.authorize(
            action=ApprovalAction.APPLY_PATCH,
            payload=inspection.payload,
            paths=inspection.changed_files,
        )


def test_workspace_rejects_patch_to_forbidden_oracle(root: Path, tmp_path: Path) -> None:
    task = load_repair_task(root, "keyboard-layout-001")
    command_runner = runner(tmp_path)
    workspace = materialize_embedded_workspace(
        project_root=root,
        task=task,
        destination=tmp_path / "workspace",
        runner=command_runner,
    )
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        """diff --git a/KeyboardTaskOracleTests/KeyboardTaskOracleTests.swift b/KeyboardTaskOracleTests/KeyboardTaskOracleTests.swift
--- a/KeyboardTaskOracleTests/KeyboardTaskOracleTests.swift
+++ b/KeyboardTaskOracleTests/KeyboardTaskOracleTests.swift
@@ -1 +1 @@
-import XCTest
+import XCTest // tampered
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"outside writable scope|forbidden"):
        workspace.inspect_patch(patch)


def test_artifact_tree_hash_changes_with_content_and_json_writer(tmp_path: Path) -> None:
    bundle = tmp_path / "sample.xcresult"
    bundle.mkdir()
    (bundle / "a.json").write_text("one", encoding="utf-8")
    first = sha256_path(bundle)
    (bundle / "a.json").write_text("two", encoding="utf-8")
    second = sha256_path(bundle)
    assert first != second

    output = tmp_path / "result.json"
    write_json(output, {"status": "PASS"})
    assert json.loads(output.read_text()) == {"status": "PASS"}
