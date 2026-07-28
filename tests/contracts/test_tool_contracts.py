from __future__ import annotations

import json
from pathlib import Path

from agent_lab.capabilities.retrieval.models import FIXTURE_SEARCH_SPEC
from agent_lab.capabilities.tools.calculator import SafeCalculator


def test_versioned_contract_dataset_matches_implementations(root: Path) -> None:
    path = root / "datasets/contracts/tool-contract-v1.jsonl"
    records = {
        item["tool"]: item
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    for spec in (FIXTURE_SEARCH_SPEC, SafeCalculator.spec):
        record = records[spec.name]
        assert record["capability"] == spec.capability
        assert record["side_effect"] == spec.side_effect
        assert record["idempotent"] is spec.idempotent
        assert record["output_is_untrusted"] is spec.output_is_untrusted
        assert set(record["errors"]) == set(spec.errors)
