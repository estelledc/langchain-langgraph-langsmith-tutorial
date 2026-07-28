#!/usr/bin/env python3
"""Fail closed when curriculum metadata, links or generated pages drift."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from generate_curriculum import expected_outputs, load_curriculum


def check(root: Path) -> list[str]:
    errors: list[str] = []
    curriculum = load_curriculum(root)
    schema = json.loads((root / "curriculum/lab.schema.json").read_text(encoding="utf-8"))
    try:
        jsonschema.validate(curriculum, schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"schema: {exc.message}")
        return errors

    labs = curriculum["labs"]
    ids = [lab["id"] for lab in labs]
    orders = [lab["order"] for lab in labs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate lab id")
    if len(orders) != len(set(orders)):
        errors.append("duplicate lab order")
    core = [lab for lab in labs if lab["track"] == "core"]
    frontier = [lab for lab in labs if lab["track"] == "frontier"]
    if len(core) != 15:
        errors.append(f"expected 15 core labs, found {len(core)}")
    if len(frontier) != 10:
        errors.append(f"expected 10 frontier labs, found {len(frontier)}")

    by_id = {lab["id"]: lab for lab in labs}
    for lab in labs:
        for prerequisite in lab["prerequisites"]:
            target = by_id.get(prerequisite)
            if target is None:
                errors.append(f"{lab['id']}: unknown prerequisite {prerequisite}")
            elif target["order"] >= lab["order"]:
                errors.append(f"{lab['id']}: prerequisite {prerequisite} must come first")
        for reference in lab["references"]:
            if not (root / reference).exists():
                errors.append(f"{lab['id']}: missing reference {reference}")

    for path, expected in expected_outputs(root).items():
        if not path.is_file():
            errors.append(f"missing generated page {path.relative_to(root)}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"stale generated page {path.relative_to(root)}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check(root)
    if errors:
        for error in errors:
            print(f"curriculum: {error}")
        return 1
    print("curriculum: OK (15 core, 10 frontier, generated pages current)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
