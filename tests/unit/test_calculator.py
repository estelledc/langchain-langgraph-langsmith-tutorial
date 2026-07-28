from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_lab.capabilities.tools.calculator import (
    CalculationStatus,
    CalculatorInput,
    SafeCalculator,
)
from agent_lab.capabilities.tools.contracts import ToolErrorCode


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("9 // 2", 4),
        ("7 % 4", 3),
        ("2 ** 8", 256),
    ],
)
def test_safe_arithmetic(expression: str, expected: int) -> None:
    result = SafeCalculator().calculate(CalculatorInput(expression=expression))
    assert result.status is CalculationStatus.OK
    assert result.value == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "(1).__class__.__mro__",
        "open('/etc/passwd').read()",
        "9 ** 999",
        "1 / 0",
        "True + 1",
    ],
)
def test_unsafe_expression_is_data_error(expression: str) -> None:
    result = SafeCalculator().calculate(CalculatorInput(expression=expression))
    assert result.status is CalculationStatus.ERROR
    assert result.error_code is ToolErrorCode.INVALID_INPUT
    assert result.value is None


def test_adversarial_dataset(root: Path) -> None:
    calculator = SafeCalculator()
    path = root / "datasets/adversarial/tool-input-v1.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        result = calculator.calculate(CalculatorInput.model_validate(case["input"]))
        assert result.status == case["expected_status"], case["case_id"]
        if "expected_value" in case:
            assert result.value == case["expected_value"]
        if "expected_error" in case:
            assert result.error_code == case["expected_error"]
