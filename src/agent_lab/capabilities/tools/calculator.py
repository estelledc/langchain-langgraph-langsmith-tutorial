"""A bounded arithmetic evaluator; model-controlled text never reaches eval/exec."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar

from pydantic import Field

from agent_lab.capabilities.tools.contracts import SideEffect, ToolErrorCode, ToolSpec
from agent_lab.domain.models import FrozenModel


class CalculationStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class CalculatorInput(FrozenModel):
    expression: str = Field(min_length=1, max_length=200)


class CalculatorResult(FrozenModel):
    status: CalculationStatus
    value: int | float | None = None
    normalized_expression: str | None = None
    error_code: ToolErrorCode | None = None
    message: str


class UnsafeExpression(ValueError):
    """Raised when an expression is outside the arithmetic contract."""


BinaryOp = Callable[[int | float, int | float], int | float]
UnaryOp = Callable[[int | float], int | float]


class SafeCalculator:
    """Interpret a small arithmetic AST with explicit resource limits."""

    spec = ToolSpec(
        name="calculate_arithmetic",
        capability="math.calculate",
        description="计算只包含数字和白名单运算符的算术表达式。",
        input_schema="CalculatorInput",
        output_schema="CalculatorResult",
        side_effect=SideEffect.NONE,
        idempotent=True,
        required_permissions=frozenset({"math.calculate"}),
        timeout_seconds=1,
        max_retries=0,
        errors=frozenset({ToolErrorCode.INVALID_INPUT, ToolErrorCode.INTERNAL}),
        output_is_untrusted=False,
        may_contain_prompt_injection=False,
    )

    _binary_ops: ClassVar[dict[type[ast.operator], BinaryOp]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_ops: ClassVar[dict[type[ast.unaryop], UnaryOp]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(
        self,
        *,
        max_nodes: int = 40,
        max_abs_value: float = 1_000_000_000_000,
        max_abs_exponent: int = 12,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_abs_value = max_abs_value
        self.max_abs_exponent = max_abs_exponent

    def calculate(self, request: CalculatorInput) -> CalculatorResult:
        try:
            tree = ast.parse(request.expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > self.max_nodes:
                raise UnsafeExpression("表达式过于复杂")
            value = self._visit(tree.body)
            return CalculatorResult(
                status=CalculationStatus.OK,
                value=value,
                normalized_expression=ast.unparse(tree),
                message="计算完成",
            )
        except (SyntaxError, UnsafeExpression, ZeroDivisionError, OverflowError) as exc:
            return CalculatorResult(
                status=CalculationStatus.ERROR,
                error_code=ToolErrorCode.INVALID_INPUT,
                message=str(exc),
            )

    def _visit(self, node: ast.expr) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsafeExpression("只允许数字常量")
            return self._bounded(node.value)

        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_ops:
            value = self._visit(node.operand)
            return self._bounded(self._unary_ops[type(node.op)](value))

        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_ops:
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > self.max_abs_exponent:
                    raise UnsafeExpression("指数超过上限")
                if left == 0 and right < 0:
                    raise UnsafeExpression("零不能取负指数")
            value = self._binary_ops[type(node.op)](left, right)
            return self._bounded(value)

        raise UnsafeExpression(f"不允许的语法：{type(node).__name__}")

    def _bounded(self, value: int | float) -> int | float:
        if not math.isfinite(float(value)) or abs(value) > self.max_abs_value:
            raise UnsafeExpression("计算结果超过上限")
        return value
