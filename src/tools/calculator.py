"""AST-based safe calculator tool shared across P1 and M pipelines.

References:
- docs/INTERFACES.md §3 (returns result string, or "ERROR: <reason>" without raising)
- docs/IMPLEMENTATION_PLAN.md §5 (PR B-2b)
- AGENTS.md (P1 and M share the exact same tool implementation and error handling)
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any, Callable

# Pre-compiled thousands-separator pattern: matches commas inside numbers like "1,200" or "1,000,000"
# without touching argument separators like "min(10, 20)".
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

_MAX_EXPRESSION_LENGTH = 500
_MAX_POW_EXPONENT = 1000
_MAX_POW_BASE = 1e12

_BINARY_OPS: dict[type[ast.operator], Callable[[float | int, float | int], float | int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float | int], float | int]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCTIONS: dict[str, Callable[..., float | int]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}


class _CalculatorError(ValueError):
    """Internal error raised during AST validation/evaluation."""


def _normalize_expression(expression: str) -> str:
    """Normalize common mathematical Unicode operators and thousands commas."""
    expr = expression.strip()
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = _THOUSANDS_COMMA_RE.sub("", expr)
    return expr


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _CalculatorError(f"Unsupported literal: {node.value!r}")
        if isinstance(node.value, float) and (math.isnan(node.value) or math.isinf(node.value)):
            raise _CalculatorError("Non-finite number literal is not allowed")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise _CalculatorError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise _CalculatorError(f"Unsupported binary operator: {type(node.op).__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise _CalculatorError("Division by zero")

        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POW_EXPONENT or abs(left) > _MAX_POW_BASE:
                raise _CalculatorError("Exponentiation exceeds safe bounds")
            if left < 0 and isinstance(right, float) and not right.is_integer():
                raise _CalculatorError("Negative base with fractional exponent produces complex number")

        return op_fn(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _CalculatorError("Only direct function calls are allowed")
        fn_name = node.func.id
        fn = _SAFE_FUNCTIONS.get(fn_name)
        if fn is None:
            raise _CalculatorError(f"Unknown or disallowed function: '{fn_name}'")
        if node.keywords:
            raise _CalculatorError("Keyword arguments are not supported")
        args = [_eval_node(arg) for arg in node.args]
        if not args:
            raise _CalculatorError(f"Function '{fn_name}' requires at least one argument")
        if fn_name == "sqrt" and args[0] < 0:
            raise _CalculatorError("Square root of negative number")
        return fn(*args)

    if isinstance(node, ast.Name):
        raise _CalculatorError(f"Unknown identifier: '{node.id}'")

    raise _CalculatorError(f"Unsupported syntax element: {type(node).__name__}")


def _format_number(value: float | int) -> str:
    """Format numeric evaluation result cleanly for LLM consumption."""
    if isinstance(value, complex):
        raise _CalculatorError("Complex numbers are not supported")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise _CalculatorError("Result is not finite")
        if value.is_integer():
            return str(int(value))
        # Clean binary float noise (e.g., 0.1 + 0.2 -> 0.3) while preserving up to 12 sig figs
        return f"{value:.12g}"
    return str(value)


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression without propagating exceptions.

    Supports `+`, `-`, `*`, `/`, `//`, `%`, `**`, parentheses, unary `+/-`,
    and safe functions (`abs`, `round`, `min`, `max`, `sqrt`, `floor`, `ceil`).
    Never raises an exception to the agent loop; returns `"ERROR: <reason>"` on failure.

    Args:
        expression: Arithmetic expression string (e.g., `"(120 + 30) * 0.15"`).

    Returns:
        str: Formatted calculation result string, or `"ERROR: <reason>"`.
    """
    if not isinstance(expression, str):
        return f"ERROR: expression must be a string, got {type(expression).__name__}"

    normalized = _normalize_expression(expression)
    if not normalized:
        return "ERROR: Empty expression"

    if len(normalized) > _MAX_EXPRESSION_LENGTH:
        return f"ERROR: Expression exceeds maximum length ({_MAX_EXPRESSION_LENGTH})"

    try:
        tree = ast.parse(normalized, mode="eval")
        result = _eval_node(tree)
        return _format_number(result)
    except _CalculatorError as exc:
        return f"ERROR: {exc}"
    except SyntaxError as exc:
        return f"ERROR: Invalid syntax ({exc.msg})"
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"ERROR: {exc}"


def get_calculator_tool() -> Any:
    """Create a LangChain Tool wrapping `calculate` for P1 ReAct / M pipelines.

    Raises:
        RuntimeError: If langchain-core is not installed.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required to construct LangChain tools. "
            "Install via: pip install -r requirements.txt"
        ) from exc

    @tool("calculator")
    def calculator_tool(expression: str) -> str:
        """Evaluate an arithmetic expression (e.g. '(12 + 8) * 3 / 4' or 'sqrt(144)')."""
        return calculate(expression)

    return calculator_tool
