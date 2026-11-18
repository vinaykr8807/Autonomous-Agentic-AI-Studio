import ast
import math
import operator
import re
import statistics
from typing import Any, Dict
from app.tools.base import ToolExecutionError


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "mean": statistics.mean,
    "median": statistics.median,
    "cagr": lambda start, end, years: ((end / start) ** (1 / years) - 1) * 100.0,
    "pct_change": lambda old, new: ((new - old) / old) * 100.0,
}


def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.List):
        return [_eval_ast(el) for el in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_ast(el) for el in node.elts)
    elif isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_ast(node.operand))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function '{func_name}'")
        args = [_eval_ast(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)
    raise ValueError(f"Unsupported syntax node: {ast.dump(node)}")


class CalculatorEvaluatorTool:
    """
    Safe AST-based Quantitative & Budget Calculator Tool.
    Evaluates mathematical expressions, percentages, CAGR, and budget breakdowns safely.
    """

    name = "calculator_evaluator"
    description = (
        "Safely evaluates mathematical, statistical, and financial expressions "
        "(e.g., '3 * 185 + 420', 'pct_change(120, 165)', 'cagr(100, 240, 3)', 'mean([120, 150, 180])')."
    )

    async def execute(self, expression: str) -> Dict[str, Any]:
        raw = (expression or "").strip()
        if not raw:
            raise ToolExecutionError(
                self.name, "InvalidInputError", "Calculator expression cannot be empty."
            )

        # Sanitize common currency symbols or thousands separators (e.g., 1,500 -> 1500) without stripping argument commas
        cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw.replace("$", ""))
        # Extract math expression if LLM passed "expression: 100 + 200"
        if ":" in cleaned and not cleaned.startswith(("cagr", "pct_change", "mean", "sum")):
            cleaned = cleaned.split(":", 1)[-1].strip()

        try:
            tree = ast.parse(cleaned, mode="eval")
            result = _eval_ast(tree)
            if isinstance(result, float):
                result = round(result, 4)
            return {
                "expression": cleaned,
                "result": result,
                "formatted": f"{result:,}" if isinstance(result, (int, float)) else str(result),
            }
        except Exception as exc:
            # Try extracting numbers from the string as fallback if natural language was passed
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]
            if len(nums) >= 2:
                total = round(sum(nums), 2)
                avg = round(statistics.mean(nums), 2)
                return {
                    "expression": raw,
                    "extracted_numbers": nums,
                    "result": total,
                    "formatted": f"Sum={total:,.2f} | Mean={avg:,.2f}",
                    "sum": total,
                    "mean": avg,
                    "min": min(nums),
                    "max": max(nums),
                    "note": "Parsed numerical series from natural language expression.",
                }
            raise ToolExecutionError(
                self.name,
                "ExpressionSyntaxError",
                f"Could not evaluate mathematical expression '{raw}': {exc}",
            )
