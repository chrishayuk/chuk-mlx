"""
Math Expert Plugin.

Provides exact arithmetic computation by routing math expressions
to Python's eval (safely via AST parsing).
"""

from __future__ import annotations

import ast
import math
import operator
import re

from ..base import VirtualExpertPlugin


class MathExpertPlugin(VirtualExpertPlugin):
    """
    Virtual expert for mathematical computation.

    Routes arithmetic expressions to Python for exact computation.
    Uses AST parsing for safe evaluation (no arbitrary code execution).

    Supported operations:
    - Arithmetic: +, -, *, /, //, %, **
    - Functions: sqrt, sin, cos, tan, log, exp, abs, round, min, max
    - Constants: pi, e, inf

    Example:
        >>> plugin = MathExpertPlugin()
        >>> plugin.execute("127 * 89 = ")
        '11303'
        >>> plugin.execute("sqrt(144)")
        '12.0'
    """

    name = "math"
    description = "Computes arithmetic expressions using Python"
    priority = 10

    OPERATORS = {
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

    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'pow': pow,
        'floor': math.floor,
        'ceil': math.ceil,
    }

    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'inf': float('inf'),
    }

    def can_handle(self, prompt: str) -> bool:
        """Check if prompt contains a computable math expression."""
        # Look for patterns like "X op Y =" or "X op Y"
        pattern = r'\d+\s*[+\-*/]\s*\d+\s*=?\s*$'
        return bool(re.search(pattern, prompt.strip()))

    def execute(self, prompt: str) -> str | None:
        """Evaluate the math expression."""
        result = self._evaluate(prompt)
        if result is not None:
            # Format as integer if whole number
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)
        return None

    def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
        """Return math vs non-math prompts for calibration."""
        positive = [
            "127 * 89 = ",
            "456 + 789 = ",
            "1000 - 250 = ",
            "What is 99 * 99?",
            "Calculate 144 / 12",
            "25 * 25 = ",
            "100 + 200 = ",
            "50 - 17 = ",
        ]
        negative = [
            "The capital of France is",
            "Hello, how are you today?",
            "Once upon a time in a land",
            "The quick brown fox jumps",
            "In the beginning, there was",
            "My favorite color is",
            "The weather today is",
            "I think that we should",
        ]
        return positive, negative

    def _evaluate(self, expr: str) -> float | int | None:
        """Safely evaluate a mathematical expression."""
        try:
            expr = self._clean_expression(expr)
            if not expr:
                return None
            tree = ast.parse(expr, mode='eval')
            return self._eval_node(tree.body)
        except Exception:
            return None

    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize expression."""
        # Remove common prefixes
        expr = re.sub(r'(?:what is|calculate|compute|evaluate|=\s*$)', '', expr, flags=re.I)

        # Normalize operators
        replacements = {'×': '*', '÷': '/', '^': '**', '√': 'sqrt'}
        for old, new in replacements.items():
            expr = expr.replace(old, new)

        # Extract numeric expression
        match = re.search(r'[\d\s+\-*/().]+', expr)
        if match:
            return match.group().strip()
        return expr.strip()

    def _eval_node(self, node: ast.AST) -> float | int:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Invalid constant: {node.value}")

        elif isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            raise ValueError(f"Unknown name: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
            raise ValueError(f"Unsupported function: {ast.dump(node.func)}")

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def extract_and_evaluate(self, text: str) -> tuple[str | None, float | int | None]:
        """
        Extract mathematical expression from text and evaluate.

        Args:
            text: Input text that may contain a math expression

        Returns:
            (expression, result) tuple
        """
        patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/^×÷])\s*(\d+(?:\.\d+)?)',
            r'what\s+is\s+(\d+(?:\.\d+)?)\s*([+\-*/^×÷])\s*(\d+(?:\.\d+)?)',
            r'calculate\s+(\d+(?:\.\d+)?)\s*([+\-*/^×÷])\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    a, op, b = groups
                    expr = f"{a} {op} {b}"
                    result = self._evaluate(expr)
                    return expr, result

        result = self._evaluate(text)
        return text if result is not None else None, result


# Backwards compatibility alias
SafeMathEvaluator = MathExpertPlugin
