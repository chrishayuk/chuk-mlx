#!/usr/bin/env python3
"""
Virtual Expert Architecture for GPT-OSS

This module implements the "capability-aware compression" strategy:
1. KEEP fluency experts (language quality)
2. REMOVE externalizable experts (math, time, APIs)
3. ROUTE to virtual experts (external tools)

Architecture:
```
Input → Embed → [L0-L2 Detector] → Route Decision
                    ↓                    ↓
              Normal Path          Virtual Expert Path
                    ↓                    ↓
              Fluency Experts      External Tool Call
                    ↓                    ↓
                Output    ←←←←    Inject Result
```

Virtual Experts:
- calculator: Arithmetic, symbolic math (sympy)
- datetime: Time, date, duration calculations
- web_api: Weather, stocks, current events
- interpreter: Code execution (sandboxed)
- unit_converter: Unit conversions (pint)

Usage:
    from virtual_expert_architecture import VirtualExpertRouter

    router = VirtualExpertRouter(model, tokenizer)
    response = router.generate("What is 127 * 89?")
    # Uses calculator virtual expert → exact answer
"""

from __future__ import annotations

import re
import ast
import math
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import mlx.core as mx


# =============================================================================
# Virtual Expert Base
# =============================================================================

@dataclass
class VirtualExpertResult:
    """Result from a virtual expert."""
    triggered: bool
    confidence: float
    result: str | None
    tool_name: str
    raw_output: Any = None


class VirtualExpert(ABC):
    """Base class for virtual experts."""

    name: str = "base"
    description: str = "Base virtual expert"

    # Trigger patterns (regex)
    trigger_patterns: list[str] = []

    # Keywords that suggest this expert should handle the input
    trigger_keywords: list[str] = []

    @abstractmethod
    def execute(self, query: str) -> VirtualExpertResult:
        """Execute the virtual expert on a query."""
        pass

    def detect(self, text: str) -> tuple[bool, float]:
        """
        Detect if this expert should handle the input.

        Returns:
            (should_trigger, confidence)
        """
        text_lower = text.lower()

        # Check patterns
        for pattern in self.trigger_patterns:
            if re.search(pattern, text_lower):
                return True, 0.9

        # Check keywords
        keyword_matches = sum(1 for kw in self.trigger_keywords if kw in text_lower)
        if keyword_matches >= 2:
            return True, 0.7 + 0.1 * min(keyword_matches, 3)
        elif keyword_matches == 1:
            return True, 0.5

        return False, 0.0


# =============================================================================
# Calculator Virtual Expert
# =============================================================================

class CalculatorExpert(VirtualExpert):
    """Virtual expert for arithmetic and symbolic math."""

    name = "calculator"
    description = "Handles arithmetic, algebra, and symbolic math"

    trigger_patterns = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Basic arithmetic
        r'calculate\s+',
        r'compute\s+',
        r'what\s+is\s+\d+',
        r'solve\s+for',
        r'sqrt\s*\(',
        r'\d+\s*%\s*of',
        r'factor\s*:',
        r'simplify\s*:',
        r'derivative\s+of',
        r'integral\s+of',
    ]

    trigger_keywords = [
        'calculate', 'compute', 'solve', 'arithmetic',
        'multiply', 'divide', 'add', 'subtract',
        'sqrt', 'square root', 'power', 'exponent',
        'percent', 'percentage', 'factor', 'simplify',
        'derivative', 'integral', 'equation',
    ]

    # Safe operators for eval
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'pow': pow,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'floor': math.floor,
        'ceil': math.ceil,
    }

    def execute(self, query: str) -> VirtualExpertResult:
        """Execute arithmetic calculation."""
        # Extract the mathematical expression
        expr = self._extract_expression(query)

        if not expr:
            return VirtualExpertResult(
                triggered=True,
                confidence=0.3,
                result=None,
                tool_name=self.name,
            )

        try:
            result = self._safe_eval(expr)
            return VirtualExpertResult(
                triggered=True,
                confidence=0.95,
                result=str(result),
                tool_name=self.name,
                raw_output=result,
            )
        except Exception as e:
            # Try sympy for symbolic math
            try:
                result = self._sympy_eval(expr)
                return VirtualExpertResult(
                    triggered=True,
                    confidence=0.9,
                    result=str(result),
                    tool_name=self.name,
                    raw_output=result,
                )
            except:
                return VirtualExpertResult(
                    triggered=True,
                    confidence=0.3,
                    result=f"Could not compute: {expr}",
                    tool_name=self.name,
                )

    def _extract_expression(self, query: str) -> str | None:
        """Extract mathematical expression from query."""
        # Remove common prefixes
        query = re.sub(r'^(calculate|compute|what is|solve|evaluate)\s*:?\s*', '', query.lower())

        # Handle percentage
        query = re.sub(r'(\d+)\s*%\s*of\s*(\d+)', r'(\1/100)*\2', query)

        # Handle "X squared" or "X cubed"
        query = re.sub(r'(\d+)\s*squared', r'\1**2', query)
        query = re.sub(r'(\d+)\s*cubed', r'\1**3', query)

        # Handle sqrt
        query = re.sub(r'sqrt\s*\(?\s*(\d+)\s*\)?', r'sqrt(\1)', query)
        query = re.sub(r'square\s*root\s*of\s*(\d+)', r'sqrt(\1)', query)

        # Handle power notation
        query = re.sub(r'(\d+)\s*\^\s*(\d+)', r'\1**\2', query)
        query = re.sub(r'(\d+)\s*to\s*the\s*power\s*of\s*(\d+)', r'\1**\2', query)

        # Extract the expression (numbers and operators)
        match = re.search(r'[\d\.\+\-\*\/\(\)\s\^sqrt\%]+', query)
        if match:
            expr = match.group().strip()
            # Clean up
            expr = re.sub(r'\s+', '', expr)
            return expr

        return None

    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Parse the expression
        tree = ast.parse(expr, mode='eval')

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # Python 3.7 compat
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op = self.SAFE_OPERATORS.get(type(node.op))
                if op:
                    return op(left, right)
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            elif isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op = self.SAFE_OPERATORS.get(type(node.op))
                if op:
                    return op(operand)
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            elif isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name in self.SAFE_FUNCTIONS:
                    args = [_eval(arg) for arg in node.args]
                    return self.SAFE_FUNCTIONS[func_name](*args)
                raise ValueError(f"Unsupported function: {func_name}")
            elif isinstance(node, ast.Name):
                # Handle constants like pi, e
                if node.id == 'pi':
                    return math.pi
                elif node.id == 'e':
                    return math.e
                raise ValueError(f"Unsupported name: {node.id}")
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        return _eval(tree)

    def _sympy_eval(self, expr: str) -> Any:
        """Evaluate using sympy for symbolic math."""
        try:
            import sympy
            result = sympy.sympify(expr)
            if result.is_number:
                return float(result)
            return result
        except ImportError:
            raise ValueError("sympy not available")


# =============================================================================
# DateTime Virtual Expert
# =============================================================================

class DateTimeExpert(VirtualExpert):
    """Virtual expert for date and time operations."""

    name = "datetime"
    description = "Handles date, time, and duration calculations"

    trigger_patterns = [
        r'what\s+(day|time|date)\s+is',
        r'how\s+many\s+(days|hours|minutes)',
        r'days\s+(until|since|between)',
        r'is\s+\d{4}\s+a\s+leap\s+year',
        r'what\s+(month|year|day)\s+(comes|is)',
    ]

    trigger_keywords = [
        'today', 'tomorrow', 'yesterday', 'current time', 'current date',
        'what day', 'what time', 'what date', 'what month', 'what year',
        'leap year', 'days until', 'days since', 'hours in',
    ]

    def execute(self, query: str) -> VirtualExpertResult:
        """Execute datetime operation."""
        query_lower = query.lower()
        now = datetime.now()

        # What day/time is it?
        if re.search(r'what\s+(day|time|date)\s+is\s+(it|today)', query_lower):
            if 'time' in query_lower:
                result = now.strftime("%I:%M %p")
            elif 'date' in query_lower:
                result = now.strftime("%B %d, %Y")
            else:
                result = now.strftime("%A, %B %d, %Y")
            return VirtualExpertResult(
                triggered=True, confidence=0.95, result=result,
                tool_name=self.name, raw_output=now,
            )

        # Days in month
        if re.search(r'how\s+many\s+days\s+in\s+(\w+)', query_lower):
            # Would implement month lookup
            pass

        # Leap year check
        match = re.search(r'is\s+(\d{4})\s+a\s+leap\s+year', query_lower)
        if match:
            year = int(match.group(1))
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            result = f"Yes, {year} is a leap year" if is_leap else f"No, {year} is not a leap year"
            return VirtualExpertResult(
                triggered=True, confidence=0.95, result=result,
                tool_name=self.name, raw_output=is_leap,
            )

        # Default: return current datetime
        return VirtualExpertResult(
            triggered=True,
            confidence=0.5,
            result=now.strftime("%A, %B %d, %Y at %I:%M %p"),
            tool_name=self.name,
            raw_output=now,
        )


# =============================================================================
# Code Interpreter Virtual Expert
# =============================================================================

class InterpreterExpert(VirtualExpert):
    """Virtual expert for code execution (sandboxed)."""

    name = "interpreter"
    description = "Executes Python code in a sandboxed environment"

    trigger_patterns = [
        r'run\s+(this|the)\s+(code|python)',
        r'execute\s*:',
        r'what\s+(does|is)\s+this\s+output',
        r'what\s+is\s+the\s+result\s+of',
    ]

    trigger_keywords = [
        'run', 'execute', 'output', 'result',
        'print', 'eval', 'code',
    ]

    # Safe builtins for sandboxed execution
    SAFE_BUILTINS = {
        'abs': abs, 'all': all, 'any': any, 'bin': bin,
        'bool': bool, 'chr': chr, 'dict': dict, 'dir': dir,
        'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
        'float': float, 'format': format, 'frozenset': frozenset,
        'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
        'issubclass': issubclass, 'iter': iter, 'len': len,
        'list': list, 'map': map, 'max': max, 'min': min,
        'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
        'range': range, 'repr': repr, 'reversed': reversed,
        'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
        'zip': zip, 'True': True, 'False': False, 'None': None,
    }

    def execute(self, query: str) -> VirtualExpertResult:
        """Execute code in sandbox."""
        # Extract code from query
        code = self._extract_code(query)

        if not code:
            return VirtualExpertResult(
                triggered=True, confidence=0.3, result=None,
                tool_name=self.name,
            )

        try:
            result = self._safe_exec(code)
            return VirtualExpertResult(
                triggered=True, confidence=0.95, result=str(result),
                tool_name=self.name, raw_output=result,
            )
        except Exception as e:
            return VirtualExpertResult(
                triggered=True, confidence=0.8, result=f"Error: {e}",
                tool_name=self.name,
            )

    def _extract_code(self, query: str) -> str | None:
        """Extract Python code from query."""
        # Look for code blocks
        match = re.search(r'```python\s*(.*?)\s*```', query, re.DOTALL)
        if match:
            return match.group(1)

        # Look for inline code
        match = re.search(r'`([^`]+)`', query)
        if match:
            return match.group(1)

        # Look for "run this: X" or "execute: X"
        match = re.search(r'(?:run\s+this|execute)\s*:\s*(.+)', query, re.IGNORECASE)
        if match:
            return match.group(1)

        # Look for function calls like sorted([...]) or len(...)
        match = re.search(r'(\w+\s*\([^)]+\))', query)
        if match:
            return match.group(1)

        # Look for code-like patterns
        match = re.search(r'([\w\[\]\(\)\.\'\"]+\s*(?:\+|\-|\*|\/|==|!=|<|>|<=|>=)\s*[\w\[\]\(\)\.\'\"]+)', query)
        if match:
            return match.group(1)

        return None

    def _safe_exec(self, code: str) -> Any:
        """Execute code in a restricted environment."""
        # Create restricted globals
        safe_globals = {"__builtins__": self.SAFE_BUILTINS}

        # Try eval first (for expressions)
        try:
            return eval(code, safe_globals)
        except SyntaxError:
            pass

        # Try exec (for statements) and capture output
        import io
        import sys

        output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output

        try:
            exec(code, safe_globals)
            return output.getvalue().strip() or "Code executed successfully"
        finally:
            sys.stdout = old_stdout


# =============================================================================
# Unit Converter Virtual Expert
# =============================================================================

class UnitConverterExpert(VirtualExpert):
    """Virtual expert for unit conversions."""

    name = "unit_converter"
    description = "Handles unit conversions"

    trigger_patterns = [
        r'convert\s+\d+',
        r'how\s+many\s+\w+\s+in\s+\d+',
        r'\d+\s*(?:meters?|feet|inches|miles|km|celsius|fahrenheit|pounds?|kg)',
    ]

    trigger_keywords = [
        'convert', 'conversion', 'meters', 'feet', 'inches',
        'miles', 'kilometers', 'celsius', 'fahrenheit',
        'pounds', 'kilograms', 'liters', 'gallons',
        'mph', 'km/h', 'how many',
    ]

    # Simple conversion factors (singular forms)
    CONVERSIONS = {
        ('meter', 'foot'): 3.28084,
        ('meter', 'feet'): 3.28084,
        ('foot', 'meter'): 0.3048,
        ('feet', 'meter'): 0.3048,
        ('mile', 'kilometer'): 1.60934,
        ('kilometer', 'mile'): 0.621371,
        ('pound', 'kilogram'): 0.453592,
        ('kilogram', 'pound'): 2.20462,
        ('gallon', 'liter'): 3.78541,
        ('liter', 'gallon'): 0.264172,
        ('inch', 'centimeter'): 2.54,
        ('centimeter', 'inch'): 0.393701,
        ('mph', 'kmh'): 1.60934,
        ('kmh', 'mph'): 0.621371,
    }

    def execute(self, query: str) -> VirtualExpertResult:
        """Execute unit conversion."""
        # Parse query for value and units
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)\s+(?:to|in)\s+(\w+)', query.lower())

        if match:
            value = float(match.group(1))
            from_unit = match.group(2).rstrip('s')  # Remove plural
            to_unit = match.group(3).rstrip('s')

            # Temperature special case
            if 'fahrenheit' in from_unit or 'celsius' in from_unit:
                if 'fahrenheit' in from_unit:
                    result = (value - 32) * 5/9
                    return VirtualExpertResult(
                        triggered=True, confidence=0.95,
                        result=f"{value}°F = {result:.1f}°C",
                        tool_name=self.name, raw_output=result,
                    )
                else:
                    result = value * 9/5 + 32
                    return VirtualExpertResult(
                        triggered=True, confidence=0.95,
                        result=f"{value}°C = {result:.1f}°F",
                        tool_name=self.name, raw_output=result,
                    )

            # Standard conversions
            key = (from_unit, to_unit)
            if key in self.CONVERSIONS:
                result = value * self.CONVERSIONS[key]
                return VirtualExpertResult(
                    triggered=True, confidence=0.95,
                    result=f"{value} {from_unit} = {result:.2f} {to_unit}",
                    tool_name=self.name, raw_output=result,
                )

        return VirtualExpertResult(
            triggered=True, confidence=0.3, result=None,
            tool_name=self.name,
        )


# =============================================================================
# Virtual Expert Router
# =============================================================================

class VirtualExpertRouter:
    """
    Routes queries to virtual experts based on detected task type.

    This is the main interface for the virtual expert system.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

        # Initialize virtual experts
        self.experts: list[VirtualExpert] = [
            CalculatorExpert(),
            DateTimeExpert(),
            InterpreterExpert(),
            UnitConverterExpert(),
        ]

    def detect_task(self, text: str) -> tuple[VirtualExpert | None, float]:
        """
        Detect which virtual expert should handle this input.

        Returns:
            (expert, confidence) or (None, 0) if no expert detected
        """
        best_expert = None
        best_confidence = 0.0

        for expert in self.experts:
            triggered, confidence = expert.detect(text)
            if triggered and confidence > best_confidence:
                best_expert = expert
                best_confidence = confidence

        return best_expert, best_confidence

    def route(self, query: str, confidence_threshold: float = 0.5) -> VirtualExpertResult | None:
        """
        Route a query to the appropriate virtual expert.

        Returns:
            VirtualExpertResult if routed to expert, None if should use LLM
        """
        expert, confidence = self.detect_task(query)

        if expert and confidence >= confidence_threshold:
            return expert.execute(query)

        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        use_virtual_experts: bool = True,
    ) -> str:
        """
        Generate response, using virtual experts when appropriate.
        """
        if use_virtual_experts:
            result = self.route(prompt)
            if result and result.result:
                return result.result

        # Fall back to LLM
        if self.model and self.tokenizer:
            return self._generate_with_llm(prompt, max_tokens)

        return "No model available and no virtual expert matched."

    def _generate_with_llm(self, prompt: str, max_tokens: int) -> str:
        """Generate using the LLM."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(max_tokens):
            logits = self.model(input_ids)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            next_token_id = int(next_token.item())
            generated.append(next_token_id)

            if next_token_id == self.tokenizer.eos_token_id:
                break

            input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
            mx.eval(input_ids)

        return self.tokenizer.decode(generated)


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demonstrate virtual expert routing."""
    print("=" * 70)
    print("VIRTUAL EXPERT ROUTER DEMO")
    print("=" * 70)
    print()

    router = VirtualExpertRouter()

    test_queries = [
        # Calculator
        ("127 * 89 = ", "calculator"),
        ("What is 15% of 200?", "calculator"),
        ("Calculate sqrt(144)", "calculator"),
        ("2^10 = ", "calculator"),

        # DateTime
        ("What day is today?", "datetime"),
        ("Is 2024 a leap year?", "datetime"),

        # Interpreter
        ("Run this: sorted([3,1,4,1,5,9])", "interpreter"),
        ("What is the result of len('hello')?", "interpreter"),

        # Unit conversion
        ("Convert 100 meters to feet", "unit_converter"),
        ("Convert 32 fahrenheit to celsius", "unit_converter"),

        # Should NOT trigger (fluency tasks)
        ("Once upon a time, in a land far away,", None),
        ("The capital of France is", None),
        ("Explain how neural networks learn", None),
    ]

    for query, expected in test_queries:
        result = router.route(query)

        if result and result.triggered:
            status = "✓" if result.tool_name == expected else "?"
            print(f"{status} [{result.tool_name}] {query}")
            print(f"  → {result.result}")
        else:
            status = "✓" if expected is None else "✗"
            print(f"{status} [LLM] {query}")
            print(f"  → (would use language model)")
        print()


if __name__ == "__main__":
    demo()
