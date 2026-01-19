"""
Compiler Expert Plugin.

Verifies and executes model-generated code, returning results or errors
for injection back into generation. Enables tight feedback loops where
the model can self-correct based on execution results.
"""

from __future__ import annotations

import ast
import re
import sys
import traceback
from dataclasses import dataclass
from io import StringIO
from typing import Any

# Import base class - adjust path as needed for your setup
try:
    from chuk_lazarus.inference.virtual_experts.base import VirtualExpertPlugin
except ImportError:
    # Standalone mode for testing
    from abc import ABC, abstractmethod

    class VirtualExpertPlugin(ABC):
        name: str = "base"
        description: str = "Base virtual expert"
        priority: int = 0

        @abstractmethod
        def can_handle(self, prompt: str) -> bool:
            pass

        @abstractmethod
        def execute(self, prompt: str) -> str | None:
            pass

        @abstractmethod
        def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
            pass


@dataclass
class ExecutionResult:
    """Result from code execution."""

    success: bool
    output: str | None = None
    error: str | None = None
    return_value: Any = None
    execution_time_ms: float | None = None


class CompilerExpertPlugin(VirtualExpertPlugin):
    """
    Virtual expert for code verification and execution.

    Detects completed code blocks, parses them, executes in a sandbox,
    and returns results/errors for feedback into generation.

    Supported:
    - Python code blocks (```python ... ```)
    - Bare code blocks (``` ... ```)
    - Syntax validation via AST
    - Sandboxed execution with timeout

    Example:
        >>> plugin = CompilerExpertPlugin()
        >>> code = '''```python
        ... def add(a, b):
        ...     return a + b
        ... print(add(2, 3))
        ... ```'''
        >>> plugin.execute(code)
        '✓ Executed successfully\\nOutput: 5'
    """

    name = "compiler"
    description = "Executes code and returns results/errors"
    priority = 8  # Lower than math (10), but still high

    # Allowed builtins for sandboxed execution
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }

    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        r"\bimport\s+os\b",
        r"\bimport\s+sys\b",
        r"\bimport\s+subprocess\b",
        r"\b__import__\b",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bopen\s*\(",
        r"\bfile\s*\(",
        r"\b__builtins__\b",
        r"\b__class__\b",
        r"\b__subclasses__\b",
        r"\bgetattr\b",
        r"\bsetattr\b",
        r"\bdelattr\b",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
    ]

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds

    def can_handle(self, prompt: str) -> bool:
        """Check if prompt contains a completed code block."""
        # Look for closed markdown code blocks
        # Pattern: ```[language]\n...code...\n```
        pattern = r"```(?:python|py)?\s*\n.*?```"
        matches = re.findall(pattern, prompt, re.DOTALL | re.IGNORECASE)

        # Must have at least one complete code block
        if not matches:
            return False

        # Check that the last code block is complete (not still being written)
        # A complete block ends with ``` followed by whitespace or end of string
        last_block_end = prompt.rfind("```")
        if last_block_end == -1:
            return False

        # Check if there's a matching opening
        code_before_last = prompt[:last_block_end]
        open_count = code_before_last.count("```")

        # Odd number of ``` before the last one means this closes a block
        return open_count % 2 == 1

    def execute(self, prompt: str) -> str | None:
        """
        Extract code from prompt, validate, execute, and return results.
        """
        # Extract the last code block
        code = self._extract_last_code_block(prompt)
        if not code:
            return None

        # Security check
        if self._is_dangerous(code):
            return "✗ Code contains potentially dangerous operations"

        # Syntax check
        syntax_result = self._check_syntax(code)
        if not syntax_result.success:
            return f"✗ SyntaxError: {syntax_result.error}"

        # Execute
        exec_result = self._execute_sandboxed(code)

        if exec_result.success:
            output_parts = ["✓ Executed successfully"]
            if exec_result.output:
                output_parts.append(f"Output:\n{exec_result.output}")
            if exec_result.return_value is not None:
                output_parts.append(f"Return: {exec_result.return_value}")
            return "\n".join(output_parts)
        else:
            return f"✗ RuntimeError: {exec_result.error}"

    def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
        """
        Return code execution vs non-code prompts for calibration.

        Positive: Completed code blocks that should be executed
        Negative: Code explanations, incomplete blocks, non-code
        """
        positive = [
            "```python\ndef add(a, b):\n    return a + b\nprint(add(2, 3))\n```",
            "Here's the solution:\n```python\nresult = sum(range(10))\nprint(result)\n```",
            "```\nfor i in range(5):\n    print(i)\n```",
            "Let me test this:\n```python\nx = [3, 1, 2]\nprint(sorted(x))\n```",
            "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nprint(factorial(5))\n```",
            "The function:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\nprint(greet('World'))\n```",
        ]

        negative = [
            "In Python, you can define a function like this:",
            "The syntax for a for loop is `for i in range(n):`",
            "```python\n# This is just a comment",  # Incomplete block
            "What does this code do? `print('hello')`",
            "127 * 89 =",
            "The capital of France is",
            "Let me explain how sorting works...",
            "In programming, a function is defined using `def`",
        ]

        return positive, negative

    def _extract_last_code_block(self, text: str) -> str | None:
        """Extract the last complete code block from text."""
        # Find all code blocks
        pattern = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[-1].strip()
        return None

    def _is_dangerous(self, code: str) -> bool:
        """Check if code contains dangerous patterns."""
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False

    def _check_syntax(self, code: str) -> ExecutionResult:
        """Validate Python syntax via AST parsing."""
        try:
            ast.parse(code)
            return ExecutionResult(success=True)
        except SyntaxError as e:
            error_msg = f"{e.msg} (line {e.lineno})"
            return ExecutionResult(success=False, error=error_msg)

    def _execute_sandboxed(self, code: str) -> ExecutionResult:
        """
        Execute code in a restricted environment.

        Uses a limited set of builtins and captures stdout.
        """
        import time

        start = time.perf_counter()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Build safe builtins dict - handle both dict and module forms
        import builtins
        safe_builtins = {}
        for name in self.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        restricted_globals = {"__builtins__": safe_builtins}

        try:
            # Execute with timeout would require threading/multiprocessing
            # For now, just execute directly (add timeout later)
            exec(code, restricted_globals)

            elapsed = (time.perf_counter() - start) * 1000
            output = captured_output.getvalue()

            return ExecutionResult(
                success=True,
                output=output.strip() if output.strip() else None,
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"

            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=elapsed,
            )

        finally:
            sys.stdout = old_stdout

    def extract_function_and_test(
        self,
        code: str,
        test_inputs: list[tuple] | None = None,
    ) -> ExecutionResult:
        """
        Execute code, find the main function, and run test cases.

        Args:
            code: Python code containing a function definition
            test_inputs: Optional list of (args, expected_output) tuples

        Returns:
            ExecutionResult with test results
        """
        # First, check syntax
        syntax_result = self._check_syntax(code)
        if not syntax_result.success:
            return syntax_result

        # Security check
        if self._is_dangerous(code):
            return ExecutionResult(success=False, error="Dangerous code detected")

        # Execute to define functions
        import builtins
        namespace: dict[str, Any] = {}
        safe_builtins = {}
        for name in self.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)
        namespace["__builtins__"] = safe_builtins

        try:
            exec(code, namespace)
        except Exception as e:
            return ExecutionResult(success=False, error=f"Execution failed: {e}")

        # Find defined functions
        functions = {
            name: obj
            for name, obj in namespace.items()
            if callable(obj) and not name.startswith("_")
        }

        if not functions:
            return ExecutionResult(
                success=True,
                output="Code executed but no functions defined",
            )

        # If test inputs provided, run tests
        if test_inputs:
            func_name = list(functions.keys())[0]
            func = functions[func_name]

            results = []
            all_passed = True

            for args, expected in test_inputs:
                try:
                    if isinstance(args, tuple):
                        actual = func(*args)
                    else:
                        actual = func(args)

                    if actual == expected:
                        results.append(f"✓ {func_name}{args} = {actual}")
                    else:
                        results.append(f"✗ {func_name}{args} = {actual}, expected {expected}")
                        all_passed = False
                except Exception as e:
                    results.append(f"✗ {func_name}{args} raised {type(e).__name__}: {e}")
                    all_passed = False

            return ExecutionResult(
                success=all_passed,
                output="\n".join(results),
            )

        return ExecutionResult(
            success=True,
            output=f"Defined functions: {list(functions.keys())}",
        )


# Convenience function for quick testing
def test_code(code: str) -> str:
    """Quick test of code execution."""
    plugin = CompilerExpertPlugin()

    # Wrap in code block if not already
    if "```" not in code:
        code = f"```python\n{code}\n```"

    if plugin.can_handle(code):
        return plugin.execute(code) or "No result"
    else:
        return "Code block not detected or incomplete"


if __name__ == "__main__":
    # Quick tests
    print("=" * 60)
    print("COMPILER EXPERT PLUGIN - Quick Test")
    print("=" * 60)

    plugin = CompilerExpertPlugin()

    # Test 1: Simple execution
    test1 = """
```python
def add(a, b):
    return a + b
print(add(2, 3))
```
"""
    print("\nTest 1: Simple function")
    print(f"Input: {test1.strip()[:50]}...")
    print(f"Can handle: {plugin.can_handle(test1)}")
    print(f"Result: {plugin.execute(test1)}")

    # Test 2: Syntax error
    test2 = """
```python
def broken(
    return x
```
"""
    print("\nTest 2: Syntax error")
    print(f"Can handle: {plugin.can_handle(test2)}")
    print(f"Result: {plugin.execute(test2)}")

    # Test 3: Runtime error
    test3 = """
```python
x = 1 / 0
```
"""
    print("\nTest 3: Runtime error")
    print(f"Can handle: {plugin.can_handle(test3)}")
    print(f"Result: {plugin.execute(test3)}")

    # Test 4: Dangerous code (should reject)
    test4 = """
```python
import os
os.system('rm -rf /')
```
"""
    print("\nTest 4: Dangerous code")
    print(f"Can handle: {plugin.can_handle(test4)}")
    print(f"Result: {plugin.execute(test4)}")

    # Test 5: With function testing
    test5_code = """
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)
"""
    print("\nTest 5: Function with test cases")
    result = plugin.extract_function_and_test(
        test5_code,
        test_inputs=[
            ((0,), 1),
            ((1,), 1),
            ((5,), 120),
            ((10,), 3628800),
        ],
    )
    print(f"Result: {result.output}")

    # Test calibration prompts
    print("\n" + "=" * 60)
    print("Calibration Prompts")
    print("=" * 60)
    pos, neg = plugin.get_calibration_prompts()
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")
