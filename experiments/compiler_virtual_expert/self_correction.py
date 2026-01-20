"""
Self-Correction Loop.

Implements iterative code generation with compiler feedback:
1. Model generates code
2. Compiler executes and detects errors
3. Error feedback injected into context
4. Model generates fix
5. Repeat until success or max iterations

This demonstrates the tight feedback loop between neural generation
and symbolic verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx

from compiler_plugin import CompilerExpertPlugin, ExecutionResult


@dataclass
class CorrectionAttempt:
    """A single attempt in the correction loop."""

    iteration: int
    code: str
    execution_result: str
    success: bool
    error_type: str | None = None  # syntax, runtime, test_failure, None if success


@dataclass
class CorrectionResult:
    """Full result of a self-correction loop."""

    task: str
    success: bool
    iterations: int
    max_iterations: int
    final_code: str | None
    attempts: list[CorrectionAttempt] = field(default_factory=list)
    test_cases: list[tuple] | None = None

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        lines = [
            f"{status} after {self.iterations}/{self.max_iterations} iterations",
            f"Task: {self.task}",
            "",
        ]

        for attempt in self.attempts:
            marker = "✓" if attempt.success else "✗"
            lines.append(f"[{attempt.iteration}] {marker} {attempt.error_type or 'passed'}")
            lines.append(f"    Code: {attempt.code[:60]}...")
            if not attempt.success:
                lines.append(f"    Error: {attempt.execution_result[:80]}...")
            lines.append("")

        if self.success and self.final_code:
            lines.append("Final working code:")
            lines.append(self.final_code)

        return "\n".join(lines)


class SelfCorrectionLoop:
    """
    Orchestrates the generate → verify → fix cycle.

    The loop:
    1. Generate initial code from task description
    2. Execute with compiler expert
    3. If error, inject feedback and prompt for fix
    4. Repeat until success or max iterations
    """

    def __init__(
        self,
        model,
        tokenizer,
        compiler: CompilerExpertPlugin | None = None,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.compiler = compiler or CompilerExpertPlugin()
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _generate(
        self,
        prompt: str,
        max_tokens: int = 300,
        stop_at_code_block_end: bool = True,
    ) -> str:
        """Generate text from prompt."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[0, -1, :])
            token_id = int(next_token.item())

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            # Check for code block end
            if stop_at_code_block_end:
                text = self.tokenizer.decode(generated)
                # Count ``` - if we have an even number >= 2, we have a complete block
                backtick_count = text.count("```")
                if backtick_count >= 2 and backtick_count % 2 == 0:
                    # Check if the last ``` is actually at the end
                    if text.rstrip().endswith("```"):
                        break

        return self.tokenizer.decode(generated)

    def _extract_code(self, text: str) -> str | None:
        """Extract code from markdown code block."""
        pattern = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None

    def _classify_error(self, result: str) -> str | None:
        """Classify the type of error from compiler output."""
        if "✓" in result:
            return None
        if "SyntaxError" in result:
            return "syntax"
        if "RuntimeError" in result or "Error:" in result:
            return "runtime"
        if "FAILED" in result or "expected" in result:
            return "test_failure"
        return "unknown"

    def _build_initial_prompt(self, task: str) -> str:
        """Build the initial prompt for code generation."""
        return f"""Write a Python function to {task}.

```python
"""

    def _build_correction_prompt(
        self,
        task: str,
        previous_code: str,
        error: str,
        iteration: int,
    ) -> str:
        """Build a prompt asking the model to fix the error."""
        # Parse error to give more specific guidance
        guidance = ""
        if "not defined" in error:
            # Variable name error
            import re
            match = re.search(r"name '(\w+)' is not defined", error)
            if match:
                var = match.group(1)
                guidance = f"The variable '{var}' is not defined. Check for typos in variable names."
        elif "RecursionError" in error:
            guidance = "The function has infinite recursion. Add a base case."
        elif "SyntaxError" in error:
            guidance = "There is a syntax error. Check brackets, colons, and indentation."
        elif "IndexError" in error:
            guidance = "List index out of range. Check array bounds."
        elif "TypeError" in error:
            guidance = "Type mismatch. Check argument types and return values."
        elif "expected" in error.lower():
            guidance = "The function returns wrong values. Check the logic."

        return f"""The following code has a bug:

```python
{previous_code}
```

Error: {error}

{guidance}

Write a FIXED version that correctly {task}:

```python
"""

    def run(
        self,
        task: str,
        test_cases: list[tuple] | None = None,
    ) -> CorrectionResult:
        """
        Run the self-correction loop.

        Args:
            task: Description of what the code should do
            test_cases: Optional list of (input, expected_output) for testing

        Returns:
            CorrectionResult with full history
        """
        attempts: list[CorrectionAttempt] = []
        current_code: str | None = None

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print("=" * 60)

            # Generate code
            if iteration == 0:
                prompt = self._build_initial_prompt(task)
            else:
                prompt = self._build_correction_prompt(
                    task,
                    current_code,
                    attempts[-1].execution_result,
                    iteration,
                )

            if self.verbose:
                print(f"\nPrompt: {prompt[:100]}...")

            generated = self._generate(prompt)

            if self.verbose:
                print(f"\nGenerated: {generated[:200]}...")

            # Extract code
            code = self._extract_code(f"```python\n{generated}")
            if not code:
                # Maybe the model didn't include closing ```
                code = generated.strip()
                if code.endswith("```"):
                    code = code[:-3].strip()

            current_code = code

            if self.verbose:
                print(f"\nExtracted code:\n{code}")

            # Execute
            if test_cases:
                exec_result = self.compiler.extract_function_and_test(code, test_cases)
                result_str = exec_result.output or str(exec_result.error)
                success = exec_result.success
            else:
                # Just check if it runs
                full_code = f"```python\n{code}\n```"
                result_str = self.compiler.execute(full_code) or "No result"
                success = "✓" in result_str

            error_type = self._classify_error(result_str)

            if self.verbose:
                print(f"\nExecution result: {result_str}")
                print(f"Success: {success}")

            attempts.append(
                CorrectionAttempt(
                    iteration=iteration + 1,
                    code=code,
                    execution_result=result_str,
                    success=success,
                    error_type=error_type,
                )
            )

            if success:
                return CorrectionResult(
                    task=task,
                    success=True,
                    iterations=iteration + 1,
                    max_iterations=self.max_iterations,
                    final_code=code,
                    attempts=attempts,
                    test_cases=test_cases,
                )

        # Failed after all iterations
        return CorrectionResult(
            task=task,
            success=False,
            iterations=self.max_iterations,
            max_iterations=self.max_iterations,
            final_code=current_code,
            attempts=attempts,
            test_cases=test_cases,
        )


# =============================================================================
# DEMO TASKS
# =============================================================================

DEMO_TASKS = [
    {
        "task": "calculate the factorial of a number",
        "test_cases": [((0,), 1), ((1,), 1), ((5,), 120), ((10,), 3628800)],
    },
    {
        "task": "reverse a string",
        "test_cases": [(("hello",), "olleh"), (("",), ""), (("a",), "a")],
    },
    {
        "task": "find the maximum value in a list",
        "test_cases": [(([1, 3, 2],), 3), (([5],), 5), (([-1, -5, -2],), -1)],
    },
    {
        "task": "check if a number is prime",
        "test_cases": [((2,), True), ((3,), True), ((4,), False), ((17,), True), ((1,), False)],
    },
    {
        "task": "count the vowels in a string",
        "test_cases": [(("hello",), 2), (("xyz",), 0), (("aeiou",), 5)],
    },
]


# =============================================================================
# SIMULATED SELF-CORRECTION (without model)
# =============================================================================


def simulate_self_correction():
    """
    Simulate the self-correction loop without a model.

    Shows the flow with hand-crafted buggy → fixed code.
    """
    print("=" * 70)
    print("SELF-CORRECTION LOOP - Simulation")
    print("=" * 70)

    compiler = CompilerExpertPlugin()

    # Simulate: Model generates buggy factorial, then fixes it
    task = "calculate the factorial of a number"
    test_cases = [((0,), 1), ((1,), 1), ((5,), 120)]

    print(f"\nTask: {task}")
    print(f"Test cases: {test_cases}")

    # Iteration 1: Buggy code (missing base case)
    print("\n" + "-" * 60)
    print("Iteration 1: Model generates buggy code")
    print("-" * 60)

    code1 = """def factorial(n):
    return n * factorial(n - 1)"""

    print(f"Code:\n{code1}")

    result1 = compiler.extract_function_and_test(code1, test_cases)
    print(f"\nCompiler says: {result1.output or result1.error}")
    print(f"Success: {result1.success}")

    # Iteration 2: Model sees error, generates fix
    print("\n" + "-" * 60)
    print("Iteration 2: Model sees error and fixes")
    print("-" * 60)

    print("\nFeedback to model:")
    print(f"  Error: {result1.error or result1.output}")
    print("  Model reasons: 'RecursionError means I need a base case'")

    code2 = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

    print(f"\nFixed code:\n{code2}")

    result2 = compiler.extract_function_and_test(code2, test_cases)
    print(f"\nCompiler says: {result2.output}")
    print(f"Success: {result2.success}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Task: {task}
Iterations: 2
Result: SUCCESS

Flow:
  1. Model generated: factorial without base case
     Error: RecursionError

  2. Model saw error, reasoned about fix
     Generated: factorial with base case
     Result: All tests passed ✓

This is the self-correction loop:
  generate → verify → feedback → fix → verify → success
""")


def run_with_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Run self-correction with actual model."""
    print("=" * 70)
    print("SELF-CORRECTION LOOP - With Model")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_name}...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(model_name)
    model = loaded.model
    tokenizer = loaded.tokenizer
    mx.eval(model.parameters())
    print("Model loaded.")

    # Create loop
    loop = SelfCorrectionLoop(
        model=model,
        tokenizer=tokenizer,
        max_iterations=5,
        verbose=True,
    )

    # Run on demo tasks
    results = []
    for demo in DEMO_TASKS[:2]:  # Just first 2 for demo
        print("\n" + "=" * 70)
        print(f"TASK: {demo['task']}")
        print("=" * 70)

        result = loop.run(
            task=demo["task"],
            test_cases=demo["test_cases"],
        )

        results.append(result)
        print("\n" + result.summary())

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    successes = sum(1 for r in results if r.success)
    total_iterations = sum(r.iterations for r in results)

    print(f"Tasks completed: {successes}/{len(results)}")
    print(f"Total iterations: {total_iterations}")
    print(f"Average iterations per task: {total_iterations / len(results):.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        model_name = sys.argv[2] if len(sys.argv) > 2 else None
        run_with_model(model_name) if model_name else run_with_model()
    else:
        # Default: simulation mode
        simulate_self_correction()
