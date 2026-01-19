"""
Benchmark: Compiler Virtual Expert vs Model-Only

Compares code generation quality with and without the compiler expert
providing feedback during generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from compiler_plugin import CompilerExpertPlugin


@dataclass
class BenchmarkTask:
    """A code generation task for benchmarking."""

    prompt: str
    expected_output: str | None = None  # Expected stdout
    test_cases: list[tuple] | None = None  # (input, expected) pairs
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result for a single benchmark task."""

    task: str
    model_output: str
    syntax_valid: bool
    execution_success: bool
    output_correct: bool
    tests_passed: int
    tests_total: int
    compiler_feedback: str | None


# Benchmark tasks
TASKS = [
    BenchmarkTask(
        prompt="Write a Python function to add two numbers:\n```python\n",
        description="Simple function",
        test_cases=[((2, 3), 5), ((0, 0), 0), ((-1, 1), 0)],
    ),
    BenchmarkTask(
        prompt="Write a function to find the maximum in a list:\n```python\n",
        description="List maximum",
        test_cases=[(([1, 3, 2],), 3), (([5],), 5), (([-1, -5, -2],), -1)],
    ),
    BenchmarkTask(
        prompt="Write a function to reverse a string:\n```python\n",
        description="String reversal",
        test_cases=[(("hello",), "olleh"), (("",), ""), (("a",), "a")],
    ),
    BenchmarkTask(
        prompt="Write a function to check if a number is even:\n```python\n",
        description="Even check",
        test_cases=[((2,), True), ((3,), False), ((0,), True)],
    ),
    BenchmarkTask(
        prompt="Write a function to calculate factorial:\n```python\n",
        description="Factorial",
        test_cases=[((0,), 1), ((1,), 1), ((5,), 120)],
    ),
    BenchmarkTask(
        prompt="Write a function to count vowels in a string:\n```python\n",
        description="Vowel count",
        test_cases=[(("hello",), 2), (("xyz",), 0), (("aeiou",), 5)],
    ),
    BenchmarkTask(
        prompt="Write a function to find the sum of a list:\n```python\n",
        description="List sum",
        test_cases=[(([1, 2, 3],), 6), (([],), 0), (([-1, 1],), 0)],
    ),
    BenchmarkTask(
        prompt="Write a function to check if a string is a palindrome:\n```python\n",
        description="Palindrome check",
        test_cases=[(("racecar",), True), (("hello",), False), (("",), True)],
    ),
]


def evaluate_code(code: str, test_cases: list[tuple] | None) -> BenchmarkResult:
    """Evaluate generated code against test cases."""
    plugin = CompilerExpertPlugin()

    # Wrap in code block
    full_code = f"```python\n{code}\n```"

    # Check syntax
    syntax_valid = plugin.can_handle(full_code)
    if not syntax_valid:
        return BenchmarkResult(
            task="",
            model_output=code,
            syntax_valid=False,
            execution_success=False,
            output_correct=False,
            tests_passed=0,
            tests_total=len(test_cases) if test_cases else 0,
            compiler_feedback="Code block not properly formed",
        )

    # Execute
    exec_result = plugin.execute(full_code)
    execution_success = exec_result is not None and "✓" in exec_result

    if not execution_success:
        return BenchmarkResult(
            task="",
            model_output=code,
            syntax_valid=True,
            execution_success=False,
            output_correct=False,
            tests_passed=0,
            tests_total=len(test_cases) if test_cases else 0,
            compiler_feedback=exec_result,
        )

    # Run test cases
    if test_cases:
        result = plugin.extract_function_and_test(code, test_cases)
        tests_passed = result.output.count("✓") if result.output else 0
        tests_total = len(test_cases)
        output_correct = tests_passed == tests_total
    else:
        tests_passed = 0
        tests_total = 0
        output_correct = True  # No tests to fail

    return BenchmarkResult(
        task="",
        model_output=code,
        syntax_valid=True,
        execution_success=True,
        output_correct=output_correct,
        tests_passed=tests_passed,
        tests_total=tests_total,
        compiler_feedback=exec_result,
    )


def run_benchmark_standalone():
    """
    Run benchmark in standalone mode (without model).

    Tests the compiler expert on hand-written code samples
    to validate the evaluation pipeline.
    """
    print("=" * 70)
    print("COMPILER EXPERT BENCHMARK - Standalone Mode")
    print("=" * 70)

    # Test with known-good implementations
    good_implementations = {
        "add": "def add(a, b):\n    return a + b",
        "max": "def find_max(lst):\n    return max(lst)",
        "reverse": "def reverse(s):\n    return s[::-1]",
        "is_even": "def is_even(n):\n    return n % 2 == 0",
        "factorial": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "list_sum": "def list_sum(lst):\n    return sum(lst)",
        "is_palindrome": "def is_palindrome(s):\n    return s == s[::-1]",
    }

    # Test with buggy implementations
    buggy_implementations = {
        "add": "def add(a, b):\n    return a - b",  # Wrong operation
        "max": "def find_max(lst):\n    return lst[0]",  # Returns first, not max
        "reverse": "def reverse(s):\n    return s",  # Doesn't reverse
        "factorial": "def factorial(n):\n    return n * factorial(n-1)",  # Missing base case
    }

    print("\n1. Testing good implementations...")
    print("-" * 70)

    good_results = []
    for i, (name, code) in enumerate(good_implementations.items()):
        task = TASKS[i]
        result = evaluate_code(code, task.test_cases)
        result.task = task.description
        good_results.append(result)

        status = "✓" if result.output_correct else "✗"
        print(
            f"{status} {name}: {result.tests_passed}/{result.tests_total} tests passed"
        )

    print("\n2. Testing buggy implementations...")
    print("-" * 70)

    buggy_results = []
    for name, code in buggy_implementations.items():
        # Find matching task
        task_idx = list(good_implementations.keys()).index(name)
        task = TASKS[task_idx]
        result = evaluate_code(code, task.test_cases)
        result.task = task.description
        buggy_results.append(result)

        status = "✓" if result.output_correct else "✗"
        print(
            f"{status} {name}: {result.tests_passed}/{result.tests_total} tests passed"
        )
        if not result.output_correct:
            print(f"   Feedback: {result.compiler_feedback}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    good_pass_rate = sum(1 for r in good_results if r.output_correct) / len(
        good_results
    )
    buggy_fail_rate = sum(1 for r in buggy_results if not r.output_correct) / len(
        buggy_results
    )

    print(f"\nGood implementations pass rate: {good_pass_rate:.1%}")
    print(f"Buggy implementations caught:   {buggy_fail_rate:.1%}")

    if good_pass_rate == 1.0 and buggy_fail_rate == 1.0:
        print("\n*** COMPILER EXPERT WORKING CORRECTLY ***")
    else:
        print("\n*** ISSUES DETECTED - CHECK IMPLEMENTATION ***")


def run_benchmark_with_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Run full benchmark with model generation."""
    print("=" * 70)
    print("COMPILER EXPERT BENCHMARK - Full Model Test")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading model: {model_name}...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(model_name)
    model = loaded.model
    tokenizer = loaded.tokenizer

    import mlx.core as mx

    mx.eval(model.parameters())
    print("   Model loaded.")

    # Generate code for each task
    print("\n2. Generating code for each task...")
    print("-" * 70)

    results = []

    for task in TASKS:
        print(f"\nTask: {task.description}")
        print(f"Prompt: {task.prompt[:50]}...")

        # Generate (simple greedy decoding)
        tokens = tokenizer.encode(task.prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(100):  # Max tokens
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[0, -1, :])
            token_id = int(next_token.item())

            if token_id == tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            # Stop at closing ```
            text = tokenizer.decode(generated)
            if text.count("```") >= 1 and text.rstrip().endswith("```"):
                break

        code = tokenizer.decode(generated)
        print(f"Generated: {code[:80]}...")

        # Evaluate
        result = evaluate_code(code, task.test_cases)
        result.task = task.description
        results.append(result)

        status = "✓" if result.output_correct else "✗"
        print(f"Result: {status} ({result.tests_passed}/{result.tests_total} tests)")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    syntax_valid = sum(1 for r in results if r.syntax_valid)
    execution_success = sum(1 for r in results if r.execution_success)
    tests_correct = sum(1 for r in results if r.output_correct)

    print(f"\nSyntax valid:       {syntax_valid}/{len(results)}")
    print(f"Execution success:  {execution_success}/{len(results)}")
    print(f"All tests passed:   {tests_correct}/{len(results)}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"benchmark_{timestamp}.json"

    save_data = {
        "model": model_name,
        "timestamp": timestamp,
        "summary": {
            "syntax_valid_rate": syntax_valid / len(results),
            "execution_success_rate": execution_success / len(results),
            "tests_passed_rate": tests_correct / len(results),
        },
        "results": [asdict(r) for r in results],
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        run_benchmark_standalone()
    elif len(sys.argv) > 1:
        run_benchmark_with_model(sys.argv[1])
    else:
        # Default: standalone mode
        run_benchmark_standalone()
