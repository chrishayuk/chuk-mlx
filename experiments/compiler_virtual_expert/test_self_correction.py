"""
Test self-correction with deliberate bug injection.

Shows the full cycle: buggy code → compiler feedback → model fix → verified
"""

import re
import mlx.core as mx
from chuk_lazarus.models_v2.loader import load_model
from compiler_plugin import CompilerExpertPlugin


def test_deliberate_bug_fix():
    """Test correction with a deliberately buggy code sample."""
    print("=" * 70)
    print("SELF-CORRECTION - Deliberate Bug Fix Demo")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    mx.eval(model.parameters())
    print("Model loaded.")

    compiler = CompilerExpertPlugin()

    # Deliberately buggy code
    buggy_code = """def sum_list(lst):
    total = 0
    for item in lst:
        total = item  # BUG: should be total += item
    return total"""

    test_cases = [(([1, 2, 3],), 6), (([], ), 0), (([5],), 5)]

    print("\n" + "-" * 70)
    print("STEP 1: Model generates buggy code")
    print("-" * 70)
    print(f"Code:\n{buggy_code}")

    result = compiler.extract_function_and_test(buggy_code, test_cases)
    print(f"\nCompiler feedback:\n{result.output or result.error}")
    print(f"Tests passed: {result.success}")

    # Build fix prompt
    print("\n" + "-" * 70)
    print("STEP 2: Feed error to model, ask for fix")
    print("-" * 70)

    fix_prompt = f"""The following code has a bug:

```python
{buggy_code}
```

Test results:
{result.output}

The function returns wrong values. Fix the bug so sum_list returns the sum of all elements.

```python
def sum_list(lst):
"""

    print(f"Prompt:\n{fix_prompt}")

    # Generate fix
    tokens = tokenizer.encode(fix_prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(100):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token = mx.argmax(logits[0, -1, :])
        token_id = int(next_token.item())

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        text = tokenizer.decode(generated)
        # Stop at code block end or double newline after return
        if "```" in text or (text.count("\n") > 3 and "return" in text):
            break

    fixed_text = tokenizer.decode(generated)
    print(f"\nModel generated:\n{fixed_text}")

    # Reconstruct full function - preserve indentation
    body = fixed_text.split("```")[0]
    # Ensure proper indentation
    lines = body.split("\n")
    indented_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Add 4-space indent if not already indented
            if not line.startswith("    ") and not line.startswith("\t"):
                indented_lines.append("    " + stripped)
            else:
                indented_lines.append(line)
    fixed_code = "def sum_list(lst):\n" + "\n".join(indented_lines)

    print("\n" + "-" * 70)
    print("STEP 3: Verify the fix")
    print("-" * 70)
    print(f"Fixed code:\n{fixed_code}")

    result2 = compiler.extract_function_and_test(fixed_code, test_cases)
    print(f"\nCompiler says:\n{result2.output or result2.error}")

    success = result2.success

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    if success:
        print("✓ SUCCESS - Model fixed the bug!")
    else:
        print("✗ FAILED - Model did not fix the bug")

    print(f"""
Self-correction flow:
  1. Buggy code: total = item (overwrites instead of accumulates)
  2. Compiler caught: sum_list([1,2,3]) = 3, expected 6
  3. Model saw feedback, generated fix
  4. Compiler verified: {"PASS" if success else "FAIL"}
""")

    return success


def test_multiple_bugs():
    """Test correction on multiple different bug types."""
    print("\n" + "=" * 70)
    print("SELF-CORRECTION - Multiple Bug Types")
    print("=" * 70)

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    mx.eval(model.parameters())

    compiler = CompilerExpertPlugin()

    bugs = [
        {
            "name": "Off-by-one",
            "buggy": "def first_n(lst, n):\n    return lst[:n-1]  # BUG: should be lst[:n]",
            "tests": [(([1, 2, 3, 4, 5], 3), [1, 2, 3])],
            "hint": "off-by-one error",
        },
        {
            "name": "Wrong operator",
            "buggy": "def multiply(a, b):\n    return a + b  # BUG: should be a * b",
            "tests": [((3, 4), 12), ((0, 5), 0)],
            "hint": "wrong operator",
        },
        {
            "name": "Missing return",
            "buggy": "def double(x):\n    result = x * 2  # BUG: missing return",
            "tests": [((5,), 10), ((0,), 0)],
            "hint": "missing return statement",
        },
    ]

    results = []
    for bug in bugs:
        print(f"\n--- {bug['name']} ---")

        # Check buggy code fails
        result = compiler.extract_function_and_test(bug["buggy"], bug["tests"])
        print(f"Buggy: {result.output}")

        # Simple check - did it fail as expected?
        if not result.success:
            print(f"✓ Bug detected: {bug['hint']}")
            results.append({"name": bug["name"], "detected": True})
        else:
            print(f"✗ Bug not detected")
            results.append({"name": bug["name"], "detected": False})

    print("\n" + "=" * 70)
    print("BUG DETECTION SUMMARY")
    print("=" * 70)
    detected = sum(1 for r in results if r["detected"])
    print(f"Bugs detected: {detected}/{len(results)}")
    for r in results:
        status = "✓" if r["detected"] else "✗"
        print(f"  {status} {r['name']}")


def test_deliberate_bug_fix_with_model(model_name: str):
    """Test correction with a deliberately buggy code sample."""
    print("=" * 70)
    print("SELF-CORRECTION - Deliberate Bug Fix Demo")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_name}...")
    loaded = load_model(model_name)
    model = loaded.model
    tokenizer = loaded.tokenizer
    mx.eval(model.parameters())
    print("Model loaded.")

    compiler = CompilerExpertPlugin()

    # Test 1: sum_list bug
    print("\n" + "=" * 70)
    print("TEST 1: sum_list - assignment instead of accumulation")
    print("=" * 70)

    buggy_code = """def sum_list(lst):
    total = 0
    for item in lst:
        total = item  # BUG: should be total += item
    return total"""

    run_single_correction(model, tokenizer, compiler, buggy_code,
                          [(([1, 2, 3],), 6), (([], ), 0), (([5],), 5)],
                          "sum_list", "return the sum of all elements")

    # Test 2: factorial bug
    print("\n" + "=" * 70)
    print("TEST 2: factorial - missing base case")
    print("=" * 70)

    buggy_code2 = """def factorial(n):
    return n * factorial(n - 1)  # BUG: missing base case"""

    run_single_correction(model, tokenizer, compiler, buggy_code2,
                          [((0,), 1), ((1,), 1), ((5,), 120)],
                          "factorial", "calculate factorial with proper base case")

    # Test 3: max_list bug
    print("\n" + "=" * 70)
    print("TEST 3: max_list - returns first instead of max")
    print("=" * 70)

    buggy_code3 = """def max_list(lst):
    return lst[0]  # BUG: returns first element, not maximum"""

    run_single_correction(model, tokenizer, compiler, buggy_code3,
                          [(([1, 3, 2],), 3), (([5],), 5), (([-1, -5],), -1)],
                          "max_list", "return the maximum value in the list")


def run_single_correction(model, tokenizer, compiler, buggy_code, test_cases,
                          func_name, task_desc):
    """Run a single correction cycle."""
    print(f"\nBuggy code:\n{buggy_code}")

    result = compiler.extract_function_and_test(buggy_code, test_cases)
    print(f"\nCompiler feedback:\n{result.output or result.error}")

    if result.success:
        print("Bug not detected (code passed tests)")
        return

    # Build fix prompt
    fix_prompt = f"""The following code has a bug:

```python
{buggy_code}
```

Test results:
{result.output or result.error}

Write ONLY the fixed function that will {task_desc}:

```python
def {func_name}"""

    print(f"\nAsking model to fix...")

    tokens = tokenizer.encode(fix_prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(150):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        next_token = mx.argmax(logits[0, -1, :])
        token_id = int(next_token.item())

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        text = tokenizer.decode(generated)
        if "```" in text or (text.count("\n") > 4 and "return" in text):
            break

    fixed_text = tokenizer.decode(generated)
    print(f"\nModel generated:\n{fixed_text}")

    # Reconstruct - add function header
    body = fixed_text.split("```")[0]
    lines = body.split("\n")
    indented_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            if not line.startswith("    ") and not line.startswith("\t"):
                indented_lines.append("    " + stripped)
            else:
                indented_lines.append(line)

    fixed_code = f"def {func_name}" + "\n".join(indented_lines)
    print(f"\nReconstructed code:\n{fixed_code}")

    result2 = compiler.extract_function_and_test(fixed_code, test_cases)
    print(f"\nCompiler says:\n{result2.output or result2.error}")

    if result2.success:
        print("\n✓ SUCCESS - Model fixed the bug!")
    else:
        print("\n✗ FAILED - Model did not fix the bug")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Using model: {model_name}")
    test_deliberate_bug_fix_with_model(model_name)
