"""
Parse GSM8K solutions to extract IR computation graphs.

GSM8K format:
  Question: "Janet's ducks lay 16 eggs per day..."
  Answer: "She eats 3 + 4 = <<3+4=7>>7 eggs...
           The remainder is 16 - 7 = <<16-7=9>>9 eggs...
           She earns 9 * 2 = <<9*2=18>>$18"
  Final: #### 18

The <<expr=result>> annotations give us the exact computation graph!
"""

import sys
from pathlib import Path
import re
import json

import functools
print = functools.partial(print, flush=True)


def extract_computations(answer_text: str) -> list[dict]:
    """Extract <<expr=result>> computations from GSM8K answer."""
    # Pattern: <<expression=result>>
    pattern = r'<<([^>]+)>>'
    matches = re.findall(pattern, answer_text)

    computations = []
    for match in matches:
        if '=' in match:
            expr, result = match.rsplit('=', 1)
            computations.append({
                "expr": expr.strip(),
                "result": result.strip()
            })

    return computations


def expr_to_ir(computations: list[dict], final_answer: int) -> str:
    """Convert computation sequence to IR format."""
    if not computations:
        return None

    lines = []
    var_map = {}  # result -> variable name
    var_counter = 0

    def get_var_name(idx: int) -> str:
        names = ["step1", "step2", "step3", "step4", "step5",
                 "step6", "step7", "step8", "step9", "step10"]
        return names[idx] if idx < len(names) else f"step{idx+1}"

    for comp in computations:
        expr = comp["expr"]
        result = comp["result"]

        # Parse the expression
        # Handle: a + b, a - b, a * b, a / b, a + b + c, etc.

        # Try to substitute known results with variable names
        for known_result, var_name in var_map.items():
            # Replace the result value with the variable name
            # Be careful with partial matches (e.g., "9" in "19")
            expr = re.sub(rf'\b{re.escape(known_result)}\b', var_name, expr)

        var_name = get_var_name(var_counter)
        lines.append(f"{var_name} = {expr}")
        var_map[result] = var_name
        var_counter += 1

    lines.append("[END]")
    return "\n".join(lines)


def parse_gsm8k_example(item: dict) -> dict | None:
    """Parse a single GSM8K example into IR format."""
    question = item["question"]
    answer = item["answer"]

    # Extract final answer
    final_match = re.search(r'####\s*(-?[\d,]+)', answer)
    if not final_match:
        return None

    final_answer = int(final_match.group(1).replace(",", ""))

    # Extract computations
    computations = extract_computations(answer)
    if not computations:
        return None

    # Convert to IR
    ir = expr_to_ir(computations, final_answer)
    if not ir:
        return None

    return {
        "q": question,
        "ir": ir,
        "ans": final_answer,
        "num_steps": len(computations),
        "raw_computations": computations
    }


def analyze_gsm8k():
    """Analyze GSM8K dataset and extract IR patterns."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")

    print(f"Total examples: {len(ds)}")

    # Parse all examples
    parsed = []
    failed = []

    for item in ds:
        result = parse_gsm8k_example(item)
        if result:
            parsed.append(result)
        else:
            failed.append(item)

    print(f"\nParsed: {len(parsed)}")
    print(f"Failed: {len(failed)}")

    # Analyze step counts
    step_counts = {}
    for item in parsed:
        n = item["num_steps"]
        step_counts[n] = step_counts.get(n, 0) + 1

    print("\nStep distribution:")
    for n in sorted(step_counts.keys()):
        pct = step_counts[n] / len(parsed) * 100
        bar = "█" * int(pct / 2)
        print(f"  {n} steps: {step_counts[n]:4d} ({pct:4.1f}%) {bar}")

    # Show samples
    print("\n" + "=" * 70)
    print("SAMPLE CONVERSIONS")
    print("=" * 70)

    for item in parsed[:5]:
        print(f"\nQ: {item['q'][:80]}...")
        print(f"Answer: {item['ans']}")
        print(f"Steps: {item['num_steps']}")
        print("IR:")
        for line in item['ir'].split('\n'):
            print(f"  {line}")
        print("Raw computations:")
        for comp in item['raw_computations']:
            print(f"  {comp['expr']} = {comp['result']}")

    return parsed, failed


def create_training_data(parsed: list[dict], output_dir: Path):
    """Create training data from parsed GSM8K examples."""
    import random

    # Shuffle and split
    random.seed(42)
    random.shuffle(parsed)

    # Use 80% train, 10% val, 10% test
    n = len(parsed)
    train = parsed[:int(n * 0.8)]
    val = parsed[int(n * 0.8):int(n * 0.9)]
    test = parsed[int(n * 0.9):]

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, 'w') as f:
            for item in data:
                # Simplify for training
                f.write(json.dumps({
                    "q": item["q"],
                    "ir": item["ir"],
                    "ans": item["ans"]
                }) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    return train, val, test


if __name__ == "__main__":
    parsed, failed = analyze_gsm8k()

    # Create training data
    output_dir = Path(__file__).parent / "gsm8k_ir_data"
    train, val, test = create_training_data(parsed, output_dir)

    print(f"\nData saved to {output_dir}")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
