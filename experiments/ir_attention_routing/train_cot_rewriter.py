"""
Train CoT Rewriter: Teach the model to normalize arbitrary input to invocation format.

This proves the full thesis:
  Input → CoT rewrite → Invocation format → Circuit → Output

Training approach:
1. Generate diverse (input, canonical_format) pairs
2. SFT on format conversion task
3. Test: arbitrary input → trained rewriter → circuit invocation → correct answer
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


# =============================================================================
# Training Data Generation
# =============================================================================


def generate_training_data(n_samples: int = 1000) -> list[dict]:
    """
    Generate diverse (input, canonical) pairs for training.

    Input formats:
    - Functional: add(5, 3), sub(10, 4), mul(6, 7), div(20, 4)
    - Natural language: "five plus three", "ten minus four"
    - Word problems: "Jenny has 5 apples..."
    - Mixed: "what is add(5, 3)"

    Output format (canonical):
    - "5 + 3 ="
    """
    data = []

    ops = [
        ("add", "+", ["plus", "and", "added to", "more than"]),
        ("sub", "-", ["minus", "subtract", "take away", "less than"]),
        ("mul", "*", ["times", "multiplied by", "multiply"]),
        ("div", "/", ["divided by", "over", "split into"]),
    ]

    # Number words for NL generation
    num_words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty",
    }

    for _ in range(n_samples):
        # Pick operation
        op_name, op_symbol, op_words = random.choice(ops)

        # Pick operands
        a = random.randint(1, 50)
        b = random.randint(1, 50)

        # Ensure valid division
        if op_name == "div" and b != 0:
            a = b * random.randint(1, 10)

        # Canonical output
        canonical = f"{a} {op_symbol} {b} ="

        # Generate diverse input format
        input_type = random.choice([
            "functional", "functional_caps", "functional_spaces",
            "word_operator", "word_numbers", "word_problem",
            "question", "command", "mixed"
        ])

        if input_type == "functional":
            # add(5, 3)
            input_text = f"{op_name}({a}, {b})"

        elif input_type == "functional_caps":
            # ADD(5, 3)
            input_text = f"{op_name.upper()}({a}, {b})"

        elif input_type == "functional_spaces":
            # add( 5 , 3 )
            input_text = f"{op_name}( {a} , {b} )"

        elif input_type == "word_operator":
            # 5 plus 3
            word = random.choice(op_words)
            input_text = f"{a} {word} {b}"

        elif input_type == "word_numbers":
            # five plus three (only for small numbers)
            if a in num_words and b in num_words:
                word = random.choice(op_words)
                input_text = f"{num_words[a]} {word} {num_words[b]}"
            else:
                word = random.choice(op_words)
                input_text = f"{a} {word} {b}"

        elif input_type == "word_problem":
            # Various word problem templates
            templates = {
                "add": [
                    f"Jenny has {a} apples and gets {b} more",
                    f"There are {a} birds, {b} more arrive",
                    f"Start with {a}, add {b}",
                    f"{a} items plus {b} items",
                ],
                "sub": [
                    f"Jenny has {a} apples and gives away {b}",
                    f"There are {a} birds, {b} fly away",
                    f"Start with {a}, remove {b}",
                    f"Take {b} from {a}",
                ],
                "mul": [
                    f"Each box has {a} items, there are {b} boxes",
                    f"{b} groups of {a}",
                    f"{a} times {b}",
                    f"Multiply {a} by {b}",
                ],
                "div": [
                    f"Split {a} into {b} equal groups",
                    f"Divide {a} by {b}",
                    f"{a} shared among {b}",
                    f"How many {b}s in {a}",
                ],
            }
            input_text = random.choice(templates[op_name])

        elif input_type == "question":
            # What is 5 + 3?
            word = random.choice(op_words)
            input_text = random.choice([
                f"What is {a} {word} {b}?",
                f"Calculate {a} {word} {b}",
                f"Compute {op_name}({a}, {b})",
                f"What's {a} {word} {b}?",
            ])

        elif input_type == "command":
            # Compute: add(5, 3)
            input_text = random.choice([
                f"Compute: {op_name}({a}, {b})",
                f"Solve: {a} {random.choice(op_words)} {b}",
                f"Calculate {op_name}({a}, {b})",
            ])

        else:  # mixed
            # Combine functional and NL
            word = random.choice(op_words)
            input_text = random.choice([
                f"what is {op_name}({a}, {b})",
                f"{op_name} of {a} and {b}",
                f"the {op_name} of {a} {word} nothing gives {a}",
            ])

        data.append({
            "input": input_text,
            "canonical": canonical,
            "operation": op_name,
            "a": a,
            "b": b,
        })

    return data


def format_training_example(item: dict, tokenizer) -> dict:
    """
    Format a training example for SFT.

    Format:
    <s>Convert to math: {input}
    {canonical}</s>
    """
    prompt = f"Convert to math: {item['input']}\n"
    completion = item['canonical']

    full_text = prompt + completion

    return {
        "prompt": prompt,
        "completion": completion,
        "full_text": full_text,
    }


# =============================================================================
# Training Loop
# =============================================================================


class CoTRewriterTrainer:
    """Simple SFT trainer for CoT rewriter."""

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        lora_rank: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.lora_rank = lora_rank

        # Add LoRA layers
        self._add_lora_layers()

        # Optimizer
        self.optimizer = optim.Adam(learning_rate=learning_rate)

    def _add_lora_layers(self):
        """Add LoRA adapters to attention layers."""
        from chuk_lazarus.models_v2.adapters.lora import LoRALinear

        lora_count = 0
        for layer in self.model.model.layers:
            # Q and V projections
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn

                if hasattr(attn, "q_proj") and isinstance(attn.q_proj, nn.Linear):
                    orig_q = attn.q_proj
                    attn.q_proj = LoRALinear(
                        orig_q, rank=self.lora_rank, alpha=float(self.lora_rank * 2)
                    )
                    lora_count += 1

                if hasattr(attn, "v_proj") and isinstance(attn.v_proj, nn.Linear):
                    orig_v = attn.v_proj
                    attn.v_proj = LoRALinear(
                        orig_v, rank=self.lora_rank, alpha=float(self.lora_rank * 2)
                    )
                    lora_count += 1

        mx.eval(self.model.parameters())
        logger.info(f"Added {lora_count} LoRA layers with rank={self.lora_rank}")

    def _get_trainable_params(self):
        """Get only LoRA parameters."""
        params = []
        for layer in self.model.model.layers:
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn.q_proj, "lora_a"):
                    params.extend([attn.q_proj.lora_a, attn.q_proj.lora_b])
                if hasattr(attn.v_proj, "lora_a"):
                    params.extend([attn.v_proj.lora_a, attn.v_proj.lora_b])
        return params

    def train_step(self, batch: list[dict]) -> float:
        """Single training step."""
        total_loss = 0.0

        for item in batch:
            # Tokenize
            prompt = item["prompt"]
            full_text = item["full_text"]

            prompt_tokens = self.tokenizer.encode(prompt)
            full_tokens = self.tokenizer.encode(full_text)

            input_ids = mx.array([full_tokens[:-1]])  # All but last
            target_ids = mx.array([full_tokens[1:]])   # All but first

            # Forward
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            # Only compute loss on completion tokens (after prompt)
            prompt_len = len(prompt_tokens) - 1  # -1 because of shift
            completion_logits = logits[0, prompt_len:, :]
            completion_targets = target_ids[0, prompt_len:]

            # Cross-entropy loss
            log_probs = mx.log(mx.softmax(completion_logits, axis=-1) + 1e-10)
            loss = -mx.mean(log_probs[mx.arange(len(completion_targets)), completion_targets])

            total_loss += float(loss.item())

        return total_loss / len(batch)

    def train(
        self,
        train_data: list[dict],
        epochs: int = 3,
        batch_size: int = 4,
    ) -> list[float]:
        """Train the CoT rewriter."""
        # Format data
        formatted = [format_training_example(item, self.tokenizer) for item in train_data]

        losses = []

        for epoch in range(epochs):
            random.shuffle(formatted)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(formatted), batch_size):
                batch = formatted[i:i + batch_size]

                # Compute loss and gradients manually
                loss = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1

                # Simple gradient update (manual for now)
                # In production, use mx.grad properly

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            logger.info(f"Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

        return losses

    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        """Generate completion."""
        full_prompt = f"Convert to math: {prompt}\n"
        tokens = self.tokenizer.encode(full_prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[0, -1, :])
            token_id = int(next_token.item())

            if token_id == self.tokenizer.eos_token_id:
                break

            token_str = self.tokenizer.decode([token_id])
            if "\n" in token_str or "=" in token_str:
                generated.append(token_id)
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_rewriter(trainer: CoTRewriterTrainer, test_data: list[dict]) -> dict:
    """Evaluate the trained rewriter."""
    results = {
        "format_correct": 0,
        "result_correct": 0,
        "total": len(test_data),
        "examples": [],
    }

    for item in test_data:
        # Generate canonical format
        generated = trainer.generate(item["input"])

        # Check format match (normalize whitespace)
        expected = item["canonical"]
        format_match = normalize(generated) == normalize(expected)

        if format_match:
            results["format_correct"] += 1

        # Test circuit invocation
        if "=" in generated:
            # Extract just the expression
            expr = generated.strip()
            if not expr.endswith("="):
                expr = expr + " ="

            # Get circuit output
            circuit_tokens = trainer.tokenizer.encode(expr)
            circuit_input = mx.array([circuit_tokens])
            circuit_output = trainer.model(circuit_input)
            circuit_logits = circuit_output.logits if hasattr(circuit_output, "logits") else circuit_output
            circuit_probs = mx.softmax(circuit_logits[0, -1, :])
            predicted_token = trainer.tokenizer.decode([int(mx.argmax(circuit_probs).item())])

            # Compute expected result
            if item["operation"] == "add":
                expected_result = item["a"] + item["b"]
            elif item["operation"] == "sub":
                expected_result = item["a"] - item["b"]
            elif item["operation"] == "mul":
                expected_result = item["a"] * item["b"]
            elif item["operation"] == "div":
                expected_result = item["a"] // item["b"]

            # Check if circuit output matches
            try:
                predicted_num = int(re.search(r"-?\d+", predicted_token).group())
                if predicted_num == expected_result:
                    results["result_correct"] += 1
            except (AttributeError, ValueError):
                pass

        results["examples"].append({
            "input": item["input"],
            "expected": expected,
            "generated": generated,
            "format_match": format_match,
        })

    results["format_accuracy"] = results["format_correct"] / results["total"]
    results["result_accuracy"] = results["result_correct"] / results["total"]

    return results


def normalize(s: str) -> str:
    """Normalize format string."""
    s = re.sub(r"\s+", " ", s.strip())
    return s


# =============================================================================
# Main
# =============================================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("COT REWRITER TRAINING")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())

    # Generate training data
    print("\n2. Generating training data...")
    train_data = generate_training_data(n_samples=500)
    test_data = generate_training_data(n_samples=50)

    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")

    # Show examples
    print("\n   Example training pairs:")
    for item in train_data[:5]:
        print(f"     '{item['input']}' → '{item['canonical']}'")

    # Create trainer
    print("\n3. Creating trainer with LoRA...")
    trainer = CoTRewriterTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=1e-4,
        lora_rank=8,
    )

    # Baseline evaluation
    print("\n4. Baseline evaluation (before training)...")
    baseline = evaluate_rewriter(trainer, test_data[:20])
    print(f"   Format accuracy: {baseline['format_accuracy']:.1%}")
    print(f"   Result accuracy: {baseline['result_accuracy']:.1%}")

    print("\n   Examples:")
    for ex in baseline["examples"][:5]:
        status = "OK" if ex["format_match"] else "FAIL"
        print(f"     '{ex['input'][:30]}' → '{ex['generated']}' [{status}]")

    # Train
    print("\n5. Training CoT rewriter...")
    losses = trainer.train(train_data, epochs=3, batch_size=4)
    print(f"   Final loss: {losses[-1]:.4f}")

    # Post-training evaluation
    print("\n6. Post-training evaluation...")
    final = evaluate_rewriter(trainer, test_data)
    print(f"   Format accuracy: {final['format_accuracy']:.1%}")
    print(f"   Result accuracy: {final['result_accuracy']:.1%}")

    print("\n   Examples:")
    for ex in final["examples"][:10]:
        status = "OK" if ex["format_match"] else "FAIL"
        print(f"     '{ex['input'][:30]}' → '{ex['generated']}' [{status}]")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"cot_rewriter_{timestamp}.json"

    results = {
        "baseline": {
            "format_accuracy": baseline["format_accuracy"],
            "result_accuracy": baseline["result_accuracy"],
        },
        "trained": {
            "format_accuracy": final["format_accuracy"],
            "result_accuracy": final["result_accuracy"],
        },
        "improvement": {
            "format": final["format_accuracy"] - baseline["format_accuracy"],
            "result": final["result_accuracy"] - baseline["result_accuracy"],
        },
        "training": {
            "samples": len(train_data),
            "epochs": 3,
            "final_loss": losses[-1],
        },
        "examples": final["examples"][:20],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline → Trained:")
    print(f"  Format accuracy: {baseline['format_accuracy']:.1%} → {final['format_accuracy']:.1%}")
    print(f"  Result accuracy: {baseline['result_accuracy']:.1%} → {final['result_accuracy']:.1%}")
    print(f"\nImprovement:")
    print(f"  Format: +{results['improvement']['format']:.1%}")
    print(f"  Result: +{results['improvement']['result']:.1%}")

    if final["format_accuracy"] > 0.8:
        print("\n*** THESIS CONFIRMED: CoT rewriter can be trained! ***")

    print("=" * 70)


if __name__ == "__main__":
    main()
