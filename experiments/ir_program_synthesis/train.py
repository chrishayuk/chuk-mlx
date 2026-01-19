"""
IR Head Training

Trains a classifier to predict which IR program to use from NL input.

Architecture:
  NL Input → LLM Encoder → Hidden State → IR Classifier → Program ID

Then we use the known IR sequence for that program, extract operands,
compile to WASM, and execute.

This is simpler than generating IR token-by-token, and validates the
core hypothesis: LLMs can learn to route to correct IR programs.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ir_emission.shared import WASMRuntime
from programs import ALL_PROGRAMS as PROGRAMS, compile_program


# =============================================================================
# MODEL
# =============================================================================

class IRProgramClassifier(nn.Module):
    """
    Classifies NL input into one of the IR programs.

    Input: Hidden state from LLM at last token position
    Output: Logits over program classes
    """

    def __init__(self, hidden_dim: int, num_programs: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_programs)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
# OPERAND EXTRACTION
# =============================================================================

def extract_operands(text: str) -> list[int]:
    """Extract numeric operands from NL text."""
    # Find all numbers in the text
    numbers = re.findall(r'\b(\d+)\b', text)
    return [int(n) for n in numbers]


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    hidden_layer: int = 12  # Which layer to extract hidden state from
    eval_every: int = 5


def load_dataset(path: Path) -> list[dict]:
    """Load dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def get_hidden_state(model, tokenizer, text: str, layer: int) -> mx.array:
    """Get hidden state from model at specified layer."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Get hidden states by running partial forward
    # We'll use the model's embedding + layers up to `layer`
    hidden = model.model.embed_tokens(input_ids)

    for i, block in enumerate(model.model.layers[:layer]):
        hidden = block(hidden, mask=None, cache=None)
        # Handle BlockOutput wrapper if present
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    # Return last token position
    return hidden[0, -1, :]


def train_epoch(
    classifier: IRProgramClassifier,
    optimizer: optim.Optimizer,
    model,
    tokenizer,
    examples: list[dict],
    program_to_idx: dict[str, int],
    config: TrainingConfig,
) -> float:
    """Train for one epoch."""
    total_loss = 0
    num_batches = 0

    # Shuffle examples
    import random
    random.shuffle(examples)

    def loss_fn(classifier, hidden_states, labels):
        logits = classifier(hidden_states)
        return nn.losses.cross_entropy(logits, labels).mean()

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    # Process in batches
    for i in range(0, len(examples), config.batch_size):
        batch = examples[i:i + config.batch_size]

        # Get hidden states for batch
        hidden_states = []
        labels = []

        for ex in batch:
            hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
            hidden_states.append(hidden)
            labels.append(program_to_idx[ex["program_name"]])

        hidden_states = mx.stack(hidden_states)
        labels = mx.array(labels)

        # Compute loss and gradients
        loss, grads = loss_and_grad(classifier, hidden_states, labels)

        # Update
        optimizer.update(classifier, grads)
        mx.eval(classifier.parameters(), optimizer.state)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    classifier: IRProgramClassifier,
    model,
    tokenizer,
    examples: list[dict],
    program_to_idx: dict[str, int],
    idx_to_program: dict[int, str],
    config: TrainingConfig,
) -> dict:
    """Evaluate classifier accuracy and end-to-end execution."""
    runtime = WASMRuntime(use_native=True)

    correct_class = 0
    correct_exec = 0
    total = 0

    details = []

    for ex in examples:
        total += 1

        # Get prediction
        hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred_idx = int(mx.argmax(logits[0]).item())
        pred_program = idx_to_program[pred_idx]

        # Check classification
        true_program = ex["program_name"]
        class_correct = (pred_program == true_program)
        if class_correct:
            correct_class += 1

        # Try end-to-end execution
        exec_correct = False
        try:
            # Extract operands from NL
            operands = extract_operands(ex["nl_input"])

            # Get program and compile
            program = PROGRAMS.get(pred_program)
            if program and len(operands) >= program.num_operands:
                wasm = compile_program(program, operands[:program.num_operands])
                result = runtime.execute(wasm)

                if result.success and result.result == ex["expected_result"]:
                    exec_correct = True
                    correct_exec += 1
        except Exception as e:
            pass

        details.append({
            "nl": ex["nl_input"],
            "true_program": true_program,
            "pred_program": pred_program,
            "class_correct": class_correct,
            "exec_correct": exec_correct,
        })

    return {
        "class_accuracy": correct_class / total if total > 0 else 0,
        "exec_accuracy": correct_exec / total if total > 0 else 0,
        "total": total,
        "correct_class": correct_class,
        "correct_exec": correct_exec,
        "details": details,
    }


def main():
    print("=" * 60)
    print("IR Program Synthesis - Training IR Classifier")
    print("=" * 60)

    # Load model
    print("\n1. Loading base model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded.")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Test: {len(test_examples)} examples")

    # Setup program mapping
    program_names = list(PROGRAMS.keys())
    program_to_idx = {name: i for i, name in enumerate(program_names)}
    idx_to_program = {i: name for i, name in enumerate(program_names)}
    print(f"   Programs: {program_names}")

    # Get hidden dimension
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"   Hidden dim: {hidden_dim}")

    # Create classifier
    print("\n3. Creating classifier...")
    config = TrainingConfig()
    classifier = IRProgramClassifier(
        hidden_dim=hidden_dim,
        num_programs=len(program_names),
    )

    # Optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print("\n4. Training...")
    print("-" * 60)

    best_test_acc = 0

    for epoch in range(config.epochs):
        start = time.perf_counter()

        # Train
        train_loss = train_epoch(
            classifier, optimizer, model, tokenizer,
            train_examples, program_to_idx, config
        )

        elapsed = time.perf_counter() - start

        # Evaluate periodically
        if (epoch + 1) % config.eval_every == 0 or epoch == 0:
            train_eval = evaluate(
                classifier, model, tokenizer,
                train_examples[:100], program_to_idx, idx_to_program, config
            )
            test_eval = evaluate(
                classifier, model, tokenizer,
                test_examples, program_to_idx, idx_to_program, config
            )

            print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_eval['class_accuracy']:.1%} | "
                  f"Test Acc: {test_eval['class_accuracy']:.1%} | "
                  f"Test Exec: {test_eval['exec_accuracy']:.1%} | "
                  f"Time: {elapsed:.1f}s")

            if test_eval['class_accuracy'] > best_test_acc:
                best_test_acc = test_eval['class_accuracy']
        else:
            print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")

    print("-" * 60)

    # Final evaluation
    print("\n5. Final Evaluation...")

    print("\n   Training set:")
    train_final = evaluate(
        classifier, model, tokenizer,
        train_examples, program_to_idx, idx_to_program, config
    )
    print(f"   Classification: {train_final['correct_class']}/{train_final['total']} = {train_final['class_accuracy']:.1%}")
    print(f"   Execution: {train_final['correct_exec']}/{train_final['total']} = {train_final['exec_accuracy']:.1%}")

    print("\n   Test set (Collatz - held out):")
    test_final = evaluate(
        classifier, model, tokenizer,
        test_examples, program_to_idx, idx_to_program, config
    )
    print(f"   Classification: {test_final['correct_class']}/{test_final['total']} = {test_final['class_accuracy']:.1%}")
    print(f"   Execution: {test_final['correct_exec']}/{test_final['total']} = {test_final['exec_accuracy']:.1%}")

    # Show some test examples
    print("\n   Sample test predictions:")
    for detail in test_final["details"][:5]:
        status = "✓" if detail["exec_correct"] else "✗"
        print(f"   {status} '{detail['nl'][:40]}...'")
        print(f"      pred: {detail['pred_program']}, true: {detail['true_program']}")

    # Save results
    print("\n6. Saving results...")
    results = {
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_layer": config.hidden_layer,
        },
        "train": {
            "class_accuracy": train_final["class_accuracy"],
            "exec_accuracy": train_final["exec_accuracy"],
            "total": train_final["total"],
        },
        "test": {
            "class_accuracy": test_final["class_accuracy"],
            "exec_accuracy": test_final["exec_accuracy"],
            "total": test_final["total"],
            "details": test_final["details"],
        },
    }

    results_path = results_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Training Results:
  Classification: {train_final['class_accuracy']:.1%}
  Execution: {train_final['exec_accuracy']:.1%}

Test Results (Collatz - UNSEEN):
  Classification: {test_final['class_accuracy']:.1%}
  Execution: {test_final['exec_accuracy']:.1%}

Key Question: Can the model generalize to Collatz?
Answer: {"YES" if test_final['class_accuracy'] > 0.5 else "NO"} ({test_final['class_accuracy']:.1%} accuracy)
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
