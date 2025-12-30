#!/usr/bin/env python3
"""
Linear Probe: Extract computed answers from hidden states.

THE KEY EXPERIMENT: If a linear probe can decode the correct answer
from Layer 20's hidden state, even when the model's final output is
wrong, it PROVES:

1. The computation happened
2. The answer exists in the representation
3. The failure is in serialization, not computation

This is the smoking gun for the "ghost in Layer 20" hypothesis.

Usage:
    # Train probe on arithmetic
    uv run python examples/introspection/linear_probe.py train \
        --model "mlx-community/gemma-3-4b-it-bf16" \
        --task arithmetic \
        --layer 20

    # Evaluate probe
    uv run python examples/introspection/linear_probe.py eval \
        --model "mlx-community/gemma-3-4b-it-bf16" \
        --probe-path probes/arithmetic_L20.npz \
        --prompt "347 * 892 = "

    # Find the best layer for probing
    uv run python examples/introspection/linear_probe.py sweep \
        --model "mlx-community/gemma-3-4b-it-bf16" \
        --task arithmetic
"""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class ProbeExample:
    """A single probe training example."""
    prompt: str
    target_token: str
    target_id: int
    hidden_state: mx.array | None = None


@dataclass
class ProbeResult:
    """Result of probing a single example."""
    prompt: str
    target_token: str
    probe_prediction: str
    probe_probability: float
    probe_rank: int
    model_prediction: str
    model_correct: bool
    probe_correct: bool
    probe_found_ghost: bool  # Probe correct when model wrong


class LinearProbe(nn.Module):
    """Simple linear probe from hidden states to vocabulary."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class ArithmeticDataset:
    """Generate arithmetic problems for probe training."""

    def __init__(self, tokenizer: Any, num_examples: int = 500):
        self.tokenizer = tokenizer
        self.examples = self._generate(num_examples)

    def _generate(self, n: int) -> list[ProbeExample]:
        examples = []
        ops = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
        ]

        for _ in range(n):
            op_sym, op_fn = random.choice(ops)

            # Mix of difficulties
            if random.random() < 0.3:
                # Easy: 1-digit
                a = random.randint(1, 9)
                b = random.randint(1, 9)
            elif random.random() < 0.6:
                # Medium: 2-digit
                a = random.randint(10, 99)
                b = random.randint(10, 99)
            else:
                # Hard: 3-digit
                a = random.randint(100, 999)
                b = random.randint(100, 999)

            # Avoid negative results for subtraction
            if op_sym == "-" and b > a:
                a, b = b, a

            result = op_fn(a, b)
            prompt = f"{a} {op_sym} {b} = "

            # Target is the first digit of the result
            first_digit = str(result)[0]
            target_ids = self.tokenizer.encode(first_digit, add_special_tokens=False)
            target_id = target_ids[0] if target_ids else 0

            examples.append(ProbeExample(
                prompt=prompt,
                target_token=first_digit,
                target_id=target_id,
            ))

        return examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class ProbeTrainer:
    """Train and evaluate linear probes."""

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any, model_id: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id

    @classmethod
    async def from_pretrained(cls, model_id: str) -> "ProbeTrainer":
        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden size: {config.hidden_size}")

        return cls(model, tokenizer, config, model_id)

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        return list(self.model.layers)

    def _get_embed(self):
        if hasattr(self.model, "model"):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    def _get_scale(self):
        return getattr(self.config, "embedding_scale", None)

    def _get_norm(self):
        if hasattr(self.model, "model"):
            return getattr(self.model.model, "norm", None)
        return getattr(self.model, "norm", None)

    def _get_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        embed = self._get_embed()
        if hasattr(embed, "as_linear"):
            return embed.as_linear
        return None

    def extract_hidden_state(self, prompt: str, layer: int) -> tuple[mx.array, mx.array]:
        """
        Extract hidden state at a specific layer.

        Returns:
            (hidden_at_layer, final_logits)
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        head = self._get_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        captured = None

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

            if idx == layer:
                captured = h[:, -1, :]  # Last position

        # Final logits
        if norm:
            h = norm(h)
        if head:
            logits = head(h)
            if hasattr(logits, "logits"):
                logits = logits.logits
        else:
            logits = h

        return captured, logits[:, -1, :]

    def train_probe(
        self,
        dataset: ArithmeticDataset,
        layer: int,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> LinearProbe:
        """Train a linear probe on the dataset."""
        print(f"\nTraining probe on Layer {layer}...")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")

        # Extract hidden states for all examples
        print("  Extracting hidden states...")
        hidden_states = []
        targets = []

        for i, ex in enumerate(dataset):
            h, _ = self.extract_hidden_state(ex.prompt, layer)
            hidden_states.append(h)
            targets.append(ex.target_id)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(dataset)}")

        X = mx.concatenate(hidden_states, axis=0)  # [N, hidden_dim]
        y = mx.array(targets)  # [N]

        # Create probe
        hidden_dim = X.shape[1]
        vocab_size = self.tokenizer.vocab_size
        probe = LinearProbe(hidden_dim, vocab_size)

        # Train
        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(probe, X, y):
            logits = probe(X)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        loss_and_grad = nn.value_and_grad(probe, loss_fn)

        print("  Training...")
        for epoch in range(epochs):
            # Shuffle
            perm = mx.array(np.random.permutation(len(X)))
            X_shuf = X[perm]
            y_shuf = y[perm]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X), batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]

                loss, grads = loss_and_grad(probe, X_batch, y_batch)
                optimizer.update(probe, grads)
                mx.eval(probe.parameters())

                total_loss += float(loss)
                n_batches += 1

            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

        return probe

    def evaluate_probe(
        self,
        probe: LinearProbe,
        dataset: ArithmeticDataset,
        layer: int,
    ) -> list[ProbeResult]:
        """Evaluate probe on dataset."""
        results = []

        for ex in dataset:
            h, model_logits = self.extract_hidden_state(ex.prompt, layer)

            # Probe prediction
            probe_logits = probe(h)
            probe_probs = mx.softmax(probe_logits[0])
            probe_top_idx = int(mx.argmax(probe_probs))
            probe_pred = self.tokenizer.decode([probe_top_idx])
            probe_prob = float(probe_probs[ex.target_id])

            # Find probe rank
            sorted_idx = mx.argsort(probe_probs)[::-1][:100].tolist()
            probe_rank = sorted_idx.index(ex.target_id) + 1 if ex.target_id in sorted_idx else 101

            # Model prediction
            model_probs = mx.softmax(model_logits[0])
            model_top_idx = int(mx.argmax(model_probs))
            model_pred = self.tokenizer.decode([model_top_idx])
            model_correct = ex.target_token in model_pred

            # Probe correct?
            probe_correct = probe_rank == 1

            # Ghost found? (probe correct when model wrong)
            ghost_found = probe_correct and not model_correct

            results.append(ProbeResult(
                prompt=ex.prompt,
                target_token=ex.target_token,
                probe_prediction=probe_pred,
                probe_probability=probe_prob,
                probe_rank=probe_rank,
                model_prediction=model_pred,
                model_correct=model_correct,
                probe_correct=probe_correct,
                probe_found_ghost=ghost_found,
            ))

        return results

    def save_probe(self, probe: LinearProbe, path: Path, layer: int, metadata: dict = None):
        """Save probe weights."""
        path.parent.mkdir(parents=True, exist_ok=True)

        weights = {
            "weight": np.array(probe.linear.weight),
            "bias": np.array(probe.linear.bias) if probe.linear.bias is not None else None,
        }

        meta = {
            "layer": layer,
            "model_id": self.model_id,
            "hidden_dim": int(probe.linear.weight.shape[1]),
            "vocab_size": int(probe.linear.weight.shape[0]),
            **(metadata or {}),
        }

        np.savez(path, **weights, metadata=json.dumps(meta))
        print(f"Saved probe to {path}")

    def load_probe(self, path: Path) -> tuple[LinearProbe, dict]:
        """Load probe weights."""
        data = np.load(path, allow_pickle=True)

        meta = json.loads(str(data["metadata"]))
        probe = LinearProbe(meta["hidden_dim"], meta["vocab_size"])

        probe.linear.weight = mx.array(data["weight"])
        if data["bias"] is not None:
            probe.linear.bias = mx.array(data["bias"])

        return probe, meta


def print_evaluation(results: list[ProbeResult], layer: int):
    """Print evaluation summary."""
    n_total = len(results)
    n_model_correct = sum(1 for r in results if r.model_correct)
    n_probe_correct = sum(1 for r in results if r.probe_correct)
    n_ghosts = sum(1 for r in results if r.probe_found_ghost)

    print(f"\n{'='*70}")
    print(f"PROBE EVALUATION (Layer {layer})")
    print(f"{'='*70}")
    print(f"Total examples: {n_total}")
    print(f"Model accuracy: {n_model_correct}/{n_total} ({100*n_model_correct/n_total:.1f}%)")
    print(f"Probe accuracy: {n_probe_correct}/{n_total} ({100*n_probe_correct/n_total:.1f}%)")
    print(f"Ghosts found:   {n_ghosts}/{n_total} (probe correct when model wrong)")

    if n_ghosts > 0:
        print(f"\n🔥 GHOST EXAMPLES (probe found answer model missed):")
        for r in results:
            if r.probe_found_ghost:
                print(f"  {r.prompt}")
                print(f"    Target: {repr(r.target_token)}")
                print(f"    Model:  {repr(r.model_prediction)} ❌")
                print(f"    Probe:  {repr(r.probe_prediction)} ✅ (rank {r.probe_rank}, {r.probe_probability:.1%})")

    # Sample of results
    print(f"\n--- Sample Results ---")
    for r in results[:10]:
        model_mark = "✅" if r.model_correct else "❌"
        probe_mark = "✅" if r.probe_correct else "❌"
        print(f"  {r.prompt:<20} target={r.target_token} model={r.model_prediction}{model_mark} probe={r.probe_prediction}{probe_mark}")


async def cmd_train(args):
    """Train a probe."""
    trainer = await ProbeTrainer.from_pretrained(args.model)

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = ArithmeticDataset(trainer.tokenizer, num_examples=args.num_examples)

    # Split train/test
    split = int(len(dataset) * 0.8)
    train_data = ArithmeticDataset.__new__(ArithmeticDataset)
    train_data.tokenizer = trainer.tokenizer
    train_data.examples = dataset.examples[:split]

    test_data = ArithmeticDataset.__new__(ArithmeticDataset)
    test_data.tokenizer = trainer.tokenizer
    test_data.examples = dataset.examples[split:]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Train
    probe = trainer.train_probe(train_data, args.layer, epochs=args.epochs, lr=args.lr)

    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate_probe(probe, test_data, args.layer)
    print_evaluation(results, args.layer)

    # Save
    probe_path = Path(args.probe_path or f"probes/arithmetic_L{args.layer}.npz")
    trainer.save_probe(probe, probe_path, args.layer, {
        "train_size": len(train_data),
        "test_size": len(test_data),
    })


async def cmd_eval(args):
    """Evaluate a saved probe."""
    trainer = await ProbeTrainer.from_pretrained(args.model)

    probe, meta = trainer.load_probe(Path(args.probe_path))
    layer = meta["layer"]

    print(f"Loaded probe for layer {layer}")

    if args.prompt:
        # Single prompt evaluation
        h, model_logits = trainer.extract_hidden_state(args.prompt, layer)

        probe_logits = probe(h)
        probe_probs = mx.softmax(probe_logits[0])

        model_probs = mx.softmax(model_logits[0])

        print(f"\nPrompt: {repr(args.prompt)}")
        print(f"\nProbe top-5 (Layer {layer}):")
        sorted_idx = mx.argsort(probe_probs)[::-1][:5].tolist()
        for i, idx in enumerate(sorted_idx):
            token = trainer.tokenizer.decode([idx])
            prob = float(probe_probs[idx])
            print(f"  {i+1}. {repr(token):10} {prob:.4f}")

        print(f"\nModel top-5 (final):")
        sorted_idx = mx.argsort(model_probs)[::-1][:5].tolist()
        for i, idx in enumerate(sorted_idx):
            token = trainer.tokenizer.decode([idx])
            prob = float(model_probs[idx])
            print(f"  {i+1}. {repr(token):10} {prob:.4f}")
    else:
        # Evaluate on generated dataset
        dataset = ArithmeticDataset(trainer.tokenizer, num_examples=100)
        results = trainer.evaluate_probe(probe, dataset, layer)
        print_evaluation(results, layer)


async def cmd_sweep(args):
    """Sweep layers to find best probe layer."""
    trainer = await ProbeTrainer.from_pretrained(args.model)

    print("\nGenerating dataset...")
    dataset = ArithmeticDataset(trainer.tokenizer, num_examples=200)

    split = int(len(dataset) * 0.8)
    train_data = ArithmeticDataset.__new__(ArithmeticDataset)
    train_data.tokenizer = trainer.tokenizer
    train_data.examples = dataset.examples[:split]

    test_data = ArithmeticDataset.__new__(ArithmeticDataset)
    test_data.tokenizer = trainer.tokenizer
    test_data.examples = dataset.examples[split:]

    num_layers = trainer.config.num_hidden_layers
    layers_to_test = list(range(0, num_layers, 4)) + [num_layers - 1]
    if 20 not in layers_to_test:
        layers_to_test.append(20)
    if 21 not in layers_to_test:
        layers_to_test.append(21)
    if 22 not in layers_to_test:
        layers_to_test.append(22)
    layers_to_test = sorted(set(l for l in layers_to_test if l < num_layers))

    print(f"\nSweeping layers: {layers_to_test}")
    print(f"{'='*60}")

    results_summary = []

    for layer in layers_to_test:
        print(f"\n--- Layer {layer} ---")
        probe = trainer.train_probe(train_data, layer, epochs=5, lr=0.01)
        results = trainer.evaluate_probe(probe, test_data, layer)

        n_probe_correct = sum(1 for r in results if r.probe_correct)
        n_ghosts = sum(1 for r in results if r.probe_found_ghost)
        accuracy = n_probe_correct / len(results)

        results_summary.append((layer, accuracy, n_ghosts))
        print(f"  Accuracy: {accuracy:.1%}, Ghosts: {n_ghosts}")

    # Summary
    print(f"\n{'='*60}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Accuracy':<12} {'Ghosts'}")
    print("-" * 35)
    for layer, acc, ghosts in results_summary:
        bar = "█" * int(acc * 20)
        print(f"L{layer:<6} {acc:.1%} {bar:<12} {ghosts}")

    best_layer, best_acc, _ = max(results_summary, key=lambda x: x[1])
    print(f"\n✓ Best layer: {best_layer} ({best_acc:.1%} accuracy)")


def main():
    parser = argparse.ArgumentParser(description="Linear probe for hidden state extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train a probe")
    train_parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    train_parser.add_argument("--layer", "-l", type=int, default=20)
    train_parser.add_argument("--num-examples", "-n", type=int, default=500)
    train_parser.add_argument("--epochs", "-e", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--probe-path", "-o", type=str)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a probe")
    eval_parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    eval_parser.add_argument("--probe-path", "-p", required=True)
    eval_parser.add_argument("--prompt", type=str)

    # Sweep
    sweep_parser = subparsers.add_parser("sweep", help="Sweep layers to find best")
    sweep_parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")

    args = parser.parse_args()

    if args.command == "train":
        asyncio.run(cmd_train(args))
    elif args.command == "eval":
        asyncio.run(cmd_eval(args))
    elif args.command == "sweep":
        asyncio.run(cmd_sweep(args))


if __name__ == "__main__":
    main()
