"""
Probe Classifier Experiment

Tests whether task information is encoded at intermediate layers,
even if not vocabulary-aligned.

Key question: Can a linear probe extract task labels from hidden states?

If yes → Routing is possible via learned projections
If no → Task information only emerges at final layers

This is critical for virtual expert architecture:
- We don't need vocabulary classifiers
- A learned routing matrix can replace logit lens
"""

import json
import logging
import random
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Results for a single layer probe."""

    layer_idx: int
    layer_pct: float
    train_accuracy: float
    test_accuracy: float
    loss_history: list[float] = field(default_factory=list)


class LinearProbe(nn.Module):
    """Simple linear probe for classification."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def __call__(self, x):
        return self.linear(x)


class ProbeClassifierExperiment(ExperimentBase):
    """Probe experiment to detect task info at each layer."""

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up probe classifier experiment...")

        self.params = self.config.parameters

        # Task labels
        self.task_to_idx = {"multiply": 0, "add": 1, "subtract": 2}
        self.idx_to_task = {v: k for k, v in self.task_to_idx.items()}

        # Generate data
        self._ensure_data()

        self.probe_results: dict[int, ProbeResult] = {}

    def _ensure_data(self) -> None:
        """Generate training data if needed."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.config.data_dir / "train.jsonl"
        if train_path.exists():
            self.log("Using existing data")
            return

        self.log("Generating training data...")
        random.seed(self.params.get("seed", 42))

        num_samples = self.params.get("num_samples", 2000)
        operations = [
            ("multiply", "*", lambda a, b: a * b),
            ("add", "+", lambda a, b: a + b),
            ("subtract", "-", lambda a, b: a - b),
        ]

        data = []
        for _ in range(num_samples):
            op_name, op_sym, op_fn = random.choice(operations)

            if op_name == "multiply":
                a, b = random.randint(2, 12), random.randint(2, 12)
            else:
                a, b = random.randint(1, 50), random.randint(1, 50)
                if op_name == "subtract":
                    a, b = max(a, b), min(a, b)

            result = op_fn(a, b)
            data.append(
                {
                    "prompt": f"{a} {op_sym} {b} = ",
                    "task": op_name,
                    "result": result,
                }
            )

        split = int(len(data) * 0.8)
        train_data, test_data = data[:split], data[split:]

        with open(train_path, "w") as f:
            for e in train_data:
                f.write(json.dumps(e) + "\n")

        with open(self.config.data_dir / "test.jsonl", "w") as f:
            for e in test_data:
                f.write(json.dumps(e) + "\n")

        self.log(f"Generated {len(train_data)} train + {len(test_data)} test samples")

    def run(self) -> dict:
        """Run probe experiment on all layers."""
        self.log("=" * 60)
        self.log("PROBE CLASSIFIER EXPERIMENT")
        self.log("Testing if task info exists at intermediate layers")
        self.log("=" * 60)

        # Load model using framework
        loaded = self.load_model()
        model, tokenizer = loaded.model, loaded.tokenizer
        num_layers = loaded.config.num_hidden_layers
        hidden_dim = loaded.config.hidden_size
        self.log(f"Model: {self.config.model}")
        self.log(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")

        # Load data
        train_data = self._load_data("train.jsonl")
        test_data = self._load_data("test.jsonl")
        self.log(f"Train: {len(train_data)}, Test: {len(test_data)}")

        # Probe each layer
        layer_pcts = self.params.get("probe_layers_pct", [0.25, 0.5, 0.75, 0.95])

        for pct in layer_pcts:
            layer_idx = int(pct * num_layers)
            layer_idx = min(layer_idx, num_layers - 1)

            self.log(f"\n--- Probing Layer {layer_idx} ({pct:.0%} depth) ---")

            result = self._probe_layer(
                model, tokenizer, layer_idx, train_data, test_data, hidden_dim, num_layers
            )
            self.probe_results[layer_idx] = result

            self.log(f"  Train accuracy: {result.train_accuracy:.1%}")
            self.log(f"  Test accuracy:  {result.test_accuracy:.1%}")

        return self._build_results()

    def _load_data(self, filename: str) -> list[dict]:
        """Load data from JSONL file."""
        data = []
        with open(self.config.data_dir / filename) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _extract_hidden_states(
        self, model, tokenizer, prompts: list[str], layer_idx: int
    ) -> mx.array:
        """Extract hidden states at specified layer for all prompts."""
        hidden_states = []

        for prompt in prompts:
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]

            # Forward through embedding and layers
            h = model.model.embed_tokens(input_ids)

            for i, layer in enumerate(model.model.layers):
                layer_out = layer(h, mask=None, cache=None)
                h = (
                    layer_out.hidden_states
                    if hasattr(layer_out, "hidden_states")
                    else (layer_out[0] if isinstance(layer_out, tuple) else layer_out)
                )

                if i == layer_idx:
                    # Take last token's hidden state
                    hidden_states.append(h[0, -1, :])
                    break

        return mx.stack(hidden_states)

    def _probe_layer(
        self,
        model,
        tokenizer,
        layer_idx: int,
        train_data: list[dict],
        test_data: list[dict],
        hidden_dim: int,
        num_layers: int,
    ) -> ProbeResult:
        """Train and evaluate a linear probe at specified layer."""
        # Extract hidden states
        train_prompts = [d["prompt"] for d in train_data]
        train_labels = mx.array([self.task_to_idx[d["task"]] for d in train_data])

        test_prompts = [d["prompt"] for d in test_data]
        test_labels = mx.array([self.task_to_idx[d["task"]] for d in test_data])

        self.log("  Extracting hidden states...")
        train_hidden = self._extract_hidden_states(model, tokenizer, train_prompts, layer_idx)
        test_hidden = self._extract_hidden_states(model, tokenizer, test_prompts, layer_idx)
        mx.eval(train_hidden, test_hidden)

        # Create and train probe
        probe = LinearProbe(hidden_dim, len(self.task_to_idx))
        optimizer = optim.Adam(learning_rate=self.params.get("probe_lr", 0.01))

        loss_and_grad_fn = nn.value_and_grad(probe, self._loss_fn)
        loss_history = []

        epochs = self.params.get("probe_epochs", 100)
        batch_size = self.params.get("probe_batch_size", 32)

        self.log(f"  Training probe for {epochs} epochs...")
        for epoch in range(epochs):
            # Shuffle data
            perm = mx.array(random.sample(range(len(train_data)), len(train_data)))
            train_hidden_shuffled = train_hidden[perm]
            train_labels_shuffled = train_labels[perm]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(train_data), batch_size):
                batch_x = train_hidden_shuffled[i : i + batch_size]
                batch_y = train_labels_shuffled[i : i + batch_size]

                loss, grads = loss_and_grad_fn(probe, batch_x, batch_y)
                optimizer.update(probe, grads)
                mx.eval(probe.parameters(), optimizer.state)

                epoch_loss += float(loss)
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                self.log(f"    Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        # Evaluate
        train_acc = self._evaluate_probe(probe, train_hidden, train_labels)
        test_acc = self._evaluate_probe(probe, test_hidden, test_labels)

        layer_pct = layer_idx / num_layers
        return ProbeResult(
            layer_idx=layer_idx,
            layer_pct=layer_pct,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            loss_history=loss_history,
        )

    def _loss_fn(self, probe: LinearProbe, x: mx.array, y: mx.array) -> mx.array:
        """Cross-entropy loss."""
        logits = probe(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    def _evaluate_probe(self, probe: LinearProbe, hidden: mx.array, labels: mx.array) -> float:
        """Evaluate probe accuracy."""
        logits = probe(hidden)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        correct = mx.sum(preds == labels)
        return float(correct) / len(labels)

    def _build_results(self) -> dict:
        """Build results dict."""
        results = {
            "model": self.config.model,
            "layers": {},
        }

        best_layer = None
        best_acc = 0.0

        for layer_idx, r in self.probe_results.items():
            results["layers"][f"L{layer_idx}"] = {
                "layer_pct": r.layer_pct,
                "train_accuracy": r.train_accuracy,
                "test_accuracy": r.test_accuracy,
            }
            if r.test_accuracy > best_acc:
                best_acc = r.test_accuracy
                best_layer = layer_idx

        results["summary"] = {
            "best_layer": best_layer,
            "best_accuracy": best_acc,
            "routing_viable": best_acc > 0.9,
        }

        # Log summary
        self.log("\n" + "=" * 60)
        self.log("SUMMARY")
        self.log("=" * 60)
        self.log(f"Best layer: L{best_layer} ({best_acc:.1%} test accuracy)")
        self.log(f"Routing viable: {'YES' if best_acc > 0.9 else 'NO'}")

        if best_acc > 0.9:
            self.log("\n>>> Task info IS encoded at intermediate layers!")
            self.log(">>> Linear routing can replace vocabulary classifiers.")
        else:
            self.log("\n>>> Task info is NOT strongly encoded.")
            self.log(">>> May need vocabulary-aligned classifiers or deeper probes.")

        return results

    def evaluate(self) -> dict:
        """Return summary metrics."""
        if self.probe_results:
            best = max(self.probe_results.values(), key=lambda r: r.test_accuracy)
            return {
                "best_layer": best.layer_idx,
                "best_accuracy": best.test_accuracy,
                "routing_viable": best.test_accuracy > 0.9,
            }
        return {"error": "No results"}

    def cleanup(self) -> None:
        """Cleanup."""
        self.probe_results = {}
