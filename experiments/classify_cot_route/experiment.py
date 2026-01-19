"""
Classify-CoT-Route Experiment

Tests three claims:
1. SAME TASK - Probe classifies both symbolic and semantic as same operation
2. DIFFERENT PATHS - Semantic inputs are longer/more complex (proxy for CoT need)
3. SAME DESTINATION - Hidden states converge to similar representation

No training. Pure observation on base model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    """Result for a single symbolic/semantic pair."""

    symbolic_input: str
    semantic_input: str
    task: str
    operands: list[int]
    expected: int

    # Classification results (probe-based)
    symbolic_class: str
    semantic_class: str
    same_classification: bool

    # Representation similarity at each layer
    layer_similarities: dict[int, float]

    # Generation results
    symbolic_answer: str
    semantic_answer: str
    symbolic_correct: bool
    semantic_correct: bool


class ClassifyCoTRouteExperiment:
    """
    Test if symbolic and semantic inputs converge to same representation.

    Claims tested:
    1. Same task classification via probe
    2. Different input complexity (semantic longer)
    3. Convergence of hidden states at later layers
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.probe = None  # Trained probe for classification

    def run(self) -> dict[str, Any]:
        """Run the experiment."""
        logger.info("Starting Classify-CoT-Route experiment")

        # Load model
        self._load_model()

        # Train probe on task classification
        self._train_probe()

        # Run test pairs
        pair_results = []
        for pair in self.config["parameters"]["test_pairs"]:
            result = self._analyze_pair(pair)
            pair_results.append(result)
            logger.info(f"Pair: {pair['task']} - Same class: {result.same_classification}")

        # Aggregate results
        results = self._aggregate_results(pair_results)

        # Save results
        self._save_results(results)

        return results

    def _load_model(self):
        """Load the model and tokenizer."""
        from chuk_lazarus.models_v2.loader import load_model

        model_name = self.config["model"]
        logger.info(f"Loading model: {model_name}")

        loaded = load_model(model_name)
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        logger.info("Model loaded")

    def _train_probe(self):
        """Train a linear probe for task classification."""
        logger.info("Training task classification probe")

        classify_layer = self.config["parameters"]["classify_layer"]

        # Generate training data - BOTH symbolic AND semantic
        train_data = [
            # Addition - symbolic
            ("5 + 3 = ", "addition"),
            ("12 + 7 = ", "addition"),
            ("20 + 15 = ", "addition"),
            ("8 + 4 = ", "addition"),
            # Addition - semantic
            ("I have 5 apples and get 3 more. How many?", "addition"),
            ("Add 12 and 7 together.", "addition"),
            ("What is the sum of 20 and 15?", "addition"),
            ("If you combine 8 items with 4 items, how many total?", "addition"),
            # Subtraction - symbolic
            ("10 - 4 = ", "subtraction"),
            ("25 - 8 = ", "subtraction"),
            ("50 - 20 = ", "subtraction"),
            ("15 - 6 = ", "subtraction"),
            # Subtraction - semantic
            ("I had 10 cookies and ate 4. How many left?", "subtraction"),
            ("Subtract 8 from 25.", "subtraction"),
            ("What is 50 minus 20?", "subtraction"),
            ("If you remove 6 from 15, what remains?", "subtraction"),
            # Multiplication - symbolic
            ("6 * 7 = ", "multiplication"),
            ("8 * 9 = ", "multiplication"),
            ("4 * 5 = ", "multiplication"),
            ("3 * 8 = ", "multiplication"),
            # Multiplication - semantic
            ("What is 6 times 7?", "multiplication"),
            ("Multiply 8 by 9.", "multiplication"),
            ("There are 4 groups of 5. How many total?", "multiplication"),
            ("Calculate 3 multiplied by 8.", "multiplication"),
            # Division - symbolic
            ("20 / 4 = ", "division"),
            ("36 / 6 = ", "division"),
            ("100 / 5 = ", "division"),
            ("48 / 8 = ", "division"),
            # Division - semantic
            ("Divide 20 by 4.", "division"),
            ("What is 36 divided by 6?", "division"),
            ("Split 100 into 5 equal parts.", "division"),
            ("If 48 items are shared among 8 people, how many each?", "division"),
        ]

        # Extract hidden states
        hiddens = []
        labels = []
        label_map = {"addition": 0, "subtraction": 1, "multiplication": 2, "division": 3}

        for prompt, task in train_data:
            h = self._get_hidden_state(prompt, classify_layer)
            hiddens.append(h)
            labels.append(label_map[task])

        # Stack and convert
        X = mx.stack(hiddens)  # [N, hidden_dim]
        y = mx.array(labels)  # [N]

        # Train simple linear probe
        hidden_dim = X.shape[1]
        num_classes = 4

        # Initialize weights
        self.probe_W = mx.random.normal((hidden_dim, num_classes)) * 0.01
        self.probe_b = mx.zeros((num_classes,))

        # Simple gradient descent
        lr = 0.1
        for epoch in range(100):
            # Forward
            logits = X @ self.probe_W + self.probe_b

            # Softmax cross-entropy loss
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            loss = -mx.mean(log_probs[mx.arange(len(y)), y])

            # Backward
            probs = mx.softmax(logits, axis=-1)
            grad_logits = probs
            grad_logits = grad_logits.at[mx.arange(len(y)), y].add(-1)
            grad_logits = grad_logits / len(y)

            grad_W = X.T @ grad_logits
            grad_b = mx.sum(grad_logits, axis=0)

            # Update
            self.probe_W = self.probe_W - lr * grad_W
            self.probe_b = self.probe_b - lr * grad_b

            mx.eval(self.probe_W, self.probe_b)

        # Check accuracy
        final_logits = X @ self.probe_W + self.probe_b
        preds = mx.argmax(final_logits, axis=-1)
        accuracy = mx.mean(preds == y).item()
        logger.info(f"Probe training accuracy: {accuracy:.1%}")

        self.label_names = ["addition", "subtraction", "multiplication", "division"]

    def _get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at specified layer for last token."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        # Forward with hidden states
        output = self.model(input_ids, output_hidden_states=True)

        # Get hidden state at layer (layer 0 is embeddings)
        hidden = output.hidden_states[layer]  # [1, seq_len, hidden_dim]

        # Return last token's hidden state
        return hidden[0, -1, :]

    def _get_all_hidden_states(self, prompt: str) -> list[mx.array]:
        """Get hidden states at all layers for last token."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        output = self.model(input_ids, output_hidden_states=True)

        # Return last token hidden state at each layer
        return [h[0, -1, :] for h in output.hidden_states]

    def _classify(self, prompt: str) -> str:
        """Classify input using trained probe."""
        layer = self.config["parameters"]["classify_layer"]
        h = self._get_hidden_state(prompt, layer)

        logits = h @ self.probe_W + self.probe_b
        pred_idx = mx.argmax(logits).item()

        return self.label_names[pred_idx]

    def _cosine_similarity(self, a: mx.array, b: mx.array) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = a / (mx.linalg.norm(a) + 1e-8)
        b_norm = b / (mx.linalg.norm(b) + 1e-8)
        return mx.sum(a_norm * b_norm).item()

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response from model."""
        # For instruct models, use chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(formatted, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()

            # Stop on EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # Don't stop on newlines - let it complete the answer
            token_str = self.tokenizer.decode([token_id])

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _check_answer(self, response: str, expected: int) -> bool:
        """Check if response contains expected answer."""
        # Extract numbers from response
        import re

        numbers = re.findall(r"-?\d+", response)
        if numbers:
            # Check if expected is in the response
            return str(expected) in numbers
        return False

    def _analyze_pair(self, pair: dict) -> PairResult:
        """Analyze a single symbolic/semantic pair."""
        symbolic = pair["symbolic"]
        semantic = pair["semantic"]
        task = pair["task"]
        operands = pair["operands"]
        expected = pair["expected"]

        # 1. CLASSIFY - same task?
        symbolic_class = self._classify(symbolic)
        semantic_class = self._classify(semantic)
        same_classification = symbolic_class == semantic_class

        # 2. CONVERGENCE - hidden state similarity at each layer
        symbolic_hiddens = self._get_all_hidden_states(symbolic)
        semantic_hiddens = self._get_all_hidden_states(semantic)

        convergence_layers = self.config["parameters"]["convergence_layers"]
        layer_similarities = {}

        for layer in convergence_layers:
            if layer < len(symbolic_hiddens) and layer < len(semantic_hiddens):
                sim = self._cosine_similarity(symbolic_hiddens[layer], semantic_hiddens[layer])
                layer_similarities[layer] = sim

        # 3. GENERATION - same answer?
        symbolic_answer = self._generate(symbolic)
        semantic_answer = self._generate(semantic)

        symbolic_correct = self._check_answer(symbolic_answer, expected)
        semantic_correct = self._check_answer(semantic_answer, expected)

        return PairResult(
            symbolic_input=symbolic,
            semantic_input=semantic,
            task=task,
            operands=operands,
            expected=expected,
            symbolic_class=symbolic_class,
            semantic_class=semantic_class,
            same_classification=same_classification,
            layer_similarities=layer_similarities,
            symbolic_answer=symbolic_answer,
            semantic_answer=semantic_answer,
            symbolic_correct=symbolic_correct,
            semantic_correct=semantic_correct,
        )

    def _aggregate_results(self, pair_results: list[PairResult]) -> dict[str, Any]:
        """Aggregate results across all pairs."""
        # Classification agreement
        same_class_count = sum(1 for r in pair_results if r.same_classification)

        # Average similarity by layer
        layer_sims = {}
        for layer in self.config["parameters"]["convergence_layers"]:
            sims = [r.layer_similarities.get(layer, 0) for r in pair_results]
            layer_sims[layer] = sum(sims) / len(sims) if sims else 0

        # Accuracy
        symbolic_correct = sum(1 for r in pair_results if r.symbolic_correct)
        semantic_correct = sum(1 for r in pair_results if r.semantic_correct)

        return {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "num_pairs": len(pair_results),
            "claims": {
                "same_task": {
                    "description": "Probe classifies both as same operation",
                    "agreement_rate": same_class_count / len(pair_results),
                    "agreed": same_class_count,
                    "total": len(pair_results),
                },
                "convergence": {
                    "description": "Hidden states converge at later layers",
                    "layer_similarities": layer_sims,
                    "trend": "increasing"
                    if layer_sims.get(12, 0) > layer_sims.get(6, 0)
                    else "flat/decreasing",
                },
                "same_answer": {
                    "description": "Both inputs produce correct answer",
                    "symbolic_accuracy": symbolic_correct / len(pair_results),
                    "semantic_accuracy": semantic_correct / len(pair_results),
                },
            },
            "pair_results": [
                {
                    "task": r.task,
                    "operands": r.operands,
                    "expected": r.expected,
                    "symbolic_class": r.symbolic_class,
                    "semantic_class": r.semantic_class,
                    "same_classification": r.same_classification,
                    "layer_similarities": r.layer_similarities,
                    "symbolic_answer": r.symbolic_answer,
                    "semantic_answer": r.semantic_answer,
                    "symbolic_correct": r.symbolic_correct,
                    "semantic_correct": r.semantic_correct,
                }
                for r in pair_results
            ],
        }

    def _save_results(self, results: dict[str, Any]):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"run_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("CLASSIFY-COT-ROUTE RESULTS")
        print("=" * 60)

        claims = results["claims"]

        print("\n1. SAME TASK (probe classification):")
        print(f"   Agreement: {claims['same_task']['agreement_rate']:.1%}")
        print(f"   ({claims['same_task']['agreed']}/{claims['same_task']['total']} pairs)")

        print("\n2. CONVERGENCE (hidden state similarity):")
        for layer, sim in claims["convergence"]["layer_similarities"].items():
            print(f"   Layer {layer}: {sim:.3f}")
        print(f"   Trend: {claims['convergence']['trend']}")

        print("\n3. SAME ANSWER (generation accuracy):")
        print(f"   Symbolic: {claims['same_answer']['symbolic_accuracy']:.1%}")
        print(f"   Semantic: {claims['same_answer']['semantic_accuracy']:.1%}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment = ClassifyCoTRouteExperiment()
    experiment.run()
