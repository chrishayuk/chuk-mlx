"""
Format Gate Detection on GPT-OSS

Research Question:
Is GPT-OSS's L13 vocab-aligned classifier the same mechanism as Llama's L8 CoT direction?

Hypothesis:
Same function, different encoding:
- Llama L8: Linear direction (CoT vector), not vocab-aligned
- GPT-OSS L13: Linear direction that happens to align with vocab (scale effect)

Tests:
1. FORMAT PROBE: Can we classify symbolic vs semantic at L13?
2. COT DIRECTION: Does (semantic - symbolic) direction work for steering?
3. VOCAB ALIGNMENT: Does L13 hidden state project onto operation tokens?
4. DUAL STEERING: Does vocab-space steering work as well as geometric steering?
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class VocabAlignmentResult:
    """Result for vocab alignment analysis."""
    layer: int
    task: str  # "addition", "multiplication", etc.
    top_tokens: list[tuple[str, float]]  # (token, projection)
    target_token_rank: int | None  # rank of expected token
    target_token_projection: float | None


class GptOssFormatGateExperiment:
    """
    Test if GPT-OSS L13 has the same format gate as Llama L8,
    just with vocab-aligned encoding.
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
        self.probes = {}

    def run(self) -> dict[str, Any]:
        """Run the full experiment."""
        logger.info("Starting GPT-OSS Format Gate experiment")

        # Load model
        self._load_model()

        results = {
            "model": self.config["model"],
            "timestamp": datetime.now().isoformat(),
            "num_layers": self.num_layers,
        }

        # 1. Format classification probes
        logger.info("=== Testing Format Classification ===")
        probe_results = self._train_format_probes()
        results["format_classification"] = probe_results

        # 2. CoT direction (geometric)
        logger.info("=== Computing CoT Direction ===")
        cot_direction_results = self._compute_cot_direction()
        results["cot_direction"] = cot_direction_results

        # 3. Vocab alignment test
        logger.info("=== Testing Vocab Alignment ===")
        vocab_results = self._test_vocab_alignment()
        results["vocab_alignment"] = vocab_results

        # 4. Generation correlation
        logger.info("=== Testing Generation Correlation ===")
        gen_results = self._test_generation_correlation()
        results["generation_correlation"] = gen_results

        # 5. Steering test (if model supports it)
        if self.config["parameters"].get("run_steering", False):
            logger.info("=== Testing Steering ===")
            steering_results = self._test_steering()
            results["steering"] = steering_results

        self._save_results(results)
        return results

    def _load_model(self):
        """Load the model and tokenizer."""
        from chuk_lazarus.models_v2.loader import load_model

        model_name = self.config["model"]
        logger.info(f"Loading model: {model_name}")

        try:
            loaded = load_model(model_name)
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            logger.info("Falling back to TinyLlama for demonstration")
            loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.model = loaded.model
            self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers")

    def _get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at specified layer for last token."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        output = self.model(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[layer]
        return hidden[0, -1, :]

    def _train_format_probes(self) -> dict[str, Any]:
        """Train format classification probes at each layer."""
        train_data = self.config["parameters"]["train_data"]
        test_data = self.config["parameters"]["test_data"]
        layers_to_test = self.config["parameters"]["layers_to_test"]

        results = {"by_layer": {}}

        for layer in layers_to_test:
            if layer >= self.num_layers:
                continue

            logger.info(f"Training probe at layer {layer}")

            # Extract hidden states
            train_hiddens = []
            train_labels = []
            for item in train_data:
                h = self._get_hidden_state(item["prompt"], layer)
                train_hiddens.append(h)
                train_labels.append(0 if item["format"] == "symbolic" else 1)

            X_train = mx.stack(train_hiddens)
            y_train = mx.array(train_labels)

            # Train probe
            W, b = self._train_probe(X_train, y_train)
            self.probes[layer] = (W, b)

            # Evaluate
            train_logits = X_train @ W + b
            train_preds = mx.argmax(train_logits, axis=-1)
            train_acc = mx.mean(train_preds == y_train).item()

            test_hiddens = []
            test_labels = []
            for item in test_data:
                h = self._get_hidden_state(item["prompt"], layer)
                test_hiddens.append(h)
                test_labels.append(0 if item["format"] == "symbolic" else 1)

            X_test = mx.stack(test_hiddens)
            y_test = mx.array(test_labels)

            test_logits = X_test @ W + b
            test_preds = mx.argmax(test_logits, axis=-1)
            test_acc = mx.mean(test_preds == y_test).item()

            results["by_layer"][layer] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }

            logger.info(f"  Layer {layer}: train={train_acc:.1%}, test={test_acc:.1%}")

        return results

    def _train_probe(self, X: mx.array, y: mx.array, epochs: int = 100) -> tuple[mx.array, mx.array]:
        """Train a linear probe."""
        hidden_dim = X.shape[1]
        W = mx.random.normal((hidden_dim, 2)) * 0.01
        b = mx.zeros((2,))
        lr = 0.1

        for _ in range(epochs):
            logits = X @ W + b
            probs = mx.softmax(logits, axis=-1)
            grad_logits = probs
            grad_logits = grad_logits.at[mx.arange(len(y)), y].add(-1)
            grad_logits = grad_logits / len(y)

            grad_W = X.T @ grad_logits
            grad_b = mx.sum(grad_logits, axis=0)

            W = W - lr * grad_W
            b = b - lr * grad_b
            mx.eval(W, b)

        return W, b

    def _compute_cot_direction(self) -> dict[str, Any]:
        """Compute CoT direction vector (same as Llama experiment)."""
        steering_layer = self.config["parameters"].get("steering_layer", 12)
        train_data = self.config["parameters"]["train_data"]

        if steering_layer >= self.num_layers:
            steering_layer = self.num_layers // 2

        symbolic_prompts = [item["prompt"] for item in train_data if item["format"] == "symbolic"]
        semantic_prompts = [item["prompt"] for item in train_data if item["format"] == "semantic"]

        symbolic_hiddens = [self._get_hidden_state(p, steering_layer) for p in symbolic_prompts]
        semantic_hiddens = [self._get_hidden_state(p, steering_layer) for p in semantic_prompts]

        symbolic_mean = mx.mean(mx.stack(symbolic_hiddens), axis=0)
        semantic_mean = mx.mean(mx.stack(semantic_hiddens), axis=0)

        cot_direction = semantic_mean - symbolic_mean
        cot_direction_norm = mx.linalg.norm(cot_direction).item()

        # Normalize
        cot_direction_unit = cot_direction / (cot_direction_norm + 1e-8)

        self.cot_direction = cot_direction_unit
        self.cot_direction_unnorm = cot_direction

        return {
            "layer": steering_layer,
            "direction_norm": cot_direction_norm,
            "symbolic_mean_norm": mx.linalg.norm(symbolic_mean).item(),
            "semantic_mean_norm": mx.linalg.norm(semantic_mean).item(),
        }

    def _test_vocab_alignment(self) -> dict[str, Any]:
        """
        Test if hidden states project onto operation tokens.

        This is the key test: GPT-OSS supposedly has vocab-aligned classifiers.
        We check if projecting hidden states onto the unembedding matrix
        gives high logits for operation words.
        """
        vocab_tokens = self.config["parameters"].get("vocab_alignment_tokens", {})
        target_layer = self.config["parameters"].get("probe_layer_for_generation", 13)

        if target_layer >= self.num_layers:
            target_layer = self.num_layers // 2

        # Get unembedding matrix (lm_head weight)
        unembed = None
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
            # Case 1: Untied weights - lm_head.lm_head is nn.Linear
            if hasattr(lm_head, 'lm_head') and lm_head.lm_head is not None:
                unembed = lm_head.lm_head.weight  # [vocab, hidden]
            # Case 2: Tied weights - use embedding
            elif hasattr(lm_head, 'tied_embeddings') and lm_head.tied_embeddings is not None:
                emb = lm_head.tied_embeddings
                if hasattr(emb, 'weight'):
                    if hasattr(emb.weight, 'weight'):
                        unembed = emb.weight.weight  # TokenEmbedding wrapper
                    else:
                        unembed = emb.weight
            # Case 3: Direct proj attribute
            elif hasattr(lm_head, 'proj'):
                unembed = lm_head.proj.weight

        if unembed is None:
            logger.warning("Could not access unembedding matrix")
            return {"error": "Could not access unembedding matrix"}

        results = {"layer": target_layer, "by_task": {}}

        train_data = self.config["parameters"]["train_data"]

        for task, expected_tokens in vocab_tokens.items():
            # Get task-specific prompts
            task_symbolic = [
                item["prompt"] for item in train_data
                if item["format"] == "symbolic" and self._infer_task(item["prompt"]) == task
            ]
            task_semantic = [
                item["prompt"] for item in train_data
                if item["format"] == "semantic" and self._infer_task(item["prompt"]) == task
            ]

            if not task_symbolic and not task_semantic:
                continue

            # Get hidden states
            all_prompts = task_symbolic + task_semantic
            hiddens = [self._get_hidden_state(p, target_layer) for p in all_prompts]
            mean_hidden = mx.mean(mx.stack(hiddens), axis=0)

            # Project onto vocab space
            # logits = hidden @ unembed.T
            logits = mean_hidden @ unembed.T
            probs = mx.softmax(logits)

            # Get top tokens
            top_k = 20
            top_indices = mx.argsort(logits)[-top_k:][::-1]
            top_tokens_list = []
            for idx in top_indices:
                token_id = int(idx.item())
                token_str = self.tokenizer.decode([token_id])
                prob = probs[token_id].item()
                top_tokens_list.append((token_str, prob))

            # Check if expected tokens are in top-k
            target_ranks = {}
            for expected in expected_tokens:
                token_ids = self.tokenizer.encode(expected, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    rank = int((logits >= logits[token_id]).sum().item())
                    target_ranks[expected] = {
                        "rank": rank,
                        "probability": probs[token_id].item(),
                    }

            results["by_task"][task] = {
                "top_tokens": top_tokens_list[:10],
                "target_token_analysis": target_ranks,
                "num_prompts": len(all_prompts),
            }

            logger.info(f"  {task}: top token = '{top_tokens_list[0][0]}' ({top_tokens_list[0][1]:.3f})")

        return results

    def _infer_task(self, prompt: str) -> str:
        """Infer task type from prompt."""
        if "*" in prompt or "times" in prompt.lower() or "multiply" in prompt.lower():
            return "multiplication"
        elif "+" in prompt or "add" in prompt.lower() or "plus" in prompt.lower() or "more" in prompt.lower():
            return "addition"
        elif "-" in prompt or "subtract" in prompt.lower() or "minus" in prompt.lower() or "gave away" in prompt.lower():
            return "subtraction"
        elif "/" in prompt or "divide" in prompt.lower() or "split" in prompt.lower() or "each" in prompt.lower():
            return "division"
        return "unknown"

    def _test_generation_correlation(self) -> dict[str, Any]:
        """Test if format probe predictions correlate with generation mode."""
        probe_layer = self.config["parameters"].get("probe_layer_for_generation", 13)
        if probe_layer >= self.num_layers:
            probe_layer = self.num_layers // 2

        if probe_layer not in self.probes:
            return {"error": f"No probe at layer {probe_layer}"}

        W, b = self.probes[probe_layer]
        test_prompts = self.config["parameters"]["generation_test"]

        results = []
        for item in test_prompts:
            prompt = item["prompt"]
            expected = item["expected_format"]

            # Probe prediction
            h = self._get_hidden_state(prompt, probe_layer)
            logits = h @ W + b
            probs = mx.softmax(logits)
            pred_idx = mx.argmax(logits).item()
            pred_format = "direct" if pred_idx == 0 else "cot"

            # Generate
            generated = self._generate(prompt)
            actual_format = self._detect_format(generated)

            results.append({
                "prompt": prompt,
                "expected": expected,
                "probe_prediction": pred_format,
                "probe_confidence": probs[pred_idx].item(),
                "actual_format": actual_format,
                "correct": pred_format == actual_format,
                "generated": generated[:200],
            })

        correlation = sum(1 for r in results if r["correct"]) / len(results) if results else 0
        return {
            "correlation": correlation,
            "details": results,
        }

    def _generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _detect_format(self, text: str) -> str:
        """Detect CoT vs direct."""
        cot_patterns = [r"to find", r"we need to", r"let's", r"first,", r"therefore", r"so,"]
        text_lower = text.lower()
        cot_matches = sum(1 for p in cot_patterns if re.search(p, text_lower))
        if cot_matches >= 2 or len(text.split()) > 20:
            return "cot"
        return "direct"

    def _test_steering(self) -> dict[str, Any]:
        """Test geometric steering at the target layer."""
        steering_layer = self.config["parameters"].get("steering_layer", 12)
        if steering_layer >= self.num_layers:
            steering_layer = self.num_layers // 2

        test_prompts = ["5 * 5 = ", "7 + 3 = ", "20 - 8 = "]
        strengths = [0.0, 3.0, 5.0]

        results = []
        for prompt in test_prompts:
            prompt_results = {"prompt": prompt, "generations": []}
            for strength in strengths:
                if strength == 0.0:
                    output = self._generate(prompt)
                else:
                    output = self._generate_with_steering(prompt, steering_layer, strength)
                detected = self._detect_format(output)
                prompt_results["generations"].append({
                    "strength": strength,
                    "output": output[:150],
                    "format": detected,
                })
            results.append(prompt_results)

        return {"layer": steering_layer, "results": results}

    def _generate_with_steering(self, prompt: str, layer: int, strength: float) -> str:
        """Generate with steering vector added."""
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])
        steering_vector = self.cot_direction * strength

        generated = []
        for _ in range(150):
            batch_size, seq_len = input_ids.shape
            hidden_states = self.model.model.embed_tokens(input_ids)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)

            for layer_idx, layer_module in enumerate(self.model.model.layers):
                output = layer_module(hidden_states, mask=mask, cache=None)
                hidden_states = output.hidden_states

                if layer_idx == layer - 1:
                    steering_add = mx.zeros_like(hidden_states)
                    steering_add = steering_add.at[0, -1, :].add(steering_vector)
                    hidden_states = hidden_states + steering_add

            hidden_states = self.model.model.norm(hidden_states)
            head_output = self.model.lm_head(hidden_states)
            next_token = mx.argmax(head_output.logits[0, -1, :])
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _save_results(self, results: dict[str, Any]):
        """Save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"run_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        self._print_summary(results)

    def _print_summary(self, results: dict[str, Any]):
        """Print summary."""
        print("\n" + "=" * 70)
        print("GPT-OSS FORMAT GATE EXPERIMENT")
        print("=" * 70)

        print(f"\nModel: {results['model']}")
        print(f"Layers: {results['num_layers']}")

        if "format_classification" in results:
            fc = results["format_classification"]
            print(f"\n1. FORMAT CLASSIFICATION")
            for layer, data in sorted(fc.get("by_layer", {}).items()):
                bar = "█" * int(data["test_accuracy"] * 20)
                print(f"   L{layer:2d}: {data['test_accuracy']:5.1%} {bar}")

        if "cot_direction" in results:
            cd = results["cot_direction"]
            print(f"\n2. CoT DIRECTION")
            print(f"   Layer: {cd['layer']}")
            print(f"   Direction norm: {cd['direction_norm']:.3f}")

        if "vocab_alignment" in results and "by_task" in results["vocab_alignment"]:
            va = results["vocab_alignment"]
            print(f"\n3. VOCAB ALIGNMENT (Layer {va.get('layer', '?')})")
            for task, data in va.get("by_task", {}).items():
                if "top_tokens" in data and data["top_tokens"]:
                    top = data["top_tokens"][0]
                    print(f"   {task}: top='{top[0]}' ({top[1]:.3f})")

        if "generation_correlation" in results:
            gc = results["generation_correlation"]
            if "correlation" in gc:
                print(f"\n4. GENERATION CORRELATION: {gc['correlation']:.1%}")

        if "steering" in results:
            st = results["steering"]
            print(f"\n5. STEERING (Layer {st.get('layer', '?')})")
            for test in st.get("results", [])[:2]:
                print(f"   '{test['prompt']}' → ", end="")
                formats = [g["format"] for g in test["generations"]]
                print(f"{formats}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment = GptOssFormatGateExperiment()
    experiment.run()
