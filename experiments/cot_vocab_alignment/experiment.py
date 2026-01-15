"""
CoT Vocabulary Alignment Experiment

Tests whether Chain-of-Thought training creates vocabulary-aligned classifiers.

Hypothesis: GPT-OSS shows "multiply" at L13 because it was trained on CoT
where "multiply" appears in the output. Training on CoT format should
create vocabulary alignment at intermediate layers.

Comparison:
- Direct format:  "7 * 8 = 56"
- CoT format:     "7 * 8 = multiply: 56"

If CoT creates vocabulary alignment, we expect:
- Before CoT training: Low vocab classifier at L8 (~10%)
- After CoT training:  High vocab classifier at L8 (~50%+)
"""

import json
import logging
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import yaml

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class VocabAlignmentResult:
    """Results for vocabulary alignment check."""

    stage: str
    answer_accuracy: float
    vocab_alignment: dict[int, float]  # layer -> avg prob of task token
    per_prompt_results: list[dict] = field(default_factory=list)


class CoTVocabAlignmentExperiment(ExperimentBase):
    """Tests if CoT training creates vocabulary-aligned classifiers."""

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up CoT vocabulary alignment experiment...")

        self.params = self.config.parameters

        # Task vocabulary
        self.task_tokens = {
            "multiply": ["multiply", "multiplication", "times", "*"],
            "add": ["add", "addition", "plus", "sum", "+"],
            "subtract": ["subtract", "subtraction", "minus", "-"],
        }

        self.test_prompts = self.params.get("test_prompts", [])

        # Generate both formats of data
        self._ensure_data()

        self.results: dict[str, VocabAlignmentResult] = {}

    def _ensure_data(self) -> None:
        """Generate training data in both formats."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        direct_path = self.config.data_dir / "train_direct.jsonl"
        cot_path = self.config.data_dir / "train_cot.jsonl"

        if direct_path.exists() and cot_path.exists():
            self.log("Using existing data")
            return

        self.log("Generating training data...")
        random.seed(self.params.get("seed", 42))

        num_samples = self.params.get("num_samples", 3000)
        operations = [
            ("multiply", "*", lambda a, b: a * b),
            ("add", "+", lambda a, b: a + b),
            ("subtract", "-", lambda a, b: a - b),
        ]

        direct_data = []
        cot_data = []

        for _ in range(num_samples):
            op_name, op_sym, op_fn = random.choice(operations)

            if op_name == "multiply":
                a, b = random.randint(2, 12), random.randint(2, 12)
            else:
                a, b = random.randint(1, 50), random.randint(1, 50)
                if op_name == "subtract":
                    a, b = max(a, b), min(a, b)

            result = op_fn(a, b)

            # Direct format: "7 * 8 = 56"
            direct_data.append(
                {
                    "text": f"{a} {op_sym} {b} = {result}",
                }
            )

            # CoT format: "7 * 8 = multiply: 56"
            cot_data.append(
                {
                    "text": f"{a} {op_sym} {b} = {op_name}: {result}",
                }
            )

        # Save direct format
        with open(direct_path, "w") as f:
            for e in direct_data:
                f.write(json.dumps(e) + "\n")

        # Save CoT format
        with open(cot_path, "w") as f:
            for e in cot_data:
                f.write(json.dumps(e) + "\n")

        # Validation sets
        split = int(num_samples * 0.1)
        with open(self.config.data_dir / "valid_direct.jsonl", "w") as f:
            for e in direct_data[:split]:
                f.write(json.dumps(e) + "\n")

        with open(self.config.data_dir / "valid_cot.jsonl", "w") as f:
            for e in cot_data[:split]:
                f.write(json.dumps(e) + "\n")

        self.log(f"Generated {num_samples} samples in both formats")
        self.log("  Direct: '7 * 8 = 56'")
        self.log("  CoT:    '7 * 8 = multiply: 56'")

    def run(self) -> dict:
        """Run the experiment."""
        self.log("=" * 60)
        self.log("COT VOCABULARY ALIGNMENT EXPERIMENT")
        self.log("Does CoT training create vocabulary classifiers?")
        self.log("=" * 60)

        # Phase 1: Baseline (no training)
        self.log("\n--- Phase 1: Baseline (no training) ---")
        baseline_result = self._check_vocab_alignment("baseline", None)
        self.results["baseline"] = baseline_result
        self._log_result(baseline_result)

        # Phase 2: Train on DIRECT format, check vocab alignment
        self.log("\n--- Phase 2: SFT on direct format ---")
        direct_adapter = self._train_sft("direct")
        direct_result = self._check_vocab_alignment("direct_sft", direct_adapter)
        self.results["direct_sft"] = direct_result
        self._log_result(direct_result)

        # Phase 3: Train on COT format, check vocab alignment
        self.log("\n--- Phase 3: SFT on CoT format ---")
        cot_adapter = self._train_sft("cot")
        cot_result = self._check_vocab_alignment("cot_sft", cot_adapter)
        self.results["cot_sft"] = cot_result
        self._log_result(cot_result)

        return self._build_results()

    def _log_result(self, result: VocabAlignmentResult):
        """Log result summary."""
        self.log(f"\n  Answer accuracy: {result.answer_accuracy:.1%}")
        self.log("  Vocabulary alignment by layer:")
        for layer, prob in sorted(result.vocab_alignment.items()):
            marker = "***" if prob > 0.3 else ""
            self.log(f"    L{layer}: {prob:.1%} {marker}")

    def _train_sft(self, format_name: str) -> Path:
        """Train SFT on specified format."""
        output_dir = self.config.checkpoint_dir / f"sft_{format_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks for train/valid files
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        train_src = self.config.data_dir / f"train_{format_name}.jsonl"
        valid_src = self.config.data_dir / f"valid_{format_name}.jsonl"

        train_dst = data_dir / "train.jsonl"
        valid_dst = data_dir / "valid.jsonl"

        if train_dst.exists():
            train_dst.unlink()
        if valid_dst.exists():
            valid_dst.unlink()

        train_dst.symlink_to(train_src)
        valid_dst.symlink_to(valid_src)

        # Create config
        lora = self.params.get("lora", {})
        train_config = {
            "model": self.config.model,
            "train": True,
            "data": str(data_dir),
            "batch_size": self.params.get("batch_size", 4),
            "learning_rate": self.params.get("learning_rate", 2e-4),
            "iters": self.params.get("max_steps", 500),
            "adapter_path": str(output_dir / "adapters"),
            "steps_per_report": 100,
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": lora.get("rank", 16),
                "alpha": lora.get("alpha", 32.0),
                "dropout": 0.0,
                "scale": lora.get("alpha", 32.0) / lora.get("rank", 16),
            },
        }

        config_path = output_dir / "train_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(train_config, f)

        self.log(f"Training on {format_name} format...")
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log(f"Training failed: {result.stderr}")

        return output_dir / "adapters"

    def _simple_generate(self, model, tokenizer, prompt: str, max_tokens: int = 20) -> str:
        """Simple greedy generation that works with the framework's model."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        generated_ids = []

        for _ in range(max_tokens):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            token_id = int(next_token[0])
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return tokenizer.decode(generated_ids)

    def _check_vocab_alignment(self, stage: str, adapter_path: Path | None) -> VocabAlignmentResult:
        """Check vocabulary alignment at each layer."""
        # Load model
        if adapter_path and adapter_path.exists():
            loaded = self.load_model_with_lora(adapter_path=str(adapter_path))
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log(f"Loaded with adapter: {adapter_path}")
        else:
            loaded = self.load_model()
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log("Loaded base model")

        num_layers = loaded.config.num_hidden_layers
        embed_weight = model.model.embed_tokens.weight.parameters()["weight"]

        # Layers to check
        layer_pcts = self.params.get("check_layers_pct", [0.25, 0.5, 0.75, 0.95])
        layer_indices = [min(int(p * num_layers), num_layers - 1) for p in layer_pcts]

        # Check each test prompt
        layer_probs = {l: [] for l in layer_indices}
        per_prompt_results = []
        correct = 0

        for prompt_info in self.test_prompts:
            input_text = prompt_info["input"]
            expected = prompt_info["expected"]
            task = prompt_info["task"]

            # Generate answer using simple greedy generation
            response = self._simple_generate(model, tokenizer, input_text, max_tokens=20)
            # Extract number from response (handle CoT format too)
            generated = self._extract_number(response)
            is_correct = generated == expected
            if is_correct:
                correct += 1

            # Check vocab alignment at each layer
            input_ids = mx.array(tokenizer.encode(input_text))[None, :]
            h = model.model.embed_tokens(input_ids)

            prompt_layer_probs = {}
            for i, layer in enumerate(model.model.layers):
                layer_out = layer(h, mask=None, cache=None)
                h = (
                    layer_out.hidden_states
                    if hasattr(layer_out, "hidden_states")
                    else (layer_out[0] if isinstance(layer_out, tuple) else layer_out)
                )

                if i in layer_indices:
                    # Project to vocabulary
                    h_normed = model.model.norm(h)
                    logits = h_normed @ embed_weight.T
                    probs = mx.softmax(logits[0, -1, :], axis=-1)
                    mx.eval(probs)

                    # Find max prob for task tokens
                    max_prob = 0.0
                    for token_word in self.task_tokens[task]:
                        token_ids = tokenizer.encode(token_word)
                        for tid in token_ids:
                            if tid < probs.shape[0]:
                                prob = float(probs[tid])
                                max_prob = max(max_prob, prob)

                    layer_probs[i].append(max_prob)
                    prompt_layer_probs[i] = max_prob

            per_prompt_results.append(
                {
                    "input": input_text,
                    "expected": expected,
                    "generated": generated,
                    "correct": is_correct,
                    "task": task,
                    "layer_probs": prompt_layer_probs,
                }
            )

        # Average probs per layer
        vocab_alignment = {l: sum(probs) / len(probs) for l, probs in layer_probs.items()}
        accuracy = correct / len(self.test_prompts)

        return VocabAlignmentResult(
            stage=stage,
            answer_accuracy=accuracy,
            vocab_alignment=vocab_alignment,
            per_prompt_results=per_prompt_results,
        )

    def _extract_number(self, text: str) -> str:
        """Extract first number from response."""
        # Handle CoT format: "multiply: 56" -> "56"
        if ":" in text:
            text = text.split(":")[-1]
        match = re.search(r"-?\d+", text)
        return match.group() if match else text.strip()

    def _build_results(self) -> dict:
        """Build results dict."""
        results = {
            "model": self.config.model,
            "stages": {},
        }

        for stage, r in self.results.items():
            results["stages"][stage] = {
                "answer_accuracy": r.answer_accuracy,
                "vocab_alignment": {f"L{l}": p for l, p in r.vocab_alignment.items()},
            }

        # Summary: Did CoT create vocabulary alignment?
        if "baseline" in self.results and "cot_sft" in self.results:
            baseline_max = max(self.results["baseline"].vocab_alignment.values())
            cot_max = max(self.results["cot_sft"].vocab_alignment.values())

            results["summary"] = {
                "baseline_max_vocab": baseline_max,
                "cot_max_vocab": cot_max,
                "vocab_alignment_increased": cot_max > baseline_max * 1.5,
                "cot_creates_vocab_alignment": cot_max > 0.3,
            }

            self.log("\n" + "=" * 60)
            self.log("CONCLUSION")
            self.log("=" * 60)
            self.log(f"Baseline max vocab alignment: {baseline_max:.1%}")
            self.log(f"CoT SFT max vocab alignment:  {cot_max:.1%}")

            if cot_max > 0.3:
                self.log("\n>>> YES! CoT training creates vocabulary alignment!")
                self.log(">>> This explains GPT-OSS L13 classifiers.")
            else:
                self.log("\n>>> NO. CoT training did NOT create vocabulary alignment.")
                self.log(">>> GPT-OSS must use something else (scale, MoE, explicit training).")

        return results

    def evaluate(self) -> dict:
        """Return summary metrics."""
        if "cot_sft" in self.results:
            return {
                "cot_accuracy": self.results["cot_sft"].answer_accuracy,
                "cot_max_vocab": max(self.results["cot_sft"].vocab_alignment.values()),
            }
        return {"error": "No results"}

    def cleanup(self) -> None:
        """Cleanup."""
        self.results = {}
