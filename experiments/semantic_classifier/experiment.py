"""
Semantic Classifier Experiment

Tests whether explicit classifiers improve accuracy when parsing is required.

Key difference from classifier_emergence:
- Input: Natural language ("seven times three")
- NOT symbolic ("7 * 3 =")

This forces the model to actually PARSE the operation, not just read a symbol.

Research question:
- Does dual-reward (explicit classifier at L8) beat SFT when classification is required?
- Or does SFT discover implicit classifiers that work just as well?
"""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)

# Number words for data generation
NUM_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
    100: "one hundred",
}


def number_to_words(n: int) -> str:
    """Convert number to words (0-100)."""
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return f"{NUM_WORDS[tens]} {NUM_WORDS[ones]}"
    return str(n)  # Fallback for larger numbers


@dataclass
class TaskResult:
    """Result for a single test prompt."""

    task: str
    prompt: str
    expected: str
    generated: str | None = None
    correct: bool = False
    classifier_layer: int | None = None
    classifier_prob: float = 0.0


@dataclass
class MethodResult:
    """Results for a training method."""

    method_name: str
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.correct) / len(self.task_results)

    @property
    def avg_classifier_prob(self) -> float:
        probs = [r.classifier_prob for r in self.task_results if r.classifier_prob > 0]
        return sum(probs) / len(probs) if probs else 0.0


class SemanticClassifierExperiment(ExperimentBase):
    """
    Tests classifier emergence on semantic (natural language) arithmetic.
    """

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up semantic classifier experiment...")

        self.params = self.config.parameters
        self.num_samples = self.params.get("num_samples", 5000)
        self.seed = self.params.get("seed", 42)

        self.training_methods = self.config.training.get("training_methods", {})
        self.input_patterns = self.params.get("input_patterns", {})
        self.test_prompts = self._build_test_prompts()

        # Operation vocabulary for classifier detection
        self.task_vocabulary = {
            "multiply": ["multiply", "times", "product"],
            "add": ["add", "plus", "sum"],
            "subtract": ["subtract", "minus", "difference"],
        }

        # Generate data if needed
        self.data_path = self.config.data_dir / "semantic_train.jsonl"
        if not self.data_path.exists():
            self.log(f"Generating {self.num_samples} semantic samples...")
            self._generate_data()
        else:
            self.log(f"Using existing data: {self.data_path}")

        self.baseline_result: MethodResult | None = None
        self.method_results: dict[str, MethodResult] = {}

    def _build_test_prompts(self) -> list[dict]:
        """Build test prompts from config."""
        prompts = []
        test_config = self.params.get("test_prompts", {})

        if not test_config:
            # Default semantic test prompts
            return [
                {"task": "multiply", "input": "seven times eight", "expected": "56"},
                {"task": "multiply", "input": "twelve multiplied by five", "expected": "60"},
                {"task": "multiply", "input": "the product of nine and nine", "expected": "81"},
                {"task": "add", "input": "twenty three plus forty five", "expected": "68"},
                {"task": "add", "input": "seventeen and thirty eight", "expected": "55"},
                {
                    "task": "add",
                    "input": "the sum of fifty five and twenty seven",
                    "expected": "82",
                },
                {"task": "subtract", "input": "eighty nine minus thirty four", "expected": "55"},
                {
                    "task": "subtract",
                    "input": "sixty five take away twenty eight",
                    "expected": "37",
                },
                {
                    "task": "subtract",
                    "input": "the difference between one hundred and forty three",
                    "expected": "57",
                },
            ]

        for task, task_prompts in test_config.items():
            for p in task_prompts:
                prompts.append(
                    {
                        "task": task,
                        "input": p["input"],
                        "expected": p["expected"],
                    }
                )

        return prompts

    def _generate_data(self) -> None:
        """Generate semantic arithmetic training data."""
        random.seed(self.seed)

        # Default patterns if not in config
        patterns = self.input_patterns or {
            "multiply": [
                "{a} times {b}",
                "{a} multiplied by {b}",
                "the product of {a} and {b}",
            ],
            "add": [
                "{a} plus {b}",
                "{a} and {b}",
                "the sum of {a} and {b}",
            ],
            "subtract": [
                "{a} minus {b}",
                "{a} take away {b}",
                "the difference between {a} and {b}",
            ],
        }

        operations = [
            ("multiply", lambda a, b: a * b),
            ("add", lambda a, b: a + b),
            ("subtract", lambda a, b: a - b),
        ]

        data = []
        for _ in range(self.num_samples):
            op_name, op_fn = random.choice(operations)

            if op_name == "multiply":
                a = random.randint(2, 12)
                b = random.randint(2, 12)
            else:
                a = random.randint(1, 50)
                b = random.randint(1, 50)
                if op_name == "subtract":
                    a, b = max(a, b), min(a, b)

            result = op_fn(a, b)

            # Convert to words
            a_words = number_to_words(a)
            b_words = number_to_words(b)

            # Pick random pattern
            pattern = random.choice(patterns[op_name])
            prompt = pattern.format(a=a_words, b=b_words)

            data.append(
                {
                    "prompt": prompt,
                    "response": str(result),
                    "text": f"{prompt} = {result}",
                    "operation": op_name,
                    # Also store canonical form for analysis
                    "canonical": f"{a} {'*' if op_name == 'multiply' else '+' if op_name == 'add' else '-'} {b} = {result}",
                }
            )

        # Split
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]

        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Format for mlx-lm SFT
        train_path = self.config.data_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for entry in train_data:
                f.write(json.dumps({"text": entry["text"]}) + "\n")

        valid_path = self.config.data_dir / "valid.jsonl"
        with open(valid_path, "w") as f:
            for entry in valid_data:
                f.write(json.dumps({"text": entry["text"]}) + "\n")

        # Format for dual-reward
        dr_train_path = self.config.data_dir / "train_dual_reward.jsonl"
        with open(dr_train_path, "w") as f:
            for entry in train_data:
                f.write(
                    json.dumps(
                        {
                            "prompt": entry["prompt"],
                            "response": entry["response"],
                            "operation": entry["operation"],
                        }
                    )
                    + "\n"
                )

        # Full data
        with open(self.data_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        self.log(f"Generated {len(train_data)} train + {len(valid_data)} valid samples")

    def run(self) -> dict:
        """Run the experiment."""
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict:
        """Async implementation."""
        self.log(f"Running semantic classifier experiment on {self.config.model}")

        # 1. Baseline
        self.log("=" * 60)
        self.log("Phase 1: Baseline (no training)")
        self.log("=" * 60)
        self.baseline_result = await self._analyze_model("baseline", None)
        self._log_summary("baseline", self.baseline_result)

        # 2. Training methods
        enabled = {k: v for k, v in self.training_methods.items() if v.get("enabled")}

        for method_name, method_config in enabled.items():
            self.log("=" * 60)
            self.log(f"Training: {method_name}")
            self.log("=" * 60)

            method = method_config.get("method", "sft")
            checkpoint_dir = self.config.checkpoint_dir / method_name

            if method == "sft":
                success = self._train_sft(checkpoint_dir, method_config)
            elif method == "dual_reward":
                success = self._train_dual_reward(checkpoint_dir, method_config)
            else:
                self.log(f"Unknown method: {method}")
                continue

            if not success:
                self.log(f"Training {method_name} failed")
                continue

            adapter_path = checkpoint_dir / "adapters"
            result = await self._analyze_model(method_name, adapter_path)
            self.method_results[method_name] = result
            self._log_summary(method_name, result)

        return self._build_results()

    def _log_summary(self, name: str, result: MethodResult):
        """Log summary for a method."""
        self.log(f"\n--- {name} Summary ---")
        self.log(f"  Accuracy: {result.accuracy:.1%}")
        self.log(f"  Avg classifier prob: {result.avg_classifier_prob:.1%}")

        for r in result.task_results:
            status = "+" if r.correct else "X"
            self.log(f"  [{status}] {r.prompt} -> {r.generated} (expected {r.expected})")

    def _simple_generate(self, model, tokenizer, prompt: str, max_tokens: int = 10) -> str:
        """Simple greedy generation that works with the framework's model."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        generated_ids = []

        for _ in range(max_tokens):
            output = model(input_ids)
            # Framework model returns ModelOutput with .logits attribute
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            token_id = int(next_token[0])
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return tokenizer.decode(generated_ids)

    async def _analyze_model(self, name: str, adapter_path: Path | None) -> MethodResult:
        """Analyze model on test prompts."""
        result = MethodResult(method_name=name)

        if adapter_path and adapter_path.exists():
            loaded = self.load_model_with_lora(adapter_path=str(adapter_path))
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log(f"Loaded with adapter: {adapter_path}")
        else:
            loaded = self.load_model()
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log("Loaded base model")

        for prompt_info in self.test_prompts:
            task = prompt_info["task"]
            prompt = prompt_info["input"]
            expected = prompt_info["expected"]

            task_result = TaskResult(task=task, prompt=prompt, expected=expected)

            # Generate answer using simple greedy generation
            full_prompt = f"{prompt} = "
            response = self._simple_generate(model, tokenizer, full_prompt, max_tokens=10)
            generated = self._extract_number(response)
            task_result.generated = generated
            task_result.correct = generated == expected

            # Check classifier at each layer
            task_vocab = self.task_vocabulary.get(task, [])
            input_ids = mx.array(tokenizer.encode(full_prompt))[None, :]
            h = model.model.embed_tokens(input_ids)
            embed_weight = model.model.embed_tokens.weight.parameters()["weight"]

            for layer_idx, layer in enumerate(model.model.layers):
                layer_out = layer(h, mask=None, cache=None)
                h = (
                    layer_out.hidden_states
                    if hasattr(layer_out, "hidden_states")
                    else (layer_out[0] if isinstance(layer_out, tuple) else layer_out)
                )

                h_normed = model.model.norm(h)
                logits = h_normed @ embed_weight.T
                probs = mx.softmax(logits[0, -1, :], axis=-1)

                top_indices = mx.argsort(probs)[-20:][::-1]
                mx.eval(top_indices, probs)

                for idx in top_indices.tolist():
                    token = tokenizer.decode([idx]).lower().strip()
                    if any(tv in token for tv in task_vocab):
                        prob = float(probs[idx])
                        if prob > task_result.classifier_prob:
                            task_result.classifier_prob = prob
                            task_result.classifier_layer = layer_idx
                        break

            result.task_results.append(task_result)

        return result

    def _extract_number(self, text: str) -> str:
        """Extract first number from text."""
        match = re.search(r"-?\d+", text)
        return match.group() if match else text.strip()

    def _train_sft(self, output_dir: Path, config: dict) -> bool:
        """Train with SFT."""
        import subprocess
        import sys

        import yaml

        output_dir.mkdir(parents=True, exist_ok=True)

        lora_config = config.get("lora", {})
        train_config = {
            "model": self.config.model,
            "train": True,
            "data": str(self.config.data_dir),
            "batch_size": config.get("batch_size", 4),
            "learning_rate": config.get("learning_rate", 2e-4),
            "iters": config.get("max_steps", 500),
            "adapter_path": str(output_dir / "adapters"),
            "steps_per_report": 50,
            "fine_tune_type": "lora",
            "lora_parameters": {
                "rank": lora_config.get("rank", 16),
                "alpha": lora_config.get("alpha", 32.0),
                "dropout": 0.0,
                "scale": lora_config.get("alpha", 32.0) / lora_config.get("rank", 16),
            },
        }

        config_path = output_dir / "train_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(train_config, f)

        cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
        self.log(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.log(f"SFT failed: {result.stderr}")
            return False
        return True

    def _train_dual_reward(self, output_dir: Path, config: dict) -> bool:
        """Train with dual-reward."""
        from chuk_lazarus.training.trainers.dual_reward_trainer import (
            DualRewardTrainer,
            DualRewardTrainerConfig,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        loaded = self.load_model()
        model, tokenizer = loaded.model, loaded.tokenizer

        lora_config = config.get("lora", {})
        trainer_config = DualRewardTrainerConfig(
            num_epochs=1,
            batch_size=1,
            learning_rate=config.get("learning_rate", 5e-4),
            max_steps=config.get("max_steps", 500),
            classifier_layer=-1,
            classifier_weight=config.get("classifier_weight", 0.7),
            classifier_targets=config.get(
                "classifier_targets",
                {
                    "multiply": "multiply",
                    "add": "add",
                    "subtract": "subtract",
                },
            ),
            lora_rank=lora_config.get("rank", 32),
            lora_targets=lora_config.get("targets", ["v_proj", "o_proj"]),
            log_interval=50,
            checkpoint_interval=config.get("max_steps", 500),
            checkpoint_dir=str(output_dir),
        )

        trainer = DualRewardTrainer(model, tokenizer, trainer_config)

        data_path = self.config.data_dir / "train_dual_reward.jsonl"
        dataset = []
        with open(data_path) as f:
            for line in f:
                dataset.append(json.loads(line))

        trainer.train(dataset)

        # Copy to expected location
        final_path = output_dir / "final"
        if final_path.exists():
            import shutil

            adapter_dest = output_dir / "adapters"
            if adapter_dest.exists():
                shutil.rmtree(adapter_dest)
            shutil.copytree(final_path, adapter_dest)

        return True

    def _build_results(self) -> dict:
        """Build results dictionary."""
        results = {
            "model": self.config.model,
            "input_type": "semantic (natural language)",
            "baseline": {
                "accuracy": self.baseline_result.accuracy if self.baseline_result else 0,
                "avg_classifier_prob": self.baseline_result.avg_classifier_prob
                if self.baseline_result
                else 0,
            },
            "methods": {},
            "summary": {},
        }

        for name, r in self.method_results.items():
            baseline_acc = self.baseline_result.accuracy if self.baseline_result else 0
            results["methods"][name] = {
                "accuracy": r.accuracy,
                "improvement": r.accuracy - baseline_acc,
                "avg_classifier_prob": r.avg_classifier_prob,
                "per_prompt": [
                    {
                        "task": t.task,
                        "prompt": t.prompt,
                        "expected": t.expected,
                        "generated": t.generated,
                        "correct": t.correct,
                        "classifier_layer": t.classifier_layer,
                        "classifier_prob": t.classifier_prob,
                    }
                    for t in r.task_results
                ],
            }

        if self.method_results:
            best = max(self.method_results.items(), key=lambda x: x[1].accuracy)
            results["summary"] = {
                "best_method": best[0],
                "best_accuracy": best[1].accuracy,
                "baseline_accuracy": self.baseline_result.accuracy if self.baseline_result else 0,
            }

        return results

    def evaluate(self) -> dict:
        """Return summary."""
        if self.method_results:
            best = max(self.method_results.items(), key=lambda x: x[1].accuracy)
            return {
                "best_method": best[0],
                "best_accuracy": best[1].accuracy,
                "baseline_accuracy": self.baseline_result.accuracy if self.baseline_result else 0,
            }
        return {"error": "No results"}

    def cleanup(self) -> None:
        """Cleanup."""
        self.baseline_result = None
        self.method_results = {}
