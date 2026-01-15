"""
Classifier Emergence Experiment: Training Method Comparison

Compares classifier emergence across different training methods:
1. SFT (Supervised Fine-Tuning) - with/without LoRA
2. GRPO (Group Relative Policy Optimization) - with/without LoRA
3. Dual-Reward (Classification + Generation Loss) - with/without LoRA

Key questions:
- Does SFT produce answer classifiers or operation classifiers?
- Does GRPO with verifiable rewards produce different classifier patterns?
- Does dual-reward training (explicit classifier loss) outperform implicit emergence?
- Does LoRA vs full fine-tuning affect classifier location or strength?
- Does having a classifier improve answer accuracy?
"""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class ClassifierSignal:
    """Track classifier signal at a layer."""

    layer: int
    top_token: str
    top_prob: float
    task_token: str | None = None
    task_prob: float | None = None
    task_rank: int | None = None


@dataclass
class TaskResult:
    """Result of analyzing a single task prompt."""

    task: str
    prompt: str
    expected_answer: str
    signals_by_layer: dict[int, ClassifierSignal] = field(default_factory=dict)
    peak_task_layer: int | None = None
    peak_task_prob: float = 0.0
    # Answer generation results
    generated_answer: str | None = None
    answer_correct: bool = False


@dataclass
class MethodResult:
    """Results for a single training method."""

    method_name: str
    training_steps: int
    task_results: list[TaskResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_classifiers(self) -> bool:
        """Does any task show a task classifier?"""
        return any(r.peak_task_prob > 0.1 for r in self.task_results)

    @property
    def average_peak_prob(self) -> float:
        """Average task token probability at peak layer."""
        probs = [r.peak_task_prob for r in self.task_results if r.peak_task_prob > 0]
        return sum(probs) / len(probs) if probs else 0.0

    @property
    def answer_accuracy(self) -> float:
        """Accuracy of generated answers."""
        if not self.task_results:
            return 0.0
        correct = sum(1 for r in self.task_results if r.answer_correct)
        return correct / len(self.task_results)


class ClassifierEmergenceExperiment(ExperimentBase):
    """
    Task classifier emergence comparison experiment.

    Compares classifier emergence and answer quality across:
    - SFT (with/without LoRA)
    - GRPO (with/without LoRA)
    - Dual-Reward (with/without LoRA)
    """

    def setup(self) -> None:
        """Initialize experiment and generate data if needed."""
        self.log("Setting up classifier emergence comparison experiment...")

        self.params = self.config.parameters
        self.num_samples = self.params.get("num_samples", 5000)
        self.seed = self.params.get("seed", 42)

        # Get training methods configuration
        self.training_methods = self.config.training.get("training_methods", {})
        self.checkpoint_steps = self.config.training.get("checkpoint_steps", [200, 500, 1000])

        # Get test prompts
        self.test_prompts = self._build_test_prompts()

        # Get task vocabulary
        self.task_vocabulary = self.params.get(
            "task_vocabulary",
            {
                "multiplication": ["multiply", "times", "product", "*", "×"],
                "addition": ["add", "plus", "sum", "+"],
                "subtraction": ["subtract", "minus", "difference", "-"],
            },
        )

        # Generate training data if needed
        self.data_path = self.config.data_dir / "arithmetic_train.jsonl"
        dr_data_path = self.config.data_dir / "train_dual_reward.jsonl"

        # Always regenerate if dual_reward data is missing (for experiments that need it)
        if not self.data_path.exists() or not dr_data_path.exists():
            self.log(f"Generating {self.num_samples} training samples...")
            self._generate_data()
        else:
            self.log(f"Using existing data: {self.data_path}")

        # Results storage
        self.baseline_result: MethodResult | None = None
        self.method_results: dict[str, MethodResult] = {}

    def _build_test_prompts(self) -> list[dict]:
        """Build list of test prompts from config."""
        prompts = []
        test_config = self.params.get("test_prompts", {})

        if not test_config:
            # Default test prompts
            return [
                {"task": "multiplication", "prompt": "7 * 8 = ", "expected": "56"},
                {"task": "multiplication", "prompt": "12 * 5 = ", "expected": "60"},
                {"task": "multiplication", "prompt": "9 * 9 = ", "expected": "81"},
                {"task": "addition", "prompt": "23 + 45 = ", "expected": "68"},
                {"task": "addition", "prompt": "17 + 38 = ", "expected": "55"},
                {"task": "subtraction", "prompt": "89 - 34 = ", "expected": "55"},
                {"task": "subtraction", "prompt": "65 - 28 = ", "expected": "37"},
                {"task": "subtraction", "prompt": "100 - 43 = ", "expected": "57"},
            ]

        for task, task_prompts in test_config.items():
            for p in task_prompts:
                prompts.append(
                    {
                        "task": task,
                        "prompt": p["prompt"],
                        "expected": p["expected"],
                    }
                )

        return prompts

    def _generate_data(self) -> None:
        """Generate arithmetic training data with operation labels."""
        random.seed(self.seed)

        operations = [
            ("multiplication", "*", lambda a, b: a * b, "multiply"),
            ("subtraction", "-", lambda a, b: a - b, "subtract"),
            ("addition", "+", lambda a, b: a + b, "add"),
        ]

        data = []
        for _ in range(self.num_samples):
            op_name, op_sym, op_fn, op_label = random.choice(operations)

            if op_name == "multiplication":
                a = random.randint(2, 20)
                b = random.randint(2, 20)
            else:
                a = random.randint(1, 99)
                b = random.randint(1, 99)
                if op_name == "subtraction":
                    a, b = max(a, b), min(a, b)

            result = op_fn(a, b)
            data.append(
                {
                    "prompt": f"{a} {op_sym} {b} = ",
                    "response": str(result),
                    "text": f"{a} {op_sym} {b} = {result}",
                    "operation": op_label,  # For dual-reward classification
                }
            )

        # Split data
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]

        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Save in multiple formats for different trainers
        # Format 1: text-only for mlx-lm SFT
        train_path = self.config.data_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for entry in train_data:
                f.write(json.dumps({"text": entry["text"]}) + "\n")

        valid_path = self.config.data_dir / "valid.jsonl"
        with open(valid_path, "w") as f:
            for entry in valid_data:
                f.write(json.dumps({"text": entry["text"]}) + "\n")

        # Format 2: prompt/response/operation for dual-reward
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

        # Save full data
        with open(self.data_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        self.log(f"Generated {len(train_data)} train + {len(valid_data)} valid samples")

    def run(self) -> dict:
        """Run the experiment."""
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict:
        """Async implementation of run."""
        self.log(f"Running classifier emergence comparison on {self.config.model}")

        # 1. Baseline analysis (no training)
        self.log("=" * 60)
        self.log("Phase 1: Baseline Analysis (no training)")
        self.log("=" * 60)
        self.baseline_result = await self._analyze_model("baseline", None)
        self._log_method_summary("baseline", self.baseline_result)

        # 2. Run each enabled training method
        enabled_methods = {
            name: cfg for name, cfg in self.training_methods.items() if cfg.get("enabled", False)
        }

        if not enabled_methods:
            self.log("No training methods enabled. Enable methods in config.yaml")
            return self._build_results()

        for method_name, method_config in enabled_methods.items():
            self.log("=" * 60)
            self.log(f"Training method: {method_name}")
            self.log("=" * 60)

            method = method_config.get("method", "sft")
            use_lora = method_config.get("use_lora", True)
            max_steps = method_config.get("max_steps", 1000)

            checkpoint_dir = self.config.checkpoint_dir / method_name

            # Train
            self.log(
                f"Training {method} ({'LoRA' if use_lora else 'Full'}) for {max_steps} steps..."
            )

            if method == "sft":
                success = self._train_sft(checkpoint_dir, method_config)
            elif method == "dual_reward":
                success = self._train_dual_reward(checkpoint_dir, method_config)
            elif method == "grpo":
                success = self._train_grpo(checkpoint_dir, method_config)
            else:
                self.log(f"Unknown method: {method}")
                continue

            if not success:
                self.log(f"Training {method_name} failed")
                continue

            # Analyze
            adapter_path = checkpoint_dir / "adapters" if use_lora else None
            result = await self._analyze_model(method_name, adapter_path)
            self.method_results[method_name] = result
            self._log_method_summary(method_name, result)

        return self._build_results()

    def _log_method_summary(self, name: str, result: MethodResult):
        """Log summary for a training method."""
        self.log(f"\n--- {name} Summary ---")
        self.log(f"  Has classifiers: {result.has_classifiers}")
        self.log(f"  Average peak prob: {result.average_peak_prob:.1%}")
        self.log(f"  Answer accuracy: {result.answer_accuracy:.1%}")

        # Per-prompt results
        for r in result.task_results:
            status = "✓" if r.answer_correct else "✗"
            classifier_info = (
                f"L{r.peak_task_layer} {r.peak_task_prob:.1%}" if r.peak_task_layer else "none"
            )
            self.log(
                f"  {r.prompt} → {r.generated_answer} ({r.expected_answer}) {status} | classifier: {classifier_info}"
            )

    def _simple_generate(self, model, tokenizer, prompt: str, max_tokens: int = 10) -> str:
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

    async def _analyze_model(self, name: str, adapter_path: Path | None) -> MethodResult:
        """Analyze model for classifier signals and answer accuracy."""
        result = MethodResult(method_name=name, training_steps=0)

        # Load model (with optional adapter) using framework
        if adapter_path and adapter_path.exists():
            loaded = self.load_model(adapter_path=str(adapter_path))
            self.log(f"Loaded model with adapter: {adapter_path}")
        else:
            loaded = self.load_model()
            self.log(f"Loaded base model: {self.config.model}")

        model, tokenizer = loaded.model, loaded.tokenizer
        num_layers = loaded.config.num_hidden_layers

        for prompt_info in self.test_prompts:
            task = prompt_info["task"]
            prompt = prompt_info["prompt"]
            expected = prompt_info["expected"]

            task_result = TaskResult(task=task, prompt=prompt, expected_answer=expected)
            task_vocab = self.task_vocabulary.get(task, [])

            # 1. Analyze classifier signals via logit lens
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            h = model.model.embed_tokens(input_ids)

            if hasattr(model.model, "embed_scale"):
                h = h * model.model.embed_scale

            embed_weight = model.model.embed_tokens.weight.parameters()["weight"]

            for layer_idx, layer in enumerate(model.model.layers):
                layer_output = layer(h, mask=None, cache=None)
                h = (
                    layer_output.hidden_states
                    if hasattr(layer_output, "hidden_states")
                    else (layer_output[0] if isinstance(layer_output, tuple) else layer_output)
                )

                # Logit lens projection
                h_normed = model.model.norm(h)
                logits = h_normed @ embed_weight.T

                probs = mx.softmax(logits[0, -1, :], axis=-1)
                top_indices = mx.argsort(probs)[-20:][::-1]
                top_probs = probs[top_indices]

                mx.eval(top_indices, top_probs)
                top_indices_list = top_indices.tolist()
                top_probs_list = top_probs.tolist()
                top_tokens = [tokenizer.decode([idx]) for idx in top_indices_list]

                top_token = top_tokens[0] if top_tokens else ""
                top_prob = top_probs_list[0] if top_probs_list else 0.0

                # Check for task vocabulary tokens
                task_token = None
                task_prob = None
                task_rank = None

                for rank, (tok, prob) in enumerate(zip(top_tokens, top_probs_list), 1):
                    token_lower = tok.lower().strip()
                    if any(tv in token_lower for tv in task_vocab):
                        task_token = tok
                        task_prob = prob
                        task_rank = rank
                        break

                signal = ClassifierSignal(
                    layer=layer_idx,
                    top_token=top_token,
                    top_prob=top_prob,
                    task_token=task_token,
                    task_prob=task_prob,
                    task_rank=task_rank,
                )
                task_result.signals_by_layer[layer_idx] = signal

                if task_prob and task_prob > task_result.peak_task_prob:
                    task_result.peak_task_prob = task_prob
                    task_result.peak_task_layer = layer_idx

            # 2. Generate answer and check correctness using simple generation
            response = self._simple_generate(model, tokenizer, prompt, max_tokens=10)
            # Extract just the first number from response
            generated = self._extract_number(response)
            task_result.generated_answer = generated
            task_result.answer_correct = generated == expected

            result.task_results.append(task_result)

        return result

    def _extract_number(self, text: str) -> str:
        """Extract the first number from generated text."""
        # Handle negative numbers and decimals
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            num_str = match.group()
            # Convert to int if it's a whole number
            try:
                num = float(num_str)
                if num == int(num):
                    return str(int(num))
                return num_str
            except ValueError:
                return num_str
        return text.strip()

    def _train_sft(self, output_dir: Path, config: dict) -> bool:
        """Train using SFT with mlx-lm."""
        import subprocess
        import sys

        output_dir.mkdir(parents=True, exist_ok=True)

        use_lora = config.get("use_lora", True)
        max_steps = config.get("max_steps", 1000)
        batch_size = config.get("batch_size", 4)
        lr = config.get("learning_rate", 1e-4)
        lora_config = config.get("lora", {})

        # mlx-lm uses config file for LoRA rank, not CLI flag
        # Create a config file for the training
        config_path = output_dir / "train_config.yaml"
        import yaml

        train_config = {
            "model": self.config.model,
            "train": True,
            "data": str(self.config.data_dir),
            "batch_size": batch_size,
            "learning_rate": lr,
            "iters": max_steps,
            "adapter_path": str(output_dir / "adapters"),
            "steps_per_report": 50,
        }

        if use_lora:
            train_config["fine_tune_type"] = "lora"
            # LoRA config in mlx-lm uses lora_parameters
            if "rank" in lora_config:
                train_config["lora_parameters"] = {
                    "rank": lora_config.get("rank", 16),
                    "alpha": lora_config.get("alpha", 32.0),
                    "dropout": 0.0,
                    "scale": lora_config.get("alpha", 32.0) / lora_config.get("rank", 16),
                }
        else:
            train_config["fine_tune_type"] = "full"

        with open(config_path, "w") as f:
            yaml.dump(train_config, f)

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "-c",
            str(config_path),
        ]

        self.log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log(f"SFT training failed: {result.stderr}")
            return False

        return True

    def _train_dual_reward(self, output_dir: Path, config: dict) -> bool:
        """Train using dual-reward loss."""
        from chuk_lazarus.training.trainers.dual_reward_trainer import (
            DualRewardTrainer,
            DualRewardTrainerConfig,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model using framework
        loaded = self.load_model()
        model, tokenizer = loaded.model, loaded.tokenizer

        # Configure trainer
        lora_config = config.get("lora", {})
        classifier_targets = config.get(
            "classifier_targets",
            {
                "multiply": "multiply",
                "add": "add",
                "subtract": "subtract",
            },
        )

        trainer_config = DualRewardTrainerConfig(
            num_epochs=1,
            batch_size=1,
            learning_rate=config.get("learning_rate", 5e-4),
            max_steps=config.get("max_steps", 1000),
            classifier_layer=-1,  # Auto-calculate 55% depth
            classifier_weight=config.get("classifier_weight", 0.7),
            classifier_targets=classifier_targets,
            lora_rank=lora_config.get("rank", 32),
            lora_targets=lora_config.get("targets", ["v_proj", "o_proj"]),
            log_interval=50,
            checkpoint_interval=config.get("max_steps", 1000),
            checkpoint_dir=str(output_dir),
        )

        trainer = DualRewardTrainer(model, tokenizer, trainer_config)

        # Load training data
        data_path = self.config.data_dir / "train_dual_reward.jsonl"
        dataset = []
        with open(data_path) as f:
            for line in f:
                dataset.append(json.loads(line))

        # Train
        trainer.train(dataset)

        # Copy adapters to expected location
        final_path = output_dir / "final"
        if final_path.exists():
            import shutil

            adapter_dest = output_dir / "adapters"
            if adapter_dest.exists():
                shutil.rmtree(adapter_dest)
            shutil.copytree(final_path, adapter_dest)

        return True

    def _train_grpo(self, output_dir: Path, config: dict) -> bool:
        """Train using GRPO with arithmetic reward."""
        from chuk_lazarus.models_v2.adapters.lora import LoRAConfig, apply_lora
        from chuk_lazarus.training.losses.grpo_loss import GRPOConfig
        from chuk_lazarus.training.trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model (need two copies - policy and reference) using framework
        loaded_policy = self.load_model()
        policy_model, tokenizer = loaded_policy.model, loaded_policy.tokenizer
        loaded_ref = self.load_model()
        reference_model = loaded_ref.model

        # Apply LoRA to policy model
        use_lora = config.get("use_lora", True)
        if use_lora:
            lora_cfg = config.get("lora", {})
            lora_config = LoRAConfig(
                rank=lora_cfg.get("rank", 16),
                alpha=lora_cfg.get("alpha", 32.0),
                dropout=0.0,
                target_modules=lora_cfg.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )
            lora_layers = apply_lora(policy_model, lora_config)
            self.log(f"Applied LoRA with rank {lora_config.rank}")

        # Arithmetic reward function
        def arithmetic_reward(prompt: str, response: str) -> float:
            """Reward function for arithmetic correctness."""
            # Parse prompt to get expected answer
            match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", prompt)
            if not match:
                return 0.0

            a, op, b = int(match.group(1)), match.group(2), int(match.group(3))

            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b
            elif op == "/":
                expected = a // b if b != 0 else 0
            else:
                return 0.0

            # Extract answer from response
            answer_match = re.search(r"-?\d+", response)
            if not answer_match:
                return 0.0

            try:
                answer = int(answer_match.group())
                return 1.0 if answer == expected else 0.0
            except ValueError:
                return 0.0

        # Configure GRPO trainer
        grpo_config = GRPOConfig(
            group_size=config.get("group_size", 4),
            kl_coeff=0.1,
            clip_range=0.2,
        )

        trainer_config = GRPOTrainerConfig(
            grpo=grpo_config,
            num_iterations=config.get("num_iterations", 500),
            prompts_per_iteration=16,
            learning_rate=config.get("learning_rate", 1e-5),
            max_response_length=10,
            temperature=1.0,
            log_interval=10,
            checkpoint_interval=100,
            checkpoint_dir=str(output_dir),
        )

        trainer = GRPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            reward_fn=arithmetic_reward,
            config=trainer_config,
        )

        # Store LoRA info on trainer for checkpointing
        if use_lora:
            trainer.lora_layers = lora_layers
            trainer.lora_config = lora_config

        # Load prompts
        prompts = []
        with open(self.config.data_dir / "train.jsonl") as f:
            for line in f:
                data = json.loads(line)
                # Extract just the prompt part
                text = data.get("text", "")
                if "=" in text:
                    prompt = text.split("=")[0] + "= "
                    prompts.append(prompt)

        def prompt_source():
            return random.sample(prompts, min(32, len(prompts)))

        # Train
        trainer.train(prompt_source)

        # Copy adapters to expected location
        if use_lora:
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
            "num_samples": self.num_samples,
            "comparison": {},
        }

        # Add baseline
        if self.baseline_result:
            results["baseline"] = self._method_result_to_dict(self.baseline_result)

        # Add each method
        for name, result in self.method_results.items():
            results["comparison"][name] = self._method_result_to_dict(result)

        # Summary comparison
        if self.baseline_result and self.method_results:
            results["summary"] = self._build_summary()

        return results

    def _method_result_to_dict(self, result: MethodResult) -> dict:
        """Convert MethodResult to dictionary."""
        return {
            "method": result.method_name,
            "has_classifiers": result.has_classifiers,
            "average_peak_prob": result.average_peak_prob,
            "answer_accuracy": result.answer_accuracy,
            "task_results": [
                {
                    "task": r.task,
                    "prompt": r.prompt,
                    "expected": r.expected_answer,
                    "generated": r.generated_answer,
                    "answer_correct": r.answer_correct,
                    "peak_layer": r.peak_task_layer,
                    "peak_prob": r.peak_task_prob,
                }
                for r in result.task_results
            ],
        }

    def _build_summary(self) -> dict:
        """Build comparison summary."""
        summary = {
            "baseline_accuracy": self.baseline_result.answer_accuracy
            if self.baseline_result
            else 0.0,
            "baseline_has_classifiers": self.baseline_result.has_classifiers
            if self.baseline_result
            else False,
            "methods": {},
        }

        for name, result in self.method_results.items():
            baseline_acc = self.baseline_result.answer_accuracy if self.baseline_result else 0.0
            summary["methods"][name] = {
                "answer_accuracy": result.answer_accuracy,
                "accuracy_improvement": result.answer_accuracy - baseline_acc,
                "has_classifiers": result.has_classifiers,
                "classifier_strength": result.average_peak_prob,
            }

        # Best method
        if summary["methods"]:
            best = max(summary["methods"].items(), key=lambda x: x[1]["answer_accuracy"])
            summary["best_method"] = best[0]
            summary["best_accuracy"] = best[1]["answer_accuracy"]

        return summary

    def evaluate(self) -> dict:
        """Summarize experiment results."""
        if not self.baseline_result and not self.method_results:
            latest = self.load_latest_results("results")
            if not latest:
                return {"error": "No results to evaluate"}
            return latest.get("summary", {"error": "No summary in results"})

        return self._build_summary()

    def cleanup(self) -> None:
        """Release resources."""
        self.log("Cleaning up...")
        self.baseline_result = None
        self.method_results = {}
