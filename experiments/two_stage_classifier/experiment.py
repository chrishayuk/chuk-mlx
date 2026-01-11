"""
Two-Stage Classifier Training

Stage 1: SFT builds computation circuits (100% accuracy on symbolic math)
Stage 2: Light dual-reward adds classifiers WITHOUT destroying computation

Key insight from previous experiments:
- Dual-reward at 70/30 creates classifiers but breaks computation
- We need computation FIRST, then add classifiers with low weight (20/80)

Expected outcome:
- After Stage 1: High accuracy, weak classifiers
- After Stage 2: High accuracy PRESERVED, strong classifiers ADDED
"""

import asyncio
import json
import logging
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Results for a training stage."""
    stage_name: str
    symbolic_accuracy: float = 0.0
    semantic_accuracy: float = 0.0
    avg_classifier_prob: float = 0.0
    results: list[dict] = field(default_factory=list)


class TwoStageClassifierExperiment(ExperimentBase):
    """Two-stage training: computation first, then classifiers."""

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up two-stage classifier experiment...")

        self.params = self.config.parameters

        # Stage configs - extra YAML fields are merged into parameters
        self.stage1_config = self.params.get("stage1", {})
        self.stage2_config = self.params.get("stage2", {})

        self.test_prompts = self._build_test_prompts()

        self.task_vocabulary = {
            "multiply": ["multiply", "times", "product", "*"],
            "add": ["add", "plus", "sum", "+"],
            "subtract": ["subtract", "minus", "difference", "-"],
        }

        # Generate data
        self._ensure_data()

        self.stage_results: dict[str, StageResult] = {}

    def _build_test_prompts(self) -> list[dict]:
        """Build test prompts from config."""
        prompts = []
        test_config = self.params.get("test_prompts", {})

        for category, category_prompts in test_config.items():
            for p in category_prompts:
                prompts.append({
                    "category": category,
                    "input": p["input"],
                    "expected": p["expected"],
                    "task": p["task"],
                })

        if not prompts:
            # Defaults
            prompts = [
                {"category": "symbolic", "input": "7 * 8 = ", "expected": "56", "task": "multiply"},
                {"category": "symbolic", "input": "12 * 5 = ", "expected": "60", "task": "multiply"},
                {"category": "symbolic", "input": "23 + 45 = ", "expected": "68", "task": "add"},
                {"category": "symbolic", "input": "89 - 34 = ", "expected": "55", "task": "subtract"},
                {"category": "semantic", "input": "seven times eight", "expected": "56", "task": "multiply"},
                {"category": "semantic", "input": "twenty three plus forty five", "expected": "68", "task": "add"},
            ]

        return prompts

    def _ensure_data(self) -> None:
        """Generate training data if needed."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        train_path = self.config.data_dir / "train.jsonl"
        dr_train_path = self.config.data_dir / "train_dual_reward.jsonl"

        if train_path.exists() and dr_train_path.exists():
            self.log("Using existing data")
            return

        self.log("Generating training data...")
        random.seed(self.params.get("seed", 42))

        num_samples = self.params.get("num_samples", 5000)
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
            data.append({
                "text": f"{a} {op_sym} {b} = {result}",
                "prompt": f"{a} {op_sym} {b} = ",
                "response": str(result),
                "operation": op_name,
            })

        split = int(len(data) * 0.9)
        train_data, valid_data = data[:split], data[split:]

        # SFT format
        with open(train_path, "w") as f:
            for e in train_data:
                f.write(json.dumps({"text": e["text"]}) + "\n")

        with open(self.config.data_dir / "valid.jsonl", "w") as f:
            for e in valid_data:
                f.write(json.dumps({"text": e["text"]}) + "\n")

        # Dual-reward format
        with open(dr_train_path, "w") as f:
            for e in train_data:
                f.write(json.dumps({
                    "prompt": e["prompt"],
                    "response": e["response"],
                    "operation": e["operation"],
                }) + "\n")

        self.log(f"Generated {len(train_data)} train samples")

    def run(self) -> dict:
        """Run two-stage training."""
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict:
        """Async implementation."""
        self.log("=" * 60)
        self.log("TWO-STAGE CLASSIFIER TRAINING")
        self.log("=" * 60)

        # Baseline
        self.log("\n--- Baseline (no training) ---")
        baseline = await self._evaluate_model("baseline", None)
        self.stage_results["baseline"] = baseline
        self._log_stage(baseline)

        # Stage 1: SFT
        self.log("\n" + "=" * 60)
        self.log("STAGE 1: SFT for computation")
        self.log("=" * 60)

        stage1_dir = self.config.checkpoint_dir / "stage1"
        success = self._train_sft(stage1_dir)
        if not success:
            self.log("Stage 1 failed!")
            return self._build_results()

        stage1_adapter = stage1_dir / "adapters"
        stage1_result = await self._evaluate_model("stage1_sft", stage1_adapter)
        self.stage_results["stage1_sft"] = stage1_result
        self._log_stage(stage1_result)

        # Stage 2: Light dual-reward ON TOP OF stage 1
        self.log("\n" + "=" * 60)
        self.log("STAGE 2: Light dual-reward for classifiers")
        self.log(f"  classifier_weight: {self.stage2_config.get('classifier_weight', 0.2)}")
        self.log("=" * 60)

        stage2_dir = self.config.checkpoint_dir / "stage2"
        success = self._train_dual_reward_on_adapter(stage2_dir, stage1_adapter)
        if not success:
            self.log("Stage 2 failed!")
            return self._build_results()

        # Evaluate stage 2: Load FUSED model (stage 1 baked in) + stage 2 adapter
        stage2_adapter = stage2_dir / "adapters"
        fused_model_path = str(stage2_dir / "fused_stage1")
        stage2_result = await self._evaluate_model(
            "stage2_dual_reward", stage2_adapter, base_model_path=fused_model_path
        )
        self.stage_results["stage2_dual_reward"] = stage2_result
        self._log_stage(stage2_result)

        return self._build_results()

    def _log_stage(self, result: StageResult):
        """Log stage results."""
        self.log(f"\n--- {result.stage_name} ---")
        self.log(f"  Symbolic accuracy: {result.symbolic_accuracy:.1%}")
        self.log(f"  Semantic accuracy: {result.semantic_accuracy:.1%}")
        self.log(f"  Avg classifier prob: {result.avg_classifier_prob:.1%}")

        for r in result.results:
            status = "+" if r["correct"] else "X"
            self.log(f"  [{status}] {r['input']} -> {r['generated']} (exp {r['expected']})")

    def _simple_generate(self, model, tokenizer, prompt: str, max_tokens: int = 10) -> str:
        """Simple greedy generation that works with the framework's model."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        generated_ids = []

        for _ in range(max_tokens):
            output = model(input_ids)
            logits = output.logits if hasattr(output, 'logits') else output
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            token_id = int(next_token[0])
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return tokenizer.decode(generated_ids)

    async def _evaluate_model(
        self, name: str, adapter_path: Path | None, base_model_path: str | None = None
    ) -> StageResult:
        """Evaluate model on test prompts.

        Args:
            name: Stage name for logging
            adapter_path: Path to adapter weights (or None for base model)
            base_model_path: Optional custom base model path (e.g., fused model for stage 2)
        """
        result = StageResult(stage_name=name)

        # Use custom base path if provided (e.g., fused stage 1 model)
        model_path = base_model_path or self.config.model

        if adapter_path and adapter_path.exists():
            loaded = self.load_model_with_lora(model_path=model_path, adapter_path=str(adapter_path))
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log(f"Loaded {model_path} with adapter: {adapter_path}")
        else:
            loaded = self.load_model(model_path=model_path)
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log(f"Loaded model: {model_path}")

        symbolic_correct, symbolic_total = 0, 0
        semantic_correct, semantic_total = 0, 0
        classifier_probs = []

        for p in self.test_prompts:
            category = p["category"]
            input_text = p["input"]
            expected = p["expected"]
            task = p["task"]

            # Generate using simple greedy generation
            prompt = input_text if input_text.endswith("= ") else f"{input_text} = "
            response = self._simple_generate(model, tokenizer, prompt, max_tokens=10)
            generated = self._extract_number(response)
            correct = (generated == expected)

            if category == "symbolic":
                symbolic_total += 1
                if correct:
                    symbolic_correct += 1
            else:
                semantic_total += 1
                if correct:
                    semantic_correct += 1

            # Check classifier
            task_vocab = self.task_vocabulary.get(task, [])
            classifier_prob = self._check_classifier(model, tokenizer, prompt, task_vocab)
            if classifier_prob > 0:
                classifier_probs.append(classifier_prob)

            result.results.append({
                "category": category,
                "input": input_text,
                "expected": expected,
                "generated": generated,
                "correct": correct,
                "classifier_prob": classifier_prob,
            })

        result.symbolic_accuracy = symbolic_correct / symbolic_total if symbolic_total else 0
        result.semantic_accuracy = semantic_correct / semantic_total if semantic_total else 0
        result.avg_classifier_prob = sum(classifier_probs) / len(classifier_probs) if classifier_probs else 0

        return result

    def _check_classifier(self, model, tokenizer, prompt: str, task_vocab: list[str]) -> float:
        """Check classifier probability at each layer, return max."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        h = model.model.embed_tokens(input_ids)
        embed_weight = model.model.embed_tokens.weight.parameters()['weight']

        max_prob = 0.0
        for layer in model.model.layers:
            layer_out = layer(h, mask=None, cache=None)
            h = layer_out.hidden_states if hasattr(layer_out, 'hidden_states') else (layer_out[0] if isinstance(layer_out, tuple) else layer_out)

            h_normed = model.model.norm(h)
            logits = h_normed @ embed_weight.T
            probs = mx.softmax(logits[0, -1, :], axis=-1)

            top_indices = mx.argsort(probs)[-20:][::-1]
            mx.eval(top_indices, probs)

            for idx in top_indices.tolist():
                token = tokenizer.decode([idx]).lower().strip()
                if any(tv in token for tv in task_vocab):
                    prob = float(probs[idx])
                    max_prob = max(max_prob, prob)
                    break

        return max_prob

    def _extract_number(self, text: str) -> str:
        """Extract first number."""
        match = re.search(r'-?\d+', text)
        return match.group() if match else text.strip()

    def _train_sft(self, output_dir: Path) -> bool:
        """Stage 1: SFT training."""
        import subprocess
        import sys
        import yaml

        output_dir.mkdir(parents=True, exist_ok=True)

        config = self.stage1_config
        lora = config.get("lora", {})

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
                "rank": lora.get("rank", 16),
                "alpha": lora.get("alpha", 32.0),
                "dropout": 0.0,
                "scale": lora.get("alpha", 32.0) / lora.get("rank", 16),
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

    def _train_dual_reward_on_adapter(self, output_dir: Path, base_adapter: Path) -> bool:
        """Stage 2: Dual-reward training with light classifier weight.

        FUSES the stage 1 adapter into base weights, then trains new LoRA on top.
        The goal is to ADD classifiers without destroying the computation learned in stage 1.
        """
        import subprocess
        import sys
        from chuk_lazarus.training.trainers.dual_reward_trainer import (
            DualRewardTrainer, DualRewardTrainerConfig
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # First, FUSE stage 1 adapter into base model weights
        fused_model_path = output_dir / "fused_stage1"
        if base_adapter.exists():
            if not fused_model_path.exists():
                self.log(f"Fusing Stage 1 adapter into base model...")
                cmd = [
                    sys.executable, "-m", "mlx_lm", "fuse",
                    "--model", self.config.model,
                    "--adapter-path", str(base_adapter),
                    "--save-path", str(fused_model_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.log(f"Fuse failed: {result.stderr}")
                    return False
                self.log(f"Fused model saved to: {fused_model_path}")
            else:
                self.log(f"Using existing fused model: {fused_model_path}")

            # Load fused model (stage 1 computation is now in the weights)
            loaded = self.load_model(model_path=str(fused_model_path))
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log(f"Loaded fused Stage 1 model")
        else:
            loaded = self.load_model()
            model, tokenizer = loaded.model, loaded.tokenizer
            self.log("Warning: Stage 1 adapter not found, using fresh model")

        config = self.stage2_config
        lora = config.get("lora", {})

        trainer_config = DualRewardTrainerConfig(
            num_epochs=1,
            batch_size=1,
            learning_rate=config.get("learning_rate", 1e-4),
            max_steps=config.get("max_steps", 200),
            classifier_layer=-1,
            classifier_weight=config.get("classifier_weight", 0.2),  # KEY: Low weight!
            classifier_targets=config.get("classifier_targets", {
                "multiply": "multiply",
                "add": "add",
                "subtract": "subtract",
            }),
            lora_rank=lora.get("rank", 32),
            lora_targets=lora.get("targets", ["v_proj", "o_proj"]),
            log_interval=50,
            checkpoint_interval=config.get("max_steps", 200),
            checkpoint_dir=str(output_dir),
        )

        trainer = DualRewardTrainer(model, tokenizer, trainer_config)

        # Load data
        data_path = self.config.data_dir / "train_dual_reward.jsonl"
        dataset = []
        with open(data_path) as f:
            for line in f:
                dataset.append(json.loads(line))

        trainer.train(dataset)

        # Copy final to adapters
        final_path = output_dir / "final"
        if final_path.exists():
            dest = output_dir / "adapters"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(final_path, dest)

        return True

    def _build_results(self) -> dict:
        """Build results dict."""
        results = {
            "model": self.config.model,
            "stages": {},
        }

        for name, r in self.stage_results.items():
            results["stages"][name] = {
                "symbolic_accuracy": r.symbolic_accuracy,
                "semantic_accuracy": r.semantic_accuracy,
                "avg_classifier_prob": r.avg_classifier_prob,
                "results": r.results,
            }

        # Summary
        if "stage2_dual_reward" in self.stage_results:
            s2 = self.stage_results["stage2_dual_reward"]
            baseline = self.stage_results.get("baseline")
            results["summary"] = {
                "final_symbolic_accuracy": s2.symbolic_accuracy,
                "final_semantic_accuracy": s2.semantic_accuracy,
                "final_classifier_prob": s2.avg_classifier_prob,
                "baseline_symbolic": baseline.symbolic_accuracy if baseline else 0,
                "computation_preserved": s2.symbolic_accuracy >= 0.9,
                "classifiers_added": s2.avg_classifier_prob > 0.1,
            }

        return results

    def evaluate(self) -> dict:
        """Return summary."""
        if "stage2_dual_reward" in self.stage_results:
            s2 = self.stage_results["stage2_dual_reward"]
            return {
                "symbolic_accuracy": s2.symbolic_accuracy,
                "semantic_accuracy": s2.semantic_accuracy,
                "classifier_prob": s2.avg_classifier_prob,
            }
        return {"error": "No results"}

    def cleanup(self) -> None:
        """Cleanup."""
        self.stage_results = {}
