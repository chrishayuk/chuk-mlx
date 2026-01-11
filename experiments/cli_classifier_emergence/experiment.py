"""
CLI Classifier Emergence Experiment.

Dual-reward training for vocabulary-aligned arithmetic classifiers.

This experiment demonstrates that training V/O projections with dual-reward
(generation + classification) creates vocabulary-aligned classifiers that
can be read via logit lens at intermediate layers.

Pipeline:
1. Generate arithmetic training data
2. Train V/O projections with dual-reward loss
3. Evaluate classifier accuracy on held-out prompts
"""

import json
import logging
import random
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase, ExperimentConfig

logger = logging.getLogger(__name__)


class CLIClassifierEmergenceExperiment(ExperimentBase):
    """
    Dual-reward classifier emergence experiment.

    Trains vocabulary-aligned operation classifiers using:
    - LoRA on v_proj and o_proj only
    - Dual-reward loss: generation + intermediate classification
    - Logit lens evaluation at 55% layer depth
    """

    def setup(self) -> None:
        """Load model and prepare training data."""
        self.log("Loading model...")
        result = self.load_model()
        self.model = result.model
        self.tokenizer = result.tokenizer
        self.model_config = result.config

        # Generate or load training data
        data_path = self.config.data_dir / "arithmetic_sft.jsonl"
        if not data_path.exists():
            self.log("Generating training data...")
            self._generate_data(data_path)
        else:
            self.log(f"Using existing data: {data_path}")

        self.data_path = data_path
        self.dataset = self._load_dataset(data_path)
        self.log(f"Loaded {len(self.dataset)} training samples")

        # Get classifier configuration
        self.classifier_config = self.config.parameters.get("classifier", {})
        self.lora_config = self.config.parameters.get("lora", {})

        # Calculate classifier layer
        layer_pct = self.classifier_config.get("layer_pct", 0.55)
        self.classifier_layer = int(self.model_config.num_hidden_layers * layer_pct)
        self.log(f"Classifier layer: {self.classifier_layer} ({layer_pct*100:.0f}% depth)")

    def _generate_data(self, output_path: Path) -> None:
        """Generate arithmetic training data."""
        data_gen_config = self.config.parameters.get("data_generation", {})
        num_samples = data_gen_config.get("samples", 1000)
        seed = data_gen_config.get("seed", 42)

        random.seed(seed)

        ops = [
            ('*', 'multiply', lambda a, b: a * b),
            ('+', 'add', lambda a, b: a + b),
            ('-', 'subtract', lambda a, b: a - b),
            ('/', 'divide', lambda a, b: a // b if b != 0 else 0),
        ]

        samples = []
        for _ in range(num_samples):
            op_sym, op_name, op_fn = random.choice(ops)

            if op_sym == '/':
                b = random.randint(1, 12)
                a = b * random.randint(1, 12)
            elif op_sym == '-':
                a = random.randint(10, 100)
                b = random.randint(1, a)
            else:
                a = random.randint(1, 50)
                b = random.randint(1, 50)

            result = op_fn(a, b)
            prompt = f"{a} {op_sym} {b} = "
            answer = str(result)

            samples.append({
                "prompt": prompt,
                "response": answer,
                "operation": op_name,
                "classification_target": op_name,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Count distribution
        op_counts = {}
        for s in samples:
            op = s["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1

        self.log(f"Generated {len(samples)} samples")
        self.log(f"Distribution: {op_counts}")

    def _load_dataset(self, path: Path) -> list:
        """Load JSONL dataset."""
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def run(self) -> dict:
        """Train the classifier using dual-reward."""
        from chuk_lazarus.training.trainers.dual_reward_trainer import (
            DualRewardTrainer,
            DualRewardTrainerConfig,
        )

        # Get training config
        training_config = self.config.training or {}
        max_steps = training_config.get("max_steps", 500)
        learning_rate = training_config.get("learning_rate", 0.001)
        log_interval = training_config.get("log_interval", 50)

        lora_rank = self.lora_config.get("rank", 16)
        lora_targets = self.lora_config.get("targets", ["v_proj", "o_proj"])
        classifier_weight = self.classifier_config.get("weight", 0.4)

        # Get classifier targets from config
        classifier_targets = self.classifier_config.get("targets", {
            "multiply": "multiply",
            "add": "add",
            "subtract": "subtract",
            "divide": "divide",
        })
        # Handle both list format (old) and dict format (new)
        if isinstance(classifier_targets, list):
            classifier_targets = {t: t for t in classifier_targets}

        self.log(f"Training for {max_steps} steps...")
        self.log(f"LoRA rank: {lora_rank}, targets: {lora_targets}")
        self.log(f"Classifier weight: {classifier_weight}")
        self.log(f"Classifier targets: {classifier_targets}")

        trainer_config = DualRewardTrainerConfig(
            max_steps=max_steps,
            classifier_layer=self.classifier_layer,
            classifier_weight=classifier_weight,
            classifier_targets=classifier_targets,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            lora_targets=lora_targets,
            log_interval=log_interval,
            checkpoint_interval=max_steps,
            checkpoint_dir=str(self.config.checkpoint_dir),
        )

        self.trainer = DualRewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=trainer_config,
            model_config=self.model_config,
        )

        # Train
        self.trainer.train(self.dataset)

        # Save training config
        training_result = {
            "model": self.config.model,
            "classifier_layer": self.classifier_layer,
            "classifier_weight": classifier_weight,
            "lora_rank": lora_rank,
            "lora_targets": lora_targets,
            "steps": max_steps,
            "classifier_token_ids": self.trainer.classifier_token_ids,
        }

        config_path = self.config.checkpoint_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(training_result, f, indent=2)

        self.log(f"Training complete. Checkpoint saved to {self.config.checkpoint_dir}")

        return training_result

    def evaluate(self) -> dict:
        """Evaluate classifier accuracy on test prompts."""
        eval_prompts = self.config.parameters.get("evaluation_prompts", [])

        if not eval_prompts:
            # Default evaluation prompts
            eval_prompts = [
                {"prompt": "7 * 8 = ", "expected": "multiply"},
                {"prompt": "12 * 5 = ", "expected": "multiply"},
                {"prompt": "23 + 45 = ", "expected": "add"},
                {"prompt": "17 + 38 = ", "expected": "add"},
                {"prompt": "50 - 23 = ", "expected": "subtract"},
                {"prompt": "89 - 34 = ", "expected": "subtract"},
                {"prompt": "48 / 6 = ", "expected": "divide"},
                {"prompt": "81 / 9 = ", "expected": "divide"},
            ]

        # Convert to tuples for trainer
        test_prompts = [(p["prompt"], p["expected"]) for p in eval_prompts]

        if hasattr(self, "trainer"):
            eval_results = self.trainer.evaluate_classifier(test_prompts)
        else:
            self.log("No trainer available, skipping evaluation")
            return {"error": "No trainer available"}

        # Format results
        self.log("\n" + "=" * 60)
        self.log("CLASSIFIER EVALUATION")
        self.log("=" * 60)

        for r in eval_results["results"]:
            status = "OK" if r["correct"] else "XX"
            self.log(f"  {r['prompt']:<13} {r['expected']:<12} {r['predicted']:<12} "
                     f"{r['confidence']:>7.1%} [{status}]")

        self.log("-" * 60)
        self.log(f"Accuracy: {eval_results['correct']}/{eval_results['total']} "
                 f"({eval_results['accuracy']:.1%})")

        return {
            "accuracy": eval_results["accuracy"],
            "correct": eval_results["correct"],
            "total": eval_results["total"],
            "results": eval_results["results"],
        }

    def cleanup(self) -> None:
        """Release resources."""
        self.log("Cleaning up...")
        self.model = None
        self.trainer = None


# For backwards compatibility
if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    config = ExperimentConfig(
        experiment_dir=Path(__file__).parent,
        **config_data,
    )

    experiment = CLIClassifierEmergenceExperiment(config)
    experiment.setup()
    results = experiment.run()
    eval_results = experiment.evaluate()
    experiment.cleanup()

    print(f"\nFinal Accuracy: {eval_results.get('accuracy', 0)*100:.1f}%")
