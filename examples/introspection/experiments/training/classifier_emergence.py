#!/usr/bin/env python3
"""
Task Classifier Emergence Experiment

Hypothesis: RL post-training on verifiable tasks creates L13-style task classifiers.

This experiment tests whether task-specific classifier signals (like those observed
in GPT-OSS Layer 13) can be induced in base models through targeted SFT training.

The experiment:
1. Baseline: Run logit lens on untrained model - expect NO task classifiers
2. Generate verifiable task data (arithmetic, synonyms)
3. SFT train for N steps
4. Re-run logit lens - look for task tokens (multiply, add, synonym) at intermediate layers
5. Track classifier confidence vs training steps

Models to test:
- meta-llama/Llama-3.2-1B (1B base - fast iteration)
- google/gemma-3-1b-pt (alternative)

Usage:
    # Full experiment
    python examples/introspection/experiments/training/classifier_emergence.py

    # Just baseline analysis
    python examples/introspection/experiments/training/classifier_emergence.py --baseline-only

    # Just training
    python examples/introspection/experiments/training/classifier_emergence.py --train-only

    # Use Lazarus native training (default: mlx-lm for stability)
    python examples/introspection/experiments/training/classifier_emergence.py --use-lazarus
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassifierSignal:
    """Track classifier signal at a layer."""

    layer: int
    top_token: str
    top_prob: float
    task_token: str | None  # e.g., "multiply", "add", "synonym"
    task_prob: float | None
    task_rank: int | None


@dataclass
class TaskResult:
    """Result of analyzing a single task prompt."""

    task: str
    prompt: str
    expected_answer: str
    signals_by_layer: dict[int, ClassifierSignal] = field(default_factory=dict)
    peak_task_layer: int | None = None
    peak_task_prob: float = 0.0


@dataclass
class ExperimentSnapshot:
    """Snapshot of model state at a training checkpoint."""

    checkpoint: str  # "baseline", "step_100", "step_500", etc.
    steps: int
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def has_classifiers(self) -> bool:
        """Does any task show a task classifier?"""
        return any(r.peak_task_prob > 0.1 for r in self.task_results)

    @property
    def average_peak_prob(self) -> float:
        """Average task token probability at peak layer."""
        probs = [r.peak_task_prob for r in self.task_results if r.peak_task_prob > 0]
        return sum(probs) / len(probs) if probs else 0.0


# Task vocabulary - tokens we expect to see as classifiers
TASK_VOCABULARY = {
    "multiplication": ["multiply", "times", "product", "*", "×"],
    "addition": ["add", "plus", "sum", "+"],
    "subtraction": ["subtract", "minus", "difference", "-"],
    "division": ["divide", "quotient", "/", "÷"],
    "synonym": ["synonym", "synonymous", "similar", "means"],
    "antonym": ["antonym", "opposite", "contrary"],
    "sentiment": ["positive", "negative", "sentiment", "good", "bad"],
}


def generate_arithmetic_data(
    n_samples: int = 5000,
    output_path: Path | None = None,
    use_lazarus_format: bool = False,
) -> list[dict]:
    """Generate arithmetic training data.

    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the data
        use_lazarus_format: If True, use {"prompt": ..., "response": ...} format
                           If False, use {"text": ...} format for mlx-lm
    """

    data = []
    operations = [
        ("addition", "+", lambda a, b: a + b),
        ("subtraction", "-", lambda a, b: a - b),
        ("multiplication", "*", lambda a, b: a * b),
    ]

    for _ in range(n_samples):
        op_name, op_sym, op_fn = random.choice(operations)

        if op_name == "multiplication":
            a = random.randint(2, 20)
            b = random.randint(2, 20)
        else:
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            if op_name == "subtraction":
                a, b = max(a, b), min(a, b)  # Ensure positive result

        result = op_fn(a, b)

        # Store both formats in internal data
        data.append(
            {
                "prompt": f"{a} {op_sym} {b} = ",
                "response": str(result),
                "text": f"{a} {op_sym} {b} = {result}",
                "task": op_name,
                "operands": [a, b],
                "answer": result,
            }
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Split data: 90% train, 10% valid
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        valid_data = data[split_idx:]

        if use_lazarus_format:
            # Lazarus SFTDataset format: {"prompt": ..., "response": ...}
            train_path = output_path.parent / "train.jsonl"
            with open(train_path, "w") as f:
                for entry in train_data:
                    f.write(
                        json.dumps({"prompt": entry["prompt"], "response": entry["response"]})
                        + "\n"
                    )

            valid_path = output_path.parent / "valid.jsonl"
            with open(valid_path, "w") as f:
                for entry in valid_data:
                    f.write(
                        json.dumps({"prompt": entry["prompt"], "response": entry["response"]})
                        + "\n"
                    )
        else:
            # mlx-lm format: {"text": ...}
            train_path = output_path.parent / "train.jsonl"
            with open(train_path, "w") as f:
                for entry in train_data:
                    f.write(json.dumps({"text": entry["text"]}) + "\n")

            valid_path = output_path.parent / "valid.jsonl"
            with open(valid_path, "w") as f:
                for entry in valid_data:
                    f.write(json.dumps({"text": entry["text"]}) + "\n")

        # Also save full data to requested path (always as text format for compatibility)
        with open(output_path, "w") as f:
            for entry in data:
                f.write(json.dumps({"text": entry["text"]}) + "\n")

        print(f"Saved {len(train_data)} train + {len(valid_data)} valid samples")

    return data


def generate_test_prompts() -> list[dict]:
    """Generate prompts for classifier testing."""

    prompts = []

    # Multiplication tests
    for a, b in [(7, 8), (12, 5), (9, 9), (45, 45)]:
        prompts.append(
            {
                "task": "multiplication",
                "prompt": f"{a} * {b} = ",
                "expected": str(a * b),
            }
        )

    # Addition tests
    for a, b in [(23, 45), (17, 38), (55, 27)]:
        prompts.append(
            {
                "task": "addition",
                "prompt": f"{a} + {b} = ",
                "expected": str(a + b),
            }
        )

    # Subtraction tests
    for a, b in [(89, 34), (65, 28), (100, 43)]:
        prompts.append(
            {
                "task": "subtraction",
                "prompt": f"{a} - {b} = ",
                "expected": str(a - b),
            }
        )

    return prompts


async def analyze_model(
    model_id: str,
    prompts: list[dict],
    checkpoint_name: str = "baseline",
    steps: int = 0,
) -> ExperimentSnapshot:
    """Run logit lens analysis on a model for all test prompts."""

    from chuk_lazarus.introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

    snapshot = ExperimentSnapshot(checkpoint=checkpoint_name, steps=steps)

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {checkpoint_name} ({model_id})")
    print(f"{'=' * 60}")

    async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
        info = analyzer.model_info
        num_layers = info.num_layers

        print(f"Model: {info.model_id}")
        print(f"Layers: {num_layers}")
        print(f"Hidden size: {info.hidden_size}")

        # Analyze all layers for each prompt
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.ALL,
            top_k=20,  # Need enough to find task tokens
            track_tokens=[],
        )

        for prompt_info in prompts:
            task = prompt_info["task"]
            prompt = prompt_info["prompt"]
            expected = prompt_info["expected"]

            print(f"\n  Analyzing: {prompt!r} (expecting {expected})")

            result = await analyzer.analyze(prompt, config)

            task_result = TaskResult(
                task=task,
                prompt=prompt,
                expected_answer=expected,
            )

            # Check each layer for task vocabulary
            task_vocab = TASK_VOCABULARY.get(task, [])

            for layer_pred in result.layer_predictions:
                layer_idx = layer_pred.layer_idx

                # Get top prediction
                top = layer_pred.predictions[0]

                # Look for task vocabulary in predictions
                task_token = None
                task_prob = None
                task_rank = None

                for rank, pred in enumerate(layer_pred.predictions, 1):
                    token_lower = pred.token.lower().strip()
                    if any(tv in token_lower for tv in task_vocab):
                        task_token = pred.token
                        task_prob = pred.probability
                        task_rank = rank
                        break

                signal = ClassifierSignal(
                    layer=layer_idx,
                    top_token=top.token,
                    top_prob=top.probability,
                    task_token=task_token,
                    task_prob=task_prob,
                    task_rank=task_rank,
                )
                task_result.signals_by_layer[layer_idx] = signal

                # Track peak task signal
                if task_prob and task_prob > task_result.peak_task_prob:
                    task_result.peak_task_prob = task_prob
                    task_result.peak_task_layer = layer_idx

            # Print summary for this prompt
            if task_result.peak_task_layer is not None:
                print(
                    f"    FOUND classifier: '{task_result.signals_by_layer[task_result.peak_task_layer].task_token}' "
                    f"at layer {task_result.peak_task_layer} (prob={task_result.peak_task_prob:.3f})"
                )
            else:
                print("    No classifier found in task vocabulary")

            snapshot.task_results.append(task_result)

    return snapshot


async def analyze_model_with_adapter(
    model_id: str,
    adapter_path: Path,
    prompts: list[dict],
    checkpoint_name: str = "trained",
    steps: int = 0,
) -> ExperimentSnapshot:
    """Run logit lens analysis on a model with LoRA adapter.

    Uses mlx-lm to load base model + adapter, then runs analysis.
    """
    import mlx.core as mx
    from mlx_lm import load

    snapshot = ExperimentSnapshot(checkpoint=checkpoint_name, steps=steps)

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {checkpoint_name} ({model_id} + {adapter_path})")
    print(f"{'=' * 60}")

    # Load model with adapter using mlx-lm
    print("Loading model with adapter...")
    model, tokenizer = load(model_id, adapter_path=str(adapter_path))

    # Get model info
    num_layers = len(model.model.layers)
    # For models with tied embeddings, project using embed_tokens.weight.T
    embed_weight = model.model.embed_tokens.weight  # (vocab_size, hidden_size)

    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"Layers: {num_layers}")

    # Analyze each prompt
    for prompt_info in prompts:
        task = prompt_info["task"]
        prompt = prompt_info["prompt"]
        expected = prompt_info["expected"]

        print(f"\n  Analyzing: {prompt!r} (expecting {expected})")

        # Tokenize
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Forward pass capturing hidden states manually
        # Get embedding
        h = model.model.embed_tokens(input_ids)

        # Scale for Gemma-style models (check if model has embed_scale)
        if hasattr(model.model, "embed_scale"):
            h = h * model.model.embed_scale

        task_result = TaskResult(
            task=task,
            prompt=prompt,
            expected_answer=expected,
        )

        # Check each layer for task vocabulary
        task_vocab = TASK_VOCABULARY.get(task, [])

        for layer_idx, layer in enumerate(model.model.layers):
            # Pass through layer
            layer_output = layer(h, mask=None, cache=None)
            # Handle both tuple returns and direct returns
            h = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # Apply final norm and project to vocab (logit lens)
            h_normed = model.model.norm(h)
            logits = h_normed @ embed_weight.T  # Tied embeddings

            # Get probabilities for last token position
            probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_k = 20
            top_indices = mx.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]

            # Convert to tokens
            mx.eval(top_indices, top_probs)
            top_indices_list = top_indices.tolist()
            top_probs_list = top_probs.tolist()

            top_tokens = [tokenizer.decode([idx]) for idx in top_indices_list]

            # Get top token info
            top_token = top_tokens[0] if top_tokens else ""
            top_prob = top_probs_list[0] if top_probs_list else 0.0

            # Look for task vocabulary in predictions
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

            # Track peak task signal
            if task_prob and task_prob > task_result.peak_task_prob:
                task_result.peak_task_prob = task_prob
                task_result.peak_task_layer = layer_idx

        # Print summary for this prompt
        if task_result.peak_task_layer is not None:
            print(
                f"    FOUND classifier: '{task_result.signals_by_layer[task_result.peak_task_layer].task_token}' "
                f"at layer {task_result.peak_task_layer} (prob={task_result.peak_task_prob:.3f})"
            )
        else:
            print("    No classifier found in task vocabulary")

        snapshot.task_results.append(task_result)

    return snapshot


def print_comparison_table(snapshots: list[ExperimentSnapshot]):
    """Print comparison table across training checkpoints."""

    print("\n" + "=" * 80)
    print("CLASSIFIER EMERGENCE SUMMARY")
    print("=" * 80)

    # Header
    header = ["Task", "Prompt"] + [s.checkpoint for s in snapshots]
    print(f"\n{'Task':<15} {'Prompt':<20} " + " ".join(f"{s.checkpoint:<15}" for s in snapshots))
    print("-" * (35 + 15 * len(snapshots)))

    # Get all unique task+prompt combinations
    if not snapshots:
        print("No snapshots to compare")
        return

    baseline = snapshots[0]
    for i, task_result in enumerate(baseline.task_results):
        task = task_result.task
        prompt = task_result.prompt[:18]

        row = f"{task:<15} {prompt:<20} "

        for snapshot in snapshots:
            if i < len(snapshot.task_results):
                result = snapshot.task_results[i]
                if result.peak_task_layer is not None:
                    signal = result.signals_by_layer[result.peak_task_layer]
                    cell = f"L{result.peak_task_layer}:{result.peak_task_prob:.2f}"
                else:
                    cell = "none"
            else:
                cell = "N/A"
            row += f"{cell:<15} "

        print(row)

    # Summary statistics
    print("\n" + "-" * 80)
    print("Summary:")
    for snapshot in snapshots:
        has_class = "YES" if snapshot.has_classifiers else "NO"
        avg_prob = snapshot.average_peak_prob
        print(f"  {snapshot.checkpoint}: classifiers={has_class}, avg_prob={avg_prob:.3f}")


async def run_baseline_experiment(model_id: str, output_dir: Path):
    """Run baseline analysis on untrained model."""

    prompts = generate_test_prompts()
    snapshot = await analyze_model(model_id, prompts, "baseline", 0)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_analysis.json"

    # Serialize snapshot
    snapshot_data = {
        "checkpoint": snapshot.checkpoint,
        "steps": snapshot.steps,
        "has_classifiers": snapshot.has_classifiers,
        "average_peak_prob": snapshot.average_peak_prob,
        "task_results": [
            {
                "task": r.task,
                "prompt": r.prompt,
                "expected": r.expected_answer,
                "peak_layer": r.peak_task_layer,
                "peak_prob": r.peak_task_prob,
            }
            for r in snapshot.task_results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(snapshot_data, f, indent=2)

    print(f"\nBaseline results saved to {output_file}")

    return snapshot


def run_training(
    model_id: str,
    data_path: Path,
    output_dir: Path,
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
):
    """Run SFT training using mlx-lm directly.

    Uses mlx-lm's LoRA training which is well-tested on Apple Silicon.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use mlx-lm's training
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_id,
        "--train",
        "--data",
        str(data_path.parent),  # mlx-lm expects directory with train.jsonl
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--iters",
        str(steps),
        "--adapter-path",
        str(output_dir / "adapters"),
        "--steps-per-report",
        "10",
    ]

    print(f"\nRunning training: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        return False

    print(f"Training complete. Adapter saved to {output_dir / 'adapters'}")
    return True


def run_training_transformers(
    model_id: str,
    data_path: Path,
    output_dir: Path,
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
):
    """Fallback training using transformers + peft."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

        print("Loading model with transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # Load dataset
        dataset = load_dataset("json", data_files=str(data_path), split="train")

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, max_length=128)

        dataset = dataset.map(tokenize, remove_columns=["text"])

        # Training args
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            max_steps=steps,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=steps,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(str(output_dir / "final"))

    except ImportError as e:
        print(f"Could not import required libraries: {e}")
        print("Skipping training - install torch, transformers, peft for training support")


def run_training_lazarus(
    model_id: str,
    data_path: Path,
    output_dir: Path,
    steps: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
) -> bool:
    """Run SFT training using native Lazarus infrastructure.

    Uses Lazarus's HFLoader, apply_lora, SFTDataset, and SFTTrainer.
    """
    import mlx.core as mx

    from chuk_lazarus.data import SFTDataset
    from chuk_lazarus.inference.loader import HFLoader
    from chuk_lazarus.models_v2 import LoRAConfig, apply_lora, count_lora_parameters
    from chuk_lazarus.training.trainers.sft_trainer import SFTConfig, SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and load model + tokenizer
    print(f"\nLoading model: {model_id}")
    download_result = HFLoader.download(model_id)
    model_path = download_result.model_path

    tokenizer = HFLoader.load_tokenizer(model_path)
    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # 2. Create model from HuggingFace config
    # Use mlx_lm for model creation since it handles the config properly
    from mlx_lm import load as mlx_load

    model, _ = mlx_load(model_id)
    print(f"  Model loaded: {len(model.model.layers)} layers")

    # 3. Apply LoRA (match mlx-lm defaults: scale=20.0)
    lora_config = LoRAConfig(
        rank=8,
        alpha=20.0,  # mlx-lm default scale
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    lora_layers = apply_lora(model, lora_config)
    n_lora_params = count_lora_parameters(lora_layers)
    print(f"  Applied LoRA: {len(lora_layers)} layers, {n_lora_params:,} trainable params")

    # 4. Load dataset
    train_path = data_path.parent / "train.jsonl"
    valid_path = data_path.parent / "valid.jsonl"

    # mask_prompt=False to match mlx-lm behavior (train on full sequence)
    train_dataset = SFTDataset(
        str(train_path),
        tokenizer,
        max_length=128,
        mask_prompt=False,
    )
    print(f"  Loaded {len(train_dataset)} training samples")

    eval_dataset = None
    if valid_path.exists():
        eval_dataset = SFTDataset(
            str(valid_path),
            tokenizer,
            max_length=128,
            mask_prompt=False,
        )
        print(f"  Loaded {len(eval_dataset)} validation samples")

    # 5. Configure trainer (match mlx-lm defaults: batch_size=8)
    trainer_config = SFTConfig(
        num_epochs=1,
        batch_size=8,  # mlx-lm default
        learning_rate=learning_rate,
        max_steps=steps,
        checkpoint_dir=str(output_dir / "checkpoints"),
        log_interval=10,
        eval_interval=100,
        checkpoint_interval=steps,  # Save at end
    )

    # 6. Train
    print(f"\nStarting training for {steps} steps...")
    trainer = SFTTrainer(model, tokenizer, trainer_config)
    trainer.train(train_dataset, eval_dataset)

    # 7. Save final adapter weights
    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Collect LoRA weights in mlx-lm format
    # mlx-lm expects keys like "model.layers.0.self_attn.q_proj.lora_a"
    lora_weights = {}
    for name, lora_layer in lora_layers.items():
        # Convert from our format to mlx-lm format
        # Our format: "layers.0.self_attn.q_proj"
        # mlx-lm format: "model.layers.0.self_attn.q_proj.lora_a"
        lora_weights[f"model.{name}.lora_a"] = lora_layer.lora_A
        lora_weights[f"model.{name}.lora_b"] = lora_layer.lora_B

    # Save as safetensors
    mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), lora_weights)

    # Save adapter config in mlx-lm format for compatibility
    num_layers = len(model.model.layers)
    adapter_config = {
        "model": model_id,
        "num_layers": num_layers,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": lora_config.rank,
            "dropout": lora_config.dropout,
            "scale": lora_config.alpha,
        },
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"\nTraining complete. Adapter saved to {adapter_dir}")
    return True


async def run_full_experiment(
    model_id: str = "meta-llama/Llama-3.2-1B",
    n_training_samples: int = 5000,
    training_steps: list[int] | None = None,
    output_dir: Path | None = None,
    use_lazarus: bool = False,
):
    """Run the full classifier emergence experiment.

    Args:
        model_id: HuggingFace model ID
        n_training_samples: Number of training samples to generate
        training_steps: List of step counts to train/checkpoint at
        output_dir: Directory for outputs
        use_lazarus: If True, use Lazarus native training (SFTTrainer)
                    If False (default), use mlx-lm for stability
    """

    if training_steps is None:
        training_steps = [100, 500, 1000, 2000]

    if output_dir is None:
        output_dir = Path("./experiments/classifier_emergence")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate training data
    print("\n" + "=" * 60)
    print("STEP 1: Generate Training Data")
    print("=" * 60)

    data_path = output_dir / "arithmetic_train.jsonl"
    # Use Lazarus format (prompt/response) when using Lazarus trainer
    generate_arithmetic_data(n_training_samples, data_path, use_lazarus_format=use_lazarus)

    # 2. Baseline analysis
    print("\n" + "=" * 60)
    print("STEP 2: Baseline Analysis (Before Training)")
    print("=" * 60)

    prompts = generate_test_prompts()
    snapshots = []

    baseline = await analyze_model(model_id, prompts, "baseline", 0)
    snapshots.append(baseline)

    # 3. Progressive training and analysis
    print("\n" + "=" * 60)
    print("STEP 3: Progressive Training")
    print("=" * 60)

    # Select training function based on flag
    train_fn = run_training_lazarus if use_lazarus else run_training
    trainer_name = "Lazarus SFTTrainer" if use_lazarus else "mlx-lm"
    print(f"Using {trainer_name} for training")

    for i, steps in enumerate(training_steps):
        checkpoint_dir = output_dir / f"checkpoint_{steps}"

        print(f"\n--- Training to step {steps} ---")
        train_fn(
            model_id=model_id,
            data_path=data_path,
            output_dir=checkpoint_dir,
            steps=steps,
        )

        # Analyze checkpoint with adapter
        adapter_path = checkpoint_dir / "adapters"

        snapshot = await analyze_model_with_adapter(
            model_id,
            adapter_path,
            prompts,
            f"step_{steps}",
            steps,
        )
        snapshots.append(snapshot)

    # 4. Print comparison
    print_comparison_table(snapshots)

    # 5. Save full results
    results_file = output_dir / "experiment_results.json"
    all_results = {
        "model": model_id,
        "training_samples": n_training_samples,
        "snapshots": [
            {
                "checkpoint": s.checkpoint,
                "steps": s.steps,
                "has_classifiers": s.has_classifiers,
                "average_peak_prob": s.average_peak_prob,
            }
            for s in snapshots
        ],
    }

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results saved to {results_file}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    if snapshots[-1].has_classifiers and not snapshots[0].has_classifiers:
        print("\n✓ SUCCESS: Task classifiers EMERGED through training!")
        print("  Baseline: no classifiers")
        print(f"  After {training_steps[-1]} steps: classifiers present")
        print(f"  Peak probability: {snapshots[-1].average_peak_prob:.3f}")
    elif snapshots[0].has_classifiers:
        print("\n? NOTE: Task classifiers already present in baseline")
        print("  Consider using a different base model")
    else:
        print("\n✗ No classifier emergence detected")
        print("  Consider: more training steps, different curriculum")


def main():
    parser = argparse.ArgumentParser(
        description="Task Classifier Emergence Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-m",
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="Base model to use (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./experiments/classifier_emergence"),
        help="Output directory for results",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=5000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Training step checkpoints to analyze",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline analysis (no training)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training (no analysis)",
    )
    parser.add_argument(
        "--use-lazarus",
        action="store_true",
        help="Use Lazarus native SFTTrainer instead of mlx-lm (experimental)",
    )

    args = parser.parse_args()

    # Setup logging for Lazarus mode
    if args.use_lazarus:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    if args.baseline_only:
        asyncio.run(run_baseline_experiment(args.model, args.output))
    elif args.train_only:
        data_path = args.output / "arithmetic_train.jsonl"
        if not data_path.exists():
            generate_arithmetic_data(args.samples, data_path, use_lazarus_format=args.use_lazarus)
        if args.use_lazarus:
            run_training_lazarus(args.model, data_path, args.output, args.steps[-1])
        else:
            run_training(args.model, data_path, args.output, args.steps[-1])
    else:
        asyncio.run(
            run_full_experiment(
                model_id=args.model,
                n_training_samples=args.samples,
                training_steps=args.steps,
                output_dir=args.output,
                use_lazarus=args.use_lazarus,
            )
        )


if __name__ == "__main__":
    main()
