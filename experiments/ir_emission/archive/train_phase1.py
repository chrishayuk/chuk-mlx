#!/usr/bin/env python3
"""
Phase 1: Train IR Emission from L13 Hidden States

This trains the model to emit WASM IR sequences from L13 activations.
Phase 1 focuses on single-operation arithmetic.

Usage:
    python experiments/ir_emission/train_phase1.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --steps 1000 \
        --output experiments/ir_emission/checkpoints/phase1
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Add project root for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.ir_emission.shared import CodebookConfig, IROpcode, IRSequenceDecoder, WASMRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for IR emission training."""

    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data_path: str = "experiments/ir_emission/data/phase1_train.jsonl"
    test_path: str = "experiments/ir_emission/data/phase1_test.jsonl"
    output_dir: str = "experiments/ir_emission/checkpoints/phase1"

    # Model architecture
    decision_layer_ratio: float = 0.55  # L13 for 24-layer model
    codebook_size: int = 64
    embedding_dim: int = 128
    max_ir_length: int = 8

    # Training
    max_steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    commitment_weight: float = 0.25

    # Logging
    log_interval: int = 50
    eval_interval: int = 200
    checkpoint_interval: int = 500


def load_samples(path: str) -> list[dict]:
    """Load JSONL dataset."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_numbers(text: str) -> list[int]:
    """Extract numbers from text for operand slots."""
    import re

    numbers = re.findall(r"\d+", text)
    return [int(n) for n in numbers]


class IREmissionTrainer:
    """
    Trainer for IR emission from L13 hidden states.

    The training loop:
    1. Forward pass through base model to get L13 hidden states
    2. Decode hidden states to IR sequence
    3. Execute IR and compare to expected result
    4. Backprop through decoder (base model frozen)
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_config,
        config: TrainingConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.config = config

        # Compute decision layer
        num_layers = model_config.num_hidden_layers
        self.decision_layer = int(num_layers * config.decision_layer_ratio)
        logger.info(f"Decision layer: {self.decision_layer} / {num_layers}")

        # Get hidden dimension from model
        hidden_dim = model_config.hidden_size

        # Create IR decoder
        codebook_config = CodebookConfig(
            codebook_size=config.codebook_size,
            hidden_dim=hidden_dim,
            embedding_dim=config.embedding_dim,
            max_ir_length=config.max_ir_length,
        )
        self.decoder = IRSequenceDecoder(codebook_config)

        # WASM runtime for execution verification
        self.runtime = WASMRuntime(use_native=True)

        # Optimizer (only for decoder, base model frozen)
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
        )

        # Metrics
        self.step = 0
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "valid_ir_rate": [],
        }

    def get_hidden_state(self, prompt: str) -> mx.array:
        """Extract hidden state at decision layer."""
        # Tokenize
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Access backbone (model.model for LlamaForCausalLM)
        backbone = self.model.model if hasattr(self.model, "model") else self.model

        # Forward through embedding
        h = backbone.embed_tokens(input_ids)

        # Forward through layers up to decision layer
        mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens))
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(backbone.layers):
            if i > self.decision_layer:
                break
            # Layer returns BlockOutput, extract hidden_states
            output = layer(h, mask=mask)
            h = output.hidden_states if hasattr(output, "hidden_states") else output

        # Return last token's hidden state
        return h[0, -1, :]  # (hidden_dim,)

    def compute_loss(
        self,
        hidden_states: mx.array,
        target_ir: mx.array,
        operands_list: list[list[int]],
        expected_results: list[int],
    ) -> tuple[mx.array, dict]:
        """
        Compute loss for a batch.

        Returns:
            loss: Scalar loss value
            metrics: Dict with accuracy, valid_ir_rate, etc.
        """
        batch_size = hidden_states.shape[0]

        # Forward through decoder
        logits, commitment_loss = self.decoder(hidden_states, target_ir)

        # Cross-entropy loss on IR sequence
        # logits: (batch, seq_len, codebook_size)
        # target_ir: (batch, seq_len)
        ce_loss = nn.losses.cross_entropy(
            logits.reshape(-1, self.config.codebook_size),
            target_ir.reshape(-1),
            reduction="mean",
        )

        # Total loss
        loss = ce_loss + self.config.commitment_weight * commitment_loss

        # Compute metrics (execution accuracy)
        predicted_ir = mx.argmax(logits, axis=-1)  # (batch, seq_len)

        correct = 0
        valid_ir = 0

        for i in range(batch_size):
            ir_indices = predicted_ir[i].tolist()
            operands = operands_list[i]
            expected = expected_results[i]

            # Convert to WASM and execute
            try:
                body = self.decoder.codebook.indices_to_wasm(ir_indices, operands)
                result = self.runtime.execute(body)

                if result.success:
                    valid_ir += 1
                    if result.result == expected:
                        correct += 1
            except Exception:
                pass

        metrics = {
            "ce_loss": float(ce_loss.item()),
            "commitment_loss": float(commitment_loss.item()),
            "accuracy": correct / batch_size,
            "valid_ir_rate": valid_ir / batch_size,
        }

        return loss, metrics

    def train_step(self, batch: list[dict]) -> dict:
        """Single training step."""
        # Extract hidden states (detached from base model graph)
        hidden_states = []
        for sample in batch:
            h = self.get_hidden_state(sample["prompt"])
            hidden_states.append(h)

        hidden_states = mx.stack(hidden_states)  # (batch, hidden_dim)
        mx.eval(hidden_states)  # Materialize to detach from base model

        # Prepare targets
        max_len = max(len(s["ir_sequence"]) for s in batch)
        target_ir = []
        for sample in batch:
            ir = sample["ir_sequence"]
            # Pad to max_len
            padded = ir + [IROpcode.PAD] * (max_len - len(ir))
            target_ir.append(padded)

        target_ir = mx.array(target_ir)  # (batch, seq_len)

        operands_list = [s["operands"] for s in batch]
        expected_results = [s["expected_result"] for s in batch]

        # Compute loss and gradients using nn.value_and_grad
        def loss_fn(decoder_params):
            # Temporarily set decoder parameters
            self.decoder.update(decoder_params)

            # Forward through decoder
            logits, commitment_loss = self.decoder(hidden_states, target_ir)

            # Cross-entropy loss on IR sequence
            ce_loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.config.codebook_size),
                target_ir.reshape(-1),
                reduction="mean",
            )

            # Total loss
            loss = ce_loss + self.config.commitment_weight * commitment_loss
            return loss

        # Get gradients
        loss, grads = nn.value_and_grad(self.decoder, loss_fn)(self.decoder.parameters())

        # Update decoder
        self.optimizer.update(self.decoder, grads)
        mx.eval(self.decoder.parameters())

        # Compute metrics separately (no gradient needed)
        logits, _ = self.decoder(hidden_states, target_ir)
        predicted_ir = mx.argmax(logits, axis=-1)

        batch_size = hidden_states.shape[0]
        correct = 0
        valid_ir = 0

        for i in range(batch_size):
            ir_indices = predicted_ir[i].tolist()
            operands = operands_list[i]
            expected = expected_results[i]

            try:
                body = self.decoder.codebook.indices_to_wasm(ir_indices, operands)
                result = self.runtime.execute(body)

                if result.success:
                    valid_ir += 1
                    if result.result == expected:
                        correct += 1
            except Exception:
                pass

        metrics = {
            "loss": float(loss.item()),
            "accuracy": correct / batch_size,
            "valid_ir_rate": valid_ir / batch_size,
        }

        self.step += 1
        return metrics

    def evaluate(self, test_samples: list[dict]) -> dict:
        """Evaluate on test set."""
        total_correct = 0
        total_valid = 0
        total = 0

        for sample in test_samples:
            h = self.get_hidden_state(sample["prompt"])
            h = h[None, :]  # Add batch dim

            # Generate IR
            ir_indices = self.decoder.generate(h, temperature=0)

            # Execute
            operands = sample["operands"]
            expected = sample["expected_result"]

            try:
                body = self.decoder.codebook.indices_to_wasm(ir_indices, operands)
                result = self.runtime.execute(body)

                if result.success:
                    total_valid += 1
                    if result.result == expected:
                        total_correct += 1
            except Exception:
                pass

            total += 1

        return {
            "accuracy": total_correct / total if total > 0 else 0,
            "valid_ir_rate": total_valid / total if total > 0 else 0,
            "total": total,
            "correct": total_correct,
        }

    def train(self, train_samples: list[dict], test_samples: list[dict]):
        """Main training loop."""
        import random

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on {len(train_samples)} samples")
        logger.info(f"Test set: {len(test_samples)} samples")

        random.shuffle(train_samples)

        for step in range(self.config.max_steps):
            # Sample batch
            batch_indices = [
                i % len(train_samples)
                for i in range(
                    step * self.config.batch_size,
                    (step + 1) * self.config.batch_size,
                )
            ]
            batch = [train_samples[i] for i in batch_indices]

            # Train step
            metrics = self.train_step(batch)

            # Log
            if (step + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Step {step + 1}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"acc={metrics['accuracy']:.2%}, "
                    f"valid_ir={metrics['valid_ir_rate']:.2%}"
                )

            # Evaluate
            if (step + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(test_samples[:100])
                logger.info(
                    f"Eval: acc={eval_metrics['accuracy']:.2%}, "
                    f"valid_ir={eval_metrics['valid_ir_rate']:.2%}"
                )

            # Checkpoint
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(output_dir / f"step_{step + 1}")

        # Final evaluation
        final_metrics = self.evaluate(test_samples)
        logger.info(f"\nFinal evaluation: {final_metrics}")

        # Save final checkpoint
        self.save_checkpoint(output_dir / "final")

        return final_metrics

    def save_checkpoint(self, path: Path):
        """Save decoder checkpoint."""
        path.mkdir(parents=True, exist_ok=True)

        # Save decoder weights - flatten nested dict
        flat_params = {}
        for k, v in self.decoder.parameters().items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_params[f"{k}.{k2}"] = v2
            else:
                flat_params[k] = v
        mx.savez(str(path / "decoder.npz"), **flat_params)

        # Save config
        config_dict = {
            "decision_layer": self.decision_layer,
            "codebook_size": self.config.codebook_size,
            "embedding_dim": self.config.embedding_dim,
            "max_ir_length": self.config.max_ir_length,
            "step": self.step,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: IR Emission Training")
    parser.add_argument("--model", "-m", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", "-d", default="experiments/ir_emission/data/phase1_train.jsonl")
    parser.add_argument("--test-data", default="experiments/ir_emission/data/phase1_test.jsonl")
    parser.add_argument("--output", "-o", default="experiments/ir_emission/checkpoints/phase1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data).exists():
        logger.error(f"Data not found: {args.data}")
        logger.info("Run: python experiments/ir_emission/generate_data.py")
        return

    # Load model
    logger.info(f"Loading model: {args.model}")

    from chuk_lazarus.models_v2.loader import load_model

    load_result = load_model(args.model)
    model = load_result.model
    tokenizer = load_result.tokenizer
    model_config = load_result.config

    # Freeze base model
    model.freeze()

    # Load data
    logger.info(f"Loading data: {args.data}")
    train_samples = load_samples(args.data)

    test_samples = []
    if Path(args.test_data).exists():
        test_samples = load_samples(args.test_data)
    else:
        # Split train data
        split = int(len(train_samples) * 0.9)
        test_samples = train_samples[split:]
        train_samples = train_samples[:split]

    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Create trainer
    config = TrainingConfig(
        model=args.model,
        data_path=args.data,
        test_path=args.test_data,
        output_dir=args.output,
        max_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    trainer = IREmissionTrainer(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        config=config,
    )

    # Train
    final_metrics = trainer.train(train_samples, test_samples)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final accuracy: {final_metrics['accuracy']:.2%}")
    print(f"Valid IR rate: {final_metrics['valid_ir_rate']:.2%}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
