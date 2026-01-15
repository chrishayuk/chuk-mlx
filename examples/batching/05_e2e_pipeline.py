#!/usr/bin/env python3
"""
Example 5: Unified Batch Pipeline — Dataset to Training

Demonstrates the unified batching architecture where BatchPlan is the
single source of truth for all execution paths:

1. Dataset → Tokenize → Length Index
2. BatchPlanBuilder → BatchPlan (universal IR)
3. Training via one of three paths:
   - Stream: plan.iter_epoch() (default, memory-efficient)
   - Cache: BatchWriter → NPZ → BatchReader (for repeated runs)
   - Distributed: plan.shard() (multi-GPU/node)

Run:
    python examples/batching/05_e2e_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np

# =============================================================================
# Mock Components (replace with real implementations)
# =============================================================================


class MockTokenizer:
    """Mock tokenizer for demonstration."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._vocab = {}
        self._next_id = 3

    def encode(self, text: str) -> list[int]:
        tokens = [1]  # BOS
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[word])
        return tokens


class MockModel:
    """Mock model for demonstration."""

    def __call__(self, batch: dict) -> float:
        return float(np.random.rand())


class MockOptimizer:
    """Mock optimizer for demonstration."""

    def step(self):
        pass

    def zero_grad(self):
        pass


# =============================================================================
# Stage 1: Dataset
# =============================================================================


def create_dataset(path: Path, num_samples: int = 100) -> Path:
    """Create a sample JSONL dataset."""
    import random

    random.seed(42)

    with open(path, "w") as f:
        for i in range(num_samples):
            response_len = random.randint(20, 150)
            sample = {
                "id": f"sample_{i:04d}",
                "instruction": f"Explain concept {i}.",
                "response": " ".join(f"word{j}" for j in range(response_len)),
            }
            f.write(json.dumps(sample) + "\n")

    return path


# =============================================================================
# Stage 2: Tokenize & Build Length Index
# =============================================================================


def tokenize_sample(sample: dict, tokenizer: MockTokenizer) -> tuple[list[int], list[int], int]:
    """Tokenize sample into (input_ids, loss_mask, length)."""
    prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"
    full_text = prompt + sample["response"]

    prompt_tokens = tokenizer.encode(prompt)
    full_tokens = tokenizer.encode(full_text)
    if full_tokens[-1] != tokenizer.eos_token_id:
        full_tokens.append(tokenizer.eos_token_id)

    # Loss mask: 0 for prompt, 1 for response
    loss_mask = [0] * len(prompt_tokens) + [1] * (len(full_tokens) - len(prompt_tokens))

    return full_tokens, loss_mask, len(full_tokens)


def load_and_tokenize(
    dataset_path: Path, tokenizer: MockTokenizer
) -> tuple[dict[str, int], dict[str, dict]]:
    """Load dataset and return (lengths, samples)."""
    lengths = {}
    samples = {}

    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            input_ids, loss_mask, length = tokenize_sample(sample, tokenizer)
            lengths[sample["id"]] = length
            samples[sample["id"]] = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
            }

    return lengths, samples


# =============================================================================
# Stage 3: Collate Function
# =============================================================================


def collate(batch_samples: list[dict], max_len: int, pad_id: int = 0) -> dict[str, np.ndarray]:
    """Collate samples into padded batch arrays."""

    def pad(seq: list[int], pad_val: int) -> list[int]:
        return seq[:max_len] + [pad_val] * max(0, max_len - len(seq))

    input_ids = np.array([pad(s["input_ids"], pad_id) for s in batch_samples], dtype=np.int32)
    loss_mask = np.array([pad(s["loss_mask"], 0) for s in batch_samples], dtype=np.int32)

    return {"input_ids": input_ids, "loss_mask": loss_mask}


# =============================================================================
# Main: Unified Pipeline Demo
# =============================================================================


async def main():
    print("=" * 70)
    print("Unified Batch Pipeline: Dataset → BatchPlan → Training")
    print("=" * 70)

    tokenizer = MockTokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # =====================================================================
        # Stage 1: Create Dataset
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 1: Dataset")
        print("=" * 70)

        dataset_path = create_dataset(tmpdir / "train.jsonl", num_samples=100)
        print(f"  Created: {dataset_path}")

        # =====================================================================
        # Stage 2: Tokenize & Build Length Index
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 2: Tokenize & Length Index")
        print("=" * 70)

        lengths, samples = load_and_tokenize(dataset_path, tokenizer)
        print(f"  Samples: {len(lengths)}")
        print(f"  Length range: {min(lengths.values())} - {max(lengths.values())}")
        print(f"  Mean length: {sum(lengths.values()) / len(lengths):.1f}")

        # =====================================================================
        # Stage 3: Build BatchPlan (Universal IR)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 3: Build BatchPlan")
        print("=" * 70)

        from chuk_lazarus.data.batching import (
            BatchingConfig,
            BatchPlanBuilder,
            BatchReader,
            BatchWriter,
        )

        config = BatchingConfig.predictable(
            token_budget=2048,
            bucket_edges=(128, 256, 512),
            overflow_max=1024,
            seed=42,
        )

        plan = await BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash="demo_v1",
            tokenizer_hash="mock_v1",
        ).build(num_epochs=2)

        print(f"  Epochs: {plan.num_epochs}")
        print(f"  Total microbatches: {plan.total_microbatches}")
        print(f"  Fingerprint: {plan.fingerprint}")

        for ep in range(plan.num_epochs):
            epoch_plan = plan.get_epoch(ep)
            print(
                f"    Epoch {ep}: {epoch_plan.num_microbatches} batches, "
                f"{epoch_plan.total_samples} samples"
            )

        # =====================================================================
        # Stage 4: Training (Three Equivalent Paths)
        # =====================================================================
        print("\n" + "=" * 70)
        print("STAGE 4: Training (Three Paths)")
        print("=" * 70)

        model = MockModel()
        optimizer = MockOptimizer()

        # -----------------------------------------------------------------
        # Path A: Stream (default)
        # -----------------------------------------------------------------
        print("\n  Path A: Stream")
        print("  " + "-" * 50)

        for epoch in range(plan.num_epochs):
            total_loss = 0.0
            num_batches = 0

            for mb in plan.iter_epoch(epoch):
                batch_samples = [samples[sid] for sid in mb.samples]
                batch = collate(batch_samples, mb.max_len, tokenizer.pad_token_id)
                loss = model(batch)
                optimizer.step()
                total_loss += loss
                num_batches += 1

            print(
                f"    Epoch {epoch}: {num_batches} batches, avg_loss={total_loss / num_batches:.4f}"
            )

        # -----------------------------------------------------------------
        # Path B: Cache to NPZ using BatchWriter/BatchReader
        # -----------------------------------------------------------------
        print("\n  Path B: Cache (NPZ) using BatchWriter/BatchReader")
        print("  " + "-" * 50)

        cache_dir = tmpdir / "batch_cache"

        # Write batches using BatchWriter
        writer = BatchWriter(
            plan=plan,
            samples=samples,
            output_dir=cache_dir,
            pad_id=tokenizer.pad_token_id,
        )
        batch_files = writer.write_all()
        print(f"    Written {len(batch_files)} batch files to {cache_dir}")
        print(f"    Fingerprint saved: {writer.plan.fingerprint}")

        # Read and train using BatchReader
        reader = BatchReader(cache_dir)
        print(f"    Reader fingerprint: {reader.fingerprint}")
        print(f"    Fingerprint verified: {reader.verify_fingerprint(plan.fingerprint)}")

        for epoch in range(reader.num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in reader.iter_epoch(epoch):
                loss = model({"input_ids": batch["input_ids"], "loss_mask": batch["loss_mask"]})
                optimizer.step()
                total_loss += loss
                num_batches += 1

            print(f"    Epoch {epoch}: {num_batches} batches (from cache)")

        # -----------------------------------------------------------------
        # Path C: Distributed (simulated)
        # -----------------------------------------------------------------
        print("\n  Path C: Distributed")
        print("  " + "-" * 50)

        world_size = 4
        print(f"    Simulating {world_size} workers:")

        for rank in range(world_size):
            shard = plan.shard(rank=rank, world_size=world_size)
            epoch0 = shard.get_epoch(0)
            print(
                f"      Rank {rank}: {epoch0.num_microbatches} batches, "
                f"{epoch0.total_samples} samples"
            )

        # Verify all samples covered
        all_samples = set()
        for rank in range(world_size):
            for mb in plan.shard(rank, world_size).iter_epoch(0):
                all_samples.update(mb.samples)

        original_samples = set()
        for mb in plan.iter_epoch(0):
            original_samples.update(mb.samples)

        if all_samples == original_samples:
            print(f"    ✓ All {len(original_samples)} samples covered")
        else:
            print(f"    ✗ Missing: {len(original_samples - all_samples)}")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
  The unified pipeline uses BatchPlan as the single source of truth:

  1. Dataset -> Tokenize -> Length Index
  2. BatchPlanBuilder -> BatchPlan (universal IR)
  3. Training via:
     - Stream: plan.iter_epoch() - memory efficient, default
     - Cache: BatchWriter -> NPZ -> BatchReader - for repeated experiments
     - Distributed: plan.shard() - multi-GPU/node

  Key benefits:
  - Single configuration (BatchingConfig)
  - Reproducible (fingerprinted plans)
  - Efficient (token-budget batching)
  - Scalable (built-in sharding)
  - Unified API (BatchWriter/BatchReader for I/O)
""")

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
