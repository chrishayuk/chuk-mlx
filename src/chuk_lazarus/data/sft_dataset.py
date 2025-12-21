"""
SFT Dataset for supervised fine-tuning.

Handles loading and batching of prompt-response pairs.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class SFTSample:
    """A single SFT training sample."""

    prompt: str
    response: str
    metadata: dict | None = None


class SFTDataset:
    """
    Dataset for SFT training.

    Expected format (JSONL):
    {"prompt": "...", "response": "..."}

    Or with messages:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, mask_prompt: bool = True):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt

        self.samples = self._load_data()
        logger.info(f"Loaded {len(self.samples)} SFT samples from {data_path}")

    def _load_data(self) -> list[SFTSample]:
        """Load samples from JSONL file."""
        samples = []

        with open(self.data_path) as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Handle different formats
                if "messages" in item:
                    # Chat format
                    messages = item["messages"]
                    prompt = ""
                    response = ""
                    for msg in messages:
                        if msg["role"] in ["user", "system"]:
                            prompt += msg["content"] + "\n"
                        elif msg["role"] == "assistant":
                            response = msg["content"]
                else:
                    # Simple format
                    prompt = item.get("prompt", item.get("input", ""))
                    response = item.get("response", item.get("output", item.get("completion", "")))

                samples.append(
                    SFTSample(
                        prompt=prompt.strip(),
                        response=response.strip(),
                        metadata=item.get("metadata"),
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single tokenized sample."""
        sample = self.samples[idx]
        return self._tokenize_sample(sample)

    def _tokenize_sample(self, sample: SFTSample) -> dict[str, Any]:
        """Tokenize a sample into model inputs."""
        # Tokenize prompt and full sequence
        prompt_tokens = self.tokenizer.encode(sample.prompt)
        full_text = sample.prompt + sample.response
        full_tokens = self.tokenizer.encode(full_text)

        # Truncate if needed
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[: self.max_length]

        # Create labels (shift by 1 for next-token prediction)
        labels = full_tokens[1:] + [self.tokenizer.eos_token_id or 0]

        # Create loss mask (1 for response tokens, 0 for prompt if masking)
        if self.mask_prompt:
            # Mask prompt tokens, only compute loss on response
            prompt_len = min(len(prompt_tokens), len(labels))
            loss_mask = [0.0] * prompt_len + [1.0] * (len(labels) - prompt_len)
        else:
            loss_mask = [1.0] * len(labels)

        return {
            "input_ids": full_tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "prompt_length": len(prompt_tokens),
        }

    def get_batch(self, indices: list[int], pad_token_id: int = 0) -> dict[str, mx.array]:
        """Get a padded batch."""
        items = [self[i] for i in indices]

        # Find max length
        max_len = max(len(item["input_ids"]) for item in items)
        batch_size = len(items)

        # Initialize arrays
        input_ids = mx.full((batch_size, max_len), pad_token_id, dtype=mx.int32)
        labels = mx.full((batch_size, max_len), pad_token_id, dtype=mx.int32)
        loss_mask = mx.zeros((batch_size, max_len), dtype=mx.float32)
        attention_mask = mx.zeros((batch_size, max_len), dtype=mx.float32)

        for i, item in enumerate(items):
            seq_len = len(item["input_ids"])

            input_ids = input_ids.at[i, :seq_len].set(mx.array(item["input_ids"], dtype=mx.int32))
            labels = labels.at[i, :seq_len].set(mx.array(item["labels"], dtype=mx.int32))
            loss_mask = loss_mask.at[i, :seq_len].set(mx.array(item["loss_mask"], dtype=mx.float32))
            attention_mask = attention_mask.at[i, :seq_len].set(1.0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, pad_token_id: int = 0
    ) -> Iterator[dict[str, mx.array]]:
        """Iterate over batches."""
        import random

        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield self.get_batch(batch_indices, pad_token_id)
