"""
Preference Dataset for DPO training.

Handles loading and batching of preference pairs (chosen, rejected).
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair."""

    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Rejected response
    metadata: dict | None = None


class PreferenceDataset:
    """
    Dataset for preference pairs.

    Expected format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

    Or with full messages:
    {"prompt": "...", "chosen": [{"role": "assistant", "content": "..."}], ...}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        self.pairs = self._load_data()
        logger.info(f"Loaded {len(self.pairs)} preference pairs from {data_path}")

    def _load_data(self) -> list[PreferencePair]:
        """Load preference pairs from JSONL file."""
        pairs = []

        with open(self.data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)

                # Handle both string and message-list formats
                chosen = item["chosen"]
                rejected = item["rejected"]

                if isinstance(chosen, list):
                    # Extract content from message format
                    chosen = chosen[-1]["content"] if chosen else ""
                if isinstance(rejected, list):
                    rejected = rejected[-1]["content"] if rejected else ""

                pairs.append(
                    PreferencePair(
                        prompt=item["prompt"],
                        chosen=chosen,
                        rejected=rejected,
                        metadata=item.get("metadata"),
                    )
                )

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Get a single tokenized preference pair."""
        pair = self.pairs[idx]
        return self._tokenize_pair(pair)

    def _tokenize_pair(self, pair: PreferencePair) -> dict:
        """Tokenize a preference pair into model inputs."""
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(pair.prompt)

        # Truncate prompt if needed
        if len(prompt_tokens) > self.max_prompt_length:
            prompt_tokens = prompt_tokens[: self.max_prompt_length]

        # Tokenize chosen and rejected (prompt + response)
        chosen_full = pair.prompt + pair.chosen
        rejected_full = pair.prompt + pair.rejected

        chosen_tokens = self.tokenizer.encode(chosen_full)
        rejected_tokens = self.tokenizer.encode(rejected_full)

        # Truncate to max length
        chosen_tokens = chosen_tokens[: self.max_length]
        rejected_tokens = rejected_tokens[: self.max_length]

        return {
            "prompt_length": len(prompt_tokens),
            "chosen_input_ids": chosen_tokens,
            "rejected_input_ids": rejected_tokens,
        }

    def get_batch(self, indices: list[int], pad_token_id: int = 0) -> dict:
        """
        Get a batch of tokenized preference pairs with padding.

        Returns dict with:
            - chosen_input_ids: (batch, max_seq_len)
            - rejected_input_ids: (batch, max_seq_len)
            - chosen_attention_mask: (batch, max_seq_len)
            - rejected_attention_mask: (batch, max_seq_len)
            - prompt_lengths: (batch,)
        """
        items = [self[i] for i in indices]

        # Find max lengths in this batch
        max_chosen_len = max(len(item["chosen_input_ids"]) for item in items)
        max_rejected_len = max(len(item["rejected_input_ids"]) for item in items)
        max_len = max(max_chosen_len, max_rejected_len)

        batch_size = len(items)

        # Initialize padded arrays
        chosen_ids = mx.full((batch_size, max_len), pad_token_id, dtype=mx.int32)
        rejected_ids = mx.full((batch_size, max_len), pad_token_id, dtype=mx.int32)
        chosen_mask = mx.zeros((batch_size, max_len), dtype=mx.float32)
        rejected_mask = mx.zeros((batch_size, max_len), dtype=mx.float32)
        prompt_lengths = mx.zeros((batch_size,), dtype=mx.int32)

        for i, item in enumerate(items):
            chosen_len = len(item["chosen_input_ids"])
            rejected_len = len(item["rejected_input_ids"])

            # Copy tokens (left-padded would be: max_len - length, but we do right-pad)
            chosen_ids = chosen_ids.at[i, :chosen_len].set(
                mx.array(item["chosen_input_ids"], dtype=mx.int32)
            )
            rejected_ids = rejected_ids.at[i, :rejected_len].set(
                mx.array(item["rejected_input_ids"], dtype=mx.int32)
            )

            # Set attention masks
            chosen_mask = chosen_mask.at[i, :chosen_len].set(1.0)
            rejected_mask = rejected_mask.at[i, :rejected_len].set(1.0)

            prompt_lengths = prompt_lengths.at[i].set(item["prompt_length"])

        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_attention_mask": rejected_mask,
            "prompt_lengths": prompt_lengths,
        }

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, pad_token_id: int = 0
    ) -> Iterator[dict]:
        """Iterate over batches of preference pairs."""
        import random

        indices = list(range(len(self.pairs)))

        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield self.get_batch(batch_indices, pad_token_id)


def load_preference_data(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    max_prompt_length: int = 256,
) -> PreferenceDataset:
    """Convenience function to load preference dataset."""
    return PreferenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
