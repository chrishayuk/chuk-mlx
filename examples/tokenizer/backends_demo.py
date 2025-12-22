#!/usr/bin/env python3
"""
Tokenizer Backends Demo

Demonstrates the pluggable backend architecture with two implementations:
- HuggingFaceBackend: Default, portable, HuggingFace-compatible
- FastBackend: High-throughput parallel tokenization using MLX Data CharTrie

The fast backend is optional and requires mlx-data:
    pip install chuk-lazarus[fast]
"""

import time

from chuk_lazarus.data.tokenizers.backends import (
    BackendType,
    FastBackend,
    HuggingFaceBackend,
    create_backend,
    get_best_backend,
    is_fast_backend_available,
)

# =============================================================================
# Mock tokenizer for demonstration (simulates HuggingFace tokenizer)
# =============================================================================


class MockHFTokenizer:
    """Mock HuggingFace-style tokenizer for demonstration."""

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "hello": 4,
            "world": 5,
            "the": 6,
            "quick": 7,
            "brown": 8,
            "fox": 9,
            "jumps": 10,
            "over": 11,
            "lazy": 12,
            "dog": 13,
            " ": 14,
        }
        self._id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(1)  # UNK
        if add_special_tokens:
            tokens = [2] + tokens + [3]  # BOS, EOS
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in [0, 2, 3]:
                continue
            if tid in self._id_to_token:
                tokens.append(self._id_to_token[tid])
        return " ".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token.get(i, "<unk>") for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1

    @property
    def bos_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 3


# =============================================================================
# Demo Functions
# =============================================================================


def demo_huggingface_backend():
    """Demonstrate the HuggingFace backend."""
    print("=" * 60)
    print("HuggingFace Backend Demo")
    print("=" * 60)

    # Create mock tokenizer
    hf_tokenizer = MockHFTokenizer()

    # Wrap with HuggingFaceBackend
    backend = HuggingFaceBackend(hf_tokenizer)

    print(f"\nBackend type: {backend.backend_type}")
    print(f"Vocab size: {backend.vocab_size}")
    print(f"BOS token ID: {backend.bos_token_id}")
    print(f"EOS token ID: {backend.eos_token_id}")

    # Encode text
    text = "hello world"
    result = backend.encode(text)
    print(f"\nEncoding '{text}':")
    print(f"  Token IDs: {result.token_ids}")
    print(f"  Tokens: {result.tokens}")

    # Decode back
    decoded = backend.decode(result.token_ids)
    print(f"  Decoded: '{decoded}'")

    # Batch encoding
    texts = ["hello world", "the quick fox", "lazy dog"]
    batch_result = backend.encode_batch(texts)
    print(f"\nBatch encoding {len(texts)} texts:")
    print(f"  Total tokens: {batch_result.total_tokens}")
    for i, r in enumerate(batch_result.results):
        print(f"  [{i}] {r.token_ids}")

    # Backend info
    info = backend.get_info()
    print("\nBackend info:")
    print(f"  Supports parallel: {info.supports_parallel}")
    print(f"  Supports offsets: {info.supports_offsets}")
    print(f"  Is available: {info.is_available}")


def demo_fast_backend():
    """Demonstrate the Fast backend (if available)."""
    print("\n" + "=" * 60)
    print("Fast Backend Demo (MLX Data CharTrie)")
    print("=" * 60)

    if not is_fast_backend_available():
        print("\nFast backend not available.")
        print("Install with: pip install 'chuk-lazarus[fast]'")
        print("  or: uv pip install 'chuk-lazarus[fast]'")
        print("(Requires mlx-data package)")
        return

    # Create vocabulary
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "hello": 4,
        "world": 5,
        "the": 6,
        "quick": 7,
        "brown": 8,
        "fox": 9,
        "jumps": 10,
        "over": 11,
        "lazy": 12,
        "dog": 13,
    }

    # Create FastBackend directly from vocabulary
    backend = FastBackend(
        vocab,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        unk_token_id=1,
    )

    print(f"\nBackend type: {backend.backend_type}")
    print(f"Vocab size: {backend.vocab_size}")

    # Encode text
    text = "hello world"
    result = backend.encode(text)
    print(f"\nEncoding '{text}':")
    print(f"  Token IDs: {result.token_ids}")
    print(f"  Tokens: {result.tokens}")

    # Decode back
    decoded = backend.decode(result.token_ids)
    print(f"  Decoded: '{decoded}'")

    # Parallel batch encoding - this is where FastBackend shines
    texts = ["hello world", "the quick fox", "lazy dog"] * 100  # 300 texts
    print(f"\nParallel batch encoding {len(texts)} texts:")

    # Sequential (num_workers=1)
    start = time.perf_counter()
    result_seq = backend.encode_batch(texts, num_workers=1)
    time_seq = time.perf_counter() - start
    print(f"  Sequential: {time_seq:.4f}s ({result_seq.total_tokens} tokens)")

    # Parallel (num_workers=4)
    start = time.perf_counter()
    result_par = backend.encode_batch(texts, num_workers=4)
    time_par = time.perf_counter() - start
    print(f"  Parallel (4 workers): {time_par:.4f}s ({result_par.total_tokens} tokens)")

    if time_par > 0:
        speedup = time_seq / time_par
        print(f"  Speedup: {speedup:.2f}x")

    # Backend info
    info = backend.get_info()
    print("\nBackend info:")
    print(f"  Supports parallel: {info.supports_parallel}")
    print(f"  Supports offsets: {info.supports_offsets}")
    print(f"  Is available: {info.is_available}")


def demo_factory_functions():
    """Demonstrate factory functions for backend creation."""
    print("\n" + "=" * 60)
    print("Factory Functions Demo")
    print("=" * 60)

    hf_tokenizer = MockHFTokenizer()

    # Create backend by type
    print("\nUsing create_backend():")
    backend = create_backend(BackendType.HUGGINGFACE, hf_tokenizer)
    print(f"  Created: {backend.backend_type}")

    # Also works with string
    backend = create_backend("huggingface", hf_tokenizer)
    print(f"  From string 'huggingface': {backend.backend_type}")

    # Get best available backend
    print("\nUsing get_best_backend():")
    backend = get_best_backend(hf_tokenizer, prefer_fast=True)
    print(f"  Best backend (prefer_fast=True): {backend.backend_type}")

    backend = get_best_backend(hf_tokenizer, prefer_fast=False)
    print(f"  Best backend (prefer_fast=False): {backend.backend_type}")

    # Check availability
    print(f"\nFast backend available: {is_fast_backend_available()}")


def demo_from_hf_tokenizer():
    """Demonstrate creating FastBackend from HuggingFace tokenizer."""
    print("\n" + "=" * 60)
    print("FastBackend from HuggingFace Tokenizer")
    print("=" * 60)

    if not is_fast_backend_available():
        print("\nFast backend not available. Skipping.")
        return

    hf_tokenizer = MockHFTokenizer()

    # Create FastBackend from HF tokenizer
    # This extracts the vocabulary and special token IDs
    fast_backend = FastBackend.from_tokenizer(hf_tokenizer)

    print("\nCreated FastBackend from HuggingFace tokenizer:")
    print(f"  Vocab size: {fast_backend.vocab_size}")
    print(f"  BOS token ID: {fast_backend.bos_token_id}")
    print(f"  EOS token ID: {fast_backend.eos_token_id}")

    # Compare encoding results
    text = "hello world"
    hf_backend = HuggingFaceBackend(hf_tokenizer)

    hf_result = hf_backend.encode(text)
    fast_result = fast_backend.encode(text)

    print(f"\nEncoding '{text}':")
    print(f"  HuggingFace: {hf_result.token_ids}")
    print(f"  Fast:        {fast_result.token_ids}")


def main():
    """Run all demos."""
    print("Tokenizer Backends Demo")
    print("=" * 60)
    print()

    demo_huggingface_backend()
    demo_fast_backend()
    demo_factory_functions()
    demo_from_hf_tokenizer()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
