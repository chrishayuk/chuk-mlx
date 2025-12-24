"""
Simple MLP Example

Demonstrates that basic model components work by creating a simple
MLP-based classifier and running inference using Pydantic configs.

This example uses:
- EmbeddingConfig + TokenEmbedding for input embeddings
- FFNConfig + MLP feed-forward network
- RMSNorm for normalization
- A simple classifier head

Run with:
    uv run python examples/models/01_simple_mlp.py
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.components.embeddings import TokenEmbedding
from chuk_lazarus.models_v2.components.ffn import MLP
from chuk_lazarus.models_v2.components.normalization import RMSNorm
from chuk_lazarus.models_v2.core.config import EmbeddingConfig, FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType


class SimpleMLP(nn.Module):
    """
    A simple MLP classifier for demonstration.

    Uses Pydantic configs for all components.

    Architecture:
        Input tokens -> Embedding -> MLP -> Norm -> Output projection
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_classes: int = 5,
    ):
        super().__init__()

        # Pydantic config for embeddings
        embed_config = EmbeddingConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.embedding = TokenEmbedding(embed_config)

        # Pydantic config for MLP
        ffn_config = FFNConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=ActivationType.GELU,
        )
        self.mlp = MLP(ffn_config)

        # Normalization
        self.norm = RMSNorm(dims=hidden_size, eps=1e-5)

        # Output projection
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Store config for introspection
        self._embed_config = embed_config
        self._ffn_config = ffn_config

    def __call__(self, input_ids: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seq_len, hidden)

        # Apply MLP
        x = self.mlp(x)  # (batch, seq_len, hidden)

        # Normalize
        x = self.norm(x)  # (batch, seq_len, hidden)

        # Mean pooling over sequence
        x = mx.mean(x, axis=1)  # (batch, hidden)

        # Classify
        logits = self.classifier(x)  # (batch, num_classes)

        return logits


def main():
    print("=" * 60)
    print("Simple MLP Example (Pydantic-native)")
    print("=" * 60)

    # Configuration
    vocab_size = 1000
    hidden_size = 256
    intermediate_size = 512
    num_classes = 5
    batch_size = 4
    seq_len = 16

    # Create model
    print("\n1. Creating model with Pydantic configs...")
    model = SimpleMLP(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_classes=num_classes,
    )

    # Show Pydantic config introspection
    print("\n   Embedding config (Pydantic):")
    print(f"     vocab_size: {model._embed_config.vocab_size}")
    print(f"     hidden_size: {model._embed_config.hidden_size}")
    print(f"     scale_factor: {model._embed_config.scale_factor}")

    print("\n   FFN config (Pydantic):")
    print(f"     hidden_size: {model._ffn_config.hidden_size}")
    print(f"     intermediate_size: {model._ffn_config.intermediate_size}")
    print(f"     activation: {model._ffn_config.activation}")

    # Count parameters
    def count_params(params):
        """Recursively count parameters in nested dict."""
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, "size"):
                total += v.size
        return total

    total_params = count_params(model.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    # Create dummy input
    print("\n2. Creating input...")
    input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")

    # Forward pass
    print("\n3. Running forward pass...")
    logits = model(input_ids)
    mx.eval(logits)  # Force evaluation
    print(f"   Output shape: {logits.shape}")

    # Get predictions
    predictions = mx.argmax(logits, axis=-1)
    print(f"   Predictions: {predictions.tolist()}")

    # Compute probabilities
    probs = mx.softmax(logits, axis=-1)
    max_probs = mx.max(probs, axis=-1)
    print(f"   Max probability per sample: {[round(float(p), 3) for p in max_probs]}")

    # Show individual component outputs
    print("\n4. Component breakdown:")

    # Embedding
    embedded = model.embedding(input_ids)
    print(f"   Embedding output: {embedded.shape}")

    # MLP
    mlp_out = model.mlp(embedded)
    print(f"   MLP output: {mlp_out.shape}")

    # Norm
    norm_out = model.norm(mlp_out)
    print(f"   Norm output: {norm_out.shape}")

    # Verify values are reasonable
    print("\n5. Sanity checks:")
    print(f"   Logits min: {float(mx.min(logits)):.4f}")
    print(f"   Logits max: {float(mx.max(logits)):.4f}")
    print(f"   Logits mean: {float(mx.mean(logits)):.4f}")

    # Check that softmax sums to 1
    prob_sums = mx.sum(probs, axis=-1)
    print(f"   Probability sums: {[round(float(s), 4) for s in prob_sums]} (should be ~1.0)")

    print("\n" + "=" * 60)
    print("SUCCESS! MLP components work correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
