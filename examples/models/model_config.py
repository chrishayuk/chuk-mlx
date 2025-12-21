"""
Model Configuration Example

Shows how to create and use model configurations.
"""

from chuk_lazarus.models import ModelConfig


def main():
    # Create a model config
    config = ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=32000,
        max_position_embeddings=2048,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )

    print("Model Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max positions: {config.max_position_embeddings}")

    # Save config
    # config.save("./my_model/config.json")

    # Load config
    # loaded = ModelConfig.load("./my_model/config.json")


if __name__ == "__main__":
    main()
