"""
LoRA Model Loading Example

Shows how to load a model with LoRA adapters.
"""

from chuk_lazarus.models import load_model, LoRAConfig


def main():
    # Configure LoRA
    lora_config = LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    )

    # Load model with LoRA
    print("Loading model with LoRA...")
    model, tokenizer = load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora=True,
        lora_config=lora_config,
    )

    # Count parameters
    total_params = sum(p.size for p in model.parameters().values())
    trainable_params = sum(
        p.size for name, p in model.parameters().items()
        if "lora" in name.lower()
    )

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable (LoRA) parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")


if __name__ == "__main__":
    main()
