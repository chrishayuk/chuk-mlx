"""
SFT (Supervised Fine-Tuning) Training Example

Shows how to fine-tune a model on instruction-following data.
"""

from chuk_lazarus.models import load_model
from chuk_lazarus.training import SFTTrainer, SFTConfig
from chuk_lazarus.data import SFTDataset


def main():
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora=True,  # Use LoRA for efficient fine-tuning
    )

    # Create dataset (expects JSONL with "prompt" and "response" fields)
    # Example data format:
    # {"prompt": "What is 2+2?", "response": "2+2 equals 4."}
    dataset = SFTDataset(
        path="./data/train.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )

    # Configure training
    config = SFTConfig(
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        warmup_steps=100,
        checkpoint_dir="./checkpoints/sft",
        log_interval=10,
    )

    # Create trainer and train
    trainer = SFTTrainer(model, tokenizer, config)
    trainer.train(dataset)

    print("Training complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
