"""
DPO (Direct Preference Optimization) Training Example

Shows how to train a model using preference pairs.
"""

from chuk_lazarus.models import load_model
from chuk_lazarus.training import DPOTrainer, DPOTrainerConfig
from chuk_lazarus.training.losses import DPOConfig
from chuk_lazarus.data import PreferenceDataset


def main():
    # Load policy and reference models
    print("Loading models...")
    policy_model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    reference_model, _ = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create preference dataset
    # Expects JSONL with "prompt", "chosen", "rejected" fields
    dataset = PreferenceDataset(
        path="./data/preferences.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )

    # Configure DPO
    config = DPOTrainerConfig(
        dpo=DPOConfig(
            beta=0.1,          # KL penalty coefficient
            label_smoothing=0.0,
        ),
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-6,
        checkpoint_dir="./checkpoints/dpo",
    )

    # Train
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.train(dataset)

    print("DPO training complete!")


if __name__ == "__main__":
    main()
