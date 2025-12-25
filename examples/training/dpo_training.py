"""
DPO (Direct Preference Optimization) Training Example

Shows how to train a model using preference pairs.

This example demonstrates the integration of:
- models_v2: Model loading and architecture
- training: DPOTrainer for preference learning
- data: PreferenceDataset for loading preference pairs
"""

from chuk_lazarus import load_model
from chuk_lazarus.data import PreferenceDataset
from chuk_lazarus.training import DPOTrainer, DPOTrainerConfig
from chuk_lazarus.training.losses import DPOConfig


def main():
    # Load policy and reference models
    # The policy model will be trained, reference model stays frozen
    print("Loading models...")
    policy_model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    reference_model, _ = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create preference dataset
    # Expects JSONL with "prompt", "chosen", "rejected" fields
    # Example: {"prompt": "...", "chosen": "good response", "rejected": "bad response"}
    dataset = PreferenceDataset(
        data_path="./data/preferences.jsonl",
        tokenizer=tokenizer,
        max_length=512,
    )

    # Configure DPO
    config = DPOTrainerConfig(
        dpo=DPOConfig(
            beta=0.1,  # KL penalty coefficient
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
