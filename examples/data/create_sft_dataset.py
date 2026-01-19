"""
SFT Dataset Creation Example

Shows how to create and use an SFT dataset.
"""

import json
from pathlib import Path

from chuk_lazarus.models import load_tokenizer

from chuk_lazarus.data import SFTDataset


def create_sample_data(output_path: str):
    """Create sample training data."""
    samples = [
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
        {
            "prompt": "Explain photosynthesis briefly.",
            "response": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        },
        {"prompt": "What is 15 + 27?", "response": "15 + 27 = 42"},
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} samples at {output_path}")


def main():
    # Create sample data
    data_path = "./data/sample_sft.jsonl"
    create_sample_data(data_path)

    # Load tokenizer
    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create dataset
    dataset = SFTDataset(
        path=data_path,
        tokenizer=tokenizer,
        max_length=256,
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Iterate through batches
    print("\nBatch iteration example:")
    for i, batch in enumerate(dataset.iter_batches(batch_size=2, shuffle=False)):
        print(f"Batch {i + 1}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        if i >= 1:
            break


if __name__ == "__main__":
    main()
