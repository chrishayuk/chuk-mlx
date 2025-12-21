"""
Math Data Generation Example

Shows how to generate synthetic math problems for training.
"""

from chuk_lazarus.data.generators import (
    MathProblemGenerator,
    ProblemType,
    generate_lazarus_dataset,
)


def main():
    # Create generator
    gen = MathProblemGenerator(seed=42)

    # Generate a few samples
    print("Generating sample problems...")
    samples = gen.generate_batch(
        num_problems=5,
        difficulty_range=(1, 3),
        problem_types=[ProblemType.ARITHMETIC, ProblemType.WORD_PROBLEM],
    )

    # Display samples
    for i, sample in enumerate(samples):
        print(f"\n--- Problem {i+1} ---")
        print(f"Type: {sample.problem.problem_type.value}")
        print(f"Problem: {sample.problem.problem_text}")
        print(f"Answer: {sample.problem.answer}")
        print(f"Correct response:\n{sample.correct_response}")

    # Generate full dataset
    print("\n\nGenerating full dataset...")
    paths = generate_lazarus_dataset(
        output_dir="./data/math",
        sft_samples=100,
        dpo_samples=50,
        seed=42,
    )

    print("Generated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
