"""
Basic Inference Example

Shows how to load a model and generate text.
"""

from chuk_lazarus.models import generate_response, load_model


def main():
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Generate text
    prompt = "Once upon a time"
    print(f"\nPrompt: {prompt}")
    print("Generating...")

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )

    print(f"Response: {response}")


if __name__ == "__main__":
    main()
