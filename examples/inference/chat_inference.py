"""
Chat Inference Example

Shows how to use chat templates for conversational inference.
"""

from chuk_lazarus.models import load_model, generate_response


def format_chat(messages: list, tokenizer) -> str:
    """Format messages using the tokenizer's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Fallback: simple format
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|{role}|>\n{content}\n"
    formatted += "<|assistant|>\n"
    return formatted


def main():
    print("Loading model...")
    model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Chat messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    prompt = format_chat(messages, tokenizer)
    print(f"Formatted prompt:\n{prompt}\n")

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
    )

    print(f"Response: {response}")


if __name__ == "__main__":
    main()
