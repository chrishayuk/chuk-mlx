from typing import Generator
import mlx.core as mx
import mlx.nn as nn

def generate_sequence(prompt: mx.array, model: nn.Module, temperature: float = 0) -> Generator[mx.array, None, None]:
    def sample(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1) if temperature == 0 else mx.random.categorical(logits * (1 / temperature))

    y = prompt
    cache = None
    step = 0
    while True:
        logits, cache = model(y[None], cache=cache)
        if logits is None or logits.shape[1] == 0:
            break

        logits = logits[:, -1, :]
        y = sample(logits)

        yield y
        step += 1

def generate_response(model, prompt, tokenizer, max_length: int = 500):
    print(prompt, end="", flush=True)
    prompt_encoded = mx.array(tokenizer.encode(prompt))
    print(f"Encoded prompt: {prompt_encoded}")

    tokens = []
    skip = 0

    for token, _ in zip(generate_sequence(prompt_encoded, model), range(max_length)):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1

    decoded_response = tokenizer.decode(tokens)[skip:]
    print(decoded_response, flush=True)
    print("=" * 10)

    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return []

    return tokens
