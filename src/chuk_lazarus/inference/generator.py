from collections.abc import Generator

import mlx.core as mx
import mlx.nn as nn


def generate_sequence(
    prompt: mx.array, model: nn.Module, temperature: float = 0
) -> Generator[mx.array, None, None]:
    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temperature == 0
            else mx.random.categorical(logits * (1 / temperature))
        )

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
    # encode the prompt
    prompt_encoded = mx.array(tokenizer.encode(prompt))

    # clear tokens
    tokens = []
    skip = 0

    # loop through the sequence
    for token, _ in zip(generate_sequence(prompt_encoded, model), range(max_length)):
        # check for end of sequence token
        if token == tokenizer.eos_token_id:
            break

        # append the token
        tokens.append(token.item())

        # decode the token
        s = tokenizer.decode(tokens)

        # check the length
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1

    # get the decoded response
    decoded_response = tokenizer.decode(tokens)[skip:]

    # print it out, in future we should yield it, so that it's in the UI's gift
    print(decoded_response, flush=True)

    # check if we got tokens
    if len(tokens) == 0:
        # no tokens
        print("No tokens generated for this prompt")
        return []

    # return tokens
    return tokens
