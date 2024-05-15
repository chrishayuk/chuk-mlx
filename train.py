from typing import Generator
import mlx.core as mx
import mlx.nn as nn
import models
import models.llama
import models.llama.llama_model
from models.load_weights import load_model_weights
import models.llama.model
from models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub
from utils.tokenizer_loader import load_tokenizer

# set the model name
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#model_name = "ibm/granite-7b-base"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "ibm-granite/granite-3b-code-instruct"

# not supported
#model_name = "google/gemma-2b"
#model_name = "google/gemma-1.1-2b-it"
#model_name = "google/code-gemma-2b"
#model_name = "google/code-gemma-7b"

# load the model from huggingface
print(f"Loading Model: {model_name}")
model_path = load_from_hub(model_name)

# load config
model_config = ModelConfig.load(model_path)

# load the model weights
weights = load_model_weights(model_path)

# create the model instance
model = models.llama.model.Model(model_config)

# Model Loaded
print(f"Model Loaded: {model_name}")


# loading weights
print("loading weights")
model.load_weights(list(weights.items()))
mx.eval(model.parameters())
print("weights loaded")

# load the model
#model = models.llama.llama_model.LlamaModel.load(model_name)

# prompt it
# Load tokenizer and define vocabulary size
tokenizer = load_tokenizer(model_name)

def generatey(
    prompt: mx.array, model: nn.Module, temp: float = 0
) -> Generator[mx.array, None, None]:
    
    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        # TODO: handle cache in the future
        #logits, cache = model(y[None])
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y

def generate(model, prompt, tokenizer):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        generatey(prompt, model),
        range(500),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return


print("prompting")
prompt = "Weite a fibonacci function in python?"
response = generate(model, prompt, tokenizer)
print(response)
