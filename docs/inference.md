# Inference Guide

Run text generation with pretrained models from HuggingFace Hub using the models_v2 architecture.

## Quick Start

### CLI Inference

```bash
# Basic inference with TinyLlama
chuk-lazarus infer --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --prompt "What is the capital of France?"

# With generation parameters
chuk-lazarus infer \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --prompt "Explain quantum computing in one sentence" \
  --max-tokens 100 \
  --temperature 0.7
```

### Python API

```python
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer
import mlx.core as mx

# Load model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = LlamaConfig.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, config)

# Generate text
prompt = "What is machine learning?"
input_ids = tokenizer.encode(prompt, return_tensors="np")
input_ids = mx.array(input_ids)

output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    stop_tokens=[tokenizer.eos_token_id],
)
mx.eval(output_ids)

response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(response)
```

## Llama Family Inference Example

The `examples/models/llama/03_llama_family_inference.py` script provides a unified interface for running inference with various Llama-architecture models:

```bash
# List available model presets
uv run python examples/models/llama/03_llama_family_inference.py --list-models

# Run with different models
uv run python examples/models/llama/03_llama_family_inference.py --model tinyllama
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-135m
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-360m
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-1.7b

# Custom prompt
uv run python examples/models/llama/03_llama_family_inference.py \
  --model tinyllama \
  --prompt "Explain relativity in simple terms" \
  --max-tokens 150 \
  --temperature 0.8
```

### Available Model Presets

| Preset | Model ID | Parameters | Notes |
|--------|----------|------------|-------|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Fast, good for testing |
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct | 135M | Tiny, runs anywhere |
| `smollm2-360m` | HuggingFaceTB/SmolLM2-360M-Instruct | 360M | Good quality/speed balance |
| `smollm2-1.7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B | High quality, still fast |
| `llama-7b` | meta-llama/Llama-2-7b-hf | 7B | Original Llama base model |
| `llama2-7b` | meta-llama/Llama-2-7b-chat-hf | 7B | Llama 2 Chat |
| `llama2-13b` | meta-llama/Llama-2-13b-chat-hf | 13B | Larger Llama 2 |
| `llama3.2-1b` | meta-llama/Llama-3.2-1B-Instruct | 1B | Smallest Llama 3 |
| `llama3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 3B | Small but capable |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | Standard size |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B | Sliding window attention |

**Note:** Meta Llama models require HuggingFace authentication. Run `huggingface-cli login` first.

## Generation Parameters

The `generate()` method supports several parameters to control text generation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens` | int | 100 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy, higher = more random) |
| `top_p` | float | 0.9 | Nucleus sampling probability threshold |
| `top_k` | int | None | Top-k sampling (limits to k most likely tokens) |
| `repetition_penalty` | float | 1.0 | Penalty for repeating tokens (>1 reduces repetition) |
| `stop_tokens` | list | None | Token IDs that stop generation |

### Example: Temperature Effects

```python
# Greedy decoding (deterministic)
output = model.generate(input_ids, temperature=0.0)

# Low temperature (focused, coherent)
output = model.generate(input_ids, temperature=0.3)

# Medium temperature (balanced)
output = model.generate(input_ids, temperature=0.7)

# High temperature (creative, diverse)
output = model.generate(input_ids, temperature=1.2)
```

## Weight Loading

Models are loaded from HuggingFace Hub and automatically converted to the models_v2 format:

1. **Download**: Weights are downloaded via `huggingface_hub.snapshot_download()`
2. **Load**: MLX's native `mx.load()` handles safetensors files
3. **Convert**: Weight names are mapped from HF format to models_v2 format
4. **Update**: Weights are loaded into the model via `model.update()`

### Weight Name Conversion

| HuggingFace Name | models_v2 Name |
|-----------------|----------------|
| `model.embed_tokens.weight` | `model.embed_tokens.weight.weight` |
| `model.norm.weight` | `model.norm.weight` |
| `model.layers.{i}.*` | `model.layers.{i}.*` |
| `lm_head.weight` | `lm_head.lm_head.weight` (if not tied) |

### Dtype Considerations

Use `bfloat16` for numerical stability with most models:

```bash
# Recommended for most models
uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-360m --dtype bfloat16

# float16 may cause numerical issues in some models
# float32 uses more memory but has highest precision
```

## Chat Templates

The inference examples use the tokenizer's built-in chat template when available:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

## Performance Tips

1. **Use bfloat16**: Default dtype for numerical stability
2. **Enable KV-cache**: Automatically enabled for autoregressive generation
3. **Batch prompts**: Process multiple prompts in a single forward pass when possible
4. **Use smaller models**: SmolLM2-135M for fast iteration, larger models for quality

## Troubleshooting

### Garbage Output

If the model produces nonsensical output:
- Ensure weights loaded correctly (check tensor count)
- Verify dtype is `bfloat16` (not `float16`)
- Check that `tie_word_embeddings` matches config

### Slow Generation

- First token is slow (model compilation)
- Subsequent tokens should be fast (~50-150 tok/s)
- Larger models need more memory bandwidth

### Missing Weights

If weight loading fails:
- Check model files exist in cache
- Verify safetensors format
- Some models may need HF authentication

## FunctionGemma (Function Calling)

FunctionGemma is a 270M parameter model from Google, designed specifically for on-device function calling. It's excellent for:
- Tool use / API calling
- MCP (Model Context Protocol) integration
- Lightweight RAG pipelines
- On-device agents

### Running FunctionGemma

```bash
# Run the FunctionGemma inference example
uv run python examples/models/gemma/01_functiongemma_inference.py
```

### How FunctionGemma Works

FunctionGemma uses special tokens for structured function calling:
- `<start_function_declaration>` / `<end_function_declaration>` - Define available tools
- `<start_function_call>` / `<end_function_call>` - Model requests tool use
- `<start_function_response>` / `<end_function_response>` - Tool results
- `<escape>` - Wraps string values in structured data

### Example with Tools

```python
from huggingface_hub import hf_hub_download
from jinja2 import Template
from mlx_lm import generate, load

# Load bf16 model (better accuracy than quantized for function calling)
model_name = "mlx-community/functiongemma-270m-it-bf16"
model, tokenizer = load(model_name)

# Load Jinja2 chat template
template_path = hf_hub_download(model_name, "chat_template.jinja")
with open(template_path) as f:
    chat_template = Template(f.read())

# Define tools in OpenAI-compatible format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    },
]

# Format messages with tools
messages = [
    {"role": "developer", "content": "You can do function calling with these functions"},
    {"role": "user", "content": "What's the weather in Tokyo?"},
]

prompt = chat_template.render(
    messages=messages,
    tools=tools,
    add_generation_prompt=True,
    bos_token="<bos>",
    eos_token="<eos>",
)

# Generate
response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
# Output: call:get_weather{location:<escape>Tokyo<escape>}
```

### Parsing Function Calls

```python
import re

def parse_function_call(response: str) -> dict | None:
    """Parse function call from FunctionGemma output."""
    pattern = r"call:(\w+)\{(.+?)\}"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (handle <escape> tokens)
        args = {}
        arg_pattern = r"(\w+):<escape>([^<]+)<escape>"
        for arg_match in re.finditer(arg_pattern, args_str):
            args[arg_match.group(1)] = arg_match.group(2)

        return {"name": func_name, "arguments": args}
    return None
```

### Model Selection

| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| `functiongemma-270m-it-bf16` | ~540MB | Best | Production function calling |
| `functiongemma-270m-it-4bit` | ~135MB | Lower | Memory-constrained devices |

**Note:** bf16 models provide significantly better function calling accuracy than quantized versions. Use 4-bit only when memory is severely constrained.

### Using chuk-lazarus Native Implementation

You can also use our native Gemma implementation directly:

```python
import mlx.core as mx
from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

# Create config for FunctionGemma 270M
config = GemmaConfig.functiongemma_270m()

# Create model
model = GemmaForCausalLM(config)

# Load weights from mlx-community
# weights = mx.load("path/to/model.safetensors")
# model.update(weights)

# Forward pass
test_input = mx.array([[1, 2, 3, 4, 5]])
output = model(test_input)
print(f"Output shape: {output.logits.shape}")
```

## See Also

- [Models Guide](models.md) - Architecture details
- [Training Guide](training.md) - Fine-tuning models
- [Examples](../examples/models/llama/) - Working inference examples
- [Examples](../examples/models/gemma/) - FunctionGemma examples
