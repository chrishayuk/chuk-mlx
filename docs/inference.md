# Inference Guide

Run text generation with pretrained models from HuggingFace Hub using the unified inference pipeline.

## Quick Start

### Inference Pipeline (Recommended)

The new `InferencePipeline` provides a simplified, one-liner API for loading and running inference:

```python
from chuk_lazarus.inference import InferencePipeline, PipelineConfig, DType
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM

# One-liner model loading
pipeline = InferencePipeline.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    LlamaForCausalLM,
    LlamaConfig,
)

# Simple chat API
result = pipeline.chat("What is the capital of France?")
print(result.text)
print(result.stats.summary)  # "25 tokens in 0.42s (59.5 tok/s)"
```

### With Custom Configuration

```python
from chuk_lazarus.inference import (
    InferencePipeline,
    PipelineConfig,
    GenerationConfig,
    DType,
)
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM

# Configure the pipeline
config = PipelineConfig(
    dtype=DType.BFLOAT16,
    default_system_message="You are a helpful coding assistant.",
    default_max_tokens=200,
    default_temperature=0.7,
)

pipeline = InferencePipeline.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    LlamaForCausalLM,
    LlamaConfig,
    pipeline_config=config,
)

# Generate with custom settings
result = pipeline.chat(
    "Write a Python function to calculate Fibonacci numbers",
    max_new_tokens=300,
    temperature=0.3,
)
print(result.text)
```

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

### Low-Level Python API

For more control, use the models directly:

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

## Inference Pipeline API

### Core Classes

| Class | Description |
|-------|-------------|
| `InferencePipeline` | High-level API for model loading and generation |
| `PipelineConfig` | Pipeline configuration (dtype, defaults) |
| `GenerationConfig` | Generation parameters (max_tokens, temperature, top_p) |
| `GenerationResult` | Generation output with text and stats |
| `ChatHistory` | Multi-turn conversation management |

### Loading Models

```python
# Synchronous loading
pipeline = InferencePipeline.from_pretrained(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_class=LlamaForCausalLM,
    config_class=LlamaConfig,
    pipeline_config=PipelineConfig(dtype=DType.BFLOAT16),
)

# Async loading
pipeline = await InferencePipeline.from_pretrained_async(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_class=LlamaForCausalLM,
    config_class=LlamaConfig,
)
```

### Chat API

```python
# Simple single-turn chat
result = pipeline.chat("What is 2+2?")

# With custom system message
result = pipeline.chat(
    "Write a haiku",
    system_message="You are a poet.",
)

# Multi-turn conversation
from chuk_lazarus.inference import ChatHistory

history = ChatHistory()
history.add_system("You are a helpful assistant.")
history.add_user("What is Python?")
history.add_assistant("Python is a programming language.")
history.add_user("What is it used for?")

result = pipeline.chat_with_history(history)
```

### Raw Generation

```python
# Direct prompt without chat formatting
result = pipeline.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.9,
)

# With full config
from chuk_lazarus.inference import GenerationConfig

config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)
result = pipeline.generate("The quick brown fox", config=config)
```

### Streaming Generation

```python
from chuk_lazarus.inference import generate_stream

# Stream tokens as they're generated
for chunk in generate_stream(model, tokenizer, "Write a story"):
    print(chunk, end="", flush=True)
```

## Simplified Examples

The `examples/inference/` directory contains streamlined examples using the new inference pipeline:

```bash
# Simple inference (any Llama-family model)
uv run python examples/inference/simple_inference.py --prompt "What is the capital of France?"

# Llama family with model presets
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/llama_inference.py --list  # Show all presets

# Gemma 3 with interactive chat
uv run python examples/inference/gemma_inference.py --chat

# Granite (IBM)
uv run python examples/inference/granite_inference.py --model granite-3.1-2b

# Llama 4 Scout (Mamba-Transformer hybrid)
uv run python examples/inference/llama4_inference.py
```

These examples replace the 400+ line model-specific examples with ~100-200 line implementations using the unified API.

## Llama Family Inference

The `examples/inference/llama_inference.py` script provides a unified interface for Llama-architecture models:

```bash
# List available model presets
uv run python examples/inference/llama_inference.py --list

# Run with different models
uv run python examples/inference/llama_inference.py --model tinyllama
uv run python examples/inference/llama_inference.py --model smollm2-360m
uv run python examples/inference/llama_inference.py --model llama3.2-1b

# Custom prompt
uv run python examples/inference/llama_inference.py \
  --model smollm2-360m \
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

## Gemma Inference

Gemma 3 is Google's latest open model family with 5 sizes (270M, 1B, 4B, 12B, 27B) and 128K context. Use bf16 models from mlx-community for direct loading.

### Running Gemma Inference

```bash
# Basic inference (simplified API)
uv run python examples/inference/gemma_inference.py --prompt "What is the capital of France?"

# Gemma 3 270M (smallest, fastest)
uv run python examples/inference/gemma_inference.py --model gemma3-270m

# FunctionGemma 270M (function calling optimized)
uv run python examples/inference/gemma_inference.py --model functiongemma

# Interactive chat mode
uv run python examples/inference/gemma_inference.py --chat

# Use larger model
uv run python examples/inference/gemma_inference.py --model gemma3-4b

# List all available models
uv run python examples/inference/gemma_inference.py --list
```

### Available Gemma Models

| Preset | Model ID | Parameters | Memory | Notes |
|--------|----------|------------|--------|-------|
| `gemma3-270m` | mlx-community/gemma-3-270m-it-bf16 | 270M | ~540MB | Smallest, fastest |
| `functiongemma` | mlx-community/functiongemma-270m-it-bf16 | 270M | ~540MB | Function calling optimized |
| `gemma3-1b` | mlx-community/gemma-3-1b-it-bf16 | 1B | ~2GB | Fast, good for testing |
| `gemma3-4b` | mlx-community/gemma-3-4b-it-bf16 | 4B | ~8GB | Good quality/speed balance |
| `gemma3-12b` | mlx-community/gemma-3-12b-it-bf16 | 12B | ~24GB | High quality |
| `gemma3-27b` | mlx-community/gemma-3-27b-it-bf16 | 27B | ~54GB | Best quality |

**Notes:**
- Use bf16 models (not 4-bit quantized) for direct loading. Quantized models require additional quantization support.
- The 4B+ models are multimodal but this example uses them for text-only inference (vision components are filtered out).

### Python API

```python
import mlx.core as mx
from mlx.utils import tree_unflatten
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import json

from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

# Download model
model_id = "mlx-community/gemma-3-1b-it-bf16"
model_path = Path(snapshot_download(repo_id=model_id, allow_patterns=["*.json", "*.safetensors"]))

# Load config
with open(model_path / "config.json") as f:
    hf_config = json.load(f)

config = GemmaConfig(
    vocab_size=hf_config["vocab_size"],
    hidden_size=hf_config["hidden_size"],
    num_hidden_layers=hf_config["num_hidden_layers"],
    num_attention_heads=hf_config["num_attention_heads"],
    num_key_value_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
    intermediate_size=hf_config["intermediate_size"],
    head_dim=hf_config.get("head_dim", 256),
)

# Create model and load weights
model = GemmaForCausalLM(config)
weights = mx.load(str(model_path / "model.safetensors"))
nested = tree_unflatten(list(weights.items()))
model.update(nested)
mx.eval(model.parameters())

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(str(model_path))

# Generate
prompt = "<bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n"
input_ids = mx.array(tokenizer.encode(prompt, return_tensors="np"))

output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    stop_tokens=[tokenizer.eos_token_id, 106],  # 106 is <end_of_turn>
)

response = tokenizer.decode(output_ids[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
print(response)
```

## Granite Inference

IBM Granite models are available in dense (3.0, 3.1) and hybrid MoE (4.0) variants.

### Running Granite Inference

```bash
# Basic inference
uv run python examples/models/granite/01_granite_inference.py --prompt "What is machine learning?"

# Use specific model
uv run python examples/models/granite/01_granite_inference.py \
  --model ibm-granite/granite-3.1-2b-instruct \
  --prompt "Explain neural networks"
```

### Available Granite Models

| Model ID | Type | Parameters | Notes |
|----------|------|------------|-------|
| `ibm-granite/granite-3.0-8b-instruct` | Dense | 8B | Original Granite 3.0 |
| `ibm-granite/granite-3.1-2b-instruct` | Dense | 2B | Long context (128K) |
| `ibm-granite/granite-3.1-8b-instruct` | Dense | 8B | Long context (128K) |

## Llama 4 Inference

Meta's Llama 4 Scout model uses a hybrid Mamba-Transformer architecture with MoE for efficient long-context processing.

### Running Llama 4 Inference

```bash
# Basic inference
uv run python examples/models/llama4/01_llama4_inference.py --prompt "What is the future of AI?"

# With custom parameters
uv run python examples/models/llama4/01_llama4_inference.py \
  --prompt "Write a story about space exploration" \
  --max-tokens 200 \
  --temperature 0.8
```

### Available Llama 4 Models

| Model ID | Parameters | Architecture | Notes |
|----------|------------|--------------|-------|
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 17B active / 109B total | Hybrid Mamba-Transformer MoE | 16 experts, 10M context |

**Note:** Llama 4 requires HuggingFace authentication. Run `huggingface-cli login` first.

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
