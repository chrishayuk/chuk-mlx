# Virtual Experts System

The Virtual Experts system enables language models to route queries to specialized expert plugins for precise computation and real-time data retrieval.

## Overview

Virtual Experts intercept model inference and route specific queries to external tools:

- **Math Expert** - Exact arithmetic computation using Python
- **Time Expert** - Real-time timezone queries

```
User Query: "What is 256 times 4?"
    ↓
CoT Rewriter (normalizes to action format)
    ↓
{"expert": "math", "operation": "evaluate", "parameters": {"expression": "256 * 4"}}
    ↓
Router (activation-space routing decision)
    ↓
MathExpert.execute() → 1024
```

## Quick Start

```bash
# Math computation
lazarus introspect virtual-expert solve \
    -m mlx-community/SmolLM-135M-fp16 \
    -p "What is 256 times 4?" \
    --use-few-shot-rewriter
# Answer: 1024

# Time query
lazarus introspect virtual-expert solve \
    -m mlx-community/SmolLM-135M-fp16 \
    -p "What time is it in Tokyo?" \
    --use-few-shot-rewriter
# Answer: 2026-01-21 10:01:33 JST
```

## Architecture

### Model Types

The system supports two model architectures:

| Model Type | Wrapper | Routing Method |
|------------|---------|----------------|
| MoE (Mixture of Experts) | `VirtualMoEWrapper` | Intercepts actual router decisions |
| Dense (LLaMA, etc.) | `VirtualDenseWrapper` | Virtual routing in activation space |

### CoT Rewriting

The Chain-of-Thought (CoT) rewriter normalizes natural language queries into structured `VirtualExpertAction` format:

```python
# Input (natural language)
"What is 256 times 4?"

# Output (structured action)
VirtualExpertAction(
    expert="math",
    operation="evaluate",
    parameters={"expression": "256 * 4"},
    confidence=1.0,
    reasoning="Natural language multiplication"
)
```

#### Two Modes

1. **CoT-Trained Models** (default): Model generates action format directly
2. **Few-Shot Rewriter** (`--use-few-shot-rewriter`): Uses in-context learning for non-CoT-trained models

```bash
# For CoT-trained models (no rewriter needed)
lazarus introspect virtual-expert solve -m my-cot-model -p "127 * 89 ="

# For standard models (use few-shot rewriter)
lazarus introspect virtual-expert solve -m mlx-community/SmolLM-135M-fp16 \
    -p "What is 256 times 4?" --use-few-shot-rewriter
```

## Available Experts

### Math Expert

Provides exact arithmetic computation using safe Python evaluation.

**Operations:**
- `evaluate` - Evaluate mathematical expression
- `extract_and_evaluate` - Extract expression from text and evaluate

**Supported:**
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Functions: `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `abs`, `round`, `min`, `max`
- Constants: `pi`, `e`, `inf`

**Examples:**
```bash
# Natural language
lazarus introspect virtual-expert solve -m MODEL \
    -p "What is 15 times 8?" --use-few-shot-rewriter
# Answer: 120

# Symbolic
lazarus introspect virtual-expert solve -m MODEL \
    -p "127 * 89 = " --use-few-shot-rewriter
# Answer: 11303

# Decimals
lazarus introspect virtual-expert solve -m MODEL \
    -p "What is 256.1 times 4.34?" --use-few-shot-rewriter
# Answer: 1111.474
```

### Time Expert

Provides real-time timezone queries using `pytz`.

**Operations:**
- `get_time` - Get current time in a timezone
- `convert_time` - Convert time between timezones
- `get_timezone_info` - Get timezone information

**Examples:**
```bash
# Current time in timezone
lazarus introspect virtual-expert solve -m MODEL \
    -p "What time is it in Tokyo?" --use-few-shot-rewriter
# Answer: 2026-01-21 10:01:33 JST

# Another timezone
lazarus introspect virtual-expert solve -m MODEL \
    -p "Current time in London" --use-few-shot-rewriter
# Answer: 2026-01-21 01:01:33 GMT

# UTC (default)
lazarus introspect virtual-expert solve -m MODEL \
    -p "What time is it?" --use-few-shot-rewriter
# Answer: 2026-01-21 01:01:33 UTC
```

## CLI Reference

```bash
lazarus introspect virtual-expert [ACTION] [OPTIONS]
```

### Actions

| Action | Description |
|--------|-------------|
| `solve` | Solve a single problem (default) |
| `benchmark` | Run benchmark on multiple problems |
| `compare` | Compare model-only vs virtual expert |
| `analyze` | Analyze expert routing patterns |
| `interactive` | Interactive REPL mode |

### Options

| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Model name or HuggingFace ID (required) |
| `-p, --prompt PROMPT` | Prompt to solve |
| `--use-few-shot-rewriter` | Use FewShotCoTRewriter for non-CoT-trained models |
| `-v, --verbose` | Show detailed routing trace |
| `-o, --output FILE` | Save results to JSON file |

### Examples

```bash
# Basic solve
lazarus introspect virtual-expert solve \
    -m mlx-community/SmolLM-135M-fp16 \
    -p "What is 256 times 4?" \
    --use-few-shot-rewriter

# Verbose mode (show routing decisions)
lazarus introspect virtual-expert solve \
    -m mlx-community/SmolLM-135M-fp16 \
    -p "127 * 89 = " \
    --use-few-shot-rewriter \
    --verbose

# Benchmark
lazarus introspect virtual-expert benchmark \
    -m mlx-community/SmolLM-135M-fp16 \
    --use-few-shot-rewriter

# Compare model vs expert
lazarus introspect virtual-expert compare \
    -m mlx-community/SmolLM-135M-fp16 \
    -p "127 * 89 = " \
    --use-few-shot-rewriter
```

## Python API

### Basic Usage

```python
from chuk_lazarus.inference.virtual_experts import (
    VirtualDenseWrapper,
    MathExpertPlugin,
)
from chuk_lazarus.inference.virtual_experts.cot_rewriter import FewShotCoTRewriter

# Load model
from chuk_lazarus.models_v2 import load_model
load_result = load_model("mlx-community/SmolLM-135M-fp16")
model, tokenizer = load_result.model, load_result.tokenizer

# Create wrapper with CoT rewriter
rewriter = FewShotCoTRewriter(model, tokenizer, max_examples_per_expert=3)
wrapper = VirtualDenseWrapper(model, tokenizer, "my-model", cot_rewriter=rewriter)

# Solve
result = wrapper.solve("What is 256 times 4?")
print(result.answer)  # "1024"
print(result.plugin_name)  # "math"
```

### Direct Expert Usage

```python
from chuk_virtual_expert import VirtualExpertAction
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpert

expert = MathExpert()

# Execute action
action = VirtualExpertAction(
    expert="math",
    operation="evaluate",
    parameters={"expression": "256 * 4"},
    confidence=1.0,
    reasoning="multiplication"
)

result = expert.execute(action)
print(result.data)  # {"result": 1024, "expression": "256 * 4", "formatted": "1024"}
```

### Custom Registry

```python
from chuk_lazarus.inference.virtual_experts.registry import VirtualExpertRegistry
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpert
from chuk_virtual_expert_time import TimeExpert

# Create custom registry
registry = VirtualExpertRegistry()
registry.register(MathExpert())
registry.register(TimeExpert())

# Use with wrapper
wrapper = VirtualDenseWrapper(model, tokenizer, "my-model", registry=registry)
```

## Routing Behavior

The routing system determines which expert handles a query:

| Query | Expert | Result |
|-------|--------|--------|
| `"What is 256 times 4?"` | math | 1024 |
| `"127 * 89 = "` | math | 11303 |
| `"What time is it in Tokyo?"` | time | Current JST time |
| `"Current time in London"` | time | Current GMT time |
| `"What is the capital of France?"` | none | Model fallback |
| `"Tell me a joke"` | none | Model fallback |

When `expert="none"`, the query falls through to standard model generation.

## Adding Custom Experts

Create a new expert by extending `VirtualExpert`:

```python
from typing import Any, ClassVar
from chuk_virtual_expert import VirtualExpert, VirtualExpertAction

class WeatherExpert(VirtualExpert):
    name: ClassVar[str] = "weather"
    description: ClassVar[str] = "Get weather information"

    # CoT examples file (relative to module)
    cot_examples_file: ClassVar[str] = "weather_cot_examples.json"

    def can_handle(self, prompt: str) -> bool:
        keywords = ["weather", "temperature", "forecast", "rain"]
        return any(kw in prompt.lower() for kw in keywords)

    def get_operations(self) -> list[str]:
        return ["get_weather", "get_forecast"]

    def execute_operation(
        self,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if operation == "get_weather":
            location = parameters.get("location", "New York")
            # Fetch weather data...
            return {"location": location, "temperature": 72, "conditions": "sunny"}
        raise ValueError(f"Unknown operation: {operation}")
```

Create `weather_cot_examples.json`:

```json
{
  "expert_name": "weather",
  "examples": [
    {
      "query": "What's the weather in Tokyo?",
      "action": {
        "expert": "weather",
        "operation": "get_weather",
        "parameters": {"location": "Tokyo"},
        "confidence": 1.0,
        "reasoning": "Weather query for Tokyo"
      }
    }
  ]
}
```

Register the expert:

```python
from chuk_lazarus.inference.virtual_experts.registry import get_default_registry

registry = get_default_registry()
registry.register(WeatherExpert())
```

## Dependencies

The virtual experts system requires:

- `chuk-virtual-expert>=2.0.0` - Base expert framework
- `chuk-virtual-expert-time>=1.0.0` - Time expert (optional)
- `mlx>=0.5.0` - Apple MLX for model inference

Install time expert:
```bash
pip install chuk-virtual-expert-time
```

## Troubleshooting

### Empty answers for math queries

Ensure `--use-few-shot-rewriter` is set when using non-CoT-trained models:
```bash
lazarus introspect virtual-expert solve -m MODEL -p "256 * 4" --use-few-shot-rewriter
```

### Incorrect routing with extra instructions

Small models may misroute queries that have extra instructions appended:

```bash
# Works correctly
"Current time in London" → time expert ✓

# May misroute with small models
"Current time in London, answer in french" → none (incorrect)
```

**Solutions:**
1. Use larger models (7B+) for better generalization
2. Keep queries simple without extra formatting instructions
3. Use a CoT-trained model that doesn't need the few-shot rewriter

### Model size recommendations

| Model Size | Routing Accuracy | Notes |
|------------|-----------------|-------|
| < 1B | Basic queries only | May misroute complex queries |
| 1B - 7B | Good | Handles most queries |
| 7B+ | Excellent | Robust with extra instructions |

### Model loading errors

Quantized models (4-bit, 8-bit) may have compatibility issues. Try fp16 models:
```bash
lazarus introspect virtual-expert solve -m mlx-community/SmolLM-135M-fp16 ...
```
