"""
Shared model loading and generation utilities for introspection examples.

Uses the unified HFLoader for consistent weight loading across all model families.
Provides generic utilities that work with any model family.

Key utilities:
- load_model(): Load any supported model family
- load_chat_template(): Load Jinja2 chat template for tool-calling
- generate(): Generate text from a prompt
- generate_with_layer_ablation(): Generate with MLP/attention ablation
- generate_with_head_ablation(): Generate with specific attention head ablation
- get_hidden_states(): Capture hidden states at each layer
- compare_activations(): Compare activations between two models
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from huggingface_hub import hf_hub_download
from jinja2 import Template

from chuk_lazarus.inference import DType, HFLoader
from chuk_lazarus.introspection import CaptureConfig, LayerSelection, ModelHooks, PositionSelection
from chuk_lazarus.models_v2.families import detect_model_family, get_family_info

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def load_model(model_id: str) -> tuple[Any, PreTrainedTokenizer, Any, Path]:
    """Load any supported model using the unified loader.

    Auto-detects the model family and uses appropriate config/model classes.
    Works with Gemma, Llama, Qwen3, Jamba, StarCoder2, Granite, etc.

    Args:
        model_id: HuggingFace model ID or local path

    Returns:
        Tuple of (model, tokenizer, config, model_path)
    """
    print(f"\nLoading {model_id}...")
    result = HFLoader.download(model_id)

    # Load raw config for family detection
    config_path = result.model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    family_type = detect_model_family(hf_config)
    if family_type is None:
        raise ValueError(f"Unable to detect model family for {model_id}")

    family_info = get_family_info(family_type)
    if family_info is None:
        raise ValueError(f"No family info registered for {family_type}")

    print(f"  Family: {family_type.value}")

    # Load config using family-specific class
    config_class = family_info.config_class
    if hasattr(config_class, "from_hf_config"):
        config = config_class.from_hf_config(hf_config)
    else:
        config = config_class(**hf_config)

    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")

    # Create model
    model_class = family_info.model_class
    model = model_class(config)

    # Apply weights using unified loader
    HFLoader.apply_weights_to_model(model, result.model_path, config, dtype=DType.BFLOAT16)

    # Load tokenizer
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"  Tokenizer vocab: {len(tokenizer)}")

    return model, tokenizer, config, result.model_path


def generate(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Generate text from a prompt.

    Works with any model that has a .generate() method.

    Args:
        model: The model instance
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)

    Returns:
        Generated text (excluding input prompt)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    input_length = input_ids.shape[1]

    # Get stop tokens
    eos = tokenizer.eos_token_id
    stop_tokens = eos if isinstance(eos, list) else [eos] if eos else []

    # Generate
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_tokens=stop_tokens,
    )
    mx.eval(output)

    # Decode new tokens only
    new_tokens = output[0, input_length:].tolist()
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def generate_with_layer_ablation(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    ablate_layer: int | None = None,
    ablation_type: str = "mlp",
    max_new_tokens: int = 60,
) -> str:
    """Generate with optional layer ablation.

    Works with any model that has model.model.layers structure.

    Args:
        model: The model instance
        tokenizer: The tokenizer
        prompt: Input prompt string
        ablate_layer: Layer index to ablate (None = no ablation)
        ablation_type: "mlp" or "attn" - which component to ablate
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    original_weight = None
    layer = None

    if ablate_layer is not None:
        layer = model.model.layers[ablate_layer]
        if ablation_type == "mlp":
            original_weight = mx.array(layer.mlp.down_proj.weight)
            layer.mlp.down_proj.weight = mx.zeros_like(original_weight)
        elif ablation_type == "attn":
            original_weight = mx.array(layer.self_attn.o_proj.weight)
            layer.self_attn.o_proj.weight = mx.zeros_like(original_weight)
        mx.eval(layer.parameters())

    try:
        result = generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0)
    finally:
        # Restore weights
        if original_weight is not None and layer is not None:
            if ablation_type == "mlp":
                layer.mlp.down_proj.weight = original_weight
            elif ablation_type == "attn":
                layer.self_attn.o_proj.weight = original_weight
            mx.eval(layer.parameters())

    return result


def format_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    system_message: str | None = None,
) -> str:
    """Format a chat prompt using the tokenizer's chat template.

    Falls back to a generic format if no template is available.

    Args:
        tokenizer: The tokenizer (may have chat_template)
        user_message: The user's message
        system_message: Optional system message

    Returns:
        Formatted prompt string
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Generic fallback
        prompt = ""
        if system_message:
            prompt += f"System: {system_message}\n\n"
        prompt += f"User: {user_message}\n\nAssistant:"
        return prompt


# =============================================================================
# Chat Template Loading
# =============================================================================


def load_chat_template(model_id: str) -> Template | None:
    """Load Jinja2 chat template if available.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Jinja2 Template or None if not available
    """
    try:
        template_path = hf_hub_download(model_id, "chat_template.jinja")
        with open(template_path) as f:
            return Template(f.read())
    except Exception:
        return None


def format_tool_prompt(
    template: Template,
    user_message: str,
    tools: list[dict],
    system_message: str = "You are a helpful assistant. Use tools when appropriate.",
) -> str:
    """Format a prompt with tool definitions.

    Args:
        template: Jinja2 chat template
        user_message: User's message
        tools: List of tool definitions
        system_message: System/developer message

    Returns:
        Formatted prompt string
    """
    messages = [
        {"role": "developer", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        bos_token="<bos>",
        eos_token="<eos>",
    )


# =============================================================================
# Activation Capture & Comparison
# =============================================================================


@dataclass
class ActivationDivergence:
    """Activation divergence between two models at a layer."""

    layer: int
    cosine_similarity: float
    l2_distance: float
    relative_l2: float  # L2 normalized by activation magnitude


def get_hidden_states(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
) -> dict[int, mx.array]:
    """Get hidden states at each layer for the last token position.

    Args:
        model: The model instance
        tokenizer: The tokenizer
        prompt: Input prompt

    Returns:
        Dict mapping layer index to hidden state array
    """
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    num_layers = model.config.num_hidden_layers

    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=LayerSelection.ALL,
            capture_hidden_states=True,
            positions=PositionSelection.LAST,
        )
    )

    hooks.forward(input_ids)

    hidden_states = {}
    for layer_idx in range(num_layers):
        h = hooks.state.hidden_states.get(layer_idx)
        if h is not None:
            # Shape is [batch, 1, hidden] due to LAST position selection
            hidden_states[layer_idx] = h[0, 0, :]  # [hidden]

    return hidden_states


def compare_activations(
    model1: Any,
    model2: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
) -> list[ActivationDivergence]:
    """Compare activations between two models.

    Args:
        model1: First model (e.g., base model)
        model2: Second model (e.g., fine-tuned model)
        tokenizer: Shared tokenizer
        prompt: Input prompt

    Returns:
        List of ActivationDivergence per layer
    """
    hidden1 = get_hidden_states(model1, tokenizer, prompt)
    hidden2 = get_hidden_states(model2, tokenizer, prompt)

    num_layers = model1.config.num_hidden_layers
    divergences = []

    for layer_idx in range(num_layers):
        h1 = hidden1.get(layer_idx)
        h2 = hidden2.get(layer_idx)

        if h1 is None or h2 is None:
            continue

        # Cosine similarity
        dot = float(mx.sum(h1 * h2))
        norm1 = float(mx.sqrt(mx.sum(h1 * h1)))
        norm2 = float(mx.sqrt(mx.sum(h2 * h2)))
        cos_sim = dot / (norm1 * norm2 + 1e-8)

        # L2 distance
        diff = h2 - h1
        l2_dist = float(mx.sqrt(mx.sum(diff * diff)))

        # Relative L2 (normalized by average magnitude)
        avg_norm = (norm1 + norm2) / 2
        rel_l2 = l2_dist / (avg_norm + 1e-8)

        divergences.append(
            ActivationDivergence(
                layer=layer_idx,
                cosine_similarity=cos_sim,
                l2_distance=l2_dist,
                relative_l2=rel_l2,
            )
        )

    return divergences


# =============================================================================
# Advanced Ablation Utilities
# =============================================================================


@contextmanager
def ablate_mlp(model: Any, layer_idx: int):
    """Context manager for temporary MLP ablation.

    Usage:
        with ablate_mlp(model, layer_idx=11):
            output = model.generate(...)
    """
    layer = model.model.layers[layer_idx]
    original_weight = mx.array(layer.mlp.down_proj.weight)
    layer.mlp.down_proj.weight = mx.zeros_like(original_weight)
    mx.eval(layer.mlp.down_proj.weight)

    try:
        yield
    finally:
        layer.mlp.down_proj.weight = original_weight
        mx.eval(layer.mlp.down_proj.weight)


@contextmanager
def ablate_attention(model: Any, layer_idx: int):
    """Context manager for temporary attention ablation.

    Usage:
        with ablate_attention(model, layer_idx=11):
            output = model.generate(...)
    """
    layer = model.model.layers[layer_idx]
    original_weight = mx.array(layer.self_attn.o_proj.weight)
    layer.self_attn.o_proj.weight = mx.zeros_like(original_weight)
    mx.eval(layer.self_attn.o_proj.weight)

    try:
        yield
    finally:
        layer.self_attn.o_proj.weight = original_weight
        mx.eval(layer.self_attn.o_proj.weight)


def create_head_mask(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    heads_to_zero: list[int],
) -> mx.array:
    """Create a mask that zeros out specific heads in o_proj.

    Args:
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        head_dim: Dimension per head
        heads_to_zero: List of head indices to zero out

    Returns:
        Mask array of shape [hidden_size, num_heads * head_dim]
    """
    total_dim = num_heads * head_dim
    mask = mx.ones((hidden_size, total_dim))

    for head_idx in heads_to_zero:
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        # Build mask by concatenating parts
        parts = []
        if start_idx > 0:
            parts.append(mx.ones((hidden_size, start_idx)))
        parts.append(mx.zeros((hidden_size, head_dim)))
        if end_idx < total_dim:
            parts.append(mx.ones((hidden_size, total_dim - end_idx)))

        head_mask = mx.concatenate(parts, axis=1)
        mask = mask * head_mask

    return mask


def generate_with_head_ablation(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    ablate_heads: list[tuple[int, int]],
    max_new_tokens: int = 50,
) -> str:
    """Generate text with specific attention heads ablated.

    Args:
        model: The model instance
        tokenizer: The tokenizer
        prompt: Input prompt
        ablate_heads: List of (layer_idx, head_idx) tuples to ablate
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    config = model.config
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size

    # Group heads by layer
    heads_by_layer: dict[int, list[int]] = {}
    for layer_idx, head_idx in ablate_heads:
        if layer_idx not in heads_by_layer:
            heads_by_layer[layer_idx] = []
        heads_by_layer[layer_idx].append(head_idx)

    # Store original weights and apply ablation
    original_weights = {}
    for layer_idx, head_indices in heads_by_layer.items():
        layer = model.model.layers[layer_idx]
        original_weights[layer_idx] = mx.array(layer.self_attn.o_proj.weight)

        mask = create_head_mask(hidden_size, num_heads, head_dim, head_indices)
        layer.self_attn.o_proj.weight = original_weights[layer_idx] * mask
        mx.eval(layer.self_attn.o_proj.weight)

    try:
        result = generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0)
    finally:
        # Restore original weights
        for layer_idx, orig_weight in original_weights.items():
            layer = model.model.layers[layer_idx]
            layer.self_attn.o_proj.weight = orig_weight
            mx.eval(layer.self_attn.o_proj.weight)

    return result


# =============================================================================
# Tool-calling Detection
# =============================================================================


def has_tool_call(text: str) -> bool:
    """Check if output contains tool-calling markers.

    Works across different model families' tool-call formats.
    """
    markers = [
        # Gemma/FunctionGemma
        "<start_function_call>",
        "<function_call>",
        # Generic JSON function calls
        '{"name":',
        '"function_call"',
        # Common patterns
        "get_weather(",
        "```tool",
        "<tool_call>",
    ]
    return any(m.lower() in text.lower() for m in markers)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


def load_gemma_model(model_id: str):
    """Load a Gemma model. Alias for load_model()."""
    return load_model(model_id)


def load_qwen3_model(model_id: str):
    """Load a Qwen3 model. Alias for load_model()."""
    return load_model(model_id)


def load_llama_model(model_id: str):
    """Load a Llama model. Alias for load_model()."""
    return load_model(model_id)
