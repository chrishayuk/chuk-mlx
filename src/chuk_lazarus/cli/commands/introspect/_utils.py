"""Shared utilities for introspect CLI commands.

This module consolidates common patterns used across CLI commands:
- Argument parsing with type conversion
- File loading with @file prefix support
- Layer depth ratio determination
- Validation helpers
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, TypeVar

from .._constants import Delimiters, LayerDepthRatio

T = TypeVar("T")


# =============================================================================
# Argument Parsing Utilities
# =============================================================================


def parse_value_list(
    values_arg: str,
    delimiter: str = Delimiters.PROMPT_SEPARATOR,
    value_type: type[T] = str,
) -> list[T]:
    """Parse values from argument string or file.

    Supports @file syntax for loading from files.

    Args:
        values_arg: Either a delimited string or @filepath
        delimiter: Delimiter for string parsing
        value_type: Type to convert values to (str, int, float)

    Returns:
        List of parsed values

    Examples:
        >>> parse_value_list("1|2|3", value_type=int)
        [1, 2, 3]
        >>> parse_value_list("@prompts.txt")  # file contents, one per line
        ['prompt1', 'prompt2', ...]
    """
    if values_arg.startswith(Delimiters.FILE_PREFIX):
        with open(values_arg[1:]) as f:
            return [value_type(line.strip()) for line in f if line.strip()]
    return [value_type(v.strip()) for v in values_arg.split(delimiter)]


def get_layer_depth_ratio(
    layer: int | None,
    default_depth: LayerDepthRatio = LayerDepthRatio.MIDDLE,
) -> float | None:
    """Get layer depth ratio if no explicit layer is specified.

    Args:
        layer: Explicit layer number (if any)
        default_depth: Default depth ratio when layer is None

    Returns:
        Depth ratio value or None if explicit layer provided
    """
    return default_depth.value if layer is None else None


def extract_arg(
    args: Namespace,
    name: str,
    default: T | None = None,
) -> T | None:
    """Safely extract an argument with default.

    Args:
        args: Parsed arguments namespace
        name: Argument name
        default: Default value if not present

    Returns:
        Argument value or default
    """
    return getattr(args, name, default)


def extract_args(
    args: Namespace,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Extract multiple args with defaults from a spec.

    Args:
        args: Parsed arguments namespace
        spec: Dict mapping arg names to default values

    Returns:
        Dict of extracted values

    Example:
        >>> extract_args(args, {'top_k': 10, 'temperature': 0.0})
        {'top_k': 5, 'temperature': 0.0}  # if args.top_k was 5
    """
    return {key: getattr(args, key, default) for key, default in spec.items()}


def load_json_or_default(
    file_arg: str | None,
    default_loader: Callable[[], T],
) -> tuple[T, bool]:
    """Load data from JSON file or use framework default.

    Supports @file syntax for loading from files.

    Args:
        file_arg: File path (with @ prefix) or None
        default_loader: Callable that returns default data

    Returns:
        Tuple of (data, is_custom) where is_custom=True if loaded from file
    """
    if file_arg and file_arg.startswith(Delimiters.FILE_PREFIX):
        with open(file_arg[1:]) as f:
            return json.load(f), True
    return default_loader(), False


def load_json_file(file_path: str) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        file_path: Path to JSON file (may have @ prefix)

    Returns:
        Parsed JSON data
    """
    path = file_path[1:] if file_path.startswith(Delimiters.FILE_PREFIX) else file_path
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Validation Utilities
# =============================================================================


def require_arg(
    args: Namespace,
    name: str,
    message: str | None = None,
) -> Any:
    """Require an argument to be present.

    Args:
        args: Parsed arguments namespace
        name: Argument name
        message: Optional custom error message

    Returns:
        Argument value

    Raises:
        ValueError: If argument is not present
    """
    value = getattr(args, name, None)
    if value is None:
        raise ValueError(message or f"--{name.replace('_', '-')} is required")
    return value


def require_one_of(
    args: Namespace,
    names: list[str],
    message: str | None = None,
) -> tuple[str, Any]:
    """Require at least one of several arguments.

    Args:
        args: Parsed arguments namespace
        names: List of argument names to check
        message: Optional custom error message

    Returns:
        Tuple of (name, value) for first present argument

    Raises:
        ValueError: If none of the arguments are present
    """
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return name, value
    formatted_names = ", ".join(f"--{n.replace('_', '-')}" for n in names)
    raise ValueError(message or f"One of {formatted_names} is required")


# =============================================================================
# Legacy Print Utilities (preserved for compatibility)
# =============================================================================


def print_analysis_result(result, tokenizer, args):
    """Print analysis result in standard format."""
    # Print tokenization
    if len(result.tokens) <= 10:
        print(f"\nTokens ({len(result.tokens)}): {result.tokens}")
    else:
        print(f"\nTokens ({len(result.tokens)}): {result.tokens[:5]}...{result.tokens[-3:]}")
    print(f"Captured layers: {result.captured_layers}")

    # Print final prediction
    print("\n=== Final Prediction ===")
    for pred in result.final_prediction[: args.top_k]:
        bar = "#" * int(pred.probability * 50)
        print(f"  {pred.probability:.4f} {bar} '{pred.token}'")

    # Print layer-by-layer predictions
    layer_top_k = min(args.top_k, 10)
    if layer_top_k > 1:
        print(f"\n=== Logit Lens (top-{layer_top_k} at each layer) ===")
    else:
        print("\n=== Logit Lens (top prediction at each layer) ===")

    # Find peak probability for final token
    final_token = result.final_prediction[0].token if result.final_prediction else None
    peak_layer = None
    peak_prob = 0.0
    for layer_pred in result.layer_predictions:
        top = layer_pred.predictions[0]
        if top.token == final_token and top.probability > peak_prob:
            peak_prob = top.probability
            peak_layer = layer_pred.layer_idx

    for layer_pred in result.layer_predictions:
        top = layer_pred.predictions[0]
        marker = ""
        if peak_layer is not None and layer_pred.layer_idx == peak_layer:
            if peak_layer != result.captured_layers[-1]:
                marker = " <- peak"
        print(f"  Layer {layer_pred.layer_idx:2d}: '{top.token}' ({top.probability:.4f}){marker}")

        if layer_top_k > 1:
            for pred in layer_pred.predictions[1:layer_top_k]:
                print(f"           '{pred.token}' ({pred.probability:.4f})")

    # Print token evolution if tracking
    if result.token_evolutions:
        print("\n=== Token Evolution ===")
        for evo in result.token_evolutions:
            print(f"\nToken '{evo.token}':")
            for layer_idx, prob in evo.layer_probabilities.items():
                rank = evo.layer_ranks.get(layer_idx)
                rank_str = f"rank {rank}" if rank else "not in top-100"
                bar = "#" * int(prob * 100)
                print(f"  Layer {layer_idx:2d}: {prob:.4f} {bar} ({rank_str})")
            if evo.emergence_layer is not None:
                print(f"  --> Becomes top-1 at layer {evo.emergence_layer}")


def load_external_chat_template(tokenizer, model_path: str) -> None:
    """Load external chat template from model directory if available."""
    from huggingface_hub import snapshot_download

    try:
        local_path = Path(snapshot_download(model_path, allow_patterns=["chat_template.jinja"]))
    except Exception:
        local_path = Path(model_path)

    chat_template_path = local_path / "chat_template.jinja"
    if chat_template_path.exists() and not tokenizer.chat_template:
        try:
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
        except Exception:
            pass


def apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool = True) -> str:
    """Apply chat template to a prompt if available."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass
    return prompt


def parse_layers(layers_str: str | None, num_layers: int | None = None) -> list[int] | None:
    """Parse comma-separated layer list with support for ranges."""
    if not layers_str:
        return None

    layers = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


def parse_prompts(prompts_arg: str) -> list[str]:
    """Parse prompts from argument string or file."""
    if prompts_arg.startswith("@"):
        with open(prompts_arg[1:]) as f:
            return [line.strip() for line in f if line.strip()]
    return [p.strip() for p in prompts_arg.split("|")]


def normalize_number(s: str) -> str:
    """Normalize a number string by removing formatting characters."""
    import re

    return re.sub(r"[\s,\u202f\u00a0]+", "", s)


def validate_prompt_args(args, require_criterion: bool = False):
    """Validate prompt-related arguments."""
    if not getattr(args, "prompt", None) and not getattr(args, "prefix", None):
        print("Error: Either --prompt/-p or --prefix is required")
        sys.exit(1)

    if require_criterion and args.prompt and not getattr(args, "criterion", None):
        print("Error: --criterion/-c is required when using --prompt/-p")
        sys.exit(1)


def get_model_layers(model):
    """Get the layers list from a model, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    return None


def get_embed_tokens(model):
    """Get the embedding layer from a model."""
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    elif hasattr(model, "embed_tokens"):
        return model.embed_tokens
    return None


def get_lm_head(model):
    """Get the LM head from a model."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    return None


def get_final_norm(model):
    """Get the final normalization layer from a model."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    elif hasattr(model, "norm"):
        return model.norm
    return None
