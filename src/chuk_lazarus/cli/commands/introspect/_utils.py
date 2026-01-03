"""Shared utilities for introspect CLI commands."""

import sys
from pathlib import Path


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
