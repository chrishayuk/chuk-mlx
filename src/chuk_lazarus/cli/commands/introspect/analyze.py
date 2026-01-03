"""Analysis command handlers for chuk-lazarus introspection CLI."""

import logging
import sys
from pathlib import Path

from ....introspection.enums import OverrideMode

logger = logging.getLogger(__name__)


def _print_analysis_result(result, tokenizer, args):
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
    layer_top_k = min(args.top_k, 10)  # Limit per-layer output
    if layer_top_k > 1:
        print(f"\n=== Logit Lens (top-{layer_top_k} at each layer) ===")
    else:
        print("\n=== Logit Lens (top prediction at each layer) ===")

    # Find peak probability for final token (to highlight)
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

        # Show additional predictions if top_k > 1
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


def _load_external_chat_template(tokenizer, model_path: str) -> None:
    """Load external chat template from model directory if available.

    Some models (like GPT-OSS) store the chat template in a separate
    chat_template.jinja file rather than in tokenizer_config.json.
    """

    from huggingface_hub import snapshot_download

    # Try to find model path
    try:
        # If it's a HF model ID, get the local cache path
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


def _apply_chat_template(tokenizer, prompt: str, add_generation_prompt: bool = True) -> str:
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


def introspect_analyze(args):
    """Run logit lens analysis on a prompt."""
    import asyncio

    from ....introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

    # Validate that either --prompt or --prefix is provided
    if not getattr(args, "prompt", None) and not getattr(args, "prefix", None):
        print("Error: Either --prompt/-p or --prefix is required")
        sys.exit(1)

    # Parse injection config if provided
    inject_config = None
    inject_layer = getattr(args, "inject_layer", None)
    inject_token = getattr(args, "inject_token", None)
    if inject_layer is not None and inject_token is not None:
        inject_blend = getattr(args, "inject_blend", 1.0)
        inject_config = {
            "layer": inject_layer,
            "token": inject_token,
            "blend": inject_blend,
        }
    elif inject_layer is not None or inject_token is not None:
        print("Error: --inject-layer and --inject-token must be used together")
        sys.exit(1)

    # Parse compute override config if provided
    compute_override_config = None
    compute_override = getattr(args, "compute_override", "none")
    if compute_override and compute_override != OverrideMode.NONE.value:
        compute_layer = getattr(args, "compute_layer", None)
        compute_override_config = {
            "mode": compute_override,
            "layer": compute_layer,  # Will default to 80% of model depth if None
        }

    # Parse steering config if provided
    steer_config = None
    steer_neuron = getattr(args, "steer_neuron", None)
    if steer_neuron is not None:
        import numpy as np

        # Single neuron steering - need to know hidden size, will be set after model loads
        steer_layer = getattr(args, "steer_layer", None)
        if steer_layer is None:
            print("Error: --steer-neuron requires --steer-layer")
            sys.exit(1)
        steer_coef = getattr(args, "strength", None) or 1.0
        steer_config = {
            "neuron": steer_neuron,
            "layer": steer_layer,
            "coefficient": steer_coef,
            "direction": None,  # Will be created after model loads
        }
    elif getattr(args, "steer", None):
        import numpy as np

        steer_arg = args.steer
        # Support both 'file.npz:coef' format and separate --strength flag
        if ":" in steer_arg:
            steer_parts = steer_arg.split(":")
            steer_file, steer_coef = steer_parts[0], float(steer_parts[1])
        else:
            steer_file = steer_arg
            steer_coef = getattr(args, "strength", None) or 1.0

        steer_data = np.load(steer_file, allow_pickle=True)
        steer_config = {
            "direction": steer_data["direction"],
            "layer": int(steer_data["layer"]),
            "coefficient": steer_coef,
            "file": steer_file,
        }
        if "label_positive" in steer_data:
            steer_config["positive"] = str(steer_data["label_positive"])
            steer_config["negative"] = str(steer_data["label_negative"])

    async def run():
        print(f"Loading model: {args.model}")

        # Determine embedding scale if specified
        embedding_scale = args.embedding_scale

        async with ModelAnalyzer.from_pretrained(
            args.model, embedding_scale=embedding_scale
        ) as analyzer:
            info = analyzer.model_info
            model_config = analyzer.config

            print(f"Model: {info.model_id}")
            if model_config is not None and hasattr(model_config, "model_type"):
                print(f"  Family: {model_config.model_type}")
            print(f"  Layers: {info.num_layers}")
            print(f"  Hidden size: {info.hidden_size}")
            print(f"  Vocab size: {info.vocab_size}")
            print(f"  Tied embeddings: {info.has_tied_embeddings}")
            if (
                model_config is not None
                and getattr(model_config, "embedding_scale", None) is not None
            ):
                print(f"  Embedding scale: {model_config.embedding_scale:.2f} (auto-detected)")

            # Apply steering if configured
            steering_wrapper = None
            if steer_config is not None:
                import mlx.core as mx
                import numpy as np

                from ....introspection.steering import SteeringHook

                steer_layer = steer_config["layer"]
                steer_coef = steer_config["coefficient"]

                # Create one-hot direction for neuron steering
                if "neuron" in steer_config:
                    neuron_idx = steer_config["neuron"]
                    direction_np = np.zeros(info.hidden_size, dtype=np.float32)
                    direction_np[neuron_idx] = 1.0
                    steer_dir = mx.array(direction_np, dtype=mx.float32)
                    print(f"\n  Steering neuron {neuron_idx} at layer {steer_layer}")
                    print(f"    Coefficient: {steer_coef:+.1f}")
                else:
                    steer_dir = mx.array(steer_config["direction"], dtype=mx.float32)
                    pos_label = steer_config.get("positive", "positive")
                    neg_label = steer_config.get("negative", "negative")
                    direction_str = (
                        f"{neg_label}->{pos_label}"
                        if steer_coef > 0
                        else f"{pos_label}->{neg_label}"
                    )
                    print(f"\n  Steering: {steer_config['file']}")
                    print(f"    Layer: {steer_layer}")
                    print(f"    Coefficient: {steer_coef:+.1f} ({direction_str})")

                # Access model layers
                model = analyzer._model
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    layers = model.model.layers
                elif hasattr(model, "layers"):
                    layers = model.layers
                else:
                    print("    WARNING: Cannot find model layers for steering")
                    layers = None

                if layers is not None:
                    original_layer = layers[steer_layer]
                    hook = SteeringHook(steer_dir, steer_coef, position=None, scale_by_norm=True)

                    class SteeredLayerWrapper:
                        def __init__(self, layer, hook):
                            self._wrapped = layer
                            self._hook = hook
                            for attr in [
                                "mlp",
                                "attn",
                                "self_attn",
                                "input_layernorm",
                                "post_attention_layernorm",
                            ]:
                                if hasattr(layer, attr):
                                    setattr(self, attr, getattr(layer, attr))

                        def __call__(self, h, **kwargs):
                            out = self._wrapped(h, **kwargs)
                            if hasattr(out, "hidden_states"):
                                out.hidden_states = self._hook(out.hidden_states)
                                return out
                            elif isinstance(out, tuple):
                                return (self._hook(out[0]),) + out[1:]
                            else:
                                return self._hook(out)

                        def __getattr__(self, name):
                            return getattr(self._wrapped, name)

                    layers[steer_layer] = SteeredLayerWrapper(original_layer, hook)
                    steering_wrapper = (layers, steer_layer, original_layer)

            # Apply token injection if configured
            injection_wrapper = None
            if inject_config is not None:
                import mlx.core as mx

                inject_layer_idx = inject_config["layer"]
                inject_token_str = inject_config["token"]
                inject_blend = inject_config["blend"]

                # Get the token embedding for the inject token
                tokenizer = analyzer._tokenizer
                inject_token_ids = tokenizer.encode(inject_token_str)
                if len(inject_token_ids) != 1:
                    print(
                        f"  Warning: '{inject_token_str}' tokenizes to {len(inject_token_ids)} tokens, using first"
                    )
                inject_token_id = inject_token_ids[0]

                # Get embedding
                model = analyzer._model
                if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                    embed = model.model.embed_tokens
                elif hasattr(model, "embed_tokens"):
                    embed = model.embed_tokens
                else:
                    print("  ERROR: Cannot find embedding layer for injection")
                    embed = None

                if embed is not None:
                    inject_embedding = embed(mx.array([inject_token_id]))[
                        0
                    ]  # Shape: (hidden_size,)

                    # Apply embedding scale if present
                    embed_scale = getattr(model_config, "embedding_scale", None)
                    if embed_scale:
                        inject_embedding = inject_embedding * embed_scale

                    print(
                        f"\n  Injecting token '{inject_token_str}' (id={inject_token_id}) at layer {inject_layer_idx}"
                    )
                    print(f"    Blend: {inject_blend:.1f} (0=original, 1=full replacement)")

                    # Access model layers
                    if hasattr(model, "model") and hasattr(model.model, "layers"):
                        layers = model.model.layers
                    elif hasattr(model, "layers"):
                        layers = model.layers
                    else:
                        print("    WARNING: Cannot find model layers for injection")
                        layers = None

                    if layers is not None:
                        original_layer = layers[inject_layer_idx]

                        class InjectedLayerWrapper:
                            def __init__(self, layer, inject_emb, blend):
                                self._wrapped = layer
                                self._inject_emb = inject_emb
                                self._blend = blend
                                for attr in [
                                    "mlp",
                                    "attn",
                                    "self_attn",
                                    "input_layernorm",
                                    "post_attention_layernorm",
                                ]:
                                    if hasattr(layer, attr):
                                        setattr(self, attr, getattr(layer, attr))

                            def __call__(self, h, **kwargs):
                                out = self._wrapped(h, **kwargs)

                                # Get the hidden states
                                if hasattr(out, "hidden_states"):
                                    hs = out.hidden_states
                                elif isinstance(out, tuple):
                                    hs = out[0]
                                else:
                                    hs = out

                                # Inject at last position: blend original with inject embedding
                                # hs shape: (batch, seq, hidden)
                                last_pos = hs[:, -1:, :]  # (batch, 1, hidden)
                                inject_expanded = self._inject_emb.reshape(
                                    1, 1, -1
                                )  # (1, 1, hidden)
                                blended = (
                                    1 - self._blend
                                ) * last_pos + self._blend * inject_expanded
                                new_hs = mx.concatenate([hs[:, :-1, :], blended], axis=1)

                                if hasattr(out, "hidden_states"):
                                    out.hidden_states = new_hs
                                    return out
                                elif isinstance(out, tuple):
                                    return (new_hs,) + out[1:]
                                else:
                                    return new_hs

                            def __getattr__(self, name):
                                return getattr(self._wrapped, name)

                        layers[inject_layer_idx] = InjectedLayerWrapper(
                            original_layer, inject_embedding, inject_blend
                        )
                        injection_wrapper = (layers, inject_layer_idx, original_layer)

            # Apply compute override if configured
            compute_wrapper = None
            if compute_override_config is not None:
                import re

                import mlx.core as mx

                override_mode = compute_override_config["mode"]
                compute_layer_idx = compute_override_config["layer"]
                if compute_layer_idx is None:
                    compute_layer_idx = int(info.num_layers * 0.8)

                # Parse the prompt for arithmetic expression
                prompt_to_check = args.prompt if args.prompt else args.prefix
                computed_answer = None

                if override_mode == OverrideMode.ARITHMETIC.value:
                    # Match patterns like "7*6=", "123+456=", "10-3=", "81/9="
                    arith_pattern = r"(\d+)\s*([+\-*/x*])\s*(\d+)\s*=\s*$"
                    match = re.search(arith_pattern, prompt_to_check)
                    if match:
                        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                        if op in ["*", "x", "*"]:
                            computed_answer = a * b
                        elif op == "+":
                            computed_answer = a + b
                        elif op == "-":
                            computed_answer = a - b
                        elif op == "/":
                            computed_answer = a // b if b != 0 else None

                if computed_answer is not None:
                    answer_str = str(computed_answer)
                    # Get the token embedding for the answer
                    tokenizer = analyzer._tokenizer
                    answer_token_ids = tokenizer.encode(answer_str)
                    if len(answer_token_ids) == 1:
                        # Get embedding
                        model = analyzer._model
                        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                            embed = model.model.embed_tokens
                        elif hasattr(model, "embed_tokens"):
                            embed = model.embed_tokens
                        else:
                            embed = None

                        if embed is not None:
                            # Strategy: Run a "reference" prompt that produces the correct answer
                            # and capture its hidden state at the target layer
                            # For multiplication, use the commutative pair (e.g., for 7*6, use 6*7)
                            import mlx.nn as nn

                            if op in ["*", "x", "*"]:
                                # Try commutative pair
                                ref_prompt = f"{b}*{a}="
                            else:
                                # For other ops, use a simple identity (answer itself)
                                ref_prompt = f"{computed_answer}"

                            print(f"\n  COMPUTE OVERRIDE: {override_mode}")
                            print(f"    Detected: {a} {op} {b} = {computed_answer}")
                            print(f"    Reference prompt: '{ref_prompt}'")
                            print(f"    Capturing hidden state at layer {compute_layer_idx}")

                            # Run reference prompt to capture hidden state
                            ref_ids = mx.array(tokenizer.encode(ref_prompt))[None, :]

                            # Get model layers
                            if hasattr(model, "model") and hasattr(model.model, "layers"):
                                ref_layers = list(model.model.layers)
                                ref_embed = model.model.embed_tokens
                            else:
                                ref_layers = list(model.layers)
                                ref_embed = model.embed_tokens

                            # Forward through layers to capture hidden state
                            h_ref = ref_embed(ref_ids)
                            embed_scale = getattr(model_config, "embedding_scale", None)
                            if embed_scale:
                                h_ref = h_ref * embed_scale

                            seq_len = ref_ids.shape[1]
                            mask = nn.MultiHeadAttention.create_additive_causal_mask(
                                seq_len
                            ).astype(h_ref.dtype)

                            for idx, lyr in enumerate(ref_layers):
                                try:
                                    out = lyr(h_ref, mask=mask)
                                except TypeError:
                                    out = lyr(h_ref)
                                h_ref = (
                                    out.hidden_states
                                    if hasattr(out, "hidden_states")
                                    else (out[0] if isinstance(out, tuple) else out)
                                )
                                if idx == compute_layer_idx:
                                    # Capture the last position hidden state
                                    reference_hidden = h_ref[0, -1, :]  # Shape: (hidden_size,)
                                    break

                            print(
                                f"    Captured reference hidden state (norm={float(mx.sqrt(mx.sum(reference_hidden**2))):.1f})"
                            )

                            # Access model layers
                            if hasattr(model, "model") and hasattr(model.model, "layers"):
                                layers = model.model.layers
                            elif hasattr(model, "layers"):
                                layers = model.layers
                            else:
                                layers = None

                            if layers is not None:
                                original_layer = layers[compute_layer_idx]

                                class ComputeOverrideWrapper:
                                    """Replaces layer output with reference hidden state from working prompt."""

                                    def __init__(self, layer, ref_hidden):
                                        self._wrapped = layer
                                        self._ref_hidden = (
                                            ref_hidden  # Hidden state from reference prompt
                                        )
                                        for attr in [
                                            "mlp",
                                            "attn",
                                            "self_attn",
                                            "input_layernorm",
                                            "post_attention_layernorm",
                                        ]:
                                            if hasattr(layer, attr):
                                                setattr(self, attr, getattr(layer, attr))

                                    def __call__(self, h, **kwargs):
                                        out = self._wrapped(h, **kwargs)

                                        # Get the hidden states
                                        if hasattr(out, "hidden_states"):
                                            hs = out.hidden_states
                                        elif isinstance(out, tuple):
                                            hs = out[0]
                                        else:
                                            hs = out

                                        # Replace last position with reference hidden state
                                        new_last = self._ref_hidden.reshape(1, 1, -1)
                                        new_hs = mx.concatenate([hs[:, :-1, :], new_last], axis=1)

                                        if hasattr(out, "hidden_states"):
                                            out.hidden_states = new_hs
                                            return out
                                        elif isinstance(out, tuple):
                                            return (new_hs,) + out[1:]
                                        else:
                                            return new_hs

                                    def __getattr__(self, name):
                                        return getattr(self._wrapped, name)

                                layers[compute_layer_idx] = ComputeOverrideWrapper(
                                    original_layer, reference_hidden
                                )
                                compute_wrapper = (layers, compute_layer_idx, original_layer)
                    else:
                        print(
                            f"\n  WARNING: Answer '{answer_str}' requires {len(answer_token_ids)} tokens, skipping override"
                        )
                else:
                    if override_mode == OverrideMode.ARITHMETIC.value:
                        print(
                            f"\n  WARNING: Could not parse arithmetic from prompt: {prompt_to_check}"
                        )

            # Check chat template mode
            use_raw = getattr(args, "raw", False)
            tokenizer = analyzer._tokenizer
            has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

            # Configure analysis
            # --layers overrides --all-layers overrides --layer-strategy
            custom_layers = None
            if getattr(args, "layers", None):
                # Parse comma-separated layer list
                custom_layers = [int(x.strip()) for x in args.layers.split(",")]
                layer_strategy = LayerStrategy.CUSTOM
            elif getattr(args, "all_layers", False):
                layer_strategy = LayerStrategy.ALL
            else:
                layer_strategy = LayerStrategy(args.layer_strategy)
            analysis_config = AnalysisConfig(
                layer_strategy=layer_strategy,
                layer_step=args.layer_step,
                top_k=args.top_k,
                track_tokens=args.track or [],
                custom_layers=custom_layers,
            )

            # Check for --prefix mode (bypasses prompt processing)
            if getattr(args, "prefix", None):
                print("  Mode: PREFIX (using exact prefix)")
                prompt = args.prefix
                print(f"\nAnalyzing prefix: {prompt!r}")
                # Go directly to analysis
                result = await analyzer.analyze(prompt, analysis_config)
                _print_analysis_result(result, tokenizer, args)
                if args.output:
                    import json

                    with open(args.output, "w") as f:
                        json.dump(result.to_dict(), f, indent=2)
                    print(f"\nResults saved to {args.output}")
                return

            if use_raw:
                print("  Mode: RAW (no chat template)")
            elif has_chat_template:
                print("  Mode: CHAT (using chat template)")
            else:
                print("  Mode: RAW (model has no chat template)")

            # Note about trailing whitespace which affects tokenization
            prompt = args.prompt

            # Apply chat template unless --raw is specified
            if not use_raw and has_chat_template:
                prompt = _apply_chat_template(tokenizer, prompt)
                print(f"\nAnalyzing (with chat template): {args.prompt!r}")
            else:
                if prompt != prompt.rstrip():
                    print("\nNote: Prompt has trailing whitespace which affects tokenization")
                    print(
                        "  This changes what the model predicts (next token after space vs after last word)"
                    )
                    print(
                        "  For arithmetic prompts like 'X + Y = ', trailing space often helps get answers"
                    )
                print(f"\nAnalyzing: {prompt!r}")

            # Find answer position - default ON for chat mode, can override with --find-answer or --no-find-answer
            find_answer_arg = getattr(args, "find_answer", None)
            no_find_answer = getattr(args, "no_find_answer", False)

            # Default: enabled in chat mode, disabled in raw mode
            if no_find_answer:
                find_answer = False
            elif find_answer_arg:
                find_answer = True
            else:
                # Default based on mode
                find_answer = has_chat_template and not use_raw

            if find_answer:
                import mlx.core as mx

                gen_tokens = getattr(args, "gen_tokens", 30)
                expected = getattr(args, "expected", None)

                print(f"\nGenerating {gen_tokens} tokens to find answer position...")

                # Use simple greedy generation without KVCache (compatible with our model)
                input_ids = mx.array(tokenizer.encode(prompt))[None, :]
                generated_ids = []

                for _ in range(gen_tokens):
                    outputs = analyzer._model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    next_token = mx.argmax(logits[:, -1, :], axis=-1)
                    generated_ids.append(int(next_token[0]))

                    # Check for EOS
                    if (
                        hasattr(tokenizer, "eos_token_id")
                        and generated_ids[-1] == tokenizer.eos_token_id
                    ):
                        break

                    input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

                generated = tokenizer.decode(generated_ids)
                print(f"Generated: {generated!r}")

                # Find where the expected answer appears (or auto-detect for arithmetic)
                if expected is None:
                    # Try to auto-detect expected answer for arithmetic
                    import re

                    match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", args.prompt.strip())
                    if match:
                        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                        if op == "+":
                            expected = str(a + b)
                        elif op == "-":
                            expected = str(a - b)
                        elif op == "*":
                            expected = str(a * b)
                        elif op == "/":
                            expected = str(a // b)
                        print(f"Auto-detected expected answer: {expected}")

                if expected:
                    # Find where expected answer first appears in generated text
                    answer_pos = generated.find(expected)
                    if answer_pos >= 0:
                        # Build prompt up to (but not including) the answer
                        prefix = generated[:answer_pos]
                        extended_prompt = prompt + prefix
                        print(f"Answer '{expected}' found at position {answer_pos}")

                        # Show the generated output with analysis point highlighted
                        before = generated[:answer_pos]
                        after = generated[answer_pos:]
                        print("\n=== Analysis Point in Response ===")
                        print(f"  {before}>{after}")
                        print(f"                {'-' * min(len(before), 40)}")
                        print(f"  Analyzing prediction at > (just before '{expected}')")

                        prompt = extended_prompt
                    else:
                        print(f"Expected answer '{expected}' not found in generated output")
                else:
                    print("No expected answer specified and couldn't auto-detect")

            result = await analyzer.analyze(prompt, analysis_config)

            # Print analysis result
            _print_analysis_result(result, tokenizer, args)

            # Export if requested
            if args.output:
                import json

                output_data = {
                    "prompt": result.prompt,
                    "tokens": result.tokens,
                    "num_layers": result.num_layers,
                    "captured_layers": result.captured_layers,
                    "final_prediction": [p.model_dump() for p in result.final_prediction],
                    "layer_predictions": [
                        {
                            "layer_idx": lp.layer_idx,
                            "predictions": [p.model_dump() for p in lp.predictions],
                        }
                        for lp in result.layer_predictions
                    ],
                }
                if result.token_evolutions:
                    output_data["token_evolutions"] = [
                        e.model_dump() for e in result.token_evolutions
                    ]
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")

            # Restore original layer if we were steering
            if steering_wrapper is not None:
                layers, steer_layer, original_layer = steering_wrapper
                layers[steer_layer] = original_layer

            # Restore original layer if we were injecting
            if injection_wrapper is not None:
                layers, inject_layer_idx, original_layer = injection_wrapper
                layers[inject_layer_idx] = original_layer

            # Restore original layer if we were compute overriding
            if compute_wrapper is not None:
                layers, compute_layer_idx, original_layer = compute_wrapper
                layers[compute_layer_idx] = original_layer

    asyncio.run(run())


def introspect_compare(args):
    """Compare two models' predictions using logit lens."""
    import asyncio

    from ....introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

    async def run():
        models = [args.model1, args.model2]
        results = []

        for model_id in models:
            print(f"\nLoading: {model_id}")
            async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
                config = AnalysisConfig(
                    layer_strategy=LayerStrategy.EVENLY_SPACED,
                    layer_step=4,
                    top_k=args.top_k,
                    track_tokens=args.track.split(",") if args.track else [],
                )
                result = await analyzer.analyze(args.prompt, config)
                results.append((model_id, result))

        # Print comparison
        print(f"\n{'=' * 70}")
        print(f"Prompt: {args.prompt!r}")
        print(f"{'=' * 70}")

        # Final predictions side by side
        print("\n=== Final Predictions ===")
        print(f"{'Model':<40} {'Top Token':<15} {'Prob':<10}")
        print("-" * 65)
        for model_id, result in results:
            if result.final_prediction:
                top = result.final_prediction[0]
                print(f"{model_id[:40]:<40} {top.token:<15} {top.probability:.4f}")

        # Token evolution comparison if tracking
        if args.track:
            print("\n=== Token Evolution Comparison ===")
            tokens = args.track.split(",")
            for token in tokens:
                print(f"\nToken '{token}':")
                for model_id, result in results:
                    for evo in result.token_evolutions:
                        if evo.token == token:
                            emergence = evo.emergence_layer
                            final_prob = (
                                list(evo.layer_probabilities.values())[-1]
                                if evo.layer_probabilities
                                else 0
                            )
                            print(
                                f"  {model_id[:35]:<35}: emerges at layer {emergence}, final prob {final_prob:.4f}"
                            )

    asyncio.run(run())


def introspect_hooks(args):
    """Low-level hook demonstration."""
    import mlx.core as mx

    from ....introspection import CaptureConfig, LogitLens, ModelHooks, PositionSelection

    # Load model
    print(f"Loading model: {args.model}")
    from mlx_lm import load

    model, tokenizer = load(args.model)

    # Load external chat template if available (e.g., GPT-OSS)
    _load_external_chat_template(tokenizer, args.model)

    # Tokenize
    input_ids = mx.array(tokenizer.encode(args.prompt))[None, :]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(0, 32, 4))  # Every 4th layer by default

    # Setup hooks
    print(f"\nCapturing layers: {layers}")
    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
            capture_attention_weights=args.capture_attention,
            positions=PositionSelection.LAST if args.last_only else PositionSelection.ALL,
        )
    )

    # Forward pass
    print("Running forward pass...")
    hooks.forward(input_ids)

    # Show captured states
    print("\n=== Captured States ===")
    print(f"Layers captured: {hooks.state.captured_layers}")
    for layer_idx, hidden in hooks.state.hidden_states.items():
        print(f"  Layer {layer_idx}: hidden shape {hidden.shape}")
    if hooks.state.attention_weights:
        for layer_idx, attn in hooks.state.attention_weights.items():
            print(f"  Layer {layer_idx}: attention shape {attn.shape}")

    # Logit lens
    if not args.no_logit_lens:
        lens = LogitLens(hooks, tokenizer)
        print("\n=== Logit Lens ===")
        lens.print_evolution(position=-1, top_k=3)
