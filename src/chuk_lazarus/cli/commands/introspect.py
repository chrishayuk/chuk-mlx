"""Introspection command handlers for chuk-lazarus CLI."""

import logging
import sys
from pathlib import Path

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
                marker = " ← peak"
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
    from pathlib import Path

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

    from ...introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

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
    if compute_override and compute_override != "none":
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

                from ...introspection.steering import SteeringHook

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
                        f"{neg_label}→{pos_label}" if steer_coef > 0 else f"{pos_label}→{neg_label}"
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

                if override_mode == "arithmetic":
                    # Match patterns like "7*6=", "123+456=", "10-3=", "81/9="
                    arith_pattern = r"(\d+)\s*([+\-*/x×])\s*(\d+)\s*=\s*$"
                    match = re.search(arith_pattern, prompt_to_check)
                    if match:
                        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                        if op in ["*", "x", "×"]:
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

                            if op in ["*", "x", "×"]:
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
                    if override_mode == "arithmetic":
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
                    print(f"\n✓ Results saved to {args.output}")
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
                    print("\n⚠ Note: Prompt has trailing whitespace which affects tokenization")
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
                        print(f"  {before}▶{after}")
                        print(f"                {'─' * min(len(before), 40)}┘")
                        print(f"  Analyzing prediction at ▶ (just before '{expected}')")

                        prompt = extended_prompt
                    else:
                        print(f"⚠ Expected answer '{expected}' not found in generated output")
                else:
                    print("⚠ No expected answer specified and couldn't auto-detect")

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

    from ...introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

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

    from ...introspection import CaptureConfig, LogitLens, ModelHooks, PositionSelection

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


def introspect_ablate(args):
    """Run ablation study to identify causal circuits.

    Supports two modes:
    1. Sweep mode (default): Test each layer independently
    2. Multi mode (--multi): Ablate all specified layers together

    Examples:
        # Sweep layers 20-23 individually on arithmetic
        lazarus introspect ablate -m openai/gpt-oss-20b -p "45 * 45 = " -c "2025" --layers 20-23

        # Ablate L22+L23 together
        lazarus introspect ablate -m openai/gpt-oss-20b -p "45 * 45 = " -c "2025" --layers 22,23 --multi

        # Test multiple prompts with difficulty gradient
        lazarus introspect ablate -m openai/gpt-oss-20b --prompts "10*10=:100|45*45=:2025|47*47=:2209" --layers 22,23 --multi
    """

    from ...introspection import AblationConfig, AblationStudy, ComponentType

    # Validate arguments: need either --prompt+--criterion OR --prompts
    prompts_arg = getattr(args, "prompts", None)
    if not prompts_arg and not args.prompt:
        print("Error: Either --prompt/-p (with --criterion/-c) or --prompts is required")
        return
    if args.prompt and not args.criterion and not prompts_arg:
        print("Error: --criterion/-c is required when using --prompt/-p")
        return

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)

    # Parse layers - support comma-separated and ranges (e.g., "0,1,2" or "0-5" or "0-5,10,15-20")
    if args.layers:
        layers = []
        for part in args.layers.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))
    else:
        layers = list(range(study.adapter.num_layers))

    multi_mode = getattr(args, "multi", False)
    use_raw = getattr(args, "raw", False)

    if multi_mode:
        print(f"Ablating layers together: {layers}")
    else:
        print(f"Sweeping layers individually: {layers}")
    print(f"Component: {args.component}")
    print(f"Mode: {'RAW' if use_raw else 'CHAT'}")

    # Create criterion function based on argument
    criterion_map = {
        "function_call": lambda x: any(
            m in x for m in ["<start_function_call>", "<function_call>", "get_weather(", '{"name":']
        ),
        "sorry": lambda x: "sorry" in x.lower() or "apologize" in x.lower(),
        "positive": lambda x: any(
            w in x.lower() for w in ["great", "good", "excellent", "wonderful", "love"]
        ),
        "negative": lambda x: any(
            w in x.lower() for w in ["bad", "terrible", "awful", "hate", "poor"]
        ),
        "refusal": lambda x: any(
            m in x.lower() for m in ["cannot", "can't", "won't", "unable", "decline"]
        ),
    }

    # Map component
    component = {
        "mlp": ComponentType.MLP,
        "attention": ComponentType.ATTENTION,
        "both": ComponentType.BOTH,
    }[args.component]

    config = AblationConfig(
        component=component,
        max_new_tokens=args.max_tokens,
    )

    # Handle multiple prompts mode (--prompts "prompt1:expected1|prompt2:expected2")
    if prompts_arg:
        prompt_pairs = []
        for item in prompts_arg.split("|"):
            item = item.strip()
            if ":" in item:
                prompt, expected = item.rsplit(":", 1)
                prompt_pairs.append((prompt.strip(), expected.strip()))
            else:
                # Prompt without expected value - use criterion if available, else error
                if args.criterion:
                    prompt_pairs.append((item, args.criterion))
                else:
                    print(
                        f"Error: Prompt '{item}' has no expected value (use 'prompt:expected' format)"
                    )
                    return

        verbose = getattr(args, "verbose", False)

        print(f"\n{'=' * 70}")
        print("MULTI-PROMPT ABLATION TEST")
        print(f"{'=' * 70}")

        # Store full outputs for verbose mode
        all_outputs: dict[str, dict[str, tuple[str, bool]]] = {}

        # Header
        header = f"{'Ablation':<20}"
        for prompt, expected in prompt_pairs:
            short_prompt = prompt[:12] + "..." if len(prompt) > 15 else prompt
            header += f" | {short_prompt:<18}"
        print(header)
        print("-" * len(header))

        # Baseline (no ablation)
        row = f"{'None (baseline)':<20}"
        all_outputs["baseline"] = {}
        for prompt, expected in prompt_pairs:
            out = study.ablate_and_generate(prompt, layers=[], config=config)
            out_short = out.strip()[:15]
            correct = expected in out
            status = f"{'Y' if correct else 'N'} {out_short}"
            row += f" | {status:<18}"
            all_outputs["baseline"][prompt] = (out, correct)
        print(row)

        if multi_mode:
            # Single test with all layers together
            layer_str = ",".join(str(layer) for layer in layers)
            row = f"L{layer_str:<19}"[:20]
            all_outputs[f"L{layer_str}"] = {}
            for prompt, expected in prompt_pairs:
                out = study.ablate_and_generate(prompt, layers=layers, config=config)
                out_short = out.strip()[:15]
                correct = expected in out
                status = f"{'Y' if correct else 'N'} {out_short}"
                row += f" | {status:<18}"
                all_outputs[f"L{layer_str}"][prompt] = (out, correct)
            print(row)
        else:
            # Sweep each layer
            for layer in layers:
                row = f"L{layer:<19}"
                all_outputs[f"L{layer}"] = {}
                for prompt, expected in prompt_pairs:
                    out = study.ablate_and_generate(prompt, layers=[layer], config=config)
                    out_short = out.strip()[:15]
                    correct = expected in out
                    status = f"{'Y' if correct else 'N'} {out_short}"
                    row += f" | {status:<18}"
                    all_outputs[f"L{layer}"][prompt] = (out, correct)
                print(row)

        # Verbose output - show full generations
        if verbose:
            print(f"\n{'=' * 70}")
            print("FULL OUTPUTS")
            print(f"{'=' * 70}")
            for prompt, expected in prompt_pairs:
                print(f"\n>>> Prompt: {prompt!r} (expected: {expected})")
                print("-" * 50)
                for ablation_name, outputs in all_outputs.items():
                    out, correct = outputs[prompt]
                    status = "PASS" if correct else "FAIL"
                    print(f"\n[{ablation_name}] ({status}):")
                    print(out.strip())

        return

    # Single prompt mode
    if args.criterion in criterion_map:
        criterion = criterion_map[args.criterion]
        criterion.__name__ = args.criterion
    else:
        # Treat as substring check
        def substring_criterion(x: str, s: str = args.criterion) -> bool:
            return s in x

        substring_criterion.__name__ = f"contains_{args.criterion}"
        criterion = substring_criterion

    if multi_mode:
        # Multi-layer ablation: ablate all layers together
        print(f"\nAblating layers {layers} together...")

        # Get baseline
        original = study.ablate_and_generate(args.prompt, layers=[], config=config)
        original_passes = criterion(original)

        # Get ablated
        ablated = study.ablate_and_generate(args.prompt, layers=layers, config=config)
        ablated_passes = criterion(ablated)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {args.prompt}")
        print(f"Criterion: {args.criterion}")
        print(f"Layers ablated: {layers}")
        print(f"{'=' * 60}")
        print(f"\nOriginal output ({['FAIL', 'PASS'][original_passes]}):")
        print(f"  {original.strip()[:200]}")
        print(f"\nAblated output ({['FAIL', 'PASS'][ablated_passes]}):")
        print(f"  {ablated.strip()[:200]}")

        if original_passes and not ablated_passes:
            print(f"\n=> CAUSAL: Ablating {layers} breaks the criterion")
        elif not original_passes and ablated_passes:
            print(f"\n=> INVERSE CAUSAL: Ablating {layers} enables the criterion")
        elif original_passes and ablated_passes:
            print(f"\n=> NOT CAUSAL: Ablating {layers} doesn't affect outcome")
        else:
            print("\n=> BASELINE FAILS: Original doesn't pass criterion")

    else:
        # Sweep mode: test each layer independently
        print("\nRunning ablation sweep...")
        result = study.run_layer_sweep(
            prompt=args.prompt,
            criterion=criterion,
            layers=layers,
            component=component,
            task_name="ablation_study",
            config=config,
        )

        # Print results
        study.print_sweep_summary(result)

        # Show verbose output if requested
        if getattr(args, "verbose", False):
            print("\n=== Detailed Outputs ===")
            for r in result.results:
                print(f"\n--- Layer {r.layer} ---")
                print(f"Original: {r.original_output[:200]}...")
                print(f"Ablated:  {r.ablated_output[:200]}...")

        # Save if requested
        if args.output:
            study.save_results({"ablation_study": result}, args.output)


def introspect_weight_diff(args):
    """Compare weight divergence between two models."""
    import json

    import mlx.core as mx
    from huggingface_hub import snapshot_download

    print(f"Loading base model: {args.base}")
    base_path = snapshot_download(args.base, allow_patterns=["*.json", "*.safetensors"])

    print(f"Loading fine-tuned model: {args.finetuned}")
    ft_path = snapshot_download(args.finetuned, allow_patterns=["*.json", "*.safetensors"])

    # Detect family and load
    from ...introspection.ablation import AblationStudy

    family = AblationStudy._detect_family(base_path)
    print(f"Detected model family: {family}")

    base_model, base_config = AblationStudy._load_model(base_path, family)
    ft_model, ft_config = AblationStudy._load_model(ft_path, family)

    # Compare weights
    from ...introspection.ablation import ModelAdapter

    base_adapter = ModelAdapter(base_model, None, base_config)
    ft_adapter = ModelAdapter(ft_model, None, ft_config)

    print(f"\nComparing {base_adapter.num_layers} layers...")

    results = []
    for layer_idx in range(base_adapter.num_layers):
        # Compare MLP
        try:
            base_mlp = base_adapter.get_mlp_down_weight(layer_idx)
            ft_mlp = ft_adapter.get_mlp_down_weight(layer_idx)

            diff = ft_mlp - base_mlp
            base_norm = float(mx.sqrt(mx.sum(base_mlp * base_mlp)))
            diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
            rel_diff = diff_norm / (base_norm + 1e-8)

            results.append(
                {
                    "layer": layer_idx,
                    "component": "mlp_down",
                    "relative_diff": rel_diff,
                }
            )
        except Exception:
            pass

        # Compare attention
        try:
            base_attn = base_adapter.get_attn_o_weight(layer_idx)
            ft_attn = ft_adapter.get_attn_o_weight(layer_idx)

            diff = ft_attn - base_attn
            base_norm = float(mx.sqrt(mx.sum(base_attn * base_attn)))
            diff_norm = float(mx.sqrt(mx.sum(diff * diff)))
            rel_diff = diff_norm / (base_norm + 1e-8)

            results.append(
                {
                    "layer": layer_idx,
                    "component": "attn_o",
                    "relative_diff": rel_diff,
                }
            )
        except Exception:
            pass

    # Print results
    print(f"\n{'Layer':<8} {'Component':<12} {'Rel. Diff':>12}")
    print("-" * 35)
    for r in results:
        marker = " ***" if r["relative_diff"] > 0.1 else ""
        print(f"{r['layer']:<8} {r['component']:<12} {r['relative_diff']:>12.6f}{marker}")

    # Find top divergent
    sorted_results = sorted(results, key=lambda x: x["relative_diff"], reverse=True)
    print("\nTop 5 divergent components:")
    for r in sorted_results[:5]:
        print(f"  Layer {r['layer']} {r['component']}: {r['relative_diff']:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_activation_diff(args):
    """Compare activation divergence between two models."""
    import json

    import mlx.core as mx

    from ...introspection import CaptureConfig, ModelHooks, PositionSelection

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split(",")]

    print(f"Testing {len(prompts)} prompts")

    # Load models
    print(f"Loading base model: {args.base}")
    from ...introspection.ablation import AblationStudy

    base_study = AblationStudy.from_pretrained(args.base)

    print(f"Loading fine-tuned model: {args.finetuned}")
    ft_study = AblationStudy.from_pretrained(args.finetuned)

    tokenizer = base_study.adapter.tokenizer

    results = []
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Get activations from both models
        base_hooks = ModelHooks(base_study.adapter.model)
        base_hooks.configure(
            CaptureConfig(
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        base_hooks.forward(input_ids)

        ft_hooks = ModelHooks(ft_study.adapter.model)
        ft_hooks.configure(
            CaptureConfig(
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        ft_hooks.forward(input_ids)

        # Compare
        for layer_idx in range(base_study.adapter.num_layers):
            base_h = base_hooks.state.hidden_states.get(layer_idx)
            ft_h = ft_hooks.state.hidden_states.get(layer_idx)

            if base_h is None or ft_h is None:
                continue

            # Flatten to last position
            base_h = base_h[0, -1] if base_h.ndim == 3 else base_h[-1]
            ft_h = ft_h[0, -1] if ft_h.ndim == 3 else ft_h[-1]

            # Cosine similarity
            dot = float(mx.sum(base_h * ft_h))
            norm_base = float(mx.sqrt(mx.sum(base_h * base_h)))
            norm_ft = float(mx.sqrt(mx.sum(ft_h * ft_h)))
            cos_sim = dot / (norm_base * norm_ft + 1e-8)

            results.append(
                {
                    "prompt": prompt[:50],
                    "layer": layer_idx,
                    "cosine_similarity": cos_sim,
                }
            )

    # Aggregate by layer
    layer_avg = {}
    for r in results:
        layer = r["layer"]
        if layer not in layer_avg:
            layer_avg[layer] = []
        layer_avg[layer].append(r["cosine_similarity"])

    print(f"\n{'Layer':<8} {'Avg Cos Sim':>12} {'Divergence':>12}")
    print("-" * 35)
    for layer in sorted(layer_avg.keys()):
        avg = sum(layer_avg[layer]) / len(layer_avg[layer])
        div = 1 - avg
        marker = " ***" if div > 0.1 else ""
        print(f"{layer:<8} {avg:>12.4f} {div:>12.4f}{marker}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_layer(args):
    """Analyze what specific layers do with representation similarity."""
    import json

    from ...introspection import LayerAnalyzer

    print(f"Loading model: {args.model}")
    analyzer = LayerAnalyzer.from_pretrained(args.model)

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(",")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels provided for {len(prompts)} prompts")
            labels = None

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None  # Use default (key layers)

    print(f"\nAnalyzing {len(prompts)} prompts at layers: {layers or 'auto'}")
    for i, p in enumerate(prompts):
        label_str = f" [{labels[i]}]" if labels else ""
        print(f"  {i + 1}. {p!r}{label_str}")

    # Run representation analysis
    result = analyzer.analyze_representations(
        prompts=prompts,
        layers=layers,
        labels=labels,
        position=-1,  # Last token position
    )

    # Print similarity matrices for each layer
    for layer_idx in result.layers:
        analyzer.print_similarity_matrix(result, layer_idx)

    # If comparing format sensitivity, show summary
    if labels and len(set(labels)) == 2:
        print("\n=== Format Sensitivity Summary ===")
        for layer_idx in result.layers:
            if result.clusters and layer_idx in result.clusters:
                cluster = result.clusters[layer_idx]
                within = cluster.within_cluster_similarity
                between = cluster.between_cluster_similarity
                sep = cluster.separation_score

                print(f"\nLayer {layer_idx}:")
                for label, sim in within.items():
                    print(f"  Within '{label}': {sim:.4f}")
                for (l1, l2), sim in between.items():
                    print(f"  Between '{l1}' <-> '{l2}': {sim:.4f}")
                print(f"  Separation score: {sep:.4f}")

                # Interpretation
                if sep > 0.02:
                    print(f"  → Layer {layer_idx} DOES distinguish between groups")
                else:
                    print(f"  → Layer {layer_idx} does NOT distinguish between groups")

    # Run attention analysis if requested
    if args.attention:
        print("\n=== Attention Analysis ===")
        attn_results = analyzer.analyze_attention(
            prompts=prompts,
            layers=layers[:2] if layers and len(layers) > 2 else layers,
        )
        for layer_idx in attn_results:
            analyzer.print_attention_comparison(attn_results, layer_idx, prompts, focus_token=-1)

    # Save if requested
    if args.output:
        output_data = {
            "prompts": prompts,
            "labels": labels,
            "layers": result.layers,
            "similarity_matrices": {
                layer: result.representations[layer].similarity_matrix for layer in result.layers
            },
        }
        if result.clusters:
            output_data["clusters"] = {
                layer: {
                    "within": result.clusters[layer].within_cluster_similarity,
                    "between": {
                        f"{l1}_{l2}": v
                        for (l1, l2), v in result.clusters[layer].between_cluster_similarity.items()
                    },
                    "separation": result.clusters[layer].separation_score,
                }
                for layer in result.clusters
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_format_sensitivity(args):
    """Quick format sensitivity check (trailing space vs no space)."""
    from ...introspection import analyze_format_sensitivity

    # Parse base prompts (without trailing space)
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            base_prompts = [line.strip().rstrip() for line in f if line.strip()]
    else:
        base_prompts = [p.strip().rstrip() for p in args.prompts.split("|")]

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None

    print(f"Format sensitivity analysis for {args.model}")
    print(f"Testing {len(base_prompts)} prompts with/without trailing space")

    result = analyze_format_sensitivity(
        model_id=args.model,
        base_prompts=base_prompts,
        layers=layers,
    )

    # Find where format matters
    print("\n=== Where Format Matters ===")
    for layer_idx in result.layers:
        if result.clusters and layer_idx in result.clusters:
            sep = result.clusters[layer_idx].separation_score
            marker = "★" if sep > 0.02 else ""
            print(f"  Layer {layer_idx}: separation = {sep:.4f} {marker}")


def _normalize_number(s: str) -> str:
    """Normalize a number string by removing formatting characters."""
    import re

    # Remove commas, thin spaces (unicode \u202f), regular spaces, and other separators
    return re.sub(r"[\s,\u202f\u00a0]+", "", s)


def _find_answer_onset(output: str, expected_answer: str | None, tokenizer) -> dict:
    """Find where the answer first appears in the output.

    Returns:
        dict with onset_index, onset_token, is_answer_first, answer_found
    """
    if expected_answer is None:
        return {
            "onset_index": None,
            "onset_token": None,
            "is_answer_first": None,
            "answer_found": False,
        }

    # Normalize expected answer (remove any formatting)
    expected_normalized = _normalize_number(expected_answer)

    # Tokenize output
    tokens = []
    output_ids = tokenizer.encode(output)
    for tid in output_ids:
        tokens.append(tokenizer.decode([tid]))

    # Find first position where expected answer appears
    # Check both in individual tokens and cumulative string
    cumulative = ""
    for i, tok in enumerate(tokens):
        cumulative += tok
        # Check if answer appears in cumulative output (normalized)
        if expected_normalized in _normalize_number(cumulative):
            return {
                "onset_index": i,
                "onset_token": tok,
                "is_answer_first": i <= 1,  # Answer in first 2 tokens
                "answer_found": True,
            }

    return {
        "onset_index": None,
        "onset_token": None,
        "is_answer_first": False,
        "answer_found": False,
    }


def _extract_expected_answer(prompt: str) -> str | None:
    """Try to compute expected answer from arithmetic prompt."""
    import re

    # Match patterns like "100 - 37 =" or "156 + 287 ="
    match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*$", prompt.strip())
    if not match:
        return None

    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    try:
        if op == "+":
            return str(a + b)
        elif op == "-":
            return str(a - b)
        elif op == "*":
            return str(a * b)
        elif op == "/":
            return str(a // b)
    except Exception:
        return None
    return None


def introspect_generate(args):
    """Generate multiple tokens to test next-token lock hypothesis.

    Tests whether format issues (like missing trailing space) cause:
    A) Simple next-token lock: model completes format, then computes
    B) Answer-onset routing: model changes WHEN to emit answer
    C) Computation blocked: model can't produce correct answer at all
    """
    from mlx_lm import generate, load

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Load external chat template if available (e.g., GPT-OSS)
    _load_external_chat_template(tokenizer, args.model)

    # Check if using raw mode (no chat template)
    use_raw = getattr(args, "raw", False)
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

    if use_raw:
        print("Mode: RAW (no chat template)")
    elif has_chat_template:
        print("Mode: CHAT (using chat template)")
        print("  Add --raw to test direct prompts without chat formatting")
    else:
        print("Mode: RAW (model has no chat template)")

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # If comparing format, create with/without space variants
    if args.compare_format:
        expanded = []
        for p in prompts:
            base = p.rstrip()
            expanded.append(base)  # without trailing space
            expanded.append(base + " ")  # with trailing space
        prompts = expanded

    print(f"\nGenerating {args.max_tokens} tokens per prompt")
    print(f"Temperature: {args.temperature}")
    print()

    results = []
    for prompt in prompts:
        # Apply chat template unless --raw is specified
        formatted_prompt = prompt
        if not use_raw and has_chat_template:
            formatted_prompt = _apply_chat_template(tokenizer, prompt)
        if args.temperature == 0:
            output = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
        else:
            output = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=args.max_tokens,
                temp=args.temperature,
                verbose=False,
            )

        # Compute expected answer and find onset
        expected = _extract_expected_answer(prompt)
        onset_info = _find_answer_onset(output, expected, tokenizer)

        # Show results
        has_space = prompt.endswith(" ")
        marker = "✓" if has_space else "✗"
        print(f"{marker} {prompt!r}")
        print(f"  → {output!r}")

        # Show answer onset info
        if expected:
            if onset_info["answer_found"]:
                onset_str = f"onset={onset_info['onset_index']}"
                if onset_info["is_answer_first"]:
                    onset_str += " (answer-first)"
                else:
                    onset_str += " (delayed)"
                print(f"  Expected: {expected}, {onset_str}")
            else:
                print(f"  Expected: {expected}, NOT FOUND in output")

        # Token-by-token breakdown if requested
        if args.show_tokens:
            prompt_ids = tokenizer.encode(formatted_prompt)
            output_ids = tokenizer.encode(formatted_prompt + output)
            gen_ids = output_ids[len(prompt_ids) :]

            print("  Tokens: ", end="")
            for i, tid in enumerate(gen_ids[:10]):
                tok = tokenizer.decode([tid])
                # Highlight the onset token
                if expected and onset_info["onset_index"] == i:
                    print(f"[{tok!r}] ", end="")
                else:
                    print(f"{tok!r} ", end="")
            if len(gen_ids) > 10:
                print("...")
            else:
                print()
        print()

        results.append(
            {
                "prompt": prompt,
                "has_trailing_space": has_space,
                "output": output,
                "expected_answer": expected,
                **onset_info,
            }
        )

    # Summary if comparing format
    if args.compare_format and len(results) >= 2:
        print("=== Format Comparison Summary ===")
        print()
        print(f"{'Prompt':<20} {'No-Space':<12} {'With-Space':<12} {'Diagnosis'}")
        print("-" * 70)

        for i in range(0, len(results), 2):
            no_space = results[i]
            with_space = results[i + 1]
            base_prompt = no_space["prompt"][:18]

            # Determine diagnosis based on onset patterns
            ns_onset = no_space.get("onset_index")
            ws_onset = with_space.get("onset_index")
            ns_found = no_space.get("answer_found", False)
            ws_found = with_space.get("answer_found", False)

            # Format onset display
            ns_str = f"onset={ns_onset}" if ns_onset is not None else "not found"
            ws_str = f"onset={ws_onset}" if ws_onset is not None else "not found"

            # Classify the behavior
            if not ns_found and not ws_found:
                diagnosis = "BOTH FAIL"
            elif not ns_found and ws_found:
                diagnosis = "COMPUTE BLOCKED"
            elif ns_found and not ws_found:
                diagnosis = "WEIRD (no-space works?)"
            elif ns_onset == ws_onset or (ns_onset <= 1 and ws_onset <= 1):
                diagnosis = "SPACE-LOCK ONLY"
            elif ns_onset is not None and ws_onset is not None and ns_onset > ws_onset + 2:
                diagnosis = "ONSET ROUTING"
            else:
                diagnosis = "MINOR DIFFERENCE"

            print(f"{base_prompt:<20} {ns_str:<12} {ws_str:<12} {diagnosis}")

        print()
        print("Legend:")
        print("  SPACE-LOCK ONLY  = Just adds space token, same answer timing")
        print("  ONSET ROUTING    = Answer delayed (mode/style switch)")
        print("  COMPUTE BLOCKED  = Answer not produced without space")
        print("  MINOR DIFFERENCE = Small onset difference")

    # Save if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_metacognitive(args):
    """Detect metacognitive strategy switch at a specific layer.

    This tool probes the model's "decision layer" (typically ~70% through the network)
    to detect whether it will use:
    - Direct computation: L24 predicts a digit → answer comes immediately
    - Chain-of-thought: L24 predicts ' ', 'To', 'Let' etc. → reasoning first

    The key insight is that token IDENTITY at the decision layer reveals the
    model's strategy, not just confidence. A digit token means "I know the answer",
    while a non-digit means "I need to think about this".
    """
    import asyncio
    import json
    import re

    from ...introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

    async def run():
        print(f"Loading model: {args.model}")

        async with ModelAnalyzer.from_pretrained(args.model) as analyzer:
            info = analyzer.model_info
            tokenizer = analyzer._tokenizer

            print(f"Model: {info.model_id}")
            print(f"  Layers: {info.num_layers}")

            # Determine decision layer (default: ~70% through network)
            if args.decision_layer:
                decision_layer = args.decision_layer
            else:
                decision_layer = int(info.num_layers * 0.7)

            print(
                f"  Decision layer: {decision_layer} (~{100 * decision_layer / info.num_layers:.0f}% depth)"
            )

            # Check chat template
            has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
            use_raw = getattr(args, "raw", False)

            if use_raw:
                print("  Mode: RAW")
            elif has_chat_template:
                print("  Mode: CHAT")
            else:
                print("  Mode: RAW (no chat template)")

            # Parse problems
            if args.problems.startswith("@"):
                with open(args.problems[1:]) as f:
                    problems = [line.strip() for line in f if line.strip()]
            else:
                problems = [p.strip() for p in args.problems.split("|")]

            # Generate problems if using --generate
            if args.generate:
                import random

                random.seed(args.seed)
                problems = []

                # Generate a variety of arithmetic problems
                for _ in range(args.num_problems):
                    op = random.choice(["+", "-", "*"])
                    if op == "+":
                        a = random.randint(1, 999)
                        b = random.randint(1, 999)
                        expected = a + b
                    elif op == "-":
                        a = random.randint(1, 999)
                        b = random.randint(1, a)  # Ensure positive result
                        expected = a - b
                    else:  # multiplication
                        if random.random() < 0.5:
                            # Simple multiplication
                            a = random.randint(2, 99)
                            b = random.randint(2, 99)
                        else:
                            # Include some squares
                            a = random.randint(2, 99)
                            b = a
                        expected = a * b

                    problems.append(f"{a} {op} {b} =")

            print(f"\nAnalyzing {len(problems)} problems...")
            print()

            # Configure to capture decision layer
            # Include a few layers around it for context
            layers_to_capture = sorted(
                {
                    0,
                    decision_layer - 4 if decision_layer >= 4 else 0,
                    decision_layer,
                    decision_layer + 4
                    if decision_layer + 4 < info.num_layers
                    else info.num_layers - 1,
                    info.num_layers - 1,
                }
            )

            config = AnalysisConfig(
                layer_strategy=LayerStrategy.CUSTOM,
                custom_layers=layers_to_capture,
                top_k=5,
            )

            results = []

            import mlx.core as mx

            for problem in problems:
                # Extract expected answer
                match = re.match(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*$", problem.strip())
                expected = None
                if match:
                    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                    if op == "+":
                        expected = str(a + b)
                    elif op == "-":
                        expected = str(a - b)
                    elif op == "*":
                        expected = str(a * b)
                    elif op == "/":
                        expected = str(a // b) if b != 0 else None

                # Apply chat template if needed
                prompt = problem
                if not use_raw and has_chat_template:
                    prompt = _apply_chat_template(tokenizer, problem)

                # Generate tokens first to find answer position
                # This is key: we need to analyze at the position where the answer starts
                input_ids = mx.array(tokenizer.encode(prompt))[None, :]
                generated_ids = []

                for _ in range(30):  # Generate up to 30 tokens
                    outputs = analyzer._model(input_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    next_token = mx.argmax(logits[:, -1, :], axis=-1)
                    generated_ids.append(int(next_token[0]))

                    if (
                        hasattr(tokenizer, "eos_token_id")
                        and generated_ids[-1] == tokenizer.eos_token_id
                    ):
                        break

                    input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

                generated = tokenizer.decode(generated_ids)

                # Find answer position and extend prompt to that point
                if expected:
                    answer_pos = generated.find(expected)
                    if answer_pos >= 0:
                        prefix = generated[:answer_pos]
                        prompt = prompt + prefix

                # Run analysis at the answer position
                result = await analyzer.analyze(prompt, config)

                # Find decision layer prediction
                decision_pred = None
                for layer_pred in result.layer_predictions:
                    if layer_pred.layer_idx == decision_layer:
                        decision_pred = layer_pred
                        break

                if decision_pred is None:
                    continue

                top_token = decision_pred.predictions[0].token
                top_prob = decision_pred.predictions[0].probability

                # Classify strategy based on token identity
                # Digit tokens indicate direct computation
                # Must be a non-empty digit after stripping
                stripped = top_token.strip()
                is_digit = len(stripped) > 0 and all(c in "0123456789" for c in stripped)
                strategy = "DIRECT" if is_digit else "CoT"

                # Get final prediction
                final_token = result.final_prediction[0].token if result.final_prediction else "?"
                final_prob = (
                    result.final_prediction[0].probability if result.final_prediction else 0.0
                )

                # Check if first digit of expected answer matches
                correct_start = False
                if expected and is_digit:
                    correct_start = expected.startswith(top_token.strip())

                results.append(
                    {
                        "problem": problem.strip(),
                        "expected": expected,
                        "generated": generated[:50],  # First 50 chars of generation
                        "decision_layer": decision_layer,
                        "decision_token": top_token,
                        "decision_prob": top_prob,
                        "strategy": strategy,
                        "is_digit": is_digit,
                        "correct_start": correct_start,
                        "final_token": final_token,
                        "final_prob": final_prob,
                    }
                )

            # Print results table
            print("=" * 90)
            print(
                f"{'Problem':<20} {'Expected':<10} {f'L{decision_layer} Token':<12} {'Conf':<8} {'Strategy':<8} {'Correct?':<8}"
            )
            print("-" * 90)

            for r in results:
                correct_marker = "✓" if r["correct_start"] else ("?" if not r["is_digit"] else "✗")
                print(
                    f"{r['problem']:<20} {r['expected'] or '?':<10} {repr(r['decision_token']):<12} {r['decision_prob']:.2f}     {r['strategy']:<8} {correct_marker:<8}"
                )

            # Summary statistics
            print("=" * 90)
            print("\n=== Strategy Distribution ===")

            direct_count = sum(1 for r in results if r["strategy"] == "DIRECT")
            cot_count = len(results) - direct_count

            print(f"  DIRECT: {direct_count} ({100 * direct_count / len(results):.1f}%)")
            print(f"  CoT:    {cot_count} ({100 * cot_count / len(results):.1f}%)")

            # Accuracy among direct answers
            direct_results = [r for r in results if r["strategy"] == "DIRECT"]
            if direct_results:
                correct_direct = sum(1 for r in direct_results if r["correct_start"])
                print(
                    f"\n  Direct answer accuracy: {correct_direct}/{len(direct_results)} ({100 * correct_direct / len(direct_results):.1f}%)"
                )

            # Confidence analysis
            print("\n=== Confidence Analysis ===")
            direct_probs = [r["decision_prob"] for r in results if r["strategy"] == "DIRECT"]
            cot_probs = [r["decision_prob"] for r in results if r["strategy"] == "CoT"]

            if direct_probs:
                print(f"  DIRECT avg confidence: {sum(direct_probs) / len(direct_probs):.3f}")
            if cot_probs:
                print(f"  CoT avg confidence:    {sum(cot_probs) / len(cot_probs):.3f}")

            # Pattern analysis for multiplication
            print("\n=== Pattern Analysis (Multiplication) ===")
            mult_results = [r for r in results if "*" in r["problem"]]
            if mult_results:
                mult_direct = [r for r in mult_results if r["strategy"] == "DIRECT"]
                mult_cot = [r for r in mult_results if r["strategy"] == "CoT"]
                print(f"  Multiplication: {len(mult_direct)} direct, {len(mult_cot)} CoT")

                # Check for patterns
                squares = [
                    r
                    for r in mult_results
                    if r["problem"].split("*")[0].strip()
                    == r["problem"].split("*")[1].split("=")[0].strip()
                ]
                if squares:
                    square_direct = sum(1 for r in squares if r["strategy"] == "DIRECT")
                    print(f"  Squares (n*n): {square_direct}/{len(squares)} direct")

            # Save if requested
            if args.output:
                output_data = {
                    "model": args.model,
                    "decision_layer": decision_layer,
                    "total_problems": len(results),
                    "direct_count": direct_count,
                    "cot_count": cot_count,
                    "results": results,
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")

    asyncio.run(run())


def introspect_steer(args):
    """Apply activation steering to manipulate model behavior.

    Supports three modes:
    1. Extract direction: Compute steering direction from contrastive prompts
    2. Apply direction: Load pre-computed direction and steer generation
    3. Compare: Show outputs at different steering coefficients
    """
    import json

    import numpy as np

    from ...introspection import ActivationSteering, SteeringConfig

    # Mode 1: Extract direction from contrastive prompts
    if args.extract:
        if not args.positive or not args.negative:
            print("Error: --extract requires --positive and --negative prompts")
            sys.exit(1)

        print(f"Loading model: {args.model}")
        steerer = ActivationSteering.from_pretrained(args.model)

        # Get hidden states at the specified layer
        layer = args.layer or steerer.num_layers // 2
        print(f"\nExtracting direction from layer {layer}...")
        print(f"  Positive: {args.positive!r}")
        print(f"  Negative: {args.negative!r}")

        # Use the internal method to get hidden states
        import mlx.core as mx

        from ...introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        # Get positive activation
        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(args.positive))[None, :]
        hooks.forward(input_ids)
        h_positive = hooks.state.hidden_states[layer][0, -1, :]

        # Get negative activation
        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(args.negative))[None, :]
        hooks.forward(input_ids)
        h_negative = hooks.state.hidden_states[layer][0, -1, :]

        # Compute direction: positive - negative
        direction = h_positive - h_negative
        direction_np = np.array(direction.tolist(), dtype=np.float32)

        # Compute statistics
        norm = float(mx.sqrt(mx.sum(direction * direction)))
        cos_sim = float(
            mx.sum(h_positive * h_negative)
            / (
                mx.sqrt(mx.sum(h_positive * h_positive)) * mx.sqrt(mx.sum(h_negative * h_negative))
                + 1e-8
            )
        )

        print("\nDirection extracted:")
        print(f"  Layer: {layer}")
        print(f"  Norm: {norm:.4f}")
        print(f"  Cosine similarity (pos, neg): {cos_sim:.4f}")
        print(f"  Separation: {1 - cos_sim:.4f}")

        # Save direction
        if args.output:
            output_path = Path(args.output)
            np.savez(
                output_path,
                direction=direction_np,
                layer=layer,
                positive_prompt=args.positive,
                negative_prompt=args.negative,
                model_id=args.model,
                norm=norm,
                cosine_similarity=cos_sim,
            )
            print(f"\nDirection saved to: {output_path}")

        return

    # Mode 2 & 3: Apply steering or compare
    print(f"Loading model: {args.model}")
    steerer = ActivationSteering.from_pretrained(args.model)

    # Load direction - from file, neuron, or contrastive prompts
    neuron_idx = getattr(args, "neuron", None)
    if neuron_idx is not None:
        # Create one-hot direction for single neuron steering
        layer = args.layer or steerer.num_layers // 2
        hidden_size = steerer.model.config.hidden_size
        direction = np.zeros(hidden_size, dtype=np.float32)
        direction[neuron_idx] = 1.0
        print(f"\nSteering neuron {neuron_idx} at layer {layer}")
        print(f"  Hidden size: {hidden_size}")
    elif args.direction:
        direction_path = Path(args.direction)
        if direction_path.suffix == ".npz":
            data = np.load(direction_path, allow_pickle=True)
            direction = data["direction"]
            layer = int(data["layer"]) if "layer" in data else args.layer

            # Show direction metadata
            print(f"\nLoaded direction from: {direction_path}")
            if "positive_prompt" in data:
                print(f"  Positive: {data['positive_prompt']}")
            if "negative_prompt" in data:
                print(f"  Negative: {data['negative_prompt']}")
            print(f"  Layer: {layer}")
            if "norm" in data:
                print(f"  Norm: {float(data['norm']):.4f}")
        elif direction_path.suffix == ".json":
            with open(direction_path) as f:
                data = json.load(f)
            direction = np.array(data["direction"], dtype=np.float32)
            layer = data.get("layer", args.layer)
        else:
            print(f"Error: Unsupported direction format: {direction_path.suffix}")
            sys.exit(1)
    else:
        # Generate direction on-the-fly from positive/negative
        if not args.positive or not args.negative:
            print("Error: Must provide --direction, --neuron, or both --positive and --negative")
            sys.exit(1)

        layer = args.layer or steerer.num_layers // 2

        import mlx.core as mx

        from ...introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[layer], capture_hidden_states=True, positions=PositionSelection.LAST
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(args.positive))[None, :]
        hooks.forward(input_ids)
        h_positive = hooks.state.hidden_states[layer][0, -1, :]

        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[layer], capture_hidden_states=True, positions=PositionSelection.LAST
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(args.negative))[None, :]
        hooks.forward(input_ids)
        h_negative = hooks.state.hidden_states[layer][0, -1, :]

        direction = np.array((h_positive - h_negative).tolist(), dtype=np.float32)
        print(f"Using on-the-fly direction from layer {layer}")

    # Add direction to steerer
    layer = layer if layer is not None else args.layer or steerer.num_layers // 2
    steerer.add_direction(
        layer=layer,
        direction=direction,
        name=args.name or "custom",
        positive_label=args.positive_label or "positive",
        negative_label=args.negative_label or "negative",
    )

    config = SteeringConfig(
        layers=[layer],
        coefficient=args.coefficient,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Mode: Compare coefficients
    if args.compare:
        coefficients = [float(c) for c in args.compare.split(",")]
        print(f"\nComparing steering at coefficients: {coefficients}")

        for prompt in prompts:
            print(f"\n{'=' * 70}")
            print(f"Prompt: {prompt!r}")
            print(f"{'=' * 70}")

            for coef in coefficients:
                output = steerer.generate(prompt, config, coefficient=coef)
                direction_label = (
                    "→ positive" if coef > 0 else "← negative" if coef < 0 else "neutral"
                )
                print(f"\n  Coef {coef:+.1f} ({direction_label}):")
                print(f"    {output!r}")

    # Mode: Single coefficient generation
    else:
        print(f"\nSteering at layer {layer} with coefficient {args.coefficient}")

        results = []
        for prompt in prompts:
            output = steerer.generate(prompt, config)

            print(f"\nPrompt: {prompt!r}")
            print(f"Output: {output!r}")

            results.append(
                {
                    "prompt": prompt,
                    "output": output,
                    "layer": layer,
                    "coefficient": args.coefficient,
                }
            )

        # Save if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


def introspect_arithmetic(args):
    """Run systematic arithmetic study to find emergence layers.

    Tests arithmetic problems of varying difficulty and tracks when
    the correct answer first emerges as the top prediction.
    """
    import asyncio
    import json

    from ...introspection import AnalysisConfig, LayerStrategy, ModelAnalyzer

    async def run():
        print(f"Loading model: {args.model}")

        async with ModelAnalyzer.from_pretrained(args.model) as analyzer:
            info = analyzer.model_info
            tokenizer = analyzer._tokenizer

            print(f"Model: {info.model_id}")
            print(f"  Layers: {info.num_layers}")

            # Check chat template
            has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template
            use_raw = getattr(args, "raw", False)

            if use_raw:
                print("  Mode: RAW")
            elif has_chat_template:
                print("  Mode: CHAT")
            else:
                print("  Mode: RAW (no chat template)")

            # Define test cases
            tests = []

            # Easy addition (1-digit)
            if not args.hard_only:
                tests.extend(
                    [
                        ("1 + 1 = ", "2", "add", "easy", 1),
                        ("2 + 3 = ", "5", "add", "easy", 1),
                        ("4 + 5 = ", "9", "add", "easy", 1),
                        ("7 + 2 = ", "9", "add", "easy", 1),
                    ]
                )

            # Medium addition (2-digit)
            if not args.easy_only:
                tests.extend(
                    [
                        ("12 + 34 = ", "46", "add", "medium", 2),
                        ("25 + 17 = ", "42", "add", "medium", 2),
                        ("99 + 11 = ", "110", "add", "medium", 2),
                    ]
                )

            # Hard addition (3-digit)
            if args.hard_only or not args.easy_only:
                tests.extend(
                    [
                        ("156 + 287 = ", "443", "add", "hard", 3),
                        ("999 + 111 = ", "1110", "add", "hard", 3),
                    ]
                )

            # Easy multiplication
            if not args.hard_only:
                tests.extend(
                    [
                        ("2 * 3 = ", "6", "mul", "easy", 1),
                        ("4 * 5 = ", "20", "mul", "easy", 1),
                        ("7 * 8 = ", "56", "mul", "easy", 1),
                    ]
                )

            # Medium multiplication
            if not args.easy_only:
                tests.extend(
                    [
                        ("12 * 12 = ", "144", "mul", "medium", 2),
                        ("25 * 4 = ", "100", "mul", "medium", 2),
                    ]
                )

            # Hard multiplication
            if args.hard_only or not args.easy_only:
                tests.extend(
                    [
                        ("123 * 456 = ", "56088", "mul", "hard", 3),
                        ("347 * 892 = ", "309524", "mul", "hard", 3),
                    ]
                )

            # Subtraction and division
            if not args.hard_only:
                tests.extend(
                    [
                        ("10 - 3 = ", "7", "sub", "easy", 1),
                        ("100 - 37 = ", "63", "sub", "medium", 2),
                        ("10 / 2 = ", "5", "div", "easy", 1),
                        ("100 / 4 = ", "25", "div", "medium", 2),
                    ]
                )

            if args.quick:
                tests = tests[::3]  # Take every 3rd test

            print(f"\nRunning {len(tests)} arithmetic tests...")

            # Configure to capture all layers
            config = AnalysisConfig(
                layer_strategy=LayerStrategy.ALL,
                top_k=10,
            )

            results = []
            stats = {"by_operation": {}, "by_difficulty": {}, "by_magnitude": {}}

            for prompt, expected, op, difficulty, magnitude in tests:
                # Apply chat template if needed
                analysis_prompt = prompt
                if not use_raw and has_chat_template:
                    analysis_prompt = _apply_chat_template(tokenizer, prompt)

                result = await analyzer.analyze(analysis_prompt, config)

                # Find emergence layer (first layer where first digit of answer is #1)
                first_digit = expected[0]
                emergence_layer = None
                peak_layer = None
                peak_prob = 0.0

                for layer_pred in result.layer_predictions:
                    for pred in layer_pred.predictions:
                        # Check if first digit appears in top prediction
                        if first_digit in pred.token.strip():
                            if pred.probability > peak_prob:
                                peak_prob = pred.probability
                                peak_layer = layer_pred.layer_idx

                        # Check if first digit is top-1
                        if layer_pred.predictions[0].token.strip() == first_digit:
                            if emergence_layer is None:
                                emergence_layer = layer_pred.layer_idx
                            break

                # Check final prediction
                final_token = result.final_prediction[0].token if result.final_prediction else "?"
                correct = first_digit in final_token.strip()

                # Print result
                status = "✓" if correct else "✗"
                emerg_str = f"L{emergence_layer}" if emergence_layer is not None else "never"
                print(
                    f"  {status} {prompt:<16} → {final_token!r:<8} (expected {expected}, emerges @ {emerg_str})"
                )

                # Aggregate stats
                for key, val, stat_dict in [
                    ("by_operation", op, stats["by_operation"]),
                    ("by_difficulty", difficulty, stats["by_difficulty"]),
                    ("by_magnitude", magnitude, stats["by_magnitude"]),
                ]:
                    if val not in stat_dict:
                        stat_dict[val] = {"correct": 0, "total": 0, "emergence_layers": []}
                    stat_dict[val]["total"] += 1
                    if correct:
                        stat_dict[val]["correct"] += 1
                    if emergence_layer is not None:
                        stat_dict[val]["emergence_layers"].append(emergence_layer)

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "operation": op,
                        "difficulty": difficulty,
                        "magnitude": magnitude,
                        "final_prediction": final_token,
                        "correct": correct,
                        "emergence_layer": emergence_layer,
                        "peak_layer": peak_layer,
                        "peak_probability": peak_prob,
                    }
                )

            # Print summary
            print(f"\n{'=' * 60}")
            print("ARITHMETIC STUDY SUMMARY")
            print(f"{'=' * 60}")
            print(f"Model: {info.model_id} ({info.num_layers} layers)")
            print(f"Total tests: {len(tests)}")

            print("\n--- By Operation ---")
            print(f"{'Operation':<10} {'Accuracy':<12} {'Avg Emergence Layer'}")
            print("-" * 45)
            for op, s in stats["by_operation"].items():
                acc = f"{100 * s['correct'] / s['total']:.1f}%" if s["total"] > 0 else "N/A"
                emerg = (
                    f"L{sum(s['emergence_layers']) / len(s['emergence_layers']):.1f}"
                    if s["emergence_layers"]
                    else "N/A"
                )
                print(f"{op:<10} {acc:<12} {emerg}")

            print("\n--- By Difficulty ---")
            print(f"{'Difficulty':<10} {'Accuracy':<12} {'Avg Emergence Layer'}")
            print("-" * 45)
            for diff, s in stats["by_difficulty"].items():
                acc = f"{100 * s['correct'] / s['total']:.1f}%" if s["total"] > 0 else "N/A"
                emerg = (
                    f"L{sum(s['emergence_layers']) / len(s['emergence_layers']):.1f}"
                    if s["emergence_layers"]
                    else "N/A"
                )
                print(f"{diff:<10} {acc:<12} {emerg}")

            print("\n--- By Magnitude ---")
            print(f"{'Digits':<10} {'Accuracy':<12} {'Avg Emergence Layer'}")
            print("-" * 45)
            for mag, s in sorted(stats["by_magnitude"].items()):
                acc = f"{100 * s['correct'] / s['total']:.1f}%" if s["total"] > 0 else "N/A"
                emerg = (
                    f"L{sum(s['emergence_layers']) / len(s['emergence_layers']):.1f}"
                    if s["emergence_layers"]
                    else "N/A"
                )
                print(f"{mag}-digit    {acc:<12} {emerg}")

            # Save if requested
            if args.output:
                output_data = {
                    "model_id": info.model_id,
                    "num_layers": info.num_layers,
                    "total_tests": len(tests),
                    "stats": {
                        k: {
                            kk: {
                                "accuracy": vv["correct"] / vv["total"] if vv["total"] > 0 else 0,
                                "avg_emergence": sum(vv["emergence_layers"])
                                / len(vv["emergence_layers"])
                                if vv["emergence_layers"]
                                else None,
                            }
                            for kk, vv in v.items()
                        }
                        for k, v in stats.items()
                    },
                    "results": results,
                }
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")

    asyncio.run(run())


def introspect_uncertainty(args):
    """Detect model uncertainty using hidden state geometry.

    Uses hidden state distance to "compute center" vs "refusal center"
    to predict whether model is confident about an answer before generation.
    """
    import asyncio
    import json

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info

    async def run():
        print(f"Loading model: {args.model}")

        result = HFLoader.download(args.model)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {args.model}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        num_layers = config.num_hidden_layers
        detection_layer = args.layer or int(num_layers * 0.7)  # ~70% depth

        print(f"  Layers: {num_layers}")
        print(f"  Detection layer: {detection_layer}")

        def get_layers():
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return list(model.model.layers)
            return list(model.layers)

        def get_embed():
            if hasattr(model, "model"):
                return model.model.embed_tokens
            return model.embed_tokens

        def get_scale():
            return getattr(config, "embedding_scale", None)

        def get_hidden_state(prompt: str) -> np.ndarray:
            """Get hidden state at detection layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            layers = get_layers()
            embed = get_embed()
            scale = get_scale()

            h = embed(input_ids)
            if scale:
                h = h * scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

            for idx, lyr in enumerate(layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )

                if idx == detection_layer:
                    return np.array(h[0, -1, :].tolist())

            return np.array(h[0, -1, :].tolist())

        # Calibrate with working vs broken prompts
        working_prompts = [
            "100 - 37 = ",
            "50 + 25 = ",
            "10 * 10 = ",
            "200 - 50 = ",
            "25 * 4 = ",
        ]
        broken_prompts = [
            "100 - 37 =",
            "50 + 25 =",
            "10 * 10 =",
            "200 - 50 =",
            "25 * 4 =",
        ]

        if args.working:
            working_prompts = [x.strip() for x in args.working.split(",")]
        if args.broken:
            broken_prompts = [x.strip() for x in args.broken.split(",")]

        print(
            f"\nCalibrating on {len(working_prompts)} working + {len(broken_prompts)} broken examples..."
        )

        working_hiddens = [get_hidden_state(p) for p in working_prompts]
        broken_hiddens = [get_hidden_state(p) for p in broken_prompts]

        compute_center = np.mean(working_hiddens, axis=0)
        refusal_center = np.mean(broken_hiddens, axis=0)

        separation = np.linalg.norm(compute_center - refusal_center)
        print(f"  Compute-Refusal separation: {separation:.0f}")
        print("  Calibration complete!")

        # Parse test prompts
        if args.prompts.startswith("@"):
            with open(args.prompts[1:]) as f:
                test_prompts = [line.strip() for line in f if line.strip()]
        else:
            test_prompts = [p.strip() for p in args.prompts.split("|")]

        # Run detection
        print(f"\n{'=' * 80}")
        print("UNCERTAINTY DETECTION RESULTS")
        print(f"{'=' * 80}")
        print(f"{'Prompt':<30} {'Score':>8} {'Prediction':<12} {'→Compute':>10} {'→Refusal':>10}")
        print("-" * 80)

        results = []
        for prompt in test_prompts:
            h = get_hidden_state(prompt)

            dist_compute = float(np.linalg.norm(h - compute_center))
            dist_refusal = float(np.linalg.norm(h - refusal_center))

            # Score: positive = closer to compute (confident)
            score = dist_refusal - dist_compute
            prediction = "CONFIDENT" if score > 0 else "UNCERTAIN"

            print(
                f"{prompt:<30} {score:>8.0f} {prediction:<12} {dist_compute:>10.0f} {dist_refusal:>10.0f}"
            )

            results.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "prediction": prediction,
                    "dist_to_compute": dist_compute,
                    "dist_to_refusal": dist_refusal,
                }
            )

        # Summary
        confident = sum(1 for r in results if r["prediction"] == "CONFIDENT")
        uncertain = len(results) - confident
        print("-" * 80)
        print(f"Summary: {confident} confident, {uncertain} uncertain")

        # Save if requested
        if args.output:
            output_data = {
                "model_id": args.model,
                "detection_layer": detection_layer,
                "separation": separation,
                "results": results,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    asyncio.run(run())


def introspect_probe(args):
    """Train linear probe on activations to find task classification layers.

    Uses logistic regression to find which layers can distinguish between
    two types of prompts (e.g., math vs factual).
    """
    import json

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {args.model}")

    result = HFLoader.download(args.model)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {args.model}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    def get_layers():
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return list(model.layers)

    def get_embed():
        if hasattr(model, "model"):
            return model.model.embed_tokens
        return model.embed_tokens

    def get_scale():
        return getattr(config, "embedding_scale", None)

    def get_all_hidden_states(prompt: str) -> list[np.ndarray]:
        """Get hidden state at each layer."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = get_layers()
        embed = get_embed()
        scale = get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        hidden_states = []
        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            hidden_states.append(np.array(h[0, -1, :].tolist()))  # Last token position

        return hidden_states

    # Parse class A and class B prompts
    if args.class_a.startswith("@"):
        with open(args.class_a[1:]) as f:
            class_a_prompts = [line.strip() for line in f if line.strip()]
    else:
        class_a_prompts = [p.strip() for p in args.class_a.split("|")]

    if args.class_b.startswith("@"):
        with open(args.class_b[1:]) as f:
            class_b_prompts = [line.strip() for line in f if line.strip()]
    else:
        class_b_prompts = [p.strip() for p in args.class_b.split("|")]

    print(f"\nClass A ({args.label_a}): {len(class_a_prompts)} prompts")
    print(f"Class B ({args.label_b}): {len(class_b_prompts)} prompts")

    # Collect activations at all layers
    print("\nCollecting activations...")
    all_activations = {layer: [] for layer in range(num_layers)}
    all_labels = []

    for prompt in class_a_prompts:
        hiddens = get_all_hidden_states(prompt)
        for layer, h in enumerate(hiddens):
            all_activations[layer].append(h)
        all_labels.append(1)

    for prompt in class_b_prompts:
        hiddens = get_all_hidden_states(prompt)
        for layer, h in enumerate(hiddens):
            all_activations[layer].append(h)
        all_labels.append(0)

    # Train probes at each layer
    print("\nTraining probes at each layer...")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("ERROR: sklearn required for probing. Install with: pip install scikit-learn")
        return

    y = np.array(all_labels)
    results = []

    for layer in range(num_layers):
        X = np.array(all_activations[layer])

        # Train with cross-validation
        probe = LogisticRegression(max_iter=1000, random_state=42)
        try:
            scores = cross_val_score(probe, X, y, cv=min(5, len(y) // 2))
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
        except ValueError:
            # Not enough samples for CV
            probe.fit(X, y)
            mean_acc = float(probe.score(X, y))
            std_acc = 0.0

        results.append(
            {
                "layer": layer,
                "accuracy": mean_acc,
                "std": std_acc,
            }
        )

    # Find best layer or use specified layer
    specified_layer = getattr(args, "layer", None)
    if specified_layer is not None:
        best_layer = specified_layer
        best = next((r for r in results if r["layer"] == best_layer), results[0])
    else:
        best = max(results, key=lambda x: x["accuracy"])
        best_layer = best["layer"]

    # Print results
    print(f"\n{'=' * 70}")
    print(f"PROBE ACCURACY BY LAYER ({args.label_a} vs {args.label_b})")
    print(f"{'=' * 70}")
    print(f"{'Layer':<8} {'Accuracy':<12} {'Std':<10} {'Bar'}")
    print("-" * 70)

    for r in results:
        bar = "#" * int(r["accuracy"] * 50)
        marker = " ← SELECTED" if r["layer"] == best_layer else ""
        print(f"  L{r['layer']:<5} {r['accuracy']:.3f}        {r['std']:.3f}     {bar}{marker}")

    print("-" * 70)
    if specified_layer is not None:
        print(f"\nSelected layer: L{best_layer} (accuracy: {best['accuracy']:.1%})")
    else:
        print(f"\nBest layer: L{best_layer} (accuracy: {best['accuracy']:.1%})")

    # Train final probe on best layer and extract direction
    X_best = np.array(all_activations[best_layer])
    final_probe = LogisticRegression(max_iter=1000, random_state=42)
    final_probe.fit(X_best, y)

    # Extract direction based on method
    method = getattr(args, "method", "logistic")
    if method == "difference":
        # Difference of means (simpler, often works well)
        class_a_mean = X_best[y == 1].mean(axis=0)
        class_b_mean = X_best[y == 0].mean(axis=0)
        direction = class_a_mean - class_b_mean
        direction = direction / np.linalg.norm(direction)  # Normalize
        print("\nDirection method: difference of means (normalized)")
    else:
        # Logistic regression weights
        direction = final_probe.coef_[0]
        print("\nDirection method: logistic regression weights")

    direction_norm = float(np.linalg.norm(direction))

    # Show projection statistics
    projections = X_best @ (direction / np.linalg.norm(direction))
    class_a_proj = projections[y == 1]
    class_b_proj = projections[y == 0]
    print("\nProjection statistics:")
    print(f"  {args.label_a}: {class_a_proj.mean():+.2f} ± {class_a_proj.std():.2f}")
    print(f"  {args.label_b}: {class_b_proj.mean():+.2f} ± {class_b_proj.std():.2f}")
    separation = abs(class_a_proj.mean() - class_b_proj.mean())
    print(f"  Separation: {separation:.2f}")

    # Find top neurons
    top_k = 10
    top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
    print(f"\nTop {top_k} neurons for {args.label_a} detection:")
    for idx in top_indices:
        print(f"  Neuron {idx}: weight {direction[idx]:.4f}")

    # Test on individual prompts
    if args.test:
        print(f"\n{'=' * 70}")
        print("TEST PREDICTIONS")
        print(f"{'=' * 70}")

        if args.test.startswith("@"):
            with open(args.test[1:]) as f:
                test_prompts = [line.strip() for line in f if line.strip()]
        else:
            test_prompts = [p.strip() for p in args.test.split("|")]

        for prompt in test_prompts:
            hiddens = get_all_hidden_states(prompt)
            h = hiddens[best_layer]
            prob = final_probe.predict_proba([h])[0]
            pred_class = args.label_a if prob[1] > 0.5 else args.label_b
            confidence = max(prob)
            print(f"  {prompt[:40]:<40} → {pred_class} ({confidence:.1%})")

    # Save if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "class_a_label": args.label_a,
            "class_b_label": args.label_b,
            "num_class_a": len(class_a_prompts),
            "num_class_b": len(class_b_prompts),
            "best_layer": best_layer,
            "best_accuracy": best["accuracy"],
            "layer_results": results,
            "direction_norm": direction_norm,
            "top_neurons": [int(i) for i in top_indices],
            "method": method,
            "separation": float(separation),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Save direction vector to npz if requested
    save_direction = getattr(args, "save_direction", None)
    if save_direction:
        np.savez(
            save_direction,
            direction=direction,
            layer=best_layer,
            label_positive=args.label_a,
            label_negative=args.label_b,
            model_id=args.model,
            method=method,
            accuracy=best["accuracy"],
            separation=separation,
            class_a_mean_projection=float(class_a_proj.mean()),
            class_b_mean_projection=float(class_b_proj.mean()),
        )
        print(f"\nDirection vector saved to: {save_direction}")
        print(f"  Shape: {direction.shape}")
        print(f"  Layer: {best_layer}")
        print(f"  Use with: lazarus introspect steer -d {save_direction} ...")


def introspect_neurons(args):
    """Analyze individual neuron activations across prompts.

    Shows how specific neurons fire across different prompts, useful for
    understanding what individual neurons encode after running a probe.

    Supports single layer (--layer) or multiple layers (--layers) for
    cross-layer neuron tracking.
    """
    import json

    import mlx.core as mx
    import numpy as np

    from ...introspection import CaptureConfig, ModelHooks, PositionSelection
    from ...introspection.ablation import AblationStudy

    # Parse layers - support both --layer and --layers
    if args.layers:
        layers_to_analyze = [int(layer.strip()) for layer in args.layers.split(",")]
    elif args.layer is not None:
        layers_to_analyze = [args.layer]
    else:
        print("ERROR: Must specify --layer or --layers")
        return

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    print(f"  Analyzing layers: {layers_to_analyze}")

    # Parse steering config if provided
    steer_config = None
    if getattr(args, "steer", None):
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

        print(
            f"  Steering: {steer_file} @ layer {steer_config['layer']} with coefficient {steer_coef}"
        )

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Parse labels if provided
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split("|")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels for {len(prompts)} prompts, ignoring labels")
            labels = None
    else:
        labels = None

    # Get neurons to analyze
    neurons = []
    neuron_weights = {}
    neuron_stats = {}  # For auto-discover stats

    # Infer auto-discover if labels are provided but no explicit neuron source
    auto_discover = getattr(args, "auto_discover", False)
    if labels and not args.neurons and not args.from_direction:
        auto_discover = True

    if args.from_direction:
        # Load from saved direction file
        data = np.load(args.from_direction)
        direction = data["direction"]
        top_k = args.top_k

        # Get top neurons by absolute weight
        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        neurons = [int(i) for i in top_indices]
        neuron_weights = {int(i): float(direction[i]) for i in top_indices}

        print(f"  Loaded top {top_k} neurons from: {args.from_direction}")
        positive_label = str(data.get("label_positive", "positive"))
        negative_label = str(data.get("label_negative", "negative"))
        print(f"  Direction: {negative_label} → {positive_label}")

    elif auto_discover:
        # Auto-discover neurons by variance/separation across label groups
        # Use first layer for discovery
        discover_layer = layers_to_analyze[0]
        if not labels:
            print("ERROR: --auto-discover requires --labels to group prompts")
            return

        print(f"\nAuto-discovering discriminative neurons at layer {discover_layer}...")
        print("  Collecting full hidden states for all prompts...")

        # Collect full hidden state for each prompt
        full_activations = []
        for prompt in prompts:
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=[discover_layer],
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )

            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            h = hooks.state.hidden_states[discover_layer][0, 0, :]
            h_np = np.array(h.astype(mx.float32), copy=False)
            full_activations.append(h_np)

        full_activations = np.array(full_activations)
        num_neurons = full_activations.shape[1]
        print(f"  Total neurons in layer: {num_neurons}")

        # Group activations by label
        unique_labels_sorted = sorted(set(labels))
        label_groups = {lbl: [] for lbl in unique_labels_sorted}
        for i, lbl in enumerate(labels):
            label_groups[lbl].append(full_activations[i])

        for lbl in unique_labels_sorted:
            label_groups[lbl] = np.array(label_groups[lbl])
            print(f"  Label '{lbl}': {len(label_groups[lbl])} prompts")

        # Calculate separation score for each neuron
        # For multi-class: use max pairwise separation
        # When single samples per group, use range/overall_std as proxy
        single_sample_mode = all(len(label_groups[lbl]) == 1 for lbl in unique_labels_sorted)
        if single_sample_mode:
            print("  Note: Single sample per label - using range-based discrimination")

        neuron_scores = []
        for neuron_idx in range(num_neurons):
            # Get activations for this neuron across all groups
            group_means = []
            group_stds = []
            for lbl in unique_labels_sorted:
                vals = label_groups[lbl][:, neuron_idx]
                group_means.append(np.mean(vals))
                group_stds.append(np.std(vals))

            # Overall std across all prompts (used as normalizer for single-sample mode)
            overall_std = np.std(full_activations[:, neuron_idx])

            # Max pairwise separation (Cohen's d style)
            max_separation = 0.0
            best_pair = None
            for i, lbl1 in enumerate(unique_labels_sorted):
                for j, lbl2 in enumerate(unique_labels_sorted):
                    if i >= j:
                        continue
                    mean_diff = abs(group_means[i] - group_means[j])

                    if single_sample_mode:
                        # With 1 sample per group, use overall_std as normalizer
                        # This finds neurons with large spread across label types
                        if overall_std > 1e-6:
                            separation = mean_diff / overall_std
                        else:
                            separation = 0.0
                    else:
                        # Standard pooled std for multi-sample groups
                        pooled_std = np.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                        if pooled_std > 1e-6:
                            separation = mean_diff / pooled_std
                        else:
                            separation = 0.0

                    if separation > max_separation:
                        max_separation = separation
                        best_pair = (lbl1, lbl2)

            # Also track the range (max - min across group means)
            mean_range = max(group_means) - min(group_means)

            neuron_scores.append(
                {
                    "idx": neuron_idx,
                    "separation": max_separation,
                    "best_pair": best_pair,
                    "overall_std": overall_std,
                    "mean_range": mean_range,
                    "group_means": {
                        lbl: group_means[i] for i, lbl in enumerate(unique_labels_sorted)
                    },
                }
            )

        # Sort by separation score
        neuron_scores.sort(key=lambda x: -x["separation"])

        # Take top-k
        top_k = args.top_k
        top_neurons = neuron_scores[:top_k]

        neurons = [n["idx"] for n in top_neurons]
        neuron_stats = {n["idx"]: n for n in top_neurons}

        print(f"\n  Top {top_k} discriminative neurons:")
        print(f"  {'Neuron':>8} {'Separation':>12} {'Range':>10} {'Best Pair'}")
        print("  " + "-" * 60)
        for n in top_neurons:
            pair_str = f"{n['best_pair'][0]} vs {n['best_pair'][1]}" if n["best_pair"] else "N/A"
            print(f"  {n['idx']:>8} {n['separation']:>12.3f} {n['mean_range']:>10.1f} {pair_str}")

    elif args.neurons:
        # Parse neuron indices
        neurons = [int(n.strip()) for n in args.neurons.split(",")]
        print(f"  Analyzing {len(neurons)} neurons: {neurons}")

    else:
        print("ERROR: Must specify --neurons, --from-direction, or --auto-discover")
        return

    # Parse neuron names if provided
    neuron_names = {}
    if getattr(args, "neuron_names", None):
        names_list = [n.strip() for n in args.neuron_names.split("|")]
        if len(names_list) != len(neurons):
            print(f"Warning: {len(names_list)} names for {len(neurons)} neurons, ignoring names")
        else:
            neuron_names = {neurons[i]: names_list[i] for i in range(len(neurons))}
            print(f"  Neuron names: {neuron_names}")

    def neuron_label(n: int) -> str:
        """Get display label for a neuron (with name if available)."""
        if n in neuron_names:
            return f"N{n}({neuron_names[n][:8]})"
        return f"N{n}"

    def neuron_header(n: int, width: int = 6) -> str:
        """Get header label for a neuron."""
        if n in neuron_names:
            name = neuron_names[n][:width]
            return f"{name:>{width}}"
        return f"N{n:>{width - 1}}"

    steer_msg = " (with steering)" if steer_config else ""
    print(
        f"\nCollecting activations for {len(prompts)} prompts across {len(layers_to_analyze)} layers{steer_msg}..."
    )

    # Collect activations for ALL layers in one pass per prompt
    # Structure: all_activations[layer][prompt_idx] = hidden_state
    all_activations_by_layer = {layer: [] for layer in layers_to_analyze}

    # If steering, we use ActivationSteering to wrap the model layers
    steerer = None
    if steer_config:
        from ...introspection import ActivationSteering

        steerer = ActivationSteering(model, tokenizer)
        steerer.add_direction(
            steer_config["layer"],
            mx.array(steer_config["direction"]),
        )
        # Wrap the steering layer so forward passes include steering
        steerer._wrap_layer(
            steer_config["layer"],
            steer_config["coefficient"],
        )

    try:
        for prompt in prompts:
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=layers_to_analyze,
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )

            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            for layer in layers_to_analyze:
                h = hooks.state.hidden_states[layer][0, 0, :]
                h_np = np.array(h.astype(mx.float32), copy=False)
                all_activations_by_layer[layer].append(h_np)
    finally:
        # Unwrap layers to restore model state
        if steerer:
            steerer._unwrap_layers()

    # Store results for all layers (for JSON output)
    all_layer_results = {}

    # Multi-layer mode: show cross-layer comparison table first
    if len(layers_to_analyze) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-LAYER NEURON TRACKING")
        print(f"{'=' * 80}")

        # Build cross-layer table: rows are prompts, columns are layers
        for neuron in neurons:
            neuron_title = neuron_names.get(neuron, f"Neuron {neuron}")
            print(f"\n--- {neuron_title} (N{neuron}) across layers ---")

            # Header with layers
            header = f"{'Prompt':<20} |"
            for layer in layers_to_analyze:
                header += f" L{layer:>2} |"
            if labels:
                header += " Label"
            print(header)
            print("-" * len(header))

            # Collect values for this neuron across all layers
            cross_layer_vals = []
            for i, prompt in enumerate(prompts):
                row_vals = []
                for layer in layers_to_analyze:
                    val = all_activations_by_layer[layer][i][neuron]
                    row_vals.append(val)
                cross_layer_vals.append(row_vals)

            cross_layer_matrix = np.array(cross_layer_vals)
            vmin, vmax = cross_layer_matrix.min(), cross_layer_matrix.max()

            # Print rows
            for i, prompt in enumerate(prompts):
                short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
                row = f"{short_prompt:<20} |"

                for j, layer in enumerate(layers_to_analyze):
                    val = cross_layer_matrix[i, j]
                    row += f" {val:+4.0f} |"

                if labels and i < len(labels):
                    row += f" {labels[i]}"

                print(row)

            # Summary stats per layer
            print("-" * len(header))
            row = f"{'mean':<20} |"
            for j in range(len(layers_to_analyze)):
                mean_val = cross_layer_matrix[:, j].mean()
                row += f" {mean_val:+4.0f} |"
            print(row)

            row = f"{'std':<20} |"
            for j in range(len(layers_to_analyze)):
                std_val = cross_layer_matrix[:, j].std()
                row += f" {std_val:4.0f} |"
            print(row)

            row = f"{'range':<20} |"
            for j in range(len(layers_to_analyze)):
                range_val = cross_layer_matrix[:, j].max() - cross_layer_matrix[:, j].min()
                row += f" {range_val:4.0f} |"
            print(row)

    # Now show per-layer detailed analysis
    for layer in layers_to_analyze:
        all_activations = all_activations_by_layer[layer]

        # Build activation matrix
        activation_matrix = np.array([[act[n] for n in neurons] for act in all_activations])

        # Print results as ASCII heatmap
        print(f"\n{'=' * 80}")
        print(f"NEURON ACTIVATION MAP AT LAYER {layer}")
        print(f"{'=' * 80}")

        # Header - use names if available
        header = f"{'Prompt':<20} |"
        for n in neurons:
            if n in neuron_names:
                name = neuron_names[n][:6]
                header += f" {name:>6} |"
            else:
                header += f" N{n:>5} |"
        if labels:
            header += " Label"
        print(header)
        print("-" * len(header))

        # Find min/max for heatmap scaling
        vmin, vmax = activation_matrix.min(), activation_matrix.max()

        # Rows
        for i, prompt in enumerate(prompts):
            short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
            row = f"{short_prompt:<20} |"

            for j, n in enumerate(neurons):
                val = activation_matrix[i, j]
                row += f" {val:+6.0f} |"

            if labels and i < len(labels):
                row += f" {labels[i]}"

            print(row)

        print("-" * 80)

        # ASCII heatmap visualization (only for single-layer or first layer to avoid too much output)
        if len(layers_to_analyze) == 1:
            print(f"\n{'=' * 80}")
            print("ASCII HEATMAP (░ = low, ▒ = medium, ▓ = high, █ = max)")
            print(f"{'=' * 80}")

            # Normalize for heatmap
            norm_matrix = (activation_matrix - vmin) / (vmax - vmin + 1e-8)

            header = f"{'Prompt':<20} |"
            for n in neurons:
                if n in neuron_names:
                    name = neuron_names[n][:6]
                    header += f" {name:>6} |"
                else:
                    header += f" N{n:>5} |"
            print(header)
            print("-" * len(header))

            heatmap_chars = " ░▒▓█"
            for i, prompt in enumerate(prompts):
                short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
                row = f"{short_prompt:<20} |"

                for j, n in enumerate(neurons):
                    norm_val = norm_matrix[i, j]
                    char_idx = min(int(norm_val * 4), 4)
                    char = heatmap_chars[char_idx]
                    row += f"  {char * 4}  |"

                if labels and i < len(labels):
                    row += f" {labels[i]}"

                print(row)

        # Neuron statistics
        print(f"\n--- Layer {layer} Statistics ---")

        for j, n in enumerate(neurons):
            vals = activation_matrix[:, j]
            extra_str = ""

            # Show weight from direction file
            if n in neuron_weights:
                w = neuron_weights[n]
                direction_str = "→ POSITIVE detector" if w > 0 else "→ NEGATIVE detector"
                extra_str = f" (weight: {w:+.3f}) {direction_str}"

            # Show separation score from auto-discover
            if n in neuron_stats:
                sep = neuron_stats[n]["separation"]
                pair = neuron_stats[n].get("best_pair")
                pair_str = f"{pair[0]} vs {pair[1]}" if pair else ""
                extra_str = f" (separation: {sep:.3f}) {pair_str}"

            # Include name if available
            name_str = f" [{neuron_names[n]}]" if n in neuron_names else ""
            print(
                f"Neuron {n:4d}{name_str}: min={vals.min():+7.1f}, max={vals.max():+7.1f}, "
                f"mean={vals.mean():+7.1f}, std={vals.std():6.1f}{extra_str}"
            )

        # Correlation with labels if provided (only for single-layer to avoid verbosity)
        if labels and len(layers_to_analyze) == 1:
            print(f"\n{'=' * 80}")
            print("LABEL CORRELATION")
            print(f"{'=' * 80}")

            unique_labels_for_corr = sorted(set(labels))
            for label in unique_labels_for_corr:
                mask = np.array([lbl == label for lbl in labels])
                if mask.sum() > 0:
                    print(f"\n{label}:")
                    for j, n in enumerate(neurons):
                        mean_val = activation_matrix[mask, j].mean()
                        name_str = f" [{neuron_names[n]}]" if n in neuron_names else ""
                        print(f"  Neuron {n:4d}{name_str}: mean={mean_val:+7.1f}")

        # Store for output
        all_layer_results[layer] = {
            "activations": activation_matrix.tolist(),
            "stats": {
                str(n): {
                    "min": float(activation_matrix[:, j].min()),
                    "max": float(activation_matrix[:, j].max()),
                    "mean": float(activation_matrix[:, j].mean()),
                    "std": float(activation_matrix[:, j].std()),
                }
                for j, n in enumerate(neurons)
            },
        }

    # Save if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "layers": layers_to_analyze,
            "neurons": neurons,
            "neuron_names": neuron_names if neuron_names else None,
            "prompts": prompts,
            "labels": labels,
            "by_layer": all_layer_results,
            "neuron_weights": neuron_weights,
            "auto_discovered": getattr(args, "auto_discover", False),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_directions(args):
    """Compare multiple direction vectors for orthogonality.

    Loads saved direction vectors (from 'introspect probe --save-direction')
    and computes the cosine similarity matrix between all pairs.

    Orthogonal directions (cosine ~ 0) indicate independent features.
    """
    import json
    from pathlib import Path

    import numpy as np

    files = args.files
    threshold = args.threshold

    if len(files) < 2:
        print("ERROR: Need at least 2 direction files to compare")
        return

    # Load all direction vectors
    directions = []
    names = []
    metadata = []

    print("Loading direction vectors...")
    for fpath in files:
        path = Path(fpath)
        if not path.exists():
            print(f"  ERROR: File not found: {fpath}")
            return

        data = np.load(fpath, allow_pickle=True)
        direction = data["direction"]

        # Get name from file or metadata
        if "label_positive" in data and "label_negative" in data:
            pos = str(data["label_positive"])
            neg = str(data["label_negative"])
            name = f"{neg}→{pos}"
        else:
            name = path.stem

        layer = int(data["layer"]) if "layer" in data else "?"
        method = str(data["method"]) if "method" in data else "?"
        accuracy = float(data["accuracy"]) if "accuracy" in data else None

        directions.append(direction)
        names.append(name)
        metadata.append(
            {
                "file": str(path),
                "name": name,
                "layer": layer,
                "method": method,
                "accuracy": accuracy,
                "dim": len(direction),
            }
        )

        acc_str = f", acc={accuracy:.1%}" if accuracy else ""
        print(f"  {name}: layer={layer}, dim={len(direction)}{acc_str}")

    # Check dimensions match
    dims = [len(d) for d in directions]
    if len(set(dims)) > 1:
        print(f"\nWARNING: Dimension mismatch: {dims}")
        print("  Directions from different models/layers may not be comparable")

    # Compute cosine similarity matrix
    n = len(directions)
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if dims[i] == dims[j]:
                d_i = directions[i] / (np.linalg.norm(directions[i]) + 1e-8)
                d_j = directions[j] / (np.linalg.norm(directions[j]) + 1e-8)
                similarity[i, j] = np.dot(d_i, d_j)
            else:
                similarity[i, j] = float("nan")

    # Print results
    print(f"\n{'=' * 80}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'=' * 80}")
    print(f"(Threshold for 'orthogonal': |cos| < {threshold})")
    print()

    # Header
    max_name_len = max(len(n) for n in names)
    col_width = max(8, max_name_len + 2)

    header = " " * (max_name_len + 2)
    for name in names:
        header += f"{name:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(n):
            val = similarity[i, j]
            if np.isnan(val):
                row += f"{'N/A':>{col_width}}"
            elif i == j:
                row += f"{'1.000':>{col_width}}"
            else:
                row += f"{val:>{col_width}.3f}"
        print(row)

    # ASCII heatmap
    print(f"\n{'=' * 80}")
    print("ORTHOGONALITY HEATMAP")
    print(f"{'=' * 80}")
    print("(■ = aligned, ▓ = correlated, ▒ = weak, ░ = near-orthogonal, · = orthogonal)")
    print()

    header = " " * (max_name_len + 2)
    for name in names:
        short = name[:6] if len(name) > 6 else name
        header += f"{short:>8}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(n):
            val = abs(similarity[i, j])
            if np.isnan(val):
                char = "?"
            elif i == j:
                char = "■"
            elif val > 0.7:
                char = "■"
            elif val > 0.5:
                char = "▓"
            elif val > 0.3:
                char = "▒"
            elif val > threshold:
                char = "░"
            else:
                char = "·"
            row += f"{char:>8}"
        print(row)

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # Get off-diagonal elements
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(similarity[i, j]):
                off_diag.append((names[i], names[j], similarity[i, j]))

    if off_diag:
        orthogonal_pairs = [(a, b, s) for a, b, s in off_diag if abs(s) < threshold]
        aligned_pairs = [(a, b, s) for a, b, s in off_diag if abs(s) > 0.5]
        correlated_pairs = [(a, b, s) for a, b, s in off_diag if threshold <= abs(s) <= 0.5]

        print(f"\nTotal pairs: {len(off_diag)}")
        print(f"Orthogonal (|cos| < {threshold}): {len(orthogonal_pairs)}")
        print(f"Correlated ({threshold} <= |cos| <= 0.5): {len(correlated_pairs)}")
        print(f"Aligned (|cos| > 0.5): {len(aligned_pairs)}")

        if orthogonal_pairs:
            print("\nOrthogonal pairs (independent dimensions):")
            for a, b, s in sorted(orthogonal_pairs, key=lambda x: abs(x[2])):
                print(f"  {a} ⊥ {b} (cos = {s:+.3f})")

        if aligned_pairs:
            print("\nAligned pairs (potentially redundant):")
            for a, b, s in sorted(aligned_pairs, key=lambda x: -abs(x[2])):
                print(f"  {a} ≈ {b} (cos = {s:+.3f})")

        # Overall assessment
        mean_abs_sim = np.mean([abs(s) for _, _, s in off_diag])
        print(f"\nMean |cosine similarity|: {mean_abs_sim:.3f}")

        if mean_abs_sim < threshold:
            print("Assessment: Directions are largely ORTHOGONAL (independent features)")
        elif mean_abs_sim < 0.3:
            print("Assessment: Directions are mostly INDEPENDENT with some correlation")
        elif mean_abs_sim < 0.5:
            print("Assessment: Directions show MODERATE correlation")
        else:
            print("Assessment: Directions are HIGHLY correlated (may be redundant)")

    # Save if requested
    if args.output:
        output_data = {
            "files": [str(f) for f in files],
            "names": names,
            "metadata": metadata,
            "similarity_matrix": similarity.tolist(),
            "threshold": threshold,
            "pairs": [
                {"a": a, "b": b, "cosine": s, "orthogonal": abs(s) < threshold}
                for a, b, s in off_diag
            ]
            if off_diag
            else [],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_activation_cluster(args):
    """Visualize activation clusters using PCA.

    Projects hidden states to 2D to see if different prompt types cluster separately.

    Supports two syntaxes:
    1. Legacy two-class: --class-a "prompts" --class-b "prompts" --label-a X --label-b Y
    2. Multi-class: --prompts "p1|p2|p3" --label L1 --prompts "p4|p5" --label L2 ...
    """
    import json

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info

    # Parse prompts with labels - support both legacy and new syntax
    prompts = []
    labels = []

    # Check for new multi-class syntax
    if args.prompt_groups and args.labels:
        if len(args.prompt_groups) != len(args.labels):
            print(
                f"ERROR: Number of --prompts ({len(args.prompt_groups)}) must match "
                f"number of --label ({len(args.labels)})"
            )
            return

        for prompt_group, label in zip(args.prompt_groups, args.labels):
            if prompt_group.startswith("@"):
                with open(prompt_group[1:]) as f:
                    group_prompts = [line.strip() for line in f if line.strip()]
            else:
                group_prompts = [p.strip() for p in prompt_group.split("|")]
            prompts.extend(group_prompts)
            labels.extend([label] * len(group_prompts))

    # Fall back to legacy two-class syntax
    elif args.class_a or args.class_b:
        if args.class_a:
            if args.class_a.startswith("@"):
                with open(args.class_a[1:]) as f:
                    class_a_prompts = [line.strip() for line in f if line.strip()]
            else:
                class_a_prompts = [p.strip() for p in args.class_a.split("|")]
            prompts.extend(class_a_prompts)
            labels.extend([args.label_a] * len(class_a_prompts))

        if args.class_b:
            if args.class_b.startswith("@"):
                with open(args.class_b[1:]) as f:
                    class_b_prompts = [line.strip() for line in f if line.strip()]
            else:
                class_b_prompts = [p.strip() for p in args.class_b.split("|")]
            prompts.extend(class_b_prompts)
            labels.extend([args.label_b] * len(class_b_prompts))
    else:
        print("ERROR: Must provide either --prompts/--label pairs or --class-a/--class-b")
        return

    if len(prompts) < 2:
        print("ERROR: Need at least 2 prompts for clustering")
        return

    print(f"Loading model: {args.model}")

    result = HFLoader.download(args.model)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {args.model}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    # Parse layers - support single int or comma-separated
    if args.layer is not None:
        if "," in str(args.layer):
            target_layers = [int(layer.strip()) for layer in str(args.layer).split(",")]
        else:
            target_layers = [int(args.layer)]
    else:
        target_layers = [int(num_layers * 0.5)]

    print(f"  Target layer(s): {target_layers}")

    def get_layers():
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return list(model.layers)

    def get_embed():
        if hasattr(model, "model"):
            return model.model.embed_tokens
        return model.embed_tokens

    def get_scale():
        return getattr(config, "embedding_scale", None)

    def get_hidden_at_layer(prompt: str, layer: int) -> np.ndarray:
        """Get hidden state at specific layer."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = get_layers()
        embed = get_embed()
        scale = get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            if idx == layer:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    # Show what we're clustering
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    print(f"\nClasses ({len(unique_labels)}):")
    for label in unique_labels:
        count = labels.count(label)
        print(f"  {label}: {count} prompts")

    print(
        f"\nCollecting activations for {len(prompts)} prompts across {len(target_layers)} layer(s)..."
    )

    # Collect activations for all layers at once (more efficient)
    activations_by_layer = {layer: [] for layer in target_layers}

    for prompt in prompts:
        # Get hidden states at all target layers in one forward pass
        for target_layer in target_layers:
            h = get_hidden_at_layer(prompt, target_layer)
            activations_by_layer[target_layer].append(h)

    # PCA import
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("ERROR: sklearn required. Install with: pip install scikit-learn")
        return

    # Create symbols for each label (use first letter, or A, B, C... if collision)
    symbols = {}
    used_symbols = set()
    fallback_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    fallback_idx = 0

    for label in unique_labels:
        symbol = label[0].upper()
        if symbol in used_symbols:
            while fallback_idx < len(fallback_symbols):
                symbol = fallback_symbols[fallback_idx]
                fallback_idx += 1
                if symbol not in used_symbols:
                    break
        symbols[label] = symbol
        used_symbols.add(symbol)

    # Process each layer
    all_results = {}
    for target_layer in target_layers:
        X = np.array(activations_by_layer[target_layer])

        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)

        # Compute cluster statistics
        cluster_stats = {}

        for label in unique_labels:
            mask = np.array([lbl == label for lbl in labels])
            points = projected[mask]
            center = np.mean(points, axis=0)
            cluster_stats[label] = {
                "center": center,
                "count": int(np.sum(mask)),
                "points": points,
            }

        # Compute pairwise separations for multi-class
        separations = {}
        for i, l1 in enumerate(unique_labels):
            for l2 in unique_labels[i + 1 :]:
                c1 = cluster_stats[l1]["center"]
                c2 = cluster_stats[l2]["center"]
                sep = float(np.linalg.norm(c1 - c2))
                separations[(l1, l2)] = sep

        # Store results
        all_results[target_layer] = {
            "pca": pca,
            "projected": projected,
            "cluster_stats": cluster_stats,
            "separations": separations,
        }

        # Print results
        print(f"\n{'=' * 70}")
        print(f"ACTIVATION CLUSTERS AT LAYER {target_layer}")
        print(f"{'=' * 70}")
        print(
            f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}"
        )

        if separations:
            print("\nCluster separations:")
            for (l1, l2), sep in sorted(separations.items(), key=lambda x: -x[1]):
                print(f"  {l1} <-> {l2}: {sep:.2f}")

        print(f"\n{'Label':<15} {'Count':<8} {'Center (PC1, PC2)'}")
        print("-" * 50)
        for label, stats in cluster_stats.items():
            print(
                f"{label:<15} {stats['count']:<8} ({stats['center'][0]:.2f}, {stats['center'][1]:.2f})"
            )

        # ASCII scatter plot
        print(f"\n{'=' * 70}")
        print(f"SCATTER PLOT (ASCII) - Layer {target_layer}")
        print(f"{'=' * 70}")

        # Normalize to grid
        x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
        y_min, y_max = projected[:, 1].min(), projected[:, 1].max()

        grid_width = 60
        grid_height = 20
        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        for i, (x, y) in enumerate(projected):
            gx = int((x - x_min) / (x_max - x_min + 1e-6) * (grid_width - 1))
            gy = int((y - y_min) / (y_max - y_min + 1e-6) * (grid_height - 1))
            gy = grid_height - 1 - gy  # Flip y
            symbol = symbols.get(labels[i], "?")
            grid[gy][gx] = symbol

        for row in grid:
            print("  " + "".join(row))

        print(f"\n  Legend: {', '.join(f'{s}={lbl}' for lbl, s in symbols.items())}")

        # Save matplotlib plot if requested
        if getattr(args, "save_plot", None):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 8))

                # Color palette for multiple classes
                colors = plt.cm.tab10.colors

                for i, label in enumerate(unique_labels):
                    mask = np.array([lbl == label for lbl in labels])
                    points = projected[mask]
                    color = colors[i % len(colors)]
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=[color],
                        label=f"{label} (n={int(np.sum(mask))})",
                        alpha=0.7,
                        s=100,
                    )
                    # Mark cluster center
                    center = cluster_stats[label]["center"]
                    ax.scatter(
                        center[0],
                        center[1],
                        c=[color],
                        marker="x",
                        s=200,
                        linewidths=3,
                    )

                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                ax.set_title(f"Activation Clusters at Layer {target_layer}\n{args.model}")
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                # For multiple layers, add layer number to filename
                if len(target_layers) > 1:
                    base, ext = (
                        args.save_plot.rsplit(".", 1)
                        if "." in args.save_plot
                        else (args.save_plot, "png")
                    )
                    plot_path = f"{base}_L{target_layer}.{ext}"
                else:
                    plot_path = args.save_plot
                plt.savefig(plot_path, dpi=150)
                print(f"\nPlot saved to: {plot_path}")
                plt.close()

            except ImportError:
                print("\nWARNING: matplotlib not available. Install with: pip install matplotlib")

    # Save JSON if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "layers": target_layers,
            "prompts": prompts,
            "labels": labels,
            "results_by_layer": {
                layer: {
                    "explained_variance": res["pca"].explained_variance_ratio_.tolist(),
                    "separations": {f"{l1}__{l2}": s for (l1, l2), s in res["separations"].items()},
                    "projected": res["projected"].tolist(),
                    "cluster_stats": {
                        label: {
                            "center": stats["center"].tolist(),
                            "count": stats["count"],
                        }
                        for label, stats in res["cluster_stats"].items()
                    },
                }
                for layer, res in all_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_memory(args):
    """Extract memory organization structure for facts.

    Analyzes how facts are stored in model memory by examining
    neighborhood activation patterns - what other facts co-activate
    when retrieving a specific fact.

    Reveals:
    - Memory organization (row vs column based, clusters)
    - Asymmetry (A->B vs B->A retrieval differences)
    - Attractor nodes (frequently co-activated facts)
    - Difficulty patterns (which facts are hardest)
    """
    import json
    from collections import defaultdict

    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ...inference.loader import DType, HFLoader
    from ...models_v2.families.registry import detect_model_family, get_family_info

    # Built-in fact generators
    def generate_multiplication_facts():
        """Generate single-digit multiplication facts."""
        facts = []
        for a in range(2, 10):
            for b in range(2, 10):
                facts.append(
                    {
                        "query": f"{a}*{b}=",
                        "answer": str(a * b),
                        "operand_a": a,
                        "operand_b": b,
                        "category": f"{a}x",  # Row category
                        "category_alt": f"x{b}",  # Column category
                    }
                )
        return facts

    def generate_addition_facts():
        """Generate single-digit addition facts."""
        facts = []
        for a in range(1, 10):
            for b in range(1, 10):
                facts.append(
                    {
                        "query": f"{a}+{b}=",
                        "answer": str(a + b),
                        "operand_a": a,
                        "operand_b": b,
                        "category": f"{a}+",
                        "category_alt": f"+{b}",
                    }
                )
        return facts

    def generate_capital_facts():
        """Generate country capital facts."""
        capitals = [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("UK", "London"),
            ("Japan", "Tokyo"),
            ("China", "Beijing"),
            ("India", "Delhi"),
            ("Brazil", "Brasilia"),
            ("Russia", "Moscow"),
            ("Canada", "Ottawa"),
            ("Australia", "Canberra"),
            ("Mexico", "Mexico City"),
            ("Egypt", "Cairo"),
            ("South Africa", "Pretoria"),
            ("Argentina", "Buenos Aires"),
            ("Poland", "Warsaw"),
            ("Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"),
            ("Sweden", "Stockholm"),
            ("Norway", "Oslo"),
            ("Denmark", "Copenhagen"),
            ("Finland", "Helsinki"),
            ("Greece", "Athens"),
            ("Turkey", "Ankara"),
            ("Iran", "Tehran"),
            ("Iraq", "Baghdad"),
            ("Saudi Arabia", "Riyadh"),
            ("Israel", "Jerusalem"),
            ("Thailand", "Bangkok"),
        ]
        facts = []
        for country, capital in capitals:
            # Get continent/region for categorization
            region = (
                "Europe"
                if country
                in [
                    "France",
                    "Germany",
                    "Italy",
                    "Spain",
                    "UK",
                    "Poland",
                    "Netherlands",
                    "Belgium",
                    "Sweden",
                    "Norway",
                    "Denmark",
                    "Finland",
                    "Greece",
                ]
                else "Asia"
                if country
                in [
                    "Japan",
                    "China",
                    "India",
                    "Turkey",
                    "Iran",
                    "Iraq",
                    "Saudi Arabia",
                    "Israel",
                    "Thailand",
                ]
                else "Americas"
                if country in ["Brazil", "Canada", "Mexico", "Argentina"]
                else "Other"
            )
            facts.append(
                {
                    "query": f"The capital of {country} is",
                    "answer": capital,
                    "country": country,
                    "category": region,
                }
            )
        return facts

    def generate_element_facts():
        """Generate periodic table element facts."""
        elements = [
            (1, "H", "Hydrogen"),
            (2, "He", "Helium"),
            (3, "Li", "Lithium"),
            (4, "Be", "Beryllium"),
            (5, "B", "Boron"),
            (6, "C", "Carbon"),
            (7, "N", "Nitrogen"),
            (8, "O", "Oxygen"),
            (9, "F", "Fluorine"),
            (10, "Ne", "Neon"),
            (11, "Na", "Sodium"),
            (12, "Mg", "Magnesium"),
            (13, "Al", "Aluminum"),
            (14, "Si", "Silicon"),
            (15, "P", "Phosphorus"),
            (16, "S", "Sulfur"),
            (17, "Cl", "Chlorine"),
            (18, "Ar", "Argon"),
            (19, "K", "Potassium"),
            (20, "Ca", "Calcium"),
        ]
        facts = []
        for num, symbol, name in elements:
            period = 1 if num <= 2 else 2 if num <= 10 else 3
            facts.append(
                {
                    "query": f"Element {num} is",
                    "answer": name,
                    "number": num,
                    "symbol": symbol,
                    "category": f"Period {period}",
                }
            )
        return facts

    # Load facts
    fact_type = args.facts
    if fact_type.startswith("@"):
        # Load from file
        with open(fact_type[1:]) as f:
            facts = json.load(f)
    elif fact_type == "multiplication":
        facts = generate_multiplication_facts()
    elif fact_type == "addition":
        facts = generate_addition_facts()
    elif fact_type == "capitals":
        facts = generate_capital_facts()
    elif fact_type == "elements":
        facts = generate_element_facts()
    else:
        print(f"ERROR: Unknown fact type: {fact_type}")
        print("Use: multiplication, addition, capitals, elements, or @file.json")
        return

    print(f"Loading model: {args.model}")

    result = HFLoader.download(args.model)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {args.model}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    target_layer = args.layer if args.layer is not None else int(num_layers * 0.8)
    top_k = args.top_k

    print(f"  Layers: {num_layers}")
    print(f"  Target layer: {target_layer}")
    print(f"  Facts to analyze: {len(facts)}")
    print(f"  Top-k predictions: {top_k}")

    def get_layers():
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return list(model.layers)

    def get_embed():
        if hasattr(model, "model"):
            return model.model.embed_tokens
        return model.embed_tokens

    def get_norm():
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        if hasattr(model, "norm"):
            return model.norm
        return None

    def get_lm_head():
        if hasattr(model, "lm_head"):
            return model.lm_head
        return None

    def get_scale():
        return getattr(config, "embedding_scale", None)

    def get_predictions_at_layer(prompt: str, layer: int, k: int) -> list:
        """Get top-k predictions at specific layer using logit lens."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = get_layers()
        embed = get_embed()
        norm = get_norm()
        lm_head = get_lm_head()
        scale = get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            if idx == layer:
                break

        # Apply norm and get logits
        if norm is not None:
            h = norm(h)
        if lm_head is not None:
            outputs = lm_head(h)
            # Handle HeadOutput wrapper vs raw logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            # Tied embeddings
            logits = h @ embed.weight.T

        # Get last position probabilities
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        top_indices = mx.argsort(probs)[-k:][::-1]
        top_probs = probs[top_indices]

        predictions = []
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            token = tokenizer.decode([idx])
            predictions.append(
                {
                    "token": token,
                    "token_id": idx,
                    "prob": prob,
                }
            )

        return predictions

    # Build answer vocabulary for categorization
    answer_vocab = {fact["answer"]: fact for fact in facts}

    print(f"\nAnalyzing {len(facts)} facts...")

    # Collect results
    results = []
    for i, fact in enumerate(facts):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(facts)}...")

        query = fact["query"]
        correct_answer = fact["answer"]

        predictions = get_predictions_at_layer(query, target_layer, top_k)

        # Find correct answer rank
        correct_rank = None
        correct_prob = None
        for j, pred in enumerate(predictions):
            if pred["token"].strip() == correct_answer or correct_answer in pred["token"]:
                correct_rank = j + 1
                correct_prob = pred["prob"]
                break

        # Categorize predictions
        neighborhood = {
            "correct_rank": correct_rank,
            "correct_prob": correct_prob,
            "same_category": [],
            "same_category_alt": [],
            "other_answers": [],
            "non_answers": [],
        }

        for pred in predictions:
            token = pred["token"].strip()
            if token == correct_answer:
                continue

            # Check if this is a known answer
            if token in answer_vocab:
                other_fact = answer_vocab[token]
                # Check category match
                if "category" in fact and "category" in other_fact:
                    if fact["category"] == other_fact["category"]:
                        neighborhood["same_category"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                    elif "category_alt" in fact and fact.get("category_alt") == other_fact.get(
                        "category_alt"
                    ):
                        neighborhood["same_category_alt"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                    else:
                        neighborhood["other_answers"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                else:
                    neighborhood["other_answers"].append(
                        {
                            "answer": token,
                            "prob": pred["prob"],
                        }
                    )
            else:
                # Not a known answer
                neighborhood["non_answers"].append(
                    {
                        "token": token,
                        "prob": pred["prob"],
                    }
                )

        results.append(
            {
                **fact,
                "predictions": predictions[:10],  # Save top 10 for reference
                "neighborhood": neighborhood,
            }
        )

    # Aggregate analysis
    print(f"\n{'=' * 70}")
    print(f"MEMORY STRUCTURE ANALYSIS: {fact_type}")
    print(f"{'=' * 70}")

    # 1. Overall accuracy
    correct_top1 = sum(1 for r in results if r["neighborhood"]["correct_rank"] == 1)
    correct_top5 = sum(
        1
        for r in results
        if r["neighborhood"]["correct_rank"] and r["neighborhood"]["correct_rank"] <= 5
    )
    not_found = sum(1 for r in results if r["neighborhood"]["correct_rank"] is None)

    print("\n1. RETRIEVAL ACCURACY")
    print(f"   Top-1: {correct_top1}/{len(results)} ({100 * correct_top1 / len(results):.1f}%)")
    print(f"   Top-5: {correct_top5}/{len(results)} ({100 * correct_top5 / len(results):.1f}%)")
    print(
        f"   Not in top-{top_k}: {not_found}/{len(results)} ({100 * not_found / len(results):.1f}%)"
    )

    # 2. Category analysis (if applicable)
    if "category" in facts[0]:
        print("\n2. ACCURACY BY CATEGORY")
        categories = list({f["category"] for f in facts})
        for cat in sorted(categories):
            cat_facts = [r for r in results if r["category"] == cat]
            cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
            cat_avg_prob = np.mean([r["neighborhood"]["correct_prob"] or 0 for r in cat_facts])
            print(f"   {cat}: {cat_top1}/{len(cat_facts)} top-1, avg_prob={cat_avg_prob:.3f}")

    # 3. Neighborhood composition
    print("\n3. NEIGHBORHOOD COMPOSITION")
    total_same_cat = sum(len(r["neighborhood"]["same_category"]) for r in results)
    total_same_cat_alt = sum(len(r["neighborhood"]["same_category_alt"]) for r in results)
    total_other = sum(len(r["neighborhood"]["other_answers"]) for r in results)
    total_non = sum(len(r["neighborhood"]["non_answers"]) for r in results)

    print(f"   Same category (primary): {total_same_cat}")
    if total_same_cat_alt > 0:
        print(f"   Same category (alt): {total_same_cat_alt}")
    print(f"   Other known answers: {total_other}")
    print(f"   Non-answer tokens: {total_non}")

    # 4. Attractor analysis
    print("\n4. ATTRACTOR NODES (most frequently co-activated)")
    answer_counts = defaultdict(int)
    answer_probs = defaultdict(list)
    for r in results:
        for cat in ["same_category", "same_category_alt", "other_answers"]:
            for item in r["neighborhood"][cat]:
                answer_counts[item["answer"]] += 1
                answer_probs[item["answer"]].append(item["prob"])

    top_attractors = sorted(answer_counts.items(), key=lambda x: -x[1])[:10]
    for answer, count in top_attractors:
        avg_prob = np.mean(answer_probs[answer])
        print(f"   '{answer}': appears {count} times, avg_prob={avg_prob:.4f}")

    # 5. Hardest facts
    print("\n5. HARDEST FACTS (lowest retrieval rank)")
    sorted_by_difficulty = sorted(results, key=lambda x: x["neighborhood"]["correct_rank"] or 999)
    for r in sorted_by_difficulty[-10:]:
        rank = r["neighborhood"]["correct_rank"] or f">{top_k}"
        prob = r["neighborhood"]["correct_prob"] or 0
        print(f"   {r['query'][:30]:<30} -> {r['answer']}: rank={rank}, prob={prob:.4f}")

    # 6. Asymmetry analysis (for facts with operand_a and operand_b)
    if "operand_a" in facts[0] and "operand_b" in facts[0]:
        print("\n6. ASYMMETRY ANALYSIS (A op B vs B op A)")
        asymmetries = []
        for r in results:
            a, b = r["operand_a"], r["operand_b"]
            if a >= b:
                continue
            # Find reverse
            reverse = next(
                (x for x in results if x["operand_a"] == b and x["operand_b"] == a), None
            )
            if reverse:
                rank_ab = r["neighborhood"]["correct_rank"] or 999
                rank_ba = reverse["neighborhood"]["correct_rank"] or 999
                prob_ab = r["neighborhood"]["correct_prob"] or 0
                prob_ba = reverse["neighborhood"]["correct_prob"] or 0
                if abs(rank_ab - rank_ba) > 2 or abs(prob_ab - prob_ba) > 0.05:
                    asymmetries.append(
                        {
                            "a": a,
                            "b": b,
                            "rank_ab": rank_ab,
                            "rank_ba": rank_ba,
                            "prob_ab": prob_ab,
                            "prob_ba": prob_ba,
                        }
                    )

        if asymmetries:
            asymmetries.sort(key=lambda x: abs(x["rank_ab"] - x["rank_ba"]), reverse=True)
            for asym in asymmetries[:10]:
                a, b = asym["a"], asym["b"]
                print(f"   {a}*{b}: rank={asym['rank_ab']}, prob={asym['prob_ab']:.3f}")
                print(f"   {b}*{a}: rank={asym['rank_ba']}, prob={asym['prob_ba']:.3f}")
                print(f"      Δrank={asym['rank_ab'] - asym['rank_ba']:+d}")
                print()
        else:
            print("   No significant asymmetries found")

    # 7. Row vs Column bias (for operand-based facts)
    if "category" in facts[0] and "category_alt" in facts[0]:
        print("\n7. ORGANIZATION BIAS (primary vs alt category)")
        row_bias = 0
        col_bias = 0
        neutral = 0
        for r in results:
            n_primary = len(r["neighborhood"]["same_category"])
            n_alt = len(r["neighborhood"]["same_category_alt"])
            if n_primary > n_alt:
                row_bias += 1
            elif n_alt > n_primary:
                col_bias += 1
            else:
                neutral += 1
        print(f"   Primary category bias: {row_bias}")
        print(f"   Alt category bias: {col_bias}")
        print(f"   Neutral: {neutral}")

    # 8. Memorization classification (if --classify flag)
    if getattr(args, "classify", False):
        print("\n8. MEMORIZATION CLASSIFICATION")
        print("-" * 50)

        memorized = []  # rank 1, prob > 0.1
        partial = []  # rank 2-5, prob > 0.01
        weak = []  # rank 6-15, prob > 0.001
        not_memorized = []  # rank > 15 or prob < 0.001

        for r in results:
            query = r["query"]
            answer = r["answer"]
            rank = r["neighborhood"]["correct_rank"]
            prob = r["neighborhood"]["correct_prob"] or 0

            if rank == 1 and prob > 0.1:
                memorized.append((query, answer, rank, prob))
            elif rank and rank <= 5 and prob > 0.01:
                partial.append((query, answer, rank, prob))
            elif rank and rank <= 15 and prob > 0.001:
                weak.append((query, answer, rank, prob))
            else:
                not_memorized.append((query, answer, rank, prob))

        print(f"\n   MEMORIZED ({len(memorized)} facts) - rank 1, prob > 10%")
        for q, a, r, p in sorted(memorized, key=lambda x: -x[3])[:5]:
            print(f"      {q:<20} = {a:<6} prob={p:.1%}")

        print(f"\n   PARTIALLY MEMORIZED ({len(partial)} facts) - rank 2-5, prob > 1%")
        for q, a, r, p in sorted(partial, key=lambda x: -x[3])[:5]:
            print(f"      {q:<20} = {a:<6} rank={r}, prob={p:.1%}")

        print(f"\n   WEAK ({len(weak)} facts) - rank 6-15, prob > 0.1%")
        for q, a, r, p in sorted(weak, key=lambda x: x[2] if x[2] else 999)[:5]:
            print(f"      {q:<20} = {a:<6} rank={r}, prob={p:.2%}")

        print(f"\n   NOT MEMORIZED ({len(not_memorized)} facts) - rank > 15 or prob < 0.1%")
        for q, a, r, p in sorted(not_memorized, key=lambda x: x[2] if x[2] else 999)[:5]:
            rank_str = str(r) if r else f">{top_k}"
            print(f"      {q:<20} = {a:<6} rank={rank_str}, prob={p:.3%}")

        # Summary bar
        print("\n   Summary: ", end="")
        print(
            f"[{'#' * len(memorized)}{'~' * len(partial)}{'?' * len(weak)}{'.' * len(not_memorized)}]"
        )
        print("            # memorized  ~ partial  ? weak  . not memorized")

    # Save results
    if args.output:
        output_data = {
            "model_id": args.model,
            "fact_type": fact_type,
            "layer": target_layer,
            "num_facts": len(facts),
            "accuracy": {
                "top1": correct_top1,
                "top5": correct_top5,
                "not_found": not_found,
            },
            "attractors": [
                {"answer": a, "count": c, "avg_prob": float(np.mean(answer_probs[a]))}
                for a, c in top_attractors
            ],
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    # Save plot
    if getattr(args, "save_plot", None):
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Accuracy by category
            if "category" in facts[0]:
                ax = axes[0, 0]
                categories = sorted({f["category"] for f in facts})
                cat_accuracy = []
                for cat in categories:
                    cat_facts = [r for r in results if r["category"] == cat]
                    cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
                    cat_accuracy.append(100 * cat_top1 / len(cat_facts))
                ax.bar(categories, cat_accuracy)
                ax.set_ylabel("Top-1 Accuracy (%)")
                ax.set_title("Accuracy by Category")
                ax.tick_params(axis="x", rotation=45)

            # Plot 2: Rank distribution
            ax = axes[0, 1]
            ranks = [r["neighborhood"]["correct_rank"] or top_k + 1 for r in results]
            ax.hist(ranks, bins=range(1, top_k + 3), edgecolor="black")
            ax.set_xlabel("Correct Answer Rank")
            ax.set_ylabel("Count")
            ax.set_title("Rank Distribution")

            # Plot 3: Top attractors
            ax = axes[1, 0]
            if top_attractors:
                answers = [a for a, _ in top_attractors[:10]]
                counts = [c for _, c in top_attractors[:10]]
                ax.barh(answers, counts)
                ax.set_xlabel("Co-activation Count")
                ax.set_title("Top Attractor Nodes")

            # Plot 4: Probability vs rank
            ax = axes[1, 1]
            probs = [r["neighborhood"]["correct_prob"] or 0 for r in results]
            ranks_plot = [r["neighborhood"]["correct_rank"] or top_k + 1 for r in results]
            ax.scatter(ranks_plot, probs, alpha=0.5)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Probability")
            ax.set_title("Probability vs Rank")

            plt.suptitle(f"Memory Structure: {fact_type} @ Layer {target_layer}\n{args.model}")
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            print(f"Plot saved to: {args.save_plot}")
            plt.close()

        except ImportError:
            print("WARNING: matplotlib not available for plotting")


def introspect_circuit_capture(args):
    """Capture circuit activations and extract computational directions.

    Runs prompts through the model and saves hidden state activations at
    specific layers. Extracts directions that encode the computation.

    Modes:
    1. Basic capture: Save raw activations for each prompt
    2. Direction extraction (--extract-direction): Find the direction that
       encodes the result value using linear regression

    Example:
        # Basic capture
        lazarus introspect circuit capture \\
            -m model \\
            --prompts "7*4=|6*8=|9*3=" \\
            --layer 19 \\
            -o mult_circuit.npz

        # Extract direction that encodes result
        lazarus introspect circuit capture \\
            -m model \\
            --prompts "7*4=|6*8=|9*3=" \\
            --results "28|48|27" \\
            --layer 19 \\
            --extract-direction \\
            -o mult_direction.npz
    """
    import re

    import mlx.core as mx
    import numpy as np

    from ...introspection import CaptureConfig, ModelHooks, PositionSelection
    from ...introspection.ablation import AblationStudy

    layer = args.layer
    if layer is None:
        print("ERROR: Must specify --layer for circuit capture")
        return

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    print(f"  Capturing at layer: {layer}")

    # Parse prompts - format: "7*4=28|6*8=48" or "7*4=|6*8=" with separate --results
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            raw_prompts = [line.strip() for line in f if line.strip()]
    else:
        raw_prompts = [p.strip() for p in args.prompts.split("|")]

    print(f"  Prompts: {len(raw_prompts)}")

    # Parse results if provided separately
    explicit_results = None
    if getattr(args, "results", None):
        if args.results.startswith("@"):
            with open(args.results[1:]) as f:
                explicit_results = [int(line.strip()) for line in f if line.strip()]
        else:
            explicit_results = [int(r.strip()) for r in args.results.split("|")]
        if len(explicit_results) != len(raw_prompts):
            print(f"ERROR: {len(explicit_results)} results for {len(raw_prompts)} prompts")
            return

    # Parse each prompt to extract operands, operator, and result
    # Regex for "A op B = C" or "A op B =" format
    pattern_with_result = re.compile(r"(\d+)\s*([+\-*/x×])\s*(\d+)\s*=\s*(\d+)")
    pattern_no_result = re.compile(r"(\d+)\s*([+\-*/x×])\s*(\d+)\s*=")

    parsed = []
    for i, prompt in enumerate(raw_prompts):
        match = pattern_with_result.search(prompt)
        if match:
            a, op, b, result = match.groups()
            parsed.append(
                {
                    "prompt": prompt,
                    "operand_a": int(a),
                    "operand_b": int(b),
                    "operator": op,
                    "result": int(result),
                }
            )
        else:
            match = pattern_no_result.search(prompt)
            if match:
                a, op, b = match.groups()
                # Use explicit result if provided
                result = explicit_results[i] if explicit_results else None
                parsed.append(
                    {
                        "prompt": prompt,
                        "operand_a": int(a),
                        "operand_b": int(b),
                        "operator": op,
                        "result": result,
                    }
                )
            else:
                # Non-arithmetic prompt
                parsed.append(
                    {
                        "prompt": prompt,
                        "operand_a": None,
                        "operand_b": None,
                        "operator": None,
                        "result": explicit_results[i] if explicit_results else None,
                    }
                )

    # Collect activations
    activations = []
    print("\nCapturing activations...")

    for item in parsed:
        prompt = item["prompt"]
        hooks = ModelHooks(model, model_config=config)
        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )

        input_ids = tokenizer.encode(prompt, return_tensors="np")
        hooks.forward(mx.array(input_ids))

        h = hooks.state.hidden_states[layer][0, 0, :]
        h_np = np.array(h.astype(mx.float32), copy=False)
        activations.append(h_np)

        # Print progress
        if item["result"] is not None:
            if item["operand_a"] is not None:
                print(
                    f"  {item['operand_a']} {item['operator']} {item['operand_b']} = {item['result']}"
                )
            else:
                print(f"  {prompt[:30]}... -> {item['result']}")
        else:
            print(f"  {prompt[:40]}...")

    activations = np.array(activations)

    # Extract direction if requested
    extract_direction = getattr(args, "extract_direction", False)
    direction = None
    direction_stats = {}

    arithmetic_items = [p for p in parsed if p["result"] is not None]
    if len(arithmetic_items) >= 2:
        print("\nAnalyzing linear predictability of results from activations...")

        try:
            from sklearn.linear_model import Ridge

            X = np.array([activations[i] for i, p in enumerate(parsed) if p["result"] is not None])
            y = np.array([p["result"] for p in parsed if p["result"] is not None])

            # Use Ridge regression to find the direction
            reg = Ridge(alpha=1.0)
            reg.fit(X, y)

            # The coefficients form the "result direction"
            direction = reg.coef_.astype(np.float32)
            direction_norm = np.linalg.norm(direction)

            # Normalize to unit vector
            direction_unit = direction / (direction_norm + 1e-8)

            # Test predictions
            preds = reg.predict(X)
            mae = np.mean(np.abs(preds - y))
            r2 = 1 - np.sum((y - preds) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-8)

            print(f"  Direction norm: {direction_norm:.4f}")
            print(f"  R² score: {r2:.3f}")
            print(f"  MAE: {mae:.2f}")

            # Show predictions
            print(f"\n  {'Actual':<10} {'Predicted':<10} {'Error':<10}")
            print("  " + "-" * 30)
            for actual, pred in zip(y, preds):
                error = pred - actual
                print(f"  {actual:<10} {pred:<10.1f} {error:+.1f}")

            # Compute projection statistics
            projections = X @ direction_unit
            print(f"\n  Projection range: {projections.min():.1f} to {projections.max():.1f}")
            print(f"  Result range: {y.min()} to {y.max()}")

            # Compute scale factor (how much to scale direction to get result)
            scale = np.mean(y / (projections + 1e-8))
            print(f"  Scale factor: {scale:.2f}")

            direction_stats = {
                "norm": float(direction_norm),
                "r2": float(r2),
                "mae": float(mae),
                "scale": float(scale),
                "intercept": float(reg.intercept_),
            }

        except ImportError:
            print("  (sklearn not available for direction extraction)")

    # Save circuit (--save / -o)
    output_path = getattr(args, "save", None) or getattr(args, "output", None)
    if output_path:
        save_data = {
            "activations": activations,
            "layer": layer,
            "model_id": args.model,
            "prompts": [p["prompt"] for p in parsed],
            "operands_a": [p["operand_a"] for p in parsed],
            "operands_b": [p["operand_b"] for p in parsed],
            "operators": [p["operator"] for p in parsed],
            "results": [p["result"] for p in parsed],
        }

        # Add direction if extracted
        if direction is not None and extract_direction:
            save_data["direction"] = direction
            save_data["direction_stats"] = direction_stats
            print("\n  Direction extracted and saved!")

        np.savez(output_path, **save_data)
        print(f"\nCircuit saved to: {output_path}")
        print(f"  Activations shape: {activations.shape}")
        if direction is not None:
            print(f"  Direction shape: {direction.shape}")
        print(f"  Use with: lazarus introspect circuit invoke -c {output_path} ...")
    else:
        print("\nWARNING: No output file specified. Use -o/--save to save the circuit.")


def introspect_circuit_invoke(args):
    """Invoke circuit with new operands.

    Given a captured circuit (from 'circuit capture'), computes new results.

    Methods:
    - steer: Use extracted direction to steer the model (most accurate)
    - linear: Weighted average based on inverse distance in operand space
    - extrapolate: Linear regression on operands to predict result

    Example:
        # Using steering (requires --extract-direction during capture)
        lazarus introspect circuit invoke \\
            -m model \\
            -c mult_circuit.npz \\
            --prompts "5*6=|8*9=|12*3=" \\
            --method steer

        # Using interpolation (no model needed)
        lazarus introspect circuit invoke \\
            -c mult_circuit.npz \\
            --operands "5,6|8,9|12,3" \\
            --method linear
    """
    import json

    import mlx.core as mx
    import numpy as np

    circuit_path = args.circuit
    if not circuit_path:
        print("ERROR: Must specify --circuit file")
        return

    # Load circuit
    print(f"Loading circuit: {circuit_path}")
    data = np.load(circuit_path, allow_pickle=True)

    layer = int(data["layer"])
    model_id = str(data["model_id"])
    prompts = list(data["prompts"])
    operands_a = list(data["operands_a"])
    operands_b = list(data["operands_b"])
    operators = list(data["operators"])
    results = list(data["results"])

    # Check for extracted direction
    has_direction = "direction" in data
    if has_direction:
        direction = data["direction"]
        direction_stats = data["direction_stats"].item() if "direction_stats" in data else {}
        print(f"  Has extracted direction: yes (R²={direction_stats.get('r2', '?'):.3f})")
    else:
        direction = None
        direction_stats = {}
        print("  Has extracted direction: no")

    print(f"  Model: {model_id}")
    print(f"  Layer: {layer}")
    print(f"  Known computations: {len(prompts)}")

    # Filter to valid arithmetic entries
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    if not valid_indices:
        print("ERROR: No valid arithmetic entries in circuit")
        return

    valid_a = [operands_a[i] for i in valid_indices]
    valid_b = [operands_b[i] for i in valid_indices]
    valid_results = [results[i] for i in valid_indices]
    valid_ops = [operators[i] for i in valid_indices]

    # Determine operator (assume all same)
    op = valid_ops[0] if valid_ops[0] else "*"
    print(f"  Operator: {op}")

    method = args.method

    # Compute true results for comparison
    def compute_true(a, b, op):
        if op in ["*", "x", "×"]:
            return a * b
        elif op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "/":
            return a / b if b != 0 else float("nan")
        return None

    results_table = []

    # Method: steer - use direction to steer model generation
    if method == "steer":
        if not has_direction:
            print("ERROR: 'steer' method requires --extract-direction during capture")
            return

        model_to_use = args.model or model_id
        print(f"\nLoading model: {model_to_use}")

        from ...introspection import ActivationSteering, SteeringConfig

        steerer = ActivationSteering.from_pretrained(model_to_use)

        # Add the circuit direction
        steerer.add_direction(
            layer=layer,
            direction=mx.array(direction),
            name="circuit",
            positive_label="high",
            negative_label="low",
        )

        # Parse prompts for steering
        if getattr(args, "invoke_prompts", None):
            if args.invoke_prompts.startswith("@"):
                with open(args.invoke_prompts[1:]) as f:
                    test_prompts = [line.strip() for line in f if line.strip()]
            else:
                test_prompts = [p.strip() for p in args.invoke_prompts.split("|")]
        elif getattr(args, "operands", None):
            # Convert operands to prompts
            if args.operands.startswith("@"):
                with open(args.operands[1:]) as f:
                    operand_strs = [line.strip() for line in f if line.strip()]
            else:
                operand_strs = [o.strip() for o in args.operands.split("|")]
            test_prompts = []
            for s in operand_strs:
                parts = s.split(",")
                if len(parts) == 2:
                    a, b = int(parts[0].strip()), int(parts[1].strip())
                    test_prompts.append(f"{a}{op}{b}=")
        else:
            print("ERROR: 'steer' method requires --prompts or --operands")
            return

        print(f"\n{'=' * 70}")
        print("CIRCUIT STEERING RESULTS")
        print(f"{'=' * 70}")

        config = SteeringConfig(
            layers=[layer],
            coefficient=0.0,  # Will vary this
            max_new_tokens=5,
            temperature=0.0,
        )

        for prompt in test_prompts:
            # Parse the prompt to get expected result
            import re

            match = re.search(r"(\d+)\s*([+\-*/x×])\s*(\d+)", prompt)
            if match:
                a, op_char, b = match.groups()
                a, b = int(a), int(b)
                expected = compute_true(a, b, op_char)
            else:
                expected = None

            # Generate with different steering strengths
            print(f"\nPrompt: {prompt!r}" + (f" (expected: {expected})" if expected else ""))

            for coef in [0, 10, 20, 50]:
                output = steerer.generate(prompt, config, coefficient=coef)
                print(f"  coef={coef:3d}: {output!r}")

            results_table.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                }
            )

    # Method: linear or interpolate or extrapolate (original behavior)
    else:
        # Parse operands
        if not getattr(args, "operands", None):
            print("ERROR: Must specify --operands for non-steer methods")
            return

        if args.operands.startswith("@"):
            with open(args.operands[1:]) as f:
                operand_strs = [line.strip() for line in f if line.strip()]
        else:
            operand_strs = [o.strip() for o in args.operands.split("|")]

        new_operands = []
        for s in operand_strs:
            parts = s.split(",")
            if len(parts) == 2:
                new_operands.append((int(parts[0].strip()), int(parts[1].strip())))
            else:
                print(f"  Warning: Invalid operand format '{s}', expected 'A,B'")

        if not new_operands:
            print("ERROR: No valid operand pairs")
            return

        print(f"\nPredicting {len(new_operands)} new computations using method: {method}")

        known_operands = np.array(list(zip(valid_a, valid_b)), dtype=np.float32)
        known_results = np.array(valid_results, dtype=np.float32)

        if method == "linear":
            for a, b in new_operands:
                query = np.array([a, b], dtype=np.float32)
                distances = np.linalg.norm(known_operands - query, axis=1)

                if np.min(distances) < 1e-6:
                    idx = np.argmin(distances)
                    pred_result = known_results[idx]
                else:
                    weights = 1.0 / (distances + 1e-6)
                    weights = weights / np.sum(weights)
                    pred_result = np.sum(weights * known_results)

                true_result = compute_true(a, b, op)
                results_table.append(
                    {
                        "operand_a": a,
                        "operand_b": b,
                        "predicted": float(pred_result),
                        "true": true_result,
                        "error": float(pred_result) - true_result if true_result else None,
                    }
                )

        elif method == "extrapolate":
            try:
                from sklearn.linear_model import LinearRegression

                reg = LinearRegression()
                reg.fit(known_operands, known_results)

                for a, b in new_operands:
                    query = np.array([[a, b]], dtype=np.float32)
                    pred_result = float(reg.predict(query)[0])
                    true_result = compute_true(a, b, op)
                    results_table.append(
                        {
                            "operand_a": a,
                            "operand_b": b,
                            "predicted": pred_result,
                            "true": true_result,
                            "error": pred_result - true_result if true_result else None,
                        }
                    )
            except ImportError:
                print("ERROR: sklearn required for extrapolate method")
                return

        elif method == "interpolate":
            k = min(3, len(valid_results))

            for a, b in new_operands:
                query = np.array([a, b], dtype=np.float32)
                distances = np.linalg.norm(known_operands - query, axis=1)
                nearest_idx = np.argsort(distances)[:k]

                nearest_dist = distances[nearest_idx]
                if np.min(nearest_dist) < 1e-6:
                    idx = nearest_idx[np.argmin(nearest_dist)]
                    pred_result = known_results[idx]
                else:
                    weights = 1.0 / (nearest_dist + 1e-6)
                    weights = weights / np.sum(weights)
                    pred_result = np.sum(weights * known_results[nearest_idx])

                true_result = compute_true(a, b, op)
                results_table.append(
                    {
                        "operand_a": a,
                        "operand_b": b,
                        "predicted": float(pred_result),
                        "true": true_result,
                        "error": float(pred_result) - true_result if true_result else None,
                    }
                )

        else:
            print(f"ERROR: Unknown method '{method}'")
            return

        # Print results for non-steer methods
        print(f"\n{'=' * 60}")
        print("CIRCUIT INVOCATION RESULTS")
        print(f"{'=' * 60}")
        print(f"{'Expression':<15} {'Predicted':<12} {'True':<12} {'Error':<10}")
        print("-" * 60)

        for r in results_table:
            expr = f"{r['operand_a']} {op} {r['operand_b']}"
            pred_str = f"{r['predicted']:.1f}"
            true_str = str(r["true"]) if r["true"] is not None else "N/A"
            error_str = f"{r['error']:+.1f}" if r["error"] is not None else "N/A"
            print(f"{expr:<15} {pred_str:<12} {true_str:<12} {error_str:<10}")

        errors = [r["error"] for r in results_table if r.get("error") is not None]
        if errors:
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(np.array(errors) ** 2))
            print("-" * 60)
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"Root Mean Square Error: {rmse:.2f}")

    # Save if requested
    if args.output:
        output_data = {
            "circuit": circuit_path,
            "method": method,
            "operator": op,
            "predictions": results_table,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_circuit_test(args):
    """Test if a trained circuit generalizes to new inputs.

    Does the model actually KNOW multiplication? Or did it just memorize?

    This command applies the circuit you extracted to NEW inputs
    and shows whether it still works.

    Example (one command):
        lazarus introspect circuit test \\
            -c mult_circuit.npz \\
            -m openai/gpt-oss-20b \\
            -p "1*1=|11*11=|10*5=" \\
            -r "1|121|50"

    Or with pre-captured activations:
        lazarus introspect circuit test \\
            -c mult_circuit.npz \\
            -t test_activations.npz
    """
    import json
    import re

    import mlx.core as mx
    import numpy as np

    from ...introspection import CaptureConfig, ModelHooks, PositionSelection

    # Load trained circuit (with direction)
    circuit_path = args.circuit
    print(f"Loading circuit: {circuit_path}")
    trained = np.load(circuit_path, allow_pickle=True)

    if "direction" not in trained:
        print("ERROR: Circuit must have a direction (use --extract-direction during capture)")
        return

    direction = trained["direction"]
    train_activations = trained["activations"]
    train_results = np.array([r for r in trained["results"] if r is not None])
    train_prompts = (
        {str(p).strip().rstrip("=") for p in trained["prompts"]} if "prompts" in trained else set()
    )
    layer = int(trained["layer"])
    model_id = str(trained["model_id"])

    # Compute scale/offset from training data
    train_scores = train_activations @ direction
    coeffs = np.polyfit(train_scores, train_results, 1)
    scale, offset = coeffs[0], coeffs[1]

    # Verify training fit
    train_preds = train_scores * scale + offset
    train_mae = np.mean(np.abs(train_preds - train_results))

    print(f"  Layer: {layer}")
    print(f"  Training examples: {len(train_results)}")
    print(f"  Training error: {train_mae:.4f}")

    # Get test activations - either from file or capture on the fly
    test_path = getattr(args, "test_activations", None)
    test_prompts_arg = getattr(args, "prompts", None)

    if test_path:
        # Load pre-captured activations
        print(f"\nLoading test data: {test_path}")
        test_data = np.load(test_path, allow_pickle=True)
        test_activations = test_data["activations"]
        test_results = np.array([r for r in test_data["results"] if r is not None])
        test_prompts = list(test_data["prompts"])

    elif test_prompts_arg:
        # Capture activations on the fly
        model_to_use = getattr(args, "model", None) or model_id

        print(f"\nLoading model: {model_to_use}")
        from ...inference.loader import HFLoader
        from ...models_v2.families.registry import detect_model_family, get_family_info

        result = HFLoader.download(model_to_use)
        model_path = result.model_path

        # Load config
        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            print(f"ERROR: Unsupported model: {model_to_use}")
            return

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)
        HFLoader.apply_weights_to_model(model, model_path, config)
        tokenizer = HFLoader.load_tokenizer(model_path)

        # Parse prompts and results
        test_prompts = [p.strip() for p in test_prompts_arg.split("|")]
        results_arg = getattr(args, "results", None)
        if results_arg:
            test_results = np.array([int(r.strip()) for r in results_arg.split("|")])
        else:
            # Try to parse results from prompts (e.g., "1*1=1")
            test_results = []
            pattern = re.compile(r"=\s*(\d+)")
            for p in test_prompts:
                match = pattern.search(p)
                if match:
                    test_results.append(int(match.group(1)))
                else:
                    print(f"ERROR: Cannot parse result from '{p}'. Use --results.")
                    return
            test_results = np.array(test_results)

        print(f"  Capturing {len(test_prompts)} test examples...")

        # Capture activations
        test_activations = []
        for prompt in test_prompts:
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=[layer],
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )

            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            h = hooks.state.hidden_states[layer][0, 0, :]
            h_np = np.array(h.astype(mx.float32), copy=False)
            test_activations.append(h_np)

        test_activations = np.array(test_activations)

    else:
        print("ERROR: Provide either --test-activations or --model with --prompts")
        return

    # Apply TRAINED direction to test activations
    test_scores = test_activations @ direction
    test_preds = test_scores * scale + offset

    # Check for overlap with training data
    overlapping = []
    novel = []
    for i, prompt in enumerate(test_prompts):
        prompt_clean = prompt.strip().rstrip("=")
        if prompt_clean in train_prompts:
            overlapping.append(i)
        else:
            novel.append(i)

    # Print results
    print(f"\nTesting {len(test_results)} inputs...")
    print(f"\n{'Input':<12} {'Expected':<10} {'Predicted':<12} {'Error':<10} {'Status':<12}")
    print("-" * 62)

    errors = []
    novel_errors = []
    results_table = []
    for i, prompt in enumerate(test_prompts):
        true_val = test_results[i]
        pred = test_preds[i]
        error = pred - true_val
        errors.append(abs(error))

        # Check if this was in training
        prompt_clean = prompt.rstrip("=")
        if i in overlapping:
            status = "(in training)"
        else:
            status = ""
            novel_errors.append(abs(error))

        print(f"{prompt_clean:<12} {true_val:<10} {pred:<12.1f} {error:+.1f}      {status}")

        results_table.append(
            {
                "prompt": prompt,
                "true": float(true_val),
                "predicted": float(pred),
                "error": float(error),
                "in_training": i in overlapping,
            }
        )

    print("-" * 62)

    # Verdict depends on whether we have novel examples
    if len(novel) == 0:
        print(f"\n⚠️  WARNING: All {len(test_prompts)} test inputs were in the training data!")
        print("This doesn't test generalization - try inputs the model hasn't seen.")
        print("\nSuggested test (two-digit numbers not in training):")
        print(f"  lazarus introspect circuit test -c {circuit_path} -m {model_id} \\")
        print('    -p "12*13=|25*4=|11*11=" -r "156|100|121"')
    elif len(overlapping) > 0:
        novel_mae = np.mean(novel_errors)
        print(
            f"\n⚠️  {len(overlapping)} of {len(test_prompts)} inputs were in training data (marked above)"
        )
        print(f"Average error on NOVEL inputs only: {novel_mae:.1f}")
        if novel_mae > 10:
            print("\nThe circuit FAILS on new inputs.")
            print("It memorized the training examples - it didn't learn the operation.")
        elif novel_mae > 3:
            print("\nThe circuit PARTIALLY works on new inputs.")
            print("Some generalization, but not reliable.")
        else:
            print("\nThe circuit WORKS on new inputs!")
            print("It learned the operation, not just memorized examples.")
    else:
        mae = np.mean(errors)
        print(f"Average error: {mae:.1f}")
        if mae > 10:
            print("\nThe circuit FAILS on new inputs.")
            print("It memorized the training examples - it didn't learn the operation.")
        elif mae > 3:
            print("\nThe circuit PARTIALLY works on new inputs.")
            print("Some generalization, but not reliable.")
        else:
            print("\nThe circuit WORKS on new inputs!")
            print("It learned the operation, not just memorized examples.")

    # Save if requested
    if args.output:
        output_data = {
            "circuit": circuit_path,
            "training_samples": len(train_results),
            "training_error": float(train_mae),
            "test_samples": len(test_results),
            "test_error": float(mae),
            "predictions": results_table,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_circuit_view(args):
    """View the contents of a captured circuit file.

    Displays circuit metadata, captured prompts/results, and optionally
    formats the data as a table (e.g., multiplication table grid).

    Example:
        lazarus introspect circuit view -c mult_complete_table.npz
        lazarus introspect circuit view -c mult_complete_table.npz --table
        lazarus introspect circuit view -c mult_complete_table.npz --stats
    """
    from pathlib import Path

    import numpy as np

    circuit_path = args.circuit
    if not circuit_path:
        print("ERROR: Must specify --circuit file")
        return

    path = Path(circuit_path)
    if not path.exists():
        print(f"ERROR: Circuit file not found: {circuit_path}")
        return

    # Load circuit
    print(f"Loading circuit: {circuit_path}")
    data = np.load(circuit_path, allow_pickle=True)

    # Show available keys
    keys = list(data.keys())
    print(f"\nKeys: {keys}")

    # Basic info
    print(f"\n{'=' * 70}")
    print("CIRCUIT INFO")
    print(f"{'=' * 70}")

    if "model_id" in data:
        print(f"  Model: {data['model_id']}")
    if "layer" in data:
        print(f"  Layer: {data['layer']}")
    if "activations" in data:
        print(f"  Activations shape: {data['activations'].shape}")
    if "direction" in data:
        print(f"  Direction shape: {data['direction'].shape}")
        direction = data["direction"]
        print(f"  Direction norm: {np.linalg.norm(direction):.4f}")

    # Direction stats if available
    if "direction_stats" in data and getattr(args, "stats", False):
        stats = (
            data["direction_stats"].item()
            if hasattr(data["direction_stats"], "item")
            else dict(data["direction_stats"])
        )
        print(f"\n{'=' * 70}")
        print("DIRECTION STATS")
        print(f"{'=' * 70}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Show prompts and results
    if "prompts" in data and "results" in data:
        prompts = list(data["prompts"])
        results = list(data["results"])

        print(f"\n{'=' * 70}")
        print(f"ENTRIES ({len(prompts)} total)")
        print(f"{'=' * 70}")

        # Check if this looks like a multiplication/arithmetic table
        show_table = getattr(args, "table", False)
        is_arithmetic = False
        operator = None

        if "operators" in data:
            operators = list(data["operators"])
            unique_ops = set(operators)
            if len(unique_ops) == 1:
                operator = list(unique_ops)[0]
                is_arithmetic = operator in ["*", "+", "-", "/"]

        # Try to detect from prompts if operators not stored
        if not is_arithmetic and len(prompts) > 0:
            for op in ["*", "+", "-", "/"]:
                if op in str(prompts[0]):
                    operator = op
                    is_arithmetic = True
                    break

        # Show as table if requested and it's arithmetic
        if show_table and is_arithmetic and "operands_a" in data and "operands_b" in data:
            operands_a = list(data["operands_a"])
            operands_b = list(data["operands_b"])

            # Find unique operands
            unique_a = sorted(set(operands_a))
            unique_b = sorted(set(operands_b))

            # Check if it's a complete grid
            expected_size = len(unique_a) * len(unique_b)
            if len(results) == expected_size:
                # Build result lookup
                result_map = {}
                for i, (a, b, r) in enumerate(zip(operands_a, operands_b, results)):
                    result_map[(a, b)] = r

                # Print as grid
                op_name = {
                    "*": "Multiplication",
                    "+": "Addition",
                    "-": "Subtraction",
                    "/": "Division",
                }.get(operator, "Arithmetic")
                print(f"\n{op_name} Table:")
                print()

                # Header
                header = "    "
                for b in unique_b:
                    header += f"{int(b):4}"
                print(header)
                print("   " + "-" * (4 * len(unique_b) + 1))

                # Rows
                for a in unique_a:
                    row = f"{int(a)} |"
                    for b in unique_b:
                        val = result_map.get((a, b), "?")
                        if val is not None:
                            row += f"{int(val):4}"
                        else:
                            row += "   ?"
                    print(row)
            else:
                show_table = False  # Fall back to list view

        # Show as list (default or fallback)
        if not show_table:
            limit = getattr(args, "limit", 20)
            for i, (p, r) in enumerate(zip(prompts, results)):
                if i >= limit and limit > 0:
                    remaining = len(prompts) - limit
                    print(f"  ... and {remaining} more entries")
                    print("  (use --limit 0 to show all, or --table for grid view)")
                    break
                result_str = f" = {r}" if r is not None else ""
                print(f"  {i:3}: {p}{result_str}")

    # Show top neurons if direction exists
    if "direction" in data and getattr(args, "stats", False):
        direction = data["direction"]
        top_k = getattr(args, "top_k", 10)

        print(f"\n{'=' * 70}")
        print(f"TOP {top_k} NEURONS (by absolute weight)")
        print(f"{'=' * 70}")

        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            weight = direction[idx]
            print(f"  {rank:2}. Neuron {idx:4}: {weight:+.6f}")


def introspect_circuit_compare(args):
    """Compare multiple circuits to see how similar/different they are.

    Shows cosine similarity and angles between circuit directions,
    revealing whether different operations use independent or overlapping
    neural pathways.

    Example:
        lazarus introspect circuit compare \\
            -c mult_circuit.npz add_circuit.npz sub_circuit.npz div_circuit.npz
    """
    import json
    from pathlib import Path

    import numpy as np

    circuit_files = args.circuits
    top_k = getattr(args, "top_k", 10)

    # Load all circuits
    circuits = []
    for circuit_file in circuit_files:
        path = Path(circuit_file)
        if not path.exists():
            print(f"ERROR: Circuit file not found: {circuit_file}")
            return

        data = np.load(circuit_file, allow_pickle=True)
        if "direction" not in data:
            print(
                f"ERROR: {circuit_file} has no direction (use --extract-direction during capture)"
            )
            return

        # Extract name from filename (e.g., "mult_circuit.npz" -> "mult")
        name = path.stem.replace("_circuit", "").replace("_neurons", "")

        circuits.append(
            {
                "name": name,
                "file": circuit_file,
                "direction": data["direction"],
                "layer": int(data["layer"]) if "layer" in data else None,
                "training_samples": len(data["results"]) if "results" in data else 0,
            }
        )

    print(f"Comparing {len(circuits)} circuits:\n")

    # Show circuit info
    print("=" * 70)
    print("CIRCUITS")
    print("=" * 70)
    for c in circuits:
        layer_str = f"L{c['layer']}" if c["layer"] is not None else "?"
        print(f"  {c['name']:<12} {c['file']:<30} ({layer_str}, {c['training_samples']} samples)")
    print()

    # Compute pairwise similarities
    print("=" * 70)
    print("SIMILARITY MATRIX (cosine similarity)")
    print("=" * 70)

    n = len(circuits)
    similarity_matrix = np.zeros((n, n))

    # Header row
    header = "              " + "".join(f"{c['name']:<12}" for c in circuits)
    print(header)
    print("-" * len(header))

    for i, c1 in enumerate(circuits):
        d1 = c1["direction"]
        d1_norm = d1 / (np.linalg.norm(d1) + 1e-8)

        row = f"{c1['name']:<12}  "
        for j, c2 in enumerate(circuits):
            d2 = c2["direction"]
            d2_norm = d2 / (np.linalg.norm(d2) + 1e-8)

            cos_sim = float(np.dot(d1_norm, d2_norm))
            similarity_matrix[i, j] = cos_sim

            if i == j:
                row += f"{'1.000':<12}"
            else:
                row += f"{cos_sim:+.3f}       "

        print(row)

    print()

    # Compute angles
    print("=" * 70)
    print("ANGLES BETWEEN CIRCUITS (90° = orthogonal/independent)")
    print("=" * 70)

    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = similarity_matrix[i, j]
            angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
            c1_name = circuits[i]["name"]
            c2_name = circuits[j]["name"]

            if angle > 80:
                interpretation = "nearly orthogonal - independent circuits"
            elif angle > 60:
                interpretation = "mostly independent"
            elif angle > 30:
                interpretation = "partially overlapping"
            else:
                interpretation = "highly similar circuits"

            print(f"  {c1_name} ↔ {c2_name}: {angle:.1f}° ({interpretation})")

    print()

    # Show top neurons for each circuit
    print("=" * 70)
    print(f"TOP {top_k} NEURONS PER CIRCUIT")
    print("=" * 70)

    all_top_neurons = {}
    for c in circuits:
        direction = c["direction"]
        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        top_weights = [(int(idx), float(direction[idx])) for idx in top_indices]
        all_top_neurons[c["name"]] = top_weights

        print(f"\n{c['name']}:")
        for idx, weight in top_weights:
            bar = (
                "+" * min(int(abs(weight) / 10), 20)
                if weight > 0
                else "-" * min(int(abs(weight) / 10), 20)
            )
            print(f"  N{idx:>4}: {weight:+8.1f} {bar}")

    # Find shared top neurons
    print()
    print("=" * 70)
    print("SHARED TOP NEURONS (appear in multiple circuits)")
    print("=" * 70)

    neuron_appearances = {}
    for name, neurons in all_top_neurons.items():
        for idx, weight in neurons:
            if idx not in neuron_appearances:
                neuron_appearances[idx] = []
            neuron_appearances[idx].append((name, weight))

    shared = [(idx, apps) for idx, apps in neuron_appearances.items() if len(apps) > 1]
    shared.sort(key=lambda x: len(x[1]), reverse=True)

    if shared:
        for idx, appearances in shared[:15]:  # Show top 15 shared neurons
            circuits_str = ", ".join(f"{name}({w:+.0f})" for name, w in appearances)
            print(f"  N{idx:>4}: {circuits_str}")
    else:
        print("  No neurons appear in multiple circuit top-k lists")

    # Save if requested
    if args.output:
        output_data = {
            "circuits": [
                {"name": c["name"], "file": c["file"], "layer": c["layer"]} for c in circuits
            ],
            "similarity_matrix": similarity_matrix.tolist(),
            "top_neurons": dict(all_top_neurons.items()),
            "shared_neurons": list(shared),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_circuit_decode(args):
    """Decode circuit activations by injecting them into a prompt.

    Injects captured activations into the model during forward pass
    and observes how it affects generation. Uses steering mechanism
    to blend original and injected activations.

    Example:
        lazarus introspect circuit decode \\
            -m model \\
            --inject mult_circuit.npz \\
            --prompt "What is 5 * 6? Answer:" \\
            --blend 1.0
    """
    import json

    import numpy as np

    from ...introspection import ActivationSteering, SteeringConfig

    # CLI uses --inject for the circuit file
    circuit_path = getattr(args, "inject", None) or getattr(args, "circuit", None)
    if not circuit_path:
        print("ERROR: Must specify --inject file")
        return

    # Load circuit
    print(f"Loading circuit: {circuit_path}")
    data = np.load(circuit_path, allow_pickle=True)

    activations = data["activations"]
    circuit_layer = int(data["layer"])
    model_id = str(data["model_id"])
    prompts = list(data["prompts"])
    results = list(data["results"])

    # Use layer from args if provided, otherwise from circuit
    layer = args.layer if args.layer is not None else circuit_layer

    print(f"  Circuit model: {model_id}")
    print(f"  Circuit layer: {circuit_layer}")
    print(f"  Injection layer: {layer}")
    print(f"  Available activations: {len(activations)}")

    # Show available activations
    print("\nAvailable circuit entries:")
    for i, (p, r) in enumerate(zip(prompts, results)):
        result_str = f" = {r}" if r is not None else ""
        print(f"  [{i}] {p[:40]}{result_str}")

    # Get injection index (default to 0, or allow --inject-idx if added later)
    inject_idx = getattr(args, "inject_idx", 0) or 0
    if inject_idx < 0 or inject_idx >= len(activations):
        print(f"ERROR: inject index must be between 0 and {len(activations) - 1}")
        return

    inject_activation = activations[inject_idx]
    inject_prompt = prompts[inject_idx]
    inject_result = results[inject_idx]

    print(f"\nInjecting activation from: {inject_prompt}")
    if inject_result is not None:
        print(f"  Original result: {inject_result}")

    # Load model for decoding
    model_to_use = args.model or model_id
    print(f"\nLoading model: {model_to_use}")
    steerer = ActivationSteering.from_pretrained(model_to_use)

    # Parse test prompts
    if args.prompt.startswith("@"):
        with open(args.prompt[1:]) as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    else:
        test_prompts = [p.strip() for p in args.prompt.split("|")]

    # CLI uses --blend for strength
    strength = getattr(args, "blend", None) or getattr(args, "strength", None) or 1.0
    max_tokens = args.max_tokens if args.max_tokens else 20

    print(f"  Injection blend: {strength}")
    print(f"  Max tokens: {max_tokens}")

    # Create a "direction" that points from origin to the captured activation
    # This is a bit of a hack - we're using steering to inject absolute activations
    # by treating the activation itself as a direction with coefficient 1.0
    direction = inject_activation.astype(np.float32)

    steerer.add_direction(
        layer=layer,
        direction=direction,
        name="circuit_injection",
        positive_label="injected",
        negative_label="original",
    )

    config = SteeringConfig(
        layers=[layer],
        coefficient=strength,
        max_new_tokens=max_tokens,
        temperature=0.0,
    )

    # Run generation with and without injection
    print(f"\n{'=' * 70}")
    print("CIRCUIT INJECTION RESULTS")
    print(f"{'=' * 70}")

    results_table = []
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")

        # Baseline (no injection)
        baseline_config = SteeringConfig(
            layers=[layer],
            coefficient=0.0,
            max_new_tokens=max_tokens,
            temperature=0.0,
        )
        baseline_output = steerer.generate(prompt, baseline_config)
        print(f"  Baseline:  {baseline_output!r}")

        # With injection
        injected_output = steerer.generate(prompt, config)
        print(f"  Injected:  {injected_output!r}")

        results_table.append(
            {
                "prompt": prompt,
                "baseline": baseline_output,
                "injected": injected_output,
                "inject_source": inject_prompt,
                "blend": strength,
            }
        )

    # Save if requested
    output_path = getattr(args, "output", None)
    if output_path:
        output_data = {
            "circuit": circuit_path,
            "inject_idx": inject_idx,
            "inject_source": inject_prompt,
            "inject_result": inject_result,
            "blend": strength,
            "layer": layer,
            "results": results_table,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def introspect_memory_inject(args):
    """
    External memory injection for fact retrieval.

    Builds an external memory store from known facts and uses it to
    inject correct answers at inference time. This can rescue queries
    that the model would otherwise get wrong.

    Examples:
        # Build memory from multiplication table and test
        lazarus introspect memory-inject -m openai/gpt-oss-20b \\
            --facts multiplication --query "7*8="

        # Test rescue on non-standard format
        lazarus introspect memory-inject -m openai/gpt-oss-20b \\
            --facts multiplication --query "seven times eight equals"

        # Load custom facts and save memory store
        lazarus introspect memory-inject -m openai/gpt-oss-20b \\
            --facts @my_facts.json --save-store memory.npz
    """
    import json

    from ...introspection.external_memory import ExternalMemory, MemoryConfig

    # Configure memory layers
    query_layer = getattr(args, "query_layer", None)
    inject_layer = getattr(args, "inject_layer", None)
    blend = getattr(args, "blend", 1.0)
    threshold = getattr(args, "threshold", 0.7)

    memory_config = None
    if query_layer is not None or inject_layer is not None:
        memory_config = MemoryConfig(
            query_layer=query_layer or 22,
            inject_layer=inject_layer or 21,
            value_layer=query_layer or 22,
            blend=blend,
            similarity_threshold=threshold,
        )

    # Create memory system
    memory = ExternalMemory.from_pretrained(args.model, memory_config)

    # Load facts
    fact_type = args.facts
    if fact_type.startswith("@"):
        # Load from file
        with open(fact_type[1:]) as f:
            facts = json.load(f)
        memory.add_facts(facts)
    elif fact_type == "multiplication":
        memory.add_multiplication_table(2, 9)
    elif fact_type == "addition":
        facts = []
        for a in range(1, 10):
            for b in range(1, 10):
                facts.append({"query": f"{a}+{b}=", "answer": str(a + b)})
        memory.add_facts(facts)
    else:
        print(f"ERROR: Unknown fact type: {fact_type}")
        print("Use: multiplication, addition, or @file.json")
        return

    # Save store if requested
    save_store = getattr(args, "save_store", None)
    if save_store:
        memory.save(save_store)

    # Load store if provided
    load_store = getattr(args, "load_store", None)
    if load_store:
        memory.load(load_store)

    # Process queries
    queries = []
    if hasattr(args, "query") and args.query:
        queries = [args.query]
    elif hasattr(args, "queries") and args.queries:
        queries = args.queries.split("|")

    if not queries:
        print("\nNo queries provided. Use --query or --queries")
        print(f"Memory store has {memory.num_entries} entries")
        return

    print(f"\n{'=' * 70}")
    print("EXTERNAL MEMORY INJECTION")
    print(f"{'=' * 70}")

    force = getattr(args, "force", False)

    for query in queries:
        result = memory.query(query, use_injection=True, force_injection=force)

        print(f"\nQuery: '{query}'")
        print(f"  Baseline: '{result.baseline_answer}' ({result.baseline_confidence:.1%})")

        if result.used_injection:
            print(f"  Injected: '{result.injected_answer}' ({result.injected_confidence:.1%})")
            if result.matched_entry:
                print(
                    f"  Matched:  '{result.matched_entry.query}' -> {result.matched_entry.answer}"
                )
                print(f"  Similarity: {result.similarity:.3f}")

            # Show if it was rescued
            if result.baseline_answer.strip() != result.injected_answer.strip():
                print("  Status: MODIFIED")
        else:
            if result.matched_entry:
                print(f"  Matched:  '{result.matched_entry.query}' (sim={result.similarity:.3f})")
                print(f"  Status: Below threshold ({threshold}), no injection")
            else:
                print("  Status: No match found")

    # Evaluate mode
    if getattr(args, "evaluate", False):
        print(f"\n{'=' * 70}")
        print("EVALUATION")
        print(f"{'=' * 70}")

        # Build test set from the facts
        if fact_type == "multiplication":
            test_facts = [
                {"query": f"{a}*{b}=", "answer": str(a * b)}
                for a in range(2, 10)
                for b in range(2, 10)
            ]
        elif fact_type == "addition":
            test_facts = [
                {"query": f"{a}+{b}=", "answer": str(a + b)}
                for a in range(1, 10)
                for b in range(1, 10)
            ]
        else:
            test_facts = facts

        metrics = memory.evaluate(test_facts, verbose=False)
        print(f"\nBaseline accuracy: {metrics['baseline_accuracy']:.1%}")
        print(f"Injected accuracy: {metrics['injected_accuracy']:.1%}")
        print(f"Rescued: {metrics['rescued']}")
        print(f"Broken: {metrics['broken']}")
