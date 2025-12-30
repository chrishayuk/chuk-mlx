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

    # Parse steering config if provided
    steer_config = None
    if getattr(args, "steer", None):
        import numpy as np
        steer_parts = args.steer.split(":")
        if len(steer_parts) != 2:
            print("Error: --steer format should be 'direction.npz:coefficient'")
            sys.exit(1)
        steer_file, steer_coef = steer_parts[0], float(steer_parts[1])
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
                from ...introspection.steering import SteeringHook

                steer_layer = steer_config["layer"]
                steer_coef = steer_config["coefficient"]
                steer_dir = mx.array(steer_config["direction"], dtype=mx.float32)

                pos_label = steer_config.get("positive", "positive")
                neg_label = steer_config.get("negative", "negative")
                direction_str = f"{neg_label}→{pos_label}" if steer_coef > 0 else f"{pos_label}→{neg_label}"

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
                            for attr in ["mlp", "attn", "self_attn", "input_layernorm", "post_attention_layernorm"]:
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
    prompts_arg = getattr(args, "prompts", None)
    if prompts_arg:
        prompt_pairs = []
        for item in prompts_arg.split("|"):
            item = item.strip()
            if ":" in item:
                prompt, expected = item.rsplit(":", 1)
                prompt_pairs.append((prompt.strip(), expected.strip()))
            else:
                # Use the criterion as expected
                prompt_pairs.append((item, args.criterion))

        verbose = getattr(args, "verbose", False)

        print(f"\n{'='*70}")
        print(f"MULTI-PROMPT ABLATION TEST")
        print(f"{'='*70}")

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
            layer_str = ",".join(str(l) for l in layers)
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
            print(f"\n{'='*70}")
            print("FULL OUTPUTS")
            print(f"{'='*70}")
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

        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt}")
        print(f"Criterion: {args.criterion}")
        print(f"Layers ablated: {layers}")
        print(f"{'='*60}")
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
            print(f"\n=> BASELINE FAILS: Original doesn't pass criterion")

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

    # Load direction
    if args.direction:
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
            print("Error: Must provide --direction or both --positive and --negative")
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
        print(f"\nDirection method: difference of means (normalized)")
    else:
        # Logistic regression weights
        direction = final_probe.coef_[0]
        print(f"\nDirection method: logistic regression weights")

    direction_norm = float(np.linalg.norm(direction))

    # Show projection statistics
    projections = X_best @ (direction / np.linalg.norm(direction))
    class_a_proj = projections[y == 1]
    class_b_proj = projections[y == 0]
    print(f"\nProjection statistics:")
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
    """
    import json

    import mlx.core as mx
    import numpy as np

    from ...introspection import CaptureConfig, ModelHooks, PositionSelection
    from ...introspection.ablation import AblationStudy

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    layer = args.layer
    print(f"  Analyzing layer: {layer}")

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Parse labels if provided
    if args.labels:
        labels = [l.strip() for l in args.labels.split("|")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels for {len(prompts)} prompts, ignoring labels")
            labels = None
    else:
        labels = None

    # Get neurons to analyze
    neurons = []
    neuron_weights = {}

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

    elif args.neurons:
        # Parse neuron indices
        neurons = [int(n.strip()) for n in args.neurons.split(",")]
        print(f"  Analyzing {len(neurons)} neurons: {neurons}")

    else:
        print("ERROR: Must specify --neurons or --from-direction")
        return

    print(f"\nCollecting activations for {len(prompts)} prompts...")

    # Collect activations
    all_activations = []
    for prompt in prompts:
        hooks = ModelHooks(model, model_config=config)
        hooks.configure(CaptureConfig(
            layers=[layer],
            capture_hidden_states=True,
            positions=PositionSelection.LAST,
        ))

        input_ids = tokenizer.encode(prompt, return_tensors="np")
        hooks.forward(mx.array(input_ids))

        h = hooks.state.hidden_states[layer][0, 0, :]
        h_np = np.array(h.astype(mx.float32), copy=False)
        all_activations.append(h_np)

    # Build activation matrix
    activation_matrix = np.array([[act[n] for n in neurons] for act in all_activations])

    # Print results as ASCII heatmap
    print(f"\n{'=' * 80}")
    print(f"NEURON ACTIVATION MAP AT LAYER {layer}")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Prompt':<20} |"
    for n in neurons:
        header += f" N{n:>4} |"
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

    # ASCII heatmap visualization
    print(f"\n{'=' * 80}")
    print("ASCII HEATMAP (░ = low, ▒ = medium, ▓ = high, █ = max)")
    print(f"{'=' * 80}")

    # Normalize for heatmap
    norm_matrix = (activation_matrix - vmin) / (vmax - vmin + 1e-8)

    header = f"{'Prompt':<20} |"
    for n in neurons:
        header += f" N{n:>4} |"
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
    print(f"\n{'=' * 80}")
    print("NEURON STATISTICS")
    print(f"{'=' * 80}")

    for j, n in enumerate(neurons):
        vals = activation_matrix[:, j]
        weight_str = ""
        if n in neuron_weights:
            w = neuron_weights[n]
            direction = "→ POSITIVE detector" if w > 0 else "→ NEGATIVE detector"
            weight_str = f" (weight: {w:+.3f}) {direction}"

        print(f"Neuron {n:4d}: min={vals.min():+7.1f}, max={vals.max():+7.1f}, "
              f"mean={vals.mean():+7.1f}, std={vals.std():6.1f}{weight_str}")

    # Correlation with labels if provided
    if labels:
        print(f"\n{'=' * 80}")
        print("LABEL CORRELATION")
        print(f"{'=' * 80}")

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = np.array([l == label for l in labels])
            if mask.sum() > 0:
                print(f"\n{label}:")
                for j, n in enumerate(neurons):
                    mean_val = activation_matrix[mask, j].mean()
                    print(f"  Neuron {n:4d}: mean={mean_val:+7.1f}")

    # Save if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "layer": layer,
            "neurons": neurons,
            "prompts": prompts,
            "labels": labels,
            "activations": activation_matrix.tolist(),
            "neuron_weights": neuron_weights,
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
        metadata.append({
            "file": str(path),
            "name": name,
            "layer": layer,
            "method": method,
            "accuracy": accuracy,
            "dim": len(direction),
        })

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
            print(f"\nOrthogonal pairs (independent dimensions):")
            for a, b, s in sorted(orthogonal_pairs, key=lambda x: abs(x[2])):
                print(f"  {a} ⊥ {b} (cos = {s:+.3f})")

        if aligned_pairs:
            print(f"\nAligned pairs (potentially redundant):")
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
            ] if off_diag else [],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_activation_cluster(args):
    """Visualize activation clusters using PCA.

    Projects hidden states to 2D to see if different prompt types cluster separately.
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
    target_layer = args.layer if args.layer is not None else int(num_layers * 0.5)
    print(f"  Layers: {num_layers}")
    print(f"  Target layer: {target_layer}")

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

    # Parse prompts with labels
    prompts = []
    labels = []

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

    print(f"\nCollecting activations for {len(prompts)} prompts...")

    # Collect activations
    activations = []
    for prompt in prompts:
        h = get_hidden_at_layer(prompt, target_layer)
        activations.append(h)

    X = np.array(activations)

    # PCA
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("ERROR: sklearn required. Install with: pip install scikit-learn")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)

    # Compute cluster statistics
    unique_labels = list(set(labels))
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

    # Inter-cluster distance
    if len(unique_labels) == 2:
        c1 = cluster_stats[unique_labels[0]]["center"]
        c2 = cluster_stats[unique_labels[1]]["center"]
        separation = float(np.linalg.norm(c1 - c2))
    else:
        separation = None

    # Print results
    print(f"\n{'=' * 70}")
    print(f"ACTIVATION CLUSTERS AT LAYER {target_layer}")
    print(f"{'=' * 70}")
    print(
        f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}"
    )

    if separation:
        print(f"Cluster separation: {separation:.2f}")

    print(f"\n{'Label':<15} {'Count':<8} {'Center (PC1, PC2)'}")
    print("-" * 50)
    for label, stats in cluster_stats.items():
        print(
            f"{label:<15} {stats['count']:<8} ({stats['center'][0]:.2f}, {stats['center'][1]:.2f})"
        )

    # ASCII scatter plot
    print(f"\n{'=' * 70}")
    print("SCATTER PLOT (ASCII)")
    print(f"{'=' * 70}")

    # Normalize to grid
    x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
    y_min, y_max = projected[:, 1].min(), projected[:, 1].max()

    grid_width = 60
    grid_height = 20
    grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

    symbols = {unique_labels[0]: "A", unique_labels[1]: "B"} if len(unique_labels) == 2 else {}

    for i, (x, y) in enumerate(projected):
        gx = int((x - x_min) / (x_max - x_min + 1e-6) * (grid_width - 1))
        gy = int((y - y_min) / (y_max - y_min + 1e-6) * (grid_height - 1))
        gy = grid_height - 1 - gy  # Flip y
        symbol = symbols.get(labels[i], labels[i][0].upper())
        grid[gy][gx] = symbol

    for row in grid:
        print("  " + "".join(row))

    print(f"\n  Legend: {', '.join(f'{s}={lbl}' for lbl, s in symbols.items())}")

    # Save if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "layer": target_layer,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "separation": separation,
            "prompts": prompts,
            "labels": labels,
            "projected": projected.tolist(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
