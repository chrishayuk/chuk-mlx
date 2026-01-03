"""Ablation study commands for introspection CLI.

Commands for causal circuit discovery through ablation studies.
"""


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

    from ....introspection import AblationConfig, AblationStudy, ComponentType

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
    from ....introspection.ablation import AblationStudy

    family = AblationStudy._detect_family(base_path)
    print(f"Detected model family: {family}")

    base_model, base_config = AblationStudy._load_model(base_path, family)
    ft_model, ft_config = AblationStudy._load_model(ft_path, family)

    # Compare weights
    from ....introspection.ablation import ModelAdapter

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

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split(",")]

    print(f"Testing {len(prompts)} prompts")

    # Load models
    print(f"Loading base model: {args.base}")
    from ....introspection.ablation import AblationStudy

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


__all__ = [
    "introspect_ablate",
    "introspect_weight_diff",
    "introspect_activation_diff",
]
