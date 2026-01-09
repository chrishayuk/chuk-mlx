"""Ablation study commands for introspection CLI.

Commands for causal circuit discovery through ablation studies.
This module is a thin CLI wrapper - all business logic is in AblationService.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ._types import AblationConfig
from ._utils import parse_layers


def introspect_ablate(args: Namespace) -> None:
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
    asyncio.run(_async_introspect_ablate(args))


async def _async_introspect_ablate(args: Namespace) -> None:
    """Async implementation of ablation command."""
    from ....introspection.ablation import (
        AblationService,
        AblationStudy,
        ComponentType,
    )

    config = AblationConfig.from_args(args)

    # Validate arguments
    if not config.prompts and not config.prompt:
        raise ValueError("Either --prompt/-p (with --criterion/-c) or --prompts is required")
    if config.prompt and not config.criterion and not config.prompts:
        raise ValueError("--criterion/-c is required when using --prompt/-p")

    print(f"Loading model: {config.model}")
    study = AblationStudy.from_pretrained(config.model)

    # Parse layers
    layers = parse_layers(config.layers) if config.layers else list(range(study.adapter.num_layers))

    # Map component
    component_map = {
        "mlp": ComponentType.MLP,
        "attention": ComponentType.ATTENTION,
        "both": ComponentType.BOTH,
    }
    component = component_map[config.component]

    if config.multi:
        print(f"Ablating layers together: {layers}")
    else:
        print(f"Sweeping layers individually: {layers}")
    print(f"Component: {config.component}")
    print(f"Mode: {'RAW' if config.raw else 'CHAT'}")

    # Handle multiple prompts mode
    if config.prompts:
        prompt_pairs = AblationService.parse_prompt_pairs(config.prompts)

        # Fill in criterion for prompts without expected values
        filled_pairs = []
        for prompt, expected in prompt_pairs:
            if not expected and config.criterion:
                filled_pairs.append((prompt, config.criterion))
            elif expected:
                filled_pairs.append((prompt, expected))
            else:
                raise ValueError(
                    f"Prompt '{prompt}' has no expected value (use 'prompt:expected' format)"
                )

        results = await AblationService.run_multi_prompt_ablation(
            model=config.model,
            prompt_pairs=filled_pairs,
            layers=layers,
            component=component,
            max_tokens=config.max_tokens,
            multi_mode=config.multi,
        )

        # Display results
        _print_multi_prompt_results(results, filled_pairs, config.verbose)
        return

    # Single prompt mode
    if config.multi:
        # Multi-layer ablation
        print(f"\nAblating layers {layers} together...")

        baseline, ablated = await AblationService.run_multi_ablation(
            model=config.model,
            prompt=config.prompt,
            layers=layers,
            criterion=config.criterion,
            component=component,
            max_tokens=config.max_tokens,
        )

        _print_multi_ablation_results(config.prompt, config.criterion, layers, baseline, ablated)

    else:
        # Sweep mode
        print("\nRunning ablation sweep...")

        result = await AblationService.run_ablation_sweep(
            model=config.model,
            prompt=config.prompt,
            criterion=config.criterion,
            layers=layers,
            component=component,
            max_tokens=config.max_tokens,
        )

        # Print results using framework
        study.print_sweep_summary_from_service(result)

        # Save if requested
        if config.output:
            study.save_results({"ablation_study": result}, config.output)


def _print_multi_prompt_results(
    results: list,
    prompt_pairs: list[tuple[str, str]],
    verbose: bool,
) -> None:
    """Print multi-prompt ablation results."""
    print(f"\n{'=' * 70}")
    print("MULTI-PROMPT ABLATION TEST")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Ablation':<20}"
    for prompt, expected in prompt_pairs:
        short_prompt = prompt[:12] + "..." if len(prompt) > 15 else prompt
        header += f" | {short_prompt:<18}"
    print(header)
    print("-" * len(header))

    # Results
    all_outputs: dict[str, dict[str, tuple[str, bool]]] = {}
    for ablation_result in results:
        row = f"{ablation_result.ablation_name:<20}"
        all_outputs[ablation_result.ablation_name] = {}

        for single_result in ablation_result.results:
            out_short = single_result.output.strip()[:15]
            status = f"{'Y' if single_result.passes_criterion else 'N'} {out_short}"
            row += f" | {status:<18}"
            all_outputs[ablation_result.ablation_name][single_result.prompt] = (
                single_result.output,
                single_result.passes_criterion,
            )
        print(row)

    # Verbose output
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


def _print_multi_ablation_results(
    prompt: str,
    criterion: str,
    layers: list[int],
    baseline,
    ablated,
) -> None:
    """Print multi-layer ablation results."""
    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"Criterion: {criterion}")
    print(f"Layers ablated: {layers}")
    print(f"{'=' * 60}")

    baseline_status = "PASS" if baseline.passes_criterion else "FAIL"
    ablated_status = "PASS" if ablated.passes_criterion else "FAIL"

    print(f"\nOriginal output ({baseline_status}):")
    print(f"  {baseline.output.strip()[:200]}")
    print(f"\nAblated output ({ablated_status}):")
    print(f"  {ablated.output.strip()[:200]}")

    if baseline.passes_criterion and not ablated.passes_criterion:
        print(f"\n=> CAUSAL: Ablating {layers} breaks the criterion")
    elif not baseline.passes_criterion and ablated.passes_criterion:
        print(f"\n=> INVERSE CAUSAL: Ablating {layers} enables the criterion")
    elif baseline.passes_criterion and ablated.passes_criterion:
        print(f"\n=> NOT CAUSAL: Ablating {layers} doesn't affect outcome")
    else:
        print("\n=> BASELINE FAILS: Original doesn't pass criterion")


def introspect_weight_diff(args: Namespace) -> None:
    """Compare weight divergence between two models."""
    asyncio.run(_async_introspect_weight_diff(args))


async def _async_introspect_weight_diff(args: Namespace) -> None:
    """Async implementation of weight diff command."""
    import json

    import mlx.core as mx
    from huggingface_hub import snapshot_download

    from ....introspection.ablation import AblationStudy, ModelAdapter

    print(f"Loading base model: {args.base}")
    base_path = snapshot_download(args.base, allow_patterns=["*.json", "*.safetensors"])

    print(f"Loading fine-tuned model: {args.finetuned}")
    ft_path = snapshot_download(args.finetuned, allow_patterns=["*.json", "*.safetensors"])

    # Detect family and load
    family = AblationStudy._detect_family(base_path)
    print(f"Detected model family: {family}")

    base_model, base_config = AblationStudy._load_model(base_path, family)
    ft_model, ft_config = AblationStudy._load_model(ft_path, family)

    # Compare weights
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


def introspect_activation_diff(args: Namespace) -> None:
    """Compare activation divergence between two models."""
    asyncio.run(_async_introspect_activation_diff(args))


async def _async_introspect_activation_diff(args: Namespace) -> None:
    """Async implementation of activation diff command."""
    import json

    import mlx.core as mx

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy
    from ._utils import parse_prompts

    # Parse prompts
    prompts = parse_prompts(args.prompts, delimiter=",")
    print(f"Testing {len(prompts)} prompts")

    # Load models
    print(f"Loading base model: {args.base}")
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
    layer_avg: dict[int, list[float]] = {}
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
