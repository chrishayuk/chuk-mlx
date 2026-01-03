"""Activation steering commands for introspection CLI.

Commands for extracting and applying activation steering directions.
"""

import sys
from pathlib import Path


def introspect_steer(args):
    """Apply activation steering to manipulate model behavior.

    Supports three modes:
    1. Extract direction: Compute steering direction from contrastive prompts
    2. Apply direction: Load pre-computed direction and steer generation
    3. Compare: Show outputs at different steering coefficients
    """
    import json

    import numpy as np

    from ....introspection import ActivationSteering, SteeringConfig

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

        from ....introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

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

        from ....introspection.hooks import CaptureConfig, ModelHooks, PositionSelection

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
                    "-> positive" if coef > 0 else "<- negative" if coef < 0 else "neutral"
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


__all__ = [
    "introspect_steer",
]
