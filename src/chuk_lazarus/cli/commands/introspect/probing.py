"""Probing and uncertainty detection commands for introspection CLI.

Commands for:
- Metacognitive strategy detection
- Uncertainty detection using hidden state geometry
- Linear probing for task classification
"""

import logging

from ....introspection.enums import DirectionMethod

logger = logging.getLogger(__name__)


def introspect_metacognitive(args):
    """Detect metacognitive strategy switch at a specific layer.

    This tool probes the model's "decision layer" (typically ~70% through the network)
    to detect whether it will use:
    - Direct computation: L24 predicts a digit -> answer comes immediately
    - Chain-of-thought: L24 predicts ' ', 'To', 'Let' etc. -> reasoning first

    The key insight is that token IDENTITY at the decision layer reveals the
    model's strategy, not just confidence. A digit token means "I know the answer",
    while a non-digit means "I need to think about this".
    """
    import asyncio
    import json

    from ....introspection import (
        AnalysisConfig,
        LayerStrategy,
        ModelAnalyzer,
        apply_chat_template,
        extract_expected_answer,
    )

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

            # Configure to capture only decision layer
            config = AnalysisConfig(
                layer_strategy=LayerStrategy.SPECIFIC,
                capture_layers=[decision_layer],
                top_k=5,
            )

            # Parse test prompts
            if args.prompts.startswith("@"):
                with open(args.prompts[1:]) as f:
                    test_prompts = [line.strip() for line in f if line.strip()]
            else:
                test_prompts = [p.strip() for p in args.prompts.split("|")]

            print(f"\nAnalyzing {len(test_prompts)} prompts...")

            results = []
            direct_count = 0
            cot_count = 0

            print(f"\n{'=' * 90}")
            print(
                f"{'Prompt':<25} {'L{:<3} Top':<12} {'Prob':>6} {'Strategy':<12} {'Digit?':<6} {'Match?':<6}"
            )
            print("-" * 90)

            for prompt in test_prompts:
                # Apply chat template if available
                analysis_prompt = prompt
                if not use_raw and has_chat_template:
                    analysis_prompt = apply_chat_template(tokenizer, prompt)

                result = await analyzer.analyze(analysis_prompt, config)

                # Get prediction at decision layer
                layer_pred = None
                for lp in result.layer_predictions:
                    if lp.layer_idx == decision_layer:
                        layer_pred = lp
                        break

                if layer_pred is None:
                    continue

                top_token = layer_pred.top_token
                top_prob = layer_pred.probability

                # Detect strategy based on token identity
                is_digit = top_token.strip().isdigit()
                if is_digit:
                    strategy = "DIRECT"
                    direct_count += 1
                else:
                    strategy = "COT"
                    cot_count += 1

                # Check if it matches expected answer
                expected = extract_expected_answer(prompt)
                correct_start = False
                if expected and is_digit:
                    correct_start = expected.startswith(top_token.strip())

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "decision_layer": decision_layer,
                        "decision_token": top_token,
                        "decision_prob": top_prob,
                        "strategy": strategy,
                        "is_digit": is_digit,
                        "correct_start": correct_start,
                        "final_token": result.predicted_token,
                        "final_prob": result.final_probability,
                    }
                )

                # Print row
                short_prompt = prompt[:23] + ".." if len(prompt) > 25 else prompt
                digit_str = "Yes" if is_digit else "No"
                match_str = "Yes" if correct_start else ("N/A" if not is_digit else "No")

                print(
                    f"{short_prompt:<25} {top_token!r:<12} {top_prob:>5.1%} "
                    f"{strategy:<12} {digit_str:<6} {match_str:<6}"
                )

            # Summary
            print("-" * 90)
            total = len(results)
            print("\nSummary:")
            print(
                f"  Direct computation: {direct_count}/{total} ({100 * direct_count / total:.0f}%)"
            )
            print(f"  Chain-of-thought: {cot_count}/{total} ({100 * cot_count / total:.0f}%)")

            # Accuracy among direct answers
            direct_results = [r for r in results if r["strategy"] == "DIRECT"]
            if direct_results:
                correct = sum(1 for r in direct_results if r["correct_start"])
                print(
                    f"  Direct accuracy: {correct}/{len(direct_results)} ({100 * correct / len(direct_results):.0f}%)"
                )

            # Save if requested
            if args.output:
                output_data = {
                    "model_id": args.model,
                    "decision_layer": decision_layer,
                    "total_prompts": total,
                    "direct_count": direct_count,
                    "cot_count": cot_count,
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

    from ....introspection import ModelAccessor

    async def run():
        # Lazy imports for heavy dependencies
        import mlx.core as mx
        import numpy as np

        from ....inference.loader import DType, HFLoader
        from ....models_v2.families.registry import detect_model_family, get_family_info

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

        # Use ModelAccessor for unified access
        accessor = ModelAccessor(model, config)
        num_layers = accessor.num_layers
        detection_layer = args.layer or int(num_layers * 0.7)  # ~70% depth

        print(f"  Layers: {num_layers}")
        print(f"  Detection layer: {detection_layer}")

        def get_hidden_state(prompt: str) -> np.ndarray:
            """Get hidden state at detection layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            h = accessor.embed(input_ids)

            seq_len = input_ids.shape[1]
            mask = accessor.create_causal_mask(seq_len, h.dtype)

            for idx, lyr in enumerate(accessor.layers):
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
        print(f"{'Prompt':<30} {'Score':>8} {'Prediction':<12} {'->Compute':>10} {'->Refusal':>10}")
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

    # Lazy imports for heavy dependencies
    import mlx.core as mx
    import numpy as np

    from ....inference.loader import DType, HFLoader
    from ....introspection import ModelAccessor
    from ....models_v2.families.registry import detect_model_family, get_family_info

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

    # Use ModelAccessor for unified access
    accessor = ModelAccessor(model, config)
    num_layers = accessor.num_layers
    print(f"  Layers: {num_layers}")

    def get_all_hidden_states(prompt: str) -> list[np.ndarray]:
        """Get hidden state at each layer."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        h = accessor.embed(input_ids)

        seq_len = input_ids.shape[1]
        mask = accessor.create_causal_mask(seq_len, h.dtype)

        hidden_states = []
        for idx, lyr in enumerate(accessor.layers):
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
        marker = " <- SELECTED" if r["layer"] == best_layer else ""
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
    method = getattr(args, "method", DirectionMethod.LOGISTIC.value)
    if method == DirectionMethod.MEAN_DIFFERENCE.value:
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
    print(f"  {args.label_a}: {class_a_proj.mean():+.2f} +/- {class_a_proj.std():.2f}")
    print(f"  {args.label_b}: {class_b_proj.mean():+.2f} +/- {class_b_proj.std():.2f}")
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
            print(f"  {prompt[:40]:<40} -> {pred_class} ({confidence:.1%})")

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


__all__ = [
    "introspect_metacognitive",
    "introspect_probe",
    "introspect_uncertainty",
]
