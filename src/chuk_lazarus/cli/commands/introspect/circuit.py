"""Circuit capture and manipulation commands for introspection CLI.

Commands for capturing, viewing, invoking, testing, comparing, and
decoding computational circuits within neural networks.
"""

from ....introspection.enums import InvocationMethod

__all__ = [
    "introspect_circuit_capture",
    "introspect_circuit_invoke",
    "introspect_circuit_test",
    "introspect_circuit_view",
    "introspect_circuit_compare",
    "introspect_circuit_decode",
]


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
    import mlx.core as mx
    import numpy as np

    from ....introspection import (
        CaptureConfig,
        ModelHooks,
        ParsedArithmeticPrompt,
        PositionSelection,
        parse_prompts_from_arg,
    )
    from ....introspection.ablation import AblationStudy

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

    # Parse prompts using framework utility
    raw_prompts = parse_prompts_from_arg(args.prompts)
    print(f"  Prompts: {len(raw_prompts)}")

    # Parse results if provided separately
    explicit_results: list[int] | None = None
    if getattr(args, "results", None):
        if args.results.startswith("@"):
            with open(args.results[1:]) as f:
                explicit_results = [int(line.strip()) for line in f if line.strip()]
        else:
            explicit_results = [int(r.strip()) for r in args.results.split("|")]
        if len(explicit_results) != len(raw_prompts):
            print(f"ERROR: {len(explicit_results)} results for {len(raw_prompts)} prompts")
            return

    # Parse each prompt using the Pydantic model
    parsed: list[ParsedArithmeticPrompt] = []
    for i, prompt in enumerate(raw_prompts):
        explicit_result = explicit_results[i] if explicit_results else None
        parsed.append(ParsedArithmeticPrompt.parse(prompt, explicit_result))

    # Collect activations
    activations = []
    print("\nCapturing activations...")

    for item in parsed:
        prompt = item.prompt
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
        if item.result is not None:
            if item.is_arithmetic:
                op_str = item.operator.value if item.operator else "?"
                print(f"  {item.operand_a} {op_str} {item.operand_b} = {item.result}")
            else:
                print(f"  {prompt[:30]}... -> {item.result}")
        else:
            print(f"  {prompt[:40]}...")

    activations = np.array(activations)

    # Extract direction if requested
    extract_direction = getattr(args, "extract_direction", False)
    direction = None
    direction_stats = {}

    arithmetic_items = [p for p in parsed if p.result is not None]
    if len(arithmetic_items) >= 2:
        print("\nAnalyzing linear predictability of results from activations...")

        try:
            from sklearn.linear_model import Ridge

            X = np.array([activations[i] for i, p in enumerate(parsed) if p.result is not None])
            y = np.array([p.result for p in parsed if p.result is not None])

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
            print(f"  R2 score: {r2:.3f}")
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
            "prompts": [p.prompt for p in parsed],
            "operands_a": [p.operand_a for p in parsed],
            "operands_b": [p.operand_b for p in parsed],
            "operators": [p.operator.value if p.operator else None for p in parsed],
            "results": [p.result for p in parsed],
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
        print(f"  Has extracted direction: yes (R2={direction_stats.get('r2', '?'):.3f})")
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
    if method == InvocationMethod.STEER.value:
        if not has_direction:
            print("ERROR: 'steer' method requires --extract-direction during capture")
            return

        model_to_use = args.model or model_id
        print(f"\nLoading model: {model_to_use}")

        from ....introspection import ActivationSteering, SteeringConfig

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

        if method == InvocationMethod.LINEAR.value:
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

        elif method == InvocationMethod.EXTRAPOLATE.value:
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

        elif method == InvocationMethod.INTERPOLATE.value:
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

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection

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
        from ....inference.loader import HFLoader
        from ....models_v2.families.registry import detect_model_family, get_family_info

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

    mae = np.mean(errors)

    # Verdict depends on whether we have novel examples
    if len(novel) == 0:
        print(f"\n[WARNING] All {len(test_prompts)} test inputs were in the training data!")
        print("This doesn't test generalization - try inputs the model hasn't seen.")
        print("\nSuggested test (two-digit numbers not in training):")
        print(f"  lazarus introspect circuit test -c {circuit_path} -m {model_id} \\")
        print('    -p "12*13=|25*4=|11*11=" -r "156|100|121"')
    elif len(overlapping) > 0:
        novel_mae = np.mean(novel_errors)
        print(
            f"\n[WARNING] {len(overlapping)} of {len(test_prompts)} inputs were in training data (marked above)"
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
    print("ANGLES BETWEEN CIRCUITS (90 deg = orthogonal/independent)")
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

            print(f"  {c1_name} <-> {c2_name}: {angle:.1f} deg ({interpretation})")

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

    from ....introspection import ActivationSteering, SteeringConfig

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
