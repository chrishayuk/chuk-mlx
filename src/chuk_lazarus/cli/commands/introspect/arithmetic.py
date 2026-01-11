"""Arithmetic study commands for introspection CLI.

Commands for systematic arithmetic testing and emergence layer analysis.
"""

__all__ = [
    "introspect_arithmetic",
]


def introspect_arithmetic(args):
    """Run systematic arithmetic study to find emergence layers.

    Tests arithmetic problems of varying difficulty and tracks when
    the correct answer first emerges as the top prediction.
    """
    import asyncio
    import json

    from ....introspection import (
        AnalysisConfig,
        ArithmeticTestSuite,
        Difficulty,
        LayerStrategy,
        ModelAnalyzer,
        apply_chat_template,
    )

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

            # Determine difficulty filter
            if args.hard_only:
                difficulty_filter = Difficulty.HARD
            elif args.easy_only:
                difficulty_filter = Difficulty.EASY
            else:
                difficulty_filter = None

            # Generate test cases using the Pydantic model
            test_suite = ArithmeticTestSuite.generate_test_cases(
                operations=["add", "mul", "sub", "div"],
                difficulty=difficulty_filter,
            )
            tests = test_suite.test_cases

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

            for test_case in tests:
                prompt = test_case.prompt
                expected = test_case.expected
                op = test_case.operator.value if test_case.operator else "unknown"
                difficulty = test_case.difficulty.value if test_case.difficulty else "unknown"
                magnitude = test_case.magnitude

                # Apply chat template if needed
                analysis_prompt = prompt
                if not use_raw and has_chat_template:
                    analysis_prompt = apply_chat_template(tokenizer, prompt)

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
                status = "[PASS]" if correct else "[FAIL]"
                emerg_str = f"L{emergence_layer}" if emergence_layer is not None else "never"
                print(
                    f"  {status} {prompt:<16} -> {final_token!r:<8} (expected {expected}, emerges @ {emerg_str})"
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
