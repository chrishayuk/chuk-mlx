"""Activation patching and causal intervention commands.

Commands for testing causal relationships through activation patching
and commutativity analysis.
"""

__all__ = [
    "introspect_commutativity",
    "introspect_patch",
]


def introspect_commutativity(args):
    """Test if the model's internal representation respects commutativity (A*B = B*A).

    For multiplication, A*B and B*A should produce the same answer. This test checks
    whether the internal representations for commutative pairs are similar, which
    would indicate a lookup table structure rather than an algorithm.

    High commutativity similarity (>0.99) suggests the model memorizes individual facts
    rather than computing them algorithmically.
    """
    import asyncio
    import json

    from ....introspection import CommutativityAnalyzer
    from ....introspection.ablation import AblationStudy

    async def run():
        print(f"Loading model: {args.model}")
        study = AblationStudy.from_pretrained(args.model)
        model = study.adapter.model
        tokenizer = study.adapter.tokenizer
        config = study.adapter.config

        # Parse layers
        layer = args.layer if args.layer else None

        # Parse pairs or let analyzer generate them
        pairs = None
        if args.pairs:
            # Parse explicit pairs: "2*3,3*2|7*8,8*7"
            pair_specs = args.pairs.split("|")
            pairs = []
            for spec in pair_specs:
                p1, p2 = spec.split(",")
                pairs.append((p1.strip(), p2.strip()))

        # Use the async-native CommutativityAnalyzer
        analyzer = CommutativityAnalyzer(model=model, tokenizer=tokenizer, config=config)
        result = await analyzer.analyze(layer=layer, pairs=pairs)

        # Print results
        print(f"Analyzing at layer {result.layer}")
        print(f"Testing {result.num_pairs} commutative pairs")

        print(f"\n{'Pair A':<12} {'Pair B':<12} {'Cosine Sim':<12}")
        print("-" * 40)

        for pair in result.pairs:
            print(f"{pair.prompt_a:<12} {pair.prompt_b:<12} {pair.similarity:.6f}")

        # Summary statistics
        print(f"\n{'=' * 50}")
        print("COMMUTATIVITY ANALYSIS")
        print(f"{'=' * 50}")
        print(f"Mean similarity: {result.mean_similarity:.6f}")
        print(f"Std similarity:  {result.std_similarity:.6f}")
        print(f"Min similarity:  {result.min_similarity:.6f}")
        print(f"Max similarity:  {result.max_similarity:.6f}")

        # Interpretation using the Pydantic model's properties
        print(f"\n[{result.level.value.upper()}] {result.interpretation}")

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

    asyncio.run(run())


def introspect_patch(args):
    """Perform activation patching: transfer activations from source to target prompt.

    Activation patching is a causal intervention technique that tests whether
    activations from one prompt can transfer computation to another prompt.

    For example, patching activations from "7*8=" into "7+8=" at the right layer
    should cause the model to output "56" instead of "15".

    This is useful for:
    - Identifying which layers encode the "computation" vs "operands"
    - Testing cross-operation transfer
    - Finding the causal layer for answer production
    """
    import asyncio
    import json

    from ....introspection import ActivationPatcher, extract_expected_answer, parse_layers_arg
    from ....introspection.ablation import AblationStudy

    async def run():
        print(f"Loading model: {args.model}")
        study = AblationStudy.from_pretrained(args.model)
        model = study.adapter.model
        tokenizer = study.adapter.tokenizer
        config = study.adapter.config

        source_prompt = args.source
        target_prompt = args.target

        print(f"Source: {source_prompt!r}")
        print(f"Target: {target_prompt!r}")

        # Compute expected answers using framework utility
        source_answer = extract_expected_answer(source_prompt)
        target_answer = extract_expected_answer(target_prompt)

        if source_answer:
            print(f"Source answer: {source_answer}")
        if target_answer:
            print(f"Target answer: {target_answer}")

        # Parse layers using framework utility
        layers = parse_layers_arg(args.layers if args.layers else None)
        if layers is None and args.layer:
            layers = [args.layer]
        elif layers is None:
            # Sweep key layers
            num_layers = study.adapter.num_layers
            layers = list(range(0, num_layers, max(1, num_layers // 10)))

        print(f"Patching at layers: {layers}")

        blend = args.blend if args.blend else 1.0

        # Use the async-native ActivationPatcher
        patcher = ActivationPatcher(model=model, tokenizer=tokenizer, config=config)
        result = await patcher.sweep_layers(
            target_prompt=target_prompt,
            source_prompt=source_prompt,
            layers=layers,
            blend=blend,
            source_answer=source_answer,
            target_answer=target_answer,
        )

        # Print results
        print(f"\nBaseline top-5: {result.baseline_token!r} ({result.baseline_prob:.3f})")

        print(f"\n{'=' * 70}")
        print("ACTIVATION PATCHING RESULTS")
        print(f"{'=' * 70}")
        print(f"{'Layer':<8} {'Top Token':<15} {'Prob':<10} {'Effect'}")
        print("-" * 70)

        for layer_result in result.layer_results:
            print(
                f"L{layer_result.layer:<7} "
                f"{layer_result.top_token!r:<15} "
                f"{layer_result.top_prob:.3f}      "
                f"{layer_result.effect.value}"
            )

        # Summary
        transferred_layers = [
            r.layer for r in result.layer_results if r.effect.value == "transferred"
        ]
        if transferred_layers:
            print(f"\n=> Source answer transferred at layers: {transferred_layers}")
        else:
            print("\n=> No transfer detected (source answer not produced)")

        # Save results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

    asyncio.run(run())
