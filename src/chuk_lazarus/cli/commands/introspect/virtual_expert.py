"""Virtual Expert CLI commands.

Adds virtual expert (tool-augmented) capabilities to models via CLI.

Usage:
    # Analyze expert routing (MoE models)
    lazarus introspect virtual-expert analyze -m openai/gpt-oss-20b

    # Solve a math problem with virtual expert
    lazarus introspect virtual-expert solve -m openai/gpt-oss-20b -p "127 * 89 = "

    # Run benchmark
    lazarus introspect virtual-expert benchmark -m model

    # Compare model-only vs virtual expert
    lazarus introspect virtual-expert compare -m model -p "127 * 89 = "

    # Interactive mode
    lazarus introspect virtual-expert interactive -m model
"""

import logging

logger = logging.getLogger(__name__)


def introspect_virtual_expert(args):
    """Virtual expert command dispatcher."""
    action = getattr(args, "action", "solve")

    if action == "analyze":
        _analyze_experts(args)
    elif action == "solve":
        _solve_with_expert(args)
    elif action == "benchmark":
        _run_benchmark(args)
    elif action == "compare":
        _compare_approaches(args)
    elif action == "interactive":
        _interactive_mode(args)
    else:
        print(f"Unknown action: {action}")


def _load_model(model_id: str):
    """Load model and tokenizer using Lazarus loader."""
    import json

    from ....inference.loader import DType, HFLoader
    from ....models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {model_id}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = len(list(model.model.layers))
    print(f"Loaded: {num_layers} layers")

    return model, tokenizer


def _is_moe_model(model) -> bool:
    """Check if model has MoE layers."""
    layers = list(model.model.layers)
    for layer in layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            return True
    return False


def _analyze_experts(args):
    """Analyze which experts activate for different prompt categories."""
    import mlx.core as mx

    from collections import defaultdict
    from ....introspection.moe import MoEHooks, MoECaptureConfig, get_moe_layer_info

    model, tokenizer = _load_model(args.model)

    if not _is_moe_model(model):
        print("Model is not MoE - use 'solve' or 'benchmark' for dense models")
        return

    print("\n" + "=" * 70)
    print("EXPERT CATEGORY ANALYSIS")
    print("=" * 70)

    # Test prompts by category
    categories = {
        "MATH": [
            "127 * 89 = ",
            "456 + 789 = ",
            "1000 - 250 = ",
            "What is 25 squared?",
        ],
        "CODE": [
            "def fibonacci(n):",
            "for i in range(10):",
            "import numpy as np",
            "class Calculator:",
        ],
        "LOGIC": [
            "If A implies B, and B implies C, then",
            "All men are mortal. Socrates is a man. Therefore",
            "NOT (A AND B) is equivalent to",
            "The contrapositive of P->Q is",
        ],
        "LANGUAGE": [
            "The capital of France is",
            "Once upon a time",
            "Hello, how are you",
            "The quick brown fox",
        ],
    }

    # Find MoE layers
    layers = list(model.model.layers)
    moe_layers = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            moe_layers.append(i)

    # Use middle MoE layer for analysis
    target_layer = moe_layers[len(moe_layers) // 2]
    info = get_moe_layer_info(model, target_layer)
    num_experts = info.num_experts if info else 32

    print(f"Model: {args.model}")
    print(f"MoE layers: {len(moe_layers)} ({moe_layers[0]} to {moe_layers[-1]})")
    print(f"Analyzing layer: {target_layer}")
    print(f"Number of experts: {num_experts}")
    print()

    # Track which experts activate for each category
    category_expert_counts = {cat: defaultdict(int) for cat in categories}

    hooks = MoEHooks(model)
    hooks.configure(MoECaptureConfig(
        capture_selected_experts=True,
        layers=[target_layer],
    ))

    for category, prompts in categories.items():
        for prompt in prompts:
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            hooks.forward(input_ids)

            if target_layer in hooks.state.selected_experts:
                experts = hooks.state.selected_experts[target_layer]
                last_experts = experts[0, -1].tolist()
                for exp_idx in last_experts:
                    category_expert_counts[category][exp_idx] += 1

    # Display results
    print(f"{'Expert':<10} ", end="")
    for cat in categories:
        print(f"{cat:<12}", end="")
    print()
    print("-" * (10 + 12 * len(categories)))

    all_experts = set()
    for counts in category_expert_counts.values():
        all_experts.update(counts.keys())

    def total_activations(exp):
        return sum(counts.get(exp, 0) for counts in category_expert_counts.values())

    sorted_experts = sorted(all_experts, key=total_activations, reverse=True)

    math_counts = category_expert_counts["MATH"]
    math_expert = max(math_counts, key=math_counts.get) if math_counts else None

    for exp_idx in sorted_experts[:15]:
        print(f"Expert {exp_idx:<3} ", end="")
        for cat in categories:
            count = category_expert_counts[cat].get(exp_idx, 0)
            print(f"{count:<12}", end="")

        annotations = []
        if exp_idx == math_expert:
            annotations.append("<- 'math expert'")

        uses = sum(1 for cat in categories if category_expert_counts[cat].get(exp_idx, 0) > 0)
        if uses >= 3:
            annotations.append("(multi-use)")

        if annotations:
            print(" ".join(annotations), end="")
        print()

    print("\n" + "=" * 70)


def _solve_with_expert(args):
    """Solve a problem with virtual expert."""
    model, tokenizer = _load_model(args.model)

    if _is_moe_model(model):
        from ....inference import VirtualMoEWrapper
        wrapper = VirtualMoEWrapper(model, tokenizer, args.model)
    else:
        from ....inference import VirtualDenseWrapper
        wrapper = VirtualDenseWrapper(model, tokenizer, args.model)

    print("Calibrating virtual expert...")
    wrapper.calibrate()

    prompt = args.prompt
    if not prompt.endswith("= ") and not prompt.endswith("="):
        prompt = prompt + " = "

    result = wrapper.solve(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Answer: {result.answer}")
    print(f"Correct: {result.is_correct}")
    if result.plugin_name:
        print(f"Plugin: {result.plugin_name}")
    print(f"Used virtual expert: {result.used_virtual_expert}")
    if result.routing_score:
        print(f"Routing score: {result.routing_score:.3f}")


def _run_benchmark(args):
    """Run benchmark comparing model vs virtual expert."""
    model, tokenizer = _load_model(args.model)

    if _is_moe_model(model):
        from ....inference import VirtualMoEWrapper
        wrapper = VirtualMoEWrapper(model, tokenizer, args.model)
    else:
        from ....inference import VirtualDenseWrapper
        wrapper = VirtualDenseWrapper(model, tokenizer, args.model)

    print("Calibrating virtual expert...")
    wrapper.calibrate()

    # Default problems
    problems = [
        "2 + 2 = ",
        "5 * 5 = ",
        "10 - 3 = ",
        "6 * 7 = ",
        "25 + 17 = ",
        "100 - 37 = ",
        "23 * 17 = ",
        "156 + 287 = ",
        "127 * 89 = ",
        "456 * 78 = ",
        "999 * 888 = ",
        "1234 + 5678 = ",
        "999 * 999 = ",
        "12345 + 67890 = ",
    ]

    if args.problems:
        if args.problems.startswith("@"):
            with open(args.problems[1:]) as f:
                problems = [line.strip() for line in f if line.strip()]
        else:
            problems = [p.strip() for p in args.problems.split("|")]

    analysis = wrapper.benchmark(problems)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {analysis.model_name}")
    print(f"Total problems: {analysis.total_problems}")
    print(f"Correct without virtual: {analysis.correct_without_virtual} ({analysis.accuracy_without:.1%})")
    print(f"Correct with virtual:    {analysis.correct_with_virtual} ({analysis.accuracy_with:.1%})")
    print(f"Improvement: {analysis.improvement:+.1%}")
    print(f"Times virtual used: {analysis.times_virtual_used}")
    print(f"Average routing score: {analysis.avg_routing_score:.3f}")

    if analysis.plugins_used:
        print(f"Plugins used: {analysis.plugins_used}")

    print("\n" + "-" * 70)
    print("PER-PROBLEM BREAKDOWN")
    print("-" * 70)

    for result in analysis.results:
        status = "OK" if result.is_correct else "X"
        virtual = "V" if result.used_virtual_expert else "M"
        print(f"  {status} [{virtual}] {result.prompt:<20} -> {result.answer:<15} (expected: {result.correct_answer})")

    print("=" * 70)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(analysis.model_dump(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


def _compare_approaches(args):
    """Compare model-only vs virtual expert."""
    model, tokenizer = _load_model(args.model)

    if _is_moe_model(model):
        from ....inference import VirtualMoEWrapper
        wrapper = VirtualMoEWrapper(model, tokenizer, args.model)
    else:
        from ....inference import VirtualDenseWrapper
        wrapper = VirtualDenseWrapper(model, tokenizer, args.model)

    print("Calibrating virtual expert...")
    wrapper.calibrate()

    prompt = args.prompt
    if not prompt.endswith("= ") and not prompt.endswith("="):
        prompt = prompt + " = "

    wrapper.compare(prompt)


def _interactive_mode(args):
    """Interactive REPL for testing virtual experts."""
    model, tokenizer = _load_model(args.model)

    if _is_moe_model(model):
        from ....inference import VirtualMoEWrapper
        wrapper = VirtualMoEWrapper(model, tokenizer, args.model)
        model_type = "MoE"
    else:
        from ....inference import VirtualDenseWrapper
        wrapper = VirtualDenseWrapper(model, tokenizer, args.model)
        model_type = "Dense"

    print("Calibrating virtual expert...")
    wrapper.calibrate()

    print("\n" + "=" * 70)
    print(f"VIRTUAL EXPERT - INTERACTIVE MODE ({model_type})")
    print("=" * 70)
    print("Commands:")
    print("  <expression>     - Solve with virtual expert")
    print("  !model           - Get model-only answer (no virtual expert)")
    print("  !compare <expr>  - Compare model vs virtual expert")
    print("  !threshold <f>   - Set routing threshold (0.0-1.0)")
    print("  !quit            - Exit")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.startswith("!"):
            parts = prompt[1:].split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "quit":
                print("Goodbye!")
                break
            elif cmd == "model" and len(parts) > 1:
                expr = parts[1]
                if not expr.endswith("= ") and not expr.endswith("="):
                    expr = expr + " = "
                answer = wrapper._generate_direct(expr)
                print(f"Model only: {answer}")
            elif cmd == "compare" and len(parts) > 1:
                expr = parts[1]
                if not expr.endswith("= ") and not expr.endswith("="):
                    expr = expr + " = "
                wrapper.compare(expr)
            elif cmd == "threshold" and len(parts) > 1:
                try:
                    t = float(parts[1])
                    wrapper.routing_threshold = t
                    print(f"Set threshold to {t}")
                except ValueError:
                    print("Invalid threshold")
            else:
                print("Unknown command. Try !quit, !model <expr>, !compare <expr>, or !threshold <n>")
            continue

        # Add "= " if missing
        if not prompt.endswith("= ") and not prompt.endswith("="):
            prompt = prompt + " = "

        result = wrapper.solve(prompt)
        print(f"Answer: {result.answer}")
        print(f"Correct: {result.is_correct}")
        if result.used_virtual_expert:
            print(f"Plugin: {result.plugin_name}")
        if result.routing_score:
            print(f"Routing score: {result.routing_score:.3f}")


__all__ = [
    "introspect_virtual_expert",
]
