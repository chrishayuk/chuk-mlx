"""Handler for 'context-window' action - test how much context the router sees.

Research question: Does the router use just the trigram, or the full attention span?

The hidden state at each position is an attention-weighted blend of ALL previous
tokens, but the trigram might capture the dominant signal.

Tests across layer phases (early/middle/late) to see if context sensitivity changes.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter

# Test cases: same trigram, varying extended context
CONTEXT_WINDOW_TESTS = {
    "arithmetic_plus": {
        "target": "+",
        "trigram": "NUM→OP→NUM",
        "contexts": [
            ("minimal", "2 + 3"),
            ("extended", "100 + 200 + 2 + 3"),
            ("instruction", "Calculate: 2 + 3"),
            ("sentence", "The sum is 2 + 3"),
            ("code", "result = 2 + 3"),
        ],
    },
    "analogy_to": {
        "target": "to",
        "trigram": "NOUN→TO→NOUN",
        "contexts": [
            ("minimal", "king to queen"),
            ("analogy_start", "King is to queen"),
            ("analogy_full", "Man is to woman as king is to queen"),
            ("instruction", "Compare: king to queen"),
        ],
    },
    "number_start": {
        "target": "127",
        "trigram": "^→NUM→OP",
        "contexts": [
            ("minimal", "127 +"),
            ("instruction", "Calculate 127 +"),
            ("sentence", "The value 127 +"),
            ("code", "x = 127 +"),
        ],
    },
    "def_keyword": {
        "target": "def",
        "trigram": "^→KW→FUNC",
        "contexts": [
            ("minimal", "def fibonacci"),
            ("comment", "# Function definition\ndef fibonacci"),
            ("docstring", '"""Helper"""\ndef fibonacci'),
            ("class", "class Math:\n    def fibonacci"),
        ],
    },
}


def handle_context_window(args: Namespace) -> None:
    """Handle the 'context-window' action - test context window effects.

    Research question: Does routing depend on:
    1. Just the trigram (local context)
    2. Extended context (sentence-level)
    3. Full attention span (everything before)

    Tests across layer phases to see if context sensitivity changes with depth.

    Example:
        lazarus introspect moe-expert context-window -m openai/gpt-oss-20b
    """
    asyncio.run(_async_context_window(args))


async def _async_context_window(args: Namespace) -> None:
    """Async implementation of context-window handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    test_name = getattr(args, "test", None)

    print()
    print("=" * 70)
    print("CONTEXT WINDOW TEST")
    print("=" * 70)
    print()
    print("=" * 70)
    print("RESEARCH QUESTION")
    print("=" * 70)
    print()
    print("  How much context does the router actually use?")
    print()
    print("  The hidden state at each position is computed by attention,")
    print("  which theoretically sees the ENTIRE preceding context.")
    print("  But does the router actually use all of it?")
    print()
    print("  Possible findings:")
    print("    1. TRIGRAM SUFFICIENT: Same trigram → same expert always")
    print("    2. EXTENDED CONTEXT:   Broader context changes routing")
    print("    3. LAYER DEPENDENT:    Context sensitivity varies by layer phase")
    print()
    print("=" * 70)
    print("EXPERIMENT DESIGN")
    print("=" * 70)
    print()
    print("  We keep the TRIGRAM constant but vary the EXTENDED context.")
    print("  If the router only sees the trigram, experts should be identical.")
    print("  We test across layer phases: Early, Middle, Late.")
    print()

    # Select which tests to run
    if test_name and test_name in CONTEXT_WINDOW_TESTS:
        tests_to_run = {test_name: CONTEXT_WINDOW_TESTS[test_name]}
    else:
        tests_to_run = CONTEXT_WINDOW_TESTS

    for name, test_config in tests_to_run.items():
        print(f"  {name.upper()}:")
        print(f"    Target token: '{test_config['target']}'")
        print(f"    Trigram type: {test_config['trigram']}")
        print("    Contexts:")
        for ctx_name, ctx in test_config["contexts"]:
            # Show first line only for multi-line contexts
            display = ctx.split("\n")[-1] if "\n" in ctx else ctx
            print(f'      {ctx_name:<12}: "{display}"')
        print()

    print("=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print()
    print(f"  Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        moe_layers = info.moe_layers

        # Determine which layers to test
        if layer is not None:
            # User specified a single layer
            test_layers = [layer]
            layer_names = {layer: "User-specified"}
        else:
            # Test across layer phases: early, middle, late
            early = moe_layers[0]
            middle = moe_layers[len(moe_layers) // 2]
            late = moe_layers[-1]
            test_layers = [early, middle, late]
            layer_names = {early: "Early", middle: "Middle", late: "Late"}

        print(f"  Testing layers: {test_layers}")
        print()

        # Results structure: {layer: {test_name: {context: experts}}}
        layer_results: dict[int, dict[str, dict[str, tuple[int, ...]]]] = {}

        for test_layer in test_layers:
            phase = layer_names.get(test_layer, f"L{test_layer}")
            print(f"  Layer {test_layer} ({phase}):")
            layer_results[test_layer] = {}

            for name, test_config in tests_to_run.items():
                target = test_config["target"]
                layer_results[test_layer][name] = {}

                for ctx_name, ctx in test_config["contexts"]:
                    weights = await router.capture_router_weights(ctx, layers=[test_layer])

                    if weights and weights[0].positions:
                        # Find the position of target token
                        experts = None
                        for pos in reversed(weights[0].positions):
                            if target.lower() in pos.token.lower():
                                experts = pos.expert_indices
                                break

                        if experts is None:
                            experts = weights[0].positions[-1].expert_indices

                        layer_results[test_layer][name][ctx_name] = experts

            # Show summary for this layer
            for name in tests_to_run:
                results = layer_results[test_layer][name]
                primary_experts = [exp[0] for exp in results.values() if exp]
                unique = len(set(primary_experts))
                status = "STABLE" if unique == 1 else f"VARIES ({unique})"
                experts_str = ", ".join(f"E{e}" for e in sorted(set(primary_experts)))
                print(f"    {name:<18}: [{status:<10}] {experts_str}")

            print()

        print("=" * 70)
        print("RESULTS BY LAYER PHASE")
        print("=" * 70)
        print()

        # Analyze each layer
        layer_verdicts: dict[int, str] = {}

        for test_layer in test_layers:
            phase = layer_names.get(test_layer, f"L{test_layer}")

            trigram_sufficient = 0
            extended_matters = 0

            for name in tests_to_run:
                results = layer_results[test_layer][name]
                primary_experts = [exp[0] for exp in results.values() if exp]
                if len(set(primary_experts)) == 1:
                    trigram_sufficient += 1
                else:
                    extended_matters += 1

            total = trigram_sufficient + extended_matters

            if trigram_sufficient == total:
                verdict = "TRIGRAM SUFFICIENT"
            elif extended_matters == total:
                verdict = "EXTENDED CONTEXT MATTERS"
            else:
                verdict = f"MIXED ({trigram_sufficient}/{total} stable)"

            layer_verdicts[test_layer] = verdict

            print(f"  Layer {test_layer} ({phase}):")
            print(f"    Trigram sufficient: {trigram_sufficient}/{total} tests")
            print(f"    Extended matters:   {extended_matters}/{total} tests")
            print(f"    Verdict: {verdict}")
            print()

        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print()

        # Check for patterns across layers
        all_extended = all(v == "EXTENDED CONTEXT MATTERS" for v in layer_verdicts.values())
        all_trigram = all(v == "TRIGRAM SUFFICIENT" for v in layer_verdicts.values())

        early_layer = test_layers[0]
        late_layer = test_layers[-1]
        early_verdict = layer_verdicts[early_layer]
        late_verdict = layer_verdicts[late_layer]

        if all_extended:
            print("  FINDING: EXTENDED CONTEXT MATTERS AT ALL LAYERS")
            print()
            print("    The router sees beyond the immediate trigram at every layer phase.")
            print("    Attention brings distant context into the routing decision.")
            print()
            print("  IMPLICATION:")
            print("    The trigram is a useful heuristic for understanding PATTERN TYPES,")
            print("    but the full attention span modulates WHICH EXPERT handles it.")

        elif all_trigram:
            print("  FINDING: TRIGRAM IS SUFFICIENT AT ALL LAYERS")
            print()
            print("    The immediate trigram determines routing regardless of extended context.")
            print("    The router focuses on local patterns.")
            print()
            print("  IMPLICATION:")
            print("    The trigram captures the dominant routing signal.")
            print("    Extended context may be 'averaged out' by attention.")

        elif "TRIGRAM" in early_verdict and "EXTENDED" in late_verdict:
            print("  FINDING: CONTEXT SENSITIVITY INCREASES WITH DEPTH")
            print()
            print(f"    Early layers (L{early_layer}): Trigram sufficient")
            print(f"    Late layers (L{late_layer}): Extended context matters")
            print()
            print("  IMPLICATION:")
            print("    Early layers use local patterns for routing.")
            print("    Later layers integrate broader context.")
            print("    The model's understanding deepens as processing continues.")

        elif "EXTENDED" in early_verdict and "TRIGRAM" in late_verdict:
            print("  FINDING: ROUTING STABILIZES WITH DEPTH")
            print()
            print(f"    Early layers (L{early_layer}): Extended context matters")
            print(f"    Late layers (L{late_layer}): Trigram sufficient")
            print()
            print("  IMPLICATION:")
            print("    Early layers are sensitive to full context.")
            print("    Later layers converge to stable patterns.")
            print("    The model resolves ambiguity as processing deepens.")

        else:
            print("  FINDING: MIXED BEHAVIOR ACROSS LAYERS")
            print()
            for test_layer in test_layers:
                phase = layer_names.get(test_layer, f"L{test_layer}")
                print(f"    Layer {test_layer} ({phase}): {layer_verdicts[test_layer]}")
            print()
            print("  IMPLICATION:")
            print("    Context sensitivity varies by layer phase and token type.")
            print("    No single model explains routing across the full network.")

        print()
        print("=" * 70)
