"""Handler for 'context-test' action - test context independence.

Demonstrates that token routing is CONTEXT-DEPENDENT:
The same token routes to different experts based on surrounding context.
Also shows how routing stabilizes across layers.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter

# Default context tests for "127"
DEFAULT_CONTEXTS = {
    "numeric": [
        "111 127",
        "255 127",
        "0 127",
    ],
    "after_word": [
        "number 127",
        "value 127",
        "code 127",
    ],
    "after_article": [
        "the 127",
        "a 127",
    ],
    "standalone": [
        "127",
    ],
    "after_operator": [
        "+ 127",
        "= 127",
    ],
}


def handle_context_test(args: Namespace) -> None:
    """Handle the 'context-test' action - test if routing is context-independent.

    Shows that expert routing depends on CONTEXT, not just token identity:
    - Same token "127" routes to different experts depending on what comes before
    - Numeric context (111 127) vs word context (number 127) vs standalone (127)
    - Also shows how context dependence varies by layer phase

    Example:
        lazarus introspect moe-expert context-test -m openai/gpt-oss-20b --token 127
    """
    asyncio.run(_async_context_test(args))


async def _async_context_test(args: Namespace) -> None:
    """Async implementation of context_test handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    target_token = getattr(args, "token", None) or "127"
    custom_contexts = getattr(args, "contexts", None)

    # Build context tests
    if custom_contexts:
        # User provided custom contexts
        context_prompts = {"custom": custom_contexts.split(",")}
    else:
        # Use default contexts
        context_prompts = DEFAULT_CONTEXTS

    print()
    print("=" * 70)
    print("CONTEXT INDEPENDENCE TEST")
    print("=" * 70)
    print()
    print("=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print()
    print("  Common assumption: Routing is determined by token identity alone")
    print(f"    - Token '{target_token}' should always route to the same expert")
    print("    - Surrounding context shouldn't matter")
    print()
    print("  If this were true, we'd expect:")
    print(f"    - '{target_token}' after a number -> Expert X")
    print(f"    - '{target_token}' after a word   -> Expert X (same!)")
    print(f"    - '{target_token}' standalone     -> Expert X (same!)")
    print()
    print("=" * 70)
    print("EXPERIMENT SETUP")
    print("=" * 70)
    print()
    print(f"  Model: {model_id}")
    print(f"  Target token: '{target_token}'")
    print()
    print("  We'll place the same token in different contexts:")
    print()

    for context_type, prompts in context_prompts.items():
        print(f"  {context_type.upper().replace('_', ' ')}:")
        for prompt in prompts:
            print(f'    - "{prompt}"')
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
        else:
            # Test across layer phases: early, middle, late
            early = moe_layers[0]
            middle = moe_layers[len(moe_layers) // 2]
            late = moe_layers[-1]
            test_layers = [early, middle, late]

        print(f"  Testing layers: {test_layers}")
        print()

        # Collect all prompts to test
        all_prompts = []
        for prompts in context_prompts.values():
            all_prompts.extend(prompts)

        # Results by layer
        layer_results: dict[int, dict[str, set[int]]] = {}

        for test_layer in test_layers:
            print(f"  Layer {test_layer}:")
            layer_results[test_layer] = {}

            for context_type, prompts in context_prompts.items():
                experts_seen: set[int] = set()

                for prompt in prompts:
                    weights = await router.capture_router_weights(prompt, layers=[test_layer])

                    if weights and weights[0].positions:
                        last_pos = weights[0].positions[-1]
                        if last_pos.expert_indices:
                            experts_seen.add(last_pos.expert_indices[0])

                layer_results[test_layer][context_type] = experts_seen

                # Show what's happening
                expert_str = ", ".join(f"E{e}" for e in sorted(experts_seen))
                print(f"    {context_type:<15}: {expert_str}")

            print()

        print("=" * 70)
        print("RESULTS BY LAYER PHASE")
        print("=" * 70)
        print()

        for test_layer in test_layers:
            # Determine layer phase
            layer_idx = moe_layers.index(test_layer) if test_layer in moe_layers else 0
            total_layers = len(moe_layers)
            if layer_idx < total_layers // 3:
                phase = "Early"
            elif layer_idx < 2 * total_layers // 3:
                phase = "Middle"
            else:
                phase = "Late"

            # Count unique experts across all contexts at this layer
            all_experts: set[int] = set()
            for experts in layer_results[test_layer].values():
                all_experts.update(experts)

            # Check if routing is consistent across contexts
            is_consistent = len(all_experts) == 1

            status = "CONSISTENT" if is_consistent else "CONTEXT-DEPENDENT"
            expert_str = ", ".join(f"E{e}" for e in sorted(all_experts))

            print(f"  Layer {test_layer} ({phase}):")
            print(f"    Status:  {status}")
            print(f"    Experts: {expert_str}")
            print(f"    Unique:  {len(all_experts)}")
            print()

        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print()

        # Check patterns across layers
        early_layer = test_layers[0]
        late_layer = test_layers[-1]

        early_experts: set[int] = set()
        late_experts: set[int] = set()

        for experts in layer_results[early_layer].values():
            early_experts.update(experts)
        for experts in layer_results[late_layer].values():
            late_experts.update(experts)

        early_varies = len(early_experts) > 1
        late_varies = len(late_experts) > 1

        if early_varies and not late_varies:
            print("  FINDING: Routing STABILIZES across layers!")
            print()
            print(f"    Early layers (L{early_layer}): Context-dependent")
            print(f"      '{target_token}' routes to {len(early_experts)} different experts")
            print()
            print(f"    Late layers (L{late_layer}): Context-independent")
            print(f"      '{target_token}' consistently routes to E{list(late_experts)[0]}")
            print()
            print("  KEY INSIGHT:")
            print("    Early layers discriminate based on syntactic context.")
            print("    Later layers converge to semantic meaning.")
            print("    The model resolves ambiguity as processing deepens.")

        elif early_varies and late_varies:
            print("  FINDING: Routing is CONTEXT-DEPENDENT at all layers!")
            print()
            print(f"    The token '{target_token}' routes to different experts")
            print("    depending on context, even in late layers.")
            print()
            print("  KEY INSIGHT:")
            print("    This token has genuinely different meanings in different contexts.")
            print("    The model treats it differently throughout the entire forward pass.")

        elif not early_varies and not late_varies:
            print("  FINDING: Routing is CONSISTENT at all layers!")
            print()
            print(f"    The token '{target_token}' routes to the same expert")
            print("    regardless of context.")
            print()
            print("  KEY INSIGHT:")
            print("    This is unusual. Most tokens show some context dependence.")
            print("    This token may have a very strong, unambiguous meaning.")

        else:
            # late_varies but not early_varies - unusual
            print("  FINDING: Routing DIVERGES in later layers!")
            print()
            print(f"    Early: Consistent routing to E{list(early_experts)[0]}")
            print(f"    Late:  Routes to {len(late_experts)} different experts")
            print()
            print("  KEY INSIGHT:")
            print("    Context becomes MORE important in deeper layers.")
            print("    The model discovers semantic differences late in processing.")

        print()
        print("=" * 70)
