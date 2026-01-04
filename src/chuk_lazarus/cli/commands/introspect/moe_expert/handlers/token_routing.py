"""Handler for 'token-routing' action - test if single tokens have stable routing.

Shows that single token classification doesn't predict routing -
the same token routes to different experts in different contexts.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter, defaultdict

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


# Test contexts for common tokens
TOKEN_CONTEXTS = {
    "127": [
        ("127", "solo - just the token"),
        ("111 127", "after another number"),
        ("abc 127", "after a word"),
        ("= 127", "after operator (assignment)"),
        ("127 * 89", "before operator (multiplication)"),
        ("The value is 127.", "in a sentence"),
    ],
    "the": [
        ("the", "solo - just the token"),
        ("the cat", "before noun"),
        ("in the house", "after preposition"),
        ("The cat sat.", "sentence start (capitalized)"),
        ("feed the dog", "after verb"),
    ],
    "def": [
        ("def", "solo - just the token"),
        ("def foo():", "function definition (code)"),
        ("Use def to define.", "in a sentence (prose)"),
    ],
}


def handle_token_routing(args: Namespace) -> None:
    """Handle the 'token-routing' action - test single token routing stability.

    Shows that single token classification doesn't work:
    - Same token "127" routes to different experts
    - Context determines routing, not token identity

    Example:
        lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b --token 127
    """
    asyncio.run(_async_token_routing(args))


async def _async_token_routing(args: Namespace) -> None:
    """Async implementation of token-routing."""
    model_id = args.model
    token = getattr(args, "token", None) or "127"
    layer = getattr(args, "layer", None) or 11

    print(format_header("SINGLE TOKEN ROUTING TEST"))
    print()
    print("=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print()
    print("  After domain-test failed, maybe we can classify by TOKEN TYPE?")
    print()
    print("  Common assumption: Each token type routes to specific experts")
    print("    - Numbers like '127' -> always route to Expert X")
    print("    - Keywords like 'def' -> always route to Expert Y")
    print("    - Articles like 'the' -> always route to Expert Z")
    print()
    print("  If this were true, we could build a lookup table:")
    print("    token -> expert assignment (context-independent)")
    print()
    print("=" * 70)
    print("EXPERIMENT SETUP")
    print("=" * 70)
    print()
    print(f"  Model: {model_id}")
    print(f"  Target token: '{token}'")
    print(f"  Layer: {layer}")
    print()
    print("  We'll test the SAME token in DIFFERENT contexts:")
    print()

    # Get contexts for this token
    if token in TOKEN_CONTEXTS:
        contexts_with_desc = TOKEN_CONTEXTS[token]
    else:
        # Generate some default contexts
        contexts_with_desc = [
            (token, "solo - just the token"),
            (f"111 {token}", "after a number"),
            (f"abc {token}", "after a word"),
            (f"The value is {token}.", "in a sentence"),
        ]

    for context, desc in contexts_with_desc:
        print(f"    \"{context}\"")
        print(f"      ^ {desc}")
        print()

    print("=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print()
    print("  For each context, we:")
    print(f"    1. Pass the full context through the model")
    print(f"    2. Find the position of token '{token}'")
    print(f"    3. Record which experts are selected for that position")
    print(f"    4. Compare across all contexts")
    print()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        # Track routing for each context
        context_routing: list[tuple[str, str, list[int]]] = []

        print("  Processing contexts...")
        for context, desc in contexts_with_desc:
            weights = await router.capture_router_weights(context, layers=[layer])

            # Find the target token in the positions
            for layer_weights in weights:
                for pos in layer_weights.positions:
                    pos_token = pos.token.strip() if pos.token else ""
                    # Check if this position contains our target token
                    if token.lower() in pos_token.lower() or pos_token.lower() in token.lower():
                        context_routing.append((context, desc, list(pos.expert_indices)))
                        print(f"    Found '{token}' in \"{context}\" -> {pos.expert_indices}")
                        break

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"ROUTING FOR TOKEN '{token}' BY CONTEXT")
        print("-" * 70)
        print()

        all_experts: set[int] = set()
        expert_counts: Counter = Counter()

        for context, desc, experts in context_routing:
            exp_str = ", ".join(f"E{e}" for e in experts)
            all_experts.update(experts)
            for e in experts:
                expert_counts[e] += 1
            print(f"  Context: \"{context}\"")
            print(f"  Purpose: {desc}")
            print(f"  Experts: [{exp_str}]")
            print()

        print("-" * 70)
        print()

        # Calculate routing variance
        num_contexts = len(context_routing)
        num_unique_experts = len(all_experts)

        print("ANALYSIS")
        print("-" * 70)
        print()
        print(f"  Same token '{token}' tested in {num_contexts} contexts")
        print(f"  Total unique experts used: {num_unique_experts}")
        print(f"  Experts: {sorted(all_experts)}")
        print()

        # Show frequency
        print("  Expert frequency across contexts:")
        for exp, count in expert_counts.most_common():
            pct = 100 * count / num_contexts
            bar = "#" * int(pct / 10)
            print(f"    E{exp:02d}: {count}/{num_contexts} ({pct:.0f}%) {bar}")

        print()
        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print()

        if num_unique_experts == 1:
            print(f"  Token '{token}' routes to SAME expert in all contexts.")
            print("  (This is rare - try other tokens like 'the' or 'def')")
        else:
            print(f"  FINDING: Token '{token}' routes to {num_unique_experts} DIFFERENT experts!")
            print()
            print("  - The SAME token routes to DIFFERENT experts")
            print("  - Context CHANGES which experts are selected")
            print("  - Single-token classification CANNOT predict routing")
            print()
            print("  IMPLICATION: We need to consider CONTEXT, not just token identity.")
            print("  This leads us to the trigram approach: PREV -> CURR -> NEXT")

        print()
        print("=" * 70)
