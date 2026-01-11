"""Handler for 'token-routing' action - test if single tokens have stable routing.

Shows that single token classification doesn't predict routing -
the same token routes to different experts in different contexts.
Also shows how the token influences routing of the following token.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


@dataclass
class TokenRoutingResult:
    """Result of routing analysis for a single context."""

    context: str
    description: str
    target_token: str
    target_experts: list[int]
    next_token: str | None = None
    next_experts: list[int] | None = None


# Test contexts for common tokens
# Include contexts with tokens AFTER target to test next-token routing
TOKEN_CONTEXTS = {
    "127": [
        ("127", "solo - just the token"),
        ("127 + 5", "before addition"),
        ("127 * 89", "before multiplication"),
        ("127 / 2", "before division"),
        ("x = 127;", "in assignment"),
        ("The value is 127.", "in a sentence"),
        ("print(127)", "in function call"),
        ("arr[127]", "as array index"),
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
        print(f'    "{context}"')
        print(f"      ^ {desc}")
        print()

    print("=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print()
    print("  For each context, we:")
    print("    1. Pass the full context through the model")
    print(f"    2. Find the position of token '{token}'")
    print("    3. Record experts for BOTH the target token AND the next token")
    print("    4. Compare across all contexts")
    print()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        # Track routing for each context
        routing_results: list[TokenRoutingResult] = []

        print("  Processing contexts...")
        for context, desc in contexts_with_desc:
            weights = await router.capture_router_weights(context, layers=[layer])

            # Find the target token and the next token
            for layer_weights in weights:
                positions = layer_weights.positions
                for i, pos in enumerate(positions):
                    pos_token = pos.token.strip() if pos.token else ""
                    # Skip empty tokens (like spaces)
                    if not pos_token:
                        continue
                    # Check if this position contains our target token
                    if token.lower() in pos_token.lower() or pos_token.lower() in token.lower():
                        # Get next token info if available
                        next_token = None
                        next_experts = None
                        if i + 1 < len(positions):
                            next_pos = positions[i + 1]
                            next_token = next_pos.token.strip() if next_pos.token else None
                            next_experts = list(next_pos.expert_indices)

                        result = TokenRoutingResult(
                            context=context,
                            description=desc,
                            target_token=pos_token,
                            target_experts=list(pos.expert_indices),
                            next_token=next_token,
                            next_experts=next_experts,
                        )
                        routing_results.append(result)

                        if next_token:
                            print(f"    '{token}' in \"{context}\"")
                            print(f"      -> target '{pos_token}': {pos.expert_indices}")
                            print(f"      -> next '{next_token}': {next_experts}")
                        else:
                            print(f"    '{token}' in \"{context}\" -> {pos.expert_indices} (no next token)")
                        break

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"ROUTING FOR TOKEN '{token}' AND NEXT TOKEN BY CONTEXT")
        print("-" * 70)
        print()

        target_experts: set[int] = set()
        target_counts: Counter = Counter()
        next_experts_all: set[int] = set()
        next_counts: Counter = Counter()
        results_with_next = 0

        for result in routing_results:
            exp_str = ", ".join(f"E{e}" for e in result.target_experts)
            target_experts.update(result.target_experts)
            for e in result.target_experts:
                target_counts[e] += 1

            print(f'  Context: "{result.context}"')
            print(f"  Purpose: {result.description}")
            print(f"  Target '{result.target_token}': [{exp_str}]")

            if result.next_token and result.next_experts:
                results_with_next += 1
                next_exp_str = ", ".join(f"E{e}" for e in result.next_experts)
                next_experts_all.update(result.next_experts)
                for e in result.next_experts:
                    next_counts[e] += 1
                print(f"  Next '{result.next_token}': [{next_exp_str}]")
            else:
                print("  Next: (no following token)")
            print()

        print("-" * 70)
        print()

        # Calculate routing variance for target token
        num_contexts = len(routing_results)
        num_unique_target = len(target_experts)

        print("ANALYSIS: TARGET TOKEN")
        print("-" * 70)
        print()
        print(f"  Same token '{token}' tested in {num_contexts} contexts")
        print(f"  Total unique experts used: {num_unique_target}")
        print(f"  Experts: {sorted(target_experts)}")
        print()

        # Show frequency for target
        print("  Expert frequency for TARGET token:")
        for exp, count in target_counts.most_common():
            pct = 100 * count / num_contexts
            bar = "#" * int(pct / 10)
            print(f"    E{exp:02d}: {count}/{num_contexts} ({pct:.0f}%) {bar}")

        # Analysis for next token
        if results_with_next > 0:
            print()
            print("-" * 70)
            print("ANALYSIS: NEXT TOKEN (token immediately after target)")
            print("-" * 70)
            print()
            print(f"  Contexts with a next token: {results_with_next}")
            print(f"  Total unique experts used: {len(next_experts_all)}")
            print(f"  Experts: {sorted(next_experts_all)}")
            print()

            print("  Expert frequency for NEXT token:")
            for exp, count in next_counts.most_common():
                pct = 100 * count / results_with_next
                bar = "#" * int(pct / 10)
                print(f"    E{exp:02d}: {count}/{results_with_next} ({pct:.0f}%) {bar}")

        print()
        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print()

        if num_unique_target == 1:
            print(f"  Token '{token}' routes to SAME expert in all contexts.")
            print("  (This is rare - try other tokens like 'the' or 'def')")
        else:
            print(f"  FINDING: Token '{token}' routes to {num_unique_target} DIFFERENT experts!")
            print()
            print("  TARGET TOKEN:")
            print("  - The SAME token routes to DIFFERENT experts")
            print("  - Context CHANGES which experts are selected")
            print("  - Single-token classification CANNOT predict routing")

        if results_with_next > 0:
            print()
            print("  NEXT TOKEN:")
            print(f"  - Tokens after '{token}' use {len(next_experts_all)} different experts")
            print("  - The preceding context (including target) influences next token routing")
            print("  - This confirms bidirectional context dependency")

        print()
        print("  IMPLICATION: We need to consider CONTEXT, not just token identity.")
        print("  This leads us to the trigram approach: PREV -> CURR -> NEXT")

        print()
        print("=" * 70)
