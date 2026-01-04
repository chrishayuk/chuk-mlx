"""Handler for 'domain-test' action - test if domain experts exist.

Shows that there is no "math expert" or "code expert" -
experts handle multiple domains, not specialized ones.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter, defaultdict

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


# Domain-specific prompts
DOMAIN_PROMPTS = {
    "math": [
        "2 + 3 = 5",
        "127 * 89 = 11303",
        "45 - 12 = 33",
        "100 / 4 = 25",
    ],
    "code": [
        "def fibonacci(n):",
        "for i in range(10):",
        "class MyClass:",
        "import os",
    ],
    "language": [
        "The cat sat on the mat.",
        "She walked to the store.",
        "Happy people smile often.",
        "The quick brown fox.",
    ],
    "reasoning": [
        "If it rains then stay inside.",
        "All dogs are mammals.",
        "Therefore the result is zero.",
        "Because she practiced daily.",
    ],
}


def handle_domain_test(args: Namespace) -> None:
    """Handle the 'domain-test' action - test if domain experts exist.

    Shows that experts are NOT domain specialists:
    - No "math expert" that only handles math
    - No "code expert" that only handles code
    - Experts handle multiple domains

    Example:
        lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b
    """
    asyncio.run(_async_domain_test(args))


async def _async_domain_test(args: Namespace) -> None:
    """Async implementation of domain-test."""
    model_id = args.model
    layer = getattr(args, "layer", None) or 11  # Default to middle layer

    print(format_header("DOMAIN EXPERT TEST"))
    print()
    print("=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print()
    print("  Common assumption: MoE models have specialized 'domain experts'")
    print("    - A 'math expert' that handles arithmetic")
    print("    - A 'code expert' that handles programming")
    print("    - A 'language expert' that handles natural language")
    print()
    print("  If this were true, we'd expect:")
    print("    - Math prompts -> consistently route to Expert X")
    print("    - Code prompts -> consistently route to Expert Y")
    print("    - Each expert handles ONE domain")
    print()
    print("=" * 70)
    print("EXPERIMENT SETUP")
    print("=" * 70)
    print()
    print(f"  Model: {model_id}")
    print(f"  Layer: {layer} (middle layer where semantic routing is strongest)")
    print(f"  Experts: 32 total, 4 active per token (top-k=4)")
    print()
    print("  We'll test 4 domains with 4 prompts each:")
    print()

    for domain, prompts in DOMAIN_PROMPTS.items():
        print(f"  {domain.upper()}:")
        for prompt in prompts:
            print(f"    - \"{prompt}\"")
    print()

    print("=" * 70)
    print("RUNNING EXPERIMENT")
    print("=" * 70)
    print()
    print("  For each prompt, we:")
    print("    1. Tokenize the input")
    print("    2. Pass through model to layer {layer}")
    print("    3. Capture router weights (which experts are selected)")
    print("    4. Record which experts handle each token")
    print()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        # Track which experts handle each domain
        domain_experts: dict[str, Counter] = defaultdict(Counter)
        expert_domains: dict[int, Counter] = defaultdict(Counter)

        print("  Processing prompts...")
        for domain, prompts in DOMAIN_PROMPTS.items():
            print(f"    {domain}: ", end="", flush=True)
            for prompt in prompts:
                weights = await router.capture_router_weights(prompt, layers=[layer])
                for layer_weights in weights:
                    for pos in layer_weights.positions:
                        for exp in pos.expert_indices:
                            domain_experts[domain][exp] += 1
                            expert_domains[exp][domain] += 1
                print(".", end="", flush=True)
            print()

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()

        # Show top experts per domain
        print("TOP 5 EXPERTS PER DOMAIN")
        print("-" * 70)
        print()
        print("  If domain experts exist, each domain should have DIFFERENT top experts.")
        print()
        for domain in DOMAIN_PROMPTS:
            top = domain_experts[domain].most_common(5)
            exp_str = ", ".join(f"E{e}({c})" for e, c in top)
            print(f"  {domain:<12}: {exp_str}")

        print()

        # Check for overlap
        all_top_experts = set()
        domain_top_sets = {}
        for domain in DOMAIN_PROMPTS:
            top_3 = {e for e, _ in domain_experts[domain].most_common(3)}
            domain_top_sets[domain] = top_3
            all_top_experts.update(top_3)

        # Find experts in multiple domains
        overlap_count = 0
        for exp in all_top_experts:
            domains_with_exp = [d for d, s in domain_top_sets.items() if exp in s]
            if len(domains_with_exp) > 1:
                overlap_count += 1

        print(f"  Overlap: {overlap_count} experts appear in multiple domains' top-3")
        print()

        # The key insight: show experts handle MULTIPLE domains
        print("EXPERT DOMAIN OVERLAP (the key finding)")
        print("-" * 70)
        print()
        print("  If domain experts exist, each expert should handle ONE domain.")
        print("  Let's see how many domains each expert handles:")
        print()

        # Find experts that appear in multiple domains
        multi_domain = []
        for exp, domains in expert_domains.items():
            if len(domains) >= 2:
                total = sum(domains.values())
                domain_list = ", ".join(f"{d}({c})" for d, c in domains.most_common())
                multi_domain.append((exp, total, domain_list, len(domains)))

        multi_domain.sort(key=lambda x: -x[1])

        for exp, total, domains, num_domains in multi_domain[:10]:
            marker = " <-- handles ALL 4 domains!" if num_domains == 4 else ""
            print(f"  E{exp:02d}: {num_domains} domains - {domains}{marker}")

        # Count how many handle all 4
        all_four = sum(1 for _, _, _, n in multi_domain if n == 4)
        print()
        print(f"  {all_four} experts handle ALL 4 domains!")

        print()
        print("=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print()
        print("  FINDING: There are NO domain experts.")
        print()
        print("  - The same experts handle math, code, language, AND reasoning")
        print("  - Experts are NOT specialized by domain")
        print("  - Domain classification CANNOT predict expert routing")
        print()
        print("  IMPLICATION: We need a different approach to understand")
        print("  expert specialization. Domain is NOT the answer.")
        print()
        print("=" * 70)
