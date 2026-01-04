"""Experiment: Do experts specialize on RELATIONSHIP MARKERS?

Look at which experts handle:
- "same as", "similar to", "equals" (synonym markers)
- "opposite of", "contrary to", "versus" (antonym markers)
- "is a", "is an" (hypernym markers)
- "is to X as Y is to" (analogy markers)
"""

import asyncio
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_lazarus.introspection.moe import ExpertRouter


# Prompts designed to isolate relationship markers
MARKER_PROMPTS = {
    "synonym_marker": [
        "X means the same as Y.",
        "A is similar to B.",
        "One equals another.",
        "This is like that.",
        "Word means phrase.",
    ],
    "antonym_marker": [
        "X is the opposite of Y.",
        "A is contrary to B.",
        "One versus another.",
        "This against that.",
        "Up is opposed to down.",
    ],
    "hypernym_marker": [
        "A dog is an animal.",
        "A car is a vehicle.",
        "X is a type of Y.",
        "One is a kind of another.",
        "This is a form of that.",
    ],
    "analogy_marker": [
        "A is to B as C is to D.",
        "X is to Y as one is to two.",
        "King is to queen as man is to woman.",
        "Up is to down as left is to right.",
        "Hot is to cold as big is to small.",
    ],
}

# Key tokens to track
MARKER_TOKENS = {
    "synonym": ["same", "similar", "like", "equals", "means"],
    "antonym": ["opposite", "contrary", "versus", "against", "opposed"],
    "hypernym": ["is a", "is an", "type", "kind", "form"],
    "analogy": ["as", "to"],
}


async def analyze_relationship_markers(model_id: str):
    """Analyze which experts handle relationship marker tokens."""

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts\n")

        # Track experts for specific marker tokens
        marker_experts: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))

        for category, prompts in MARKER_PROMPTS.items():
            print(f"Analyzing {category}...")

            for prompt in prompts:
                weights = await router.capture_router_weights(prompt)

                for layer_weights in weights:
                    layer = layer_weights.layer_idx

                    for pos in layer_weights.positions:
                        token = pos.token.strip().lower() if pos.token else ""

                        # Check if this is a marker token
                        for marker_type, markers in MARKER_TOKENS.items():
                            if any(m in token for m in markers):
                                for exp in pos.expert_indices:
                                    marker_experts[marker_type][layer][exp] += 1

        # Analyze results
        print("\n" + "="*80)
        print("RELATIONSHIP MARKER SPECIALISTS")
        print("="*80)

        # Compare synonym vs antonym marker handling
        print("\n--- SYNONYM vs ANTONYM MARKERS ---")
        for layer in [0, 5, 11, 17, 23]:
            syn_counts = marker_experts["synonym"][layer]
            ant_counts = marker_experts["antonym"][layer]

            syn_top = set(e for e, _ in syn_counts.most_common(5))
            ant_top = set(e for e, _ in ant_counts.most_common(5))

            shared = syn_top & ant_top
            syn_only = syn_top - ant_top
            ant_only = ant_top - syn_only

            print(f"\nLayer {layer:2d}:")
            print(f"  Synonym markers: {dict(syn_counts.most_common(3))}")
            print(f"  Antonym markers: {dict(ant_counts.most_common(3))}")
            print(f"  Overlap: {shared}, Syn-only: {syn_only}, Ant-only: {ant_only}")

        # Analogy "as...to" specialists
        print("\n--- ANALOGY MARKER SPECIALISTS ---")
        for layer in [0, 11, 23]:
            analogy_counts = marker_experts["analogy"][layer]
            print(f"Layer {layer:2d}: {dict(analogy_counts.most_common(5))}")

        # Compare all marker types
        print("\n--- MARKER TYPE COMPARISON (Layer 11) ---")
        for marker_type in MARKER_TOKENS:
            counts = marker_experts[marker_type][11]
            top_experts = counts.most_common(3)
            print(f"{marker_type:12s}: {dict(top_experts)}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    asyncio.run(analyze_relationship_markers(model))
