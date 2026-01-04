"""Experiment: Do experts specialize on semantic relationships?

Test if experts handle:
- Synonyms (happy/joyful, big/large)
- Antonyms (hot/cold, up/down)
- Hypernyms (dog→animal, car→vehicle)
- Word associations (doctor→hospital, king→queen)
"""

import asyncio
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_lazarus.introspection.moe import ExpertRouter


# Semantic relationship test prompts
SEMANTIC_PROMPTS = {
    "synonyms": [
        "Happy means the same as joyful.",
        "Big is similar to large.",
        "Fast equals quick.",
        "Smart is like intelligent.",
        "Beautiful means pretty.",
        "Angry is the same as furious.",
        "Cold means chilly.",
        "Hot means warm.",
        "Old means ancient.",
        "New means fresh.",
    ],
    "antonyms": [
        "Hot is the opposite of cold.",
        "Up is contrary to down.",
        "Big versus small.",
        "Fast against slow.",
        "Happy is the opposite of sad.",
        "Light versus dark.",
        "Old is the opposite of young.",
        "Good versus bad.",
        "Love is the opposite of hate.",
        "Rich versus poor.",
    ],
    "hypernyms": [
        "A dog is an animal.",
        "A car is a vehicle.",
        "A rose is a flower.",
        "A hammer is a tool.",
        "An apple is a fruit.",
        "A sparrow is a bird.",
        "A salmon is a fish.",
        "Oak is a tree.",
        "Python is a language.",
        "Paris is a city.",
    ],
    "associations": [
        "Doctor works at hospital.",
        "King and queen rule.",
        "Teacher teaches students.",
        "Chef cooks food.",
        "Pilot flies plane.",
        "Author writes books.",
        "Farmer grows crops.",
        "Artist paints pictures.",
        "Musician plays music.",
        "Scientist conducts experiments.",
    ],
    "analogies": [
        "King is to queen as man is to woman.",
        "Hot is to cold as up is to down.",
        "Dog is to puppy as cat is to kitten.",
        "Book is to read as song is to listen.",
        "Doctor is to patient as teacher is to student.",
        "Bird is to fly as fish is to swim.",
        "Pen is to write as brush is to paint.",
        "Eye is to see as ear is to hear.",
        "Sun is to day as moon is to night.",
        "Hand is to glove as foot is to shoe.",
    ],
}


async def analyze_semantic_experts(model_id: str):
    """Analyze which experts handle semantic relationships."""

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts, {len(info.moe_layers)} MoE layers\n")

        # Track which experts activate for each semantic category
        category_experts: dict[str, dict[int, Counter]] = {
            cat: defaultdict(Counter) for cat in SEMANTIC_PROMPTS
        }

        # Track token-level activations
        token_experts: dict[str, dict[int, list]] = {
            cat: defaultdict(list) for cat in SEMANTIC_PROMPTS
        }

        for category, prompts in SEMANTIC_PROMPTS.items():
            print(f"Analyzing {category}...")

            for prompt in prompts:
                weights = await router.capture_router_weights(prompt)

                for layer_weights in weights:
                    layer = layer_weights.layer_idx

                    for pos in layer_weights.positions:
                        for exp in pos.expert_indices:
                            category_experts[category][layer][exp] += 1
                            if len(token_experts[category][layer]) < 50:
                                token_experts[category][layer].append(
                                    (exp, pos.token, prompt[:40])
                                )

        # Analyze results
        print("\n" + "="*80)
        print("SEMANTIC PATTERN ANALYSIS")
        print("="*80)

        # Find experts that specialize in specific semantic categories
        for category in SEMANTIC_PROMPTS:
            print(f"\n{category.upper()} - Top experts per layer:")
            print("-" * 60)

            for layer in [0, 5, 11, 17, 23]:  # Sample layers
                counts = category_experts[category][layer]
                if counts:
                    top_experts = counts.most_common(3)
                    exp_str = ", ".join(f"E{e}({c})" for e, c in top_experts)
                    print(f"  Layer {layer:2d}: {exp_str}")

        # Cross-category comparison: which experts overlap?
        print("\n" + "="*80)
        print("CROSS-CATEGORY EXPERT OVERLAP")
        print("="*80)

        # For key layers, show which experts handle multiple categories
        for layer in [0, 11, 23]:
            print(f"\nLayer {layer}:")

            # Get top 5 experts for each category
            cat_top_experts = {}
            for cat in SEMANTIC_PROMPTS:
                counts = category_experts[cat][layer]
                cat_top_experts[cat] = set(e for e, _ in counts.most_common(5))

            # Find overlaps
            all_experts = set()
            for experts in cat_top_experts.values():
                all_experts.update(experts)

            for exp in sorted(all_experts)[:10]:
                cats_with_exp = [cat for cat, experts in cat_top_experts.items() if exp in experts]
                if len(cats_with_exp) > 1:
                    print(f"  E{exp:02d}: {', '.join(cats_with_exp)}")

        # Compare synonym vs antonym handling
        print("\n" + "="*80)
        print("SYNONYM vs ANTONYM EXPERT DIVERGENCE")
        print("="*80)

        for layer in [0, 5, 11, 17, 23]:
            syn_experts = set(e for e, _ in category_experts["synonyms"][layer].most_common(8))
            ant_experts = set(e for e, _ in category_experts["antonyms"][layer].most_common(8))

            shared = syn_experts & ant_experts
            syn_only = syn_experts - ant_experts
            ant_only = ant_experts - syn_experts

            print(f"\nLayer {layer:2d}:")
            print(f"  Shared (both):      {sorted(shared)}")
            print(f"  Synonym-specific:   {sorted(syn_only)}")
            print(f"  Antonym-specific:   {sorted(ant_only)}")

        # Show actual token activations for key relationships
        print("\n" + "="*80)
        print("SAMPLE ACTIVATIONS")
        print("="*80)

        for category in ["synonyms", "antonyms", "analogies"]:
            print(f"\n{category.upper()} (Layer 11):")
            samples = token_experts[category][11][:10]
            for exp, token, prompt in samples:
                print(f"  E{exp:02d} on '{token}' from: {prompt}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    asyncio.run(analyze_semantic_experts(model))
