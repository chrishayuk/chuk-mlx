"""Handler for 'vocab-contrib' action - analyze expert vocabulary contributions."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ......introspection.moe.logit_lens import (
    compute_expert_vocab_contribution,
    compute_token_expert_mapping,
    find_expert_specialists,
    print_expert_vocab_summary,
    print_token_expert_preferences,
)
from ..formatters import format_header


def handle_vocab_contrib(args: Namespace) -> None:
    """Handle the 'vocab-contrib' action - analyze expert vocabulary contributions.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert vocab-contrib -m openai/gpt-oss-20b
        lazarus introspect moe-expert vocab-contrib -m openai/gpt-oss-20b --layer 10 --top-k 30
    """
    asyncio.run(_async_vocab_contrib(args))


async def _async_vocab_contrib(args: Namespace) -> None:
    """Async implementation of vocab-contrib handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)
    top_k = getattr(args, "top_k", 50)
    vocab_sample = getattr(args, "vocab_sample", 5000)
    show_tokens = getattr(args, "show_tokens", False)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        model = router.model
        tokenizer = router.tokenizer

        print(format_header("EXPERT VOCABULARY CONTRIBUTION ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Experts: {info.num_experts}")
        print(f"Analyzing top-{top_k} tokens per expert")
        print()

        # Determine target layer
        target_layer = layer if layer is not None else info.moe_layers[len(info.moe_layers) // 2]

        print(f"Analyzing layer {target_layer}...")
        print()

        # Compute vocabulary contributions
        analysis = compute_expert_vocab_contribution(
            model,
            tokenizer,
            layer_idx=target_layer,
            top_k=top_k,
            vocab_sample_size=vocab_sample,
        )

        if not analysis.expert_contributions:
            print("Could not analyze vocabulary contributions")
            print("(Model may not have standard MoE structure)")
            return

        # Print summary
        print_expert_vocab_summary(analysis)

        # Find specialists
        specialists = find_expert_specialists(analysis, min_specialization=0.2)

        if specialists:
            print()
            print("Vocabulary Specialists:")
            print("-" * 50)
            for expert_idx, category, score in specialists[:10]:
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  Expert {expert_idx:2d}: [{bar}] {score:.2f} ({category})")

        # Optionally show token-to-expert mapping
        if show_tokens:
            print()
            mapping = compute_token_expert_mapping(
                model,
                tokenizer,
                layer_idx=target_layer,
            )
            print_token_expert_preferences(mapping)

        print()
        print("=" * 70)
