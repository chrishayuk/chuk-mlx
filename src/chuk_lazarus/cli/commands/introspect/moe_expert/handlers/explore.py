"""Handler for 'explore' action - interactive MoE expert explorer.

Provides an interactive REPL for exploring expert routing patterns.
This module is a thin CLI wrapper - business logic is in ExploreService.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, ExploreService
from ...._constants import MoEDefaults
from ..formatters import format_header


def handle_explore(args: Namespace) -> None:
    """Handle the 'explore' action - interactive MoE expert explorer.

    Provides an interactive REPL for exploring expert routing:
    - Enter any prompt to see tokenization and expert routing
    - Compare prompts to see how patterns differ
    - Drill down into specific positions
    - View layer evolution

    Commands:
        [prompt]     Analyze a new prompt
        l N          Switch to layer N
        c "prompt"   Compare with another prompt
        a            Show all layers for current prompt
        d N          Deep dive on position N
        q            Quit

    Example:
        lazarus introspect moe-expert explore -m openai/gpt-oss-20b
    """
    asyncio.run(_async_explore(args))


async def _async_explore(args: Namespace) -> None:
    """Async implementation of interactive explorer."""
    model_id = args.model
    default_layer = getattr(args, "layer", None) or MoEDefaults.DEFAULT_LAYER

    print(format_header("MOE EXPERT EXPLORER"))
    print()
    print(f"Loading model: {model_id}...")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print()
        print("=" * 70)
        print(f"Model: {model_id}")
        print(f"Experts: {info.num_experts} total, {info.num_experts_per_tok} active per token")
        print(f"MoE Layers: {len(info.moe_layers)}")
        print("=" * 70)
        print()
        print("Commands:")
        print("  [prompt]     Analyze a new prompt")
        print("  l N          Switch to layer N")
        print('  c "prompt"   Compare with another prompt')
        print("  a            Show all layers for current prompt")
        print("  d N          Deep dive on position N")
        print("  q            Quit")
        print()

        current_prompt = None
        current_layer = default_layer

        while True:
            try:
                layer_str = f"L{current_layer:02d}"
                prompt_str = (
                    f' | "{current_prompt[:30]}..."'
                    if current_prompt and len(current_prompt) > 30
                    else (f' | "{current_prompt}"' if current_prompt else "")
                )
                cmd = input(f"[{layer_str}{prompt_str}]> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n")
                break

            if not cmd:
                continue

            if cmd.lower() == "q":
                print("Goodbye!")
                break

            if cmd.lower().startswith("l "):
                try:
                    new_layer = int(cmd[2:].strip())
                    if 0 <= new_layer < len(info.moe_layers):
                        current_layer = new_layer
                        print(f"Switched to layer {current_layer}")
                        if current_prompt:
                            await _show_analysis(router, current_prompt, current_layer)
                    else:
                        print(f"Invalid layer. Valid range: 0-{len(info.moe_layers) - 1}")
                except ValueError:
                    print("Usage: l <layer_number>")
                continue

            if cmd.lower().startswith("c "):
                compare_prompt = cmd[2:].strip().strip("\"'")
                if current_prompt:
                    await _compare_prompts(router, current_prompt, compare_prompt, current_layer)
                else:
                    print("No current prompt. Enter a prompt first.")
                continue

            if cmd.lower() == "a":
                if current_prompt:
                    await _show_all_layers(router, current_prompt, info.moe_layers)
                else:
                    print("No current prompt. Enter a prompt first.")
                continue

            if cmd.lower().startswith("d "):
                try:
                    pos = int(cmd[2:].strip())
                    if current_prompt:
                        await _deep_dive(router, current_prompt, pos, info.moe_layers)
                    else:
                        print("No current prompt. Enter a prompt first.")
                except ValueError:
                    print("Usage: d <position_number>")
                continue

            current_prompt = cmd
            await _show_analysis(router, current_prompt, current_layer)


async def _show_analysis(router: ExpertRouter, prompt: str, layer: int) -> None:
    """Show tokenization and expert routing for a prompt."""
    print()
    print("=" * 70)
    print("TOKENIZATION & ROUTING")
    print("=" * 70)
    print()
    print(f'Prompt: "{prompt}"')
    print(f"Layer: {layer}")
    print()

    weights = await router.capture_router_weights(prompt, layers=[layer])

    if not weights:
        print("No routing data captured.")
        return

    layer_weights = weights[0]
    positions = layer_weights.positions
    tokens = [p.token for p in positions]

    # Use service to analyze routing
    analysis = ExploreService.analyze_routing(tokens, positions)

    # Print tokenization table
    print("TOKENIZATION")
    print("-" * 70)
    print(f"{'Pos':<4} {'Token':<15} {'Type':<8} {'Trigram':<24}")
    print("-" * 70)

    for item in analysis:
        tok_display = item.token.strip()[:14] if item.token else ""
        print(f"{item.position:<4} {tok_display:<15} {item.token_type:<8} {item.trigram:<24}")

    print()

    # Print expert routing table
    print(f"EXPERT ROUTING (Layer {layer})")
    print("-" * 70)
    print(f"{'Pos':<4} {'Token':<12} {'Trigram':<22} {'Top-4 Experts':<30}")
    print("-" * 70)

    for item in analysis:
        tok_display = item.token.strip()[:11] if item.token else ""
        if item.expert_weights:
            exp_weights = sorted(zip(item.all_experts, item.expert_weights), key=lambda x: -x[1])
            experts_str = " ".join(f"E{e}({w:.0%})" for e, w in exp_weights[:4])
        else:
            experts_str = " ".join(f"E{e}" for e in item.all_experts[:4])
        print(f"{item.position:<4} {tok_display:<12} {item.trigram:<22} {experts_str:<30}")

    print()

    # Pattern summary using service
    patterns = ExploreService.find_patterns(tokens, positions)
    print("PATTERN SUMMARY")
    print("-" * 70)

    if patterns:
        for p in patterns[:5]:
            print(
                f'  Pos {p.position} "{p.token}" ({p.trigram}): {p.pattern_type} -> E{p.top_expert}'
            )
    else:
        print("  No notable patterns detected.")

    print()


async def _compare_prompts(router: ExpertRouter, prompt1: str, prompt2: str, layer: int) -> None:
    """Compare expert routing between two prompts."""
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    print(f'Prompt 1: "{prompt1}"')
    print(f'Prompt 2: "{prompt2}"')
    print(f"Layer: {layer}")
    print()

    weights1 = await router.capture_router_weights(prompt1, layers=[layer])
    weights2 = await router.capture_router_weights(prompt2, layers=[layer])

    if not weights1 or not weights2:
        print("Could not capture routing for one or both prompts.")
        return

    positions1 = weights1[0].positions
    positions2 = weights2[0].positions
    tokens1 = [p.token for p in positions1]
    tokens2 = [p.token for p in positions2]

    # Use service for comparison
    result = ExploreService.compare_routing(
        tokens1, positions1, tokens2, positions2, prompt1, prompt2, layer
    )

    # Display prompt 1
    print(f'"{prompt1}"')
    print("-" * 70)
    for item in result.tokens1:
        tok = item.token.strip()[:10]
        print(f"  {item.position}: {tok:<10} {item.trigram:<20} -> E{item.top_expert}")

    print()

    # Display prompt 2
    print(f'"{prompt2}"')
    print("-" * 70)
    for item in result.tokens2:
        tok = item.token.strip()[:10]
        print(f"  {item.position}: {tok:<10} {item.trigram:<20} -> E{item.top_expert}")

    print()
    print("EXPERT OVERLAP")
    print("-" * 70)
    print(f"  Shared experts: {result.shared_experts}")
    print(f"  Only in prompt 1: {result.only_prompt1}")
    print(f"  Only in prompt 2: {result.only_prompt2}")
    print(f"  Overlap: {result.overlap_ratio:.0%}")
    print()


async def _show_all_layers(router: ExpertRouter, prompt: str, moe_layers: list[int]) -> None:
    """Show expert routing across all layers for a prompt."""
    print()
    print("=" * 70)
    print("LAYER EVOLUTION")
    print("=" * 70)
    print()
    print(f'Prompt: "{prompt}"')
    print()

    weights = await router.capture_router_weights(prompt)

    if not weights:
        print("No routing data captured.")
        return

    first_layer = weights[0]
    tokens = [p.token.strip() for p in first_layer.positions]

    # Find interesting positions using service
    interesting = ExploreService.find_interesting_positions(tokens, top_k=4)

    # Show focused view for interesting positions
    for pos_idx in interesting:
        evolution = ExploreService.analyze_layer_evolution(tokens, weights, pos_idx)

        tok = evolution.token[:12]
        print(f'Position {pos_idx}: "{tok}" ({evolution.trigram})')
        print("-" * 60)

        def format_phase(phase: any) -> str:
            if not phase.layer_experts:
                return f"  {phase.phase_name.capitalize():<8} ({phase.layer_range}): --"
            pairs = [f"L{layer}:E{exp}" for layer, exp in phase.layer_experts[:4]]
            if len(phase.layer_experts) > 4:
                pairs.append("...")
            return f"  {phase.phase_name.capitalize():<8} ({phase.layer_range}): {' '.join(pairs)}  (E{phase.dominant_expert} dominates)"

        print(format_phase(evolution.early))
        print(format_phase(evolution.middle))
        print(format_phase(evolution.late))
        print()

    # Summary: show expert changes
    print("EXPERT TRANSITIONS")
    print("-" * 60)
    print("Positions where top expert changes between phases:")
    print()

    for pos_idx in interesting:
        evolution = ExploreService.analyze_layer_evolution(tokens, weights, pos_idx)
        tok = evolution.token[:8]

        if evolution.has_transition:
            print(f'  "{tok}": {" then ".join(evolution.transitions)}')
        else:
            dom = (
                evolution.early.dominant_expert
                or evolution.middle.dominant_expert
                or evolution.late.dominant_expert
            )
            print(f'  "{tok}": E{dom} (stable)')

    print()


async def _deep_dive(
    router: ExpertRouter, prompt: str, pos_idx: int, moe_layers: list[int]
) -> None:
    """Deep dive into a specific position."""
    print()
    print("=" * 70)
    print(f"DEEP DIVE: Position {pos_idx}")
    print("=" * 70)
    print()

    weights = await router.capture_router_weights(prompt)

    if not weights:
        print("No routing data captured.")
        return

    first_layer = weights[0]
    if pos_idx >= len(first_layer.positions):
        print(f"Invalid position. Valid range: 0-{len(first_layer.positions) - 1}")
        return

    tokens = [p.token for p in first_layer.positions]

    # Use service for deep dive
    result = ExploreService.deep_dive_position(tokens, weights, pos_idx)

    print(f'Token: "{result.token}"')
    print(f"Type: {result.token_type}")
    print(f"Trigram: {result.trigram}")
    print()
    print("Context:")
    print(f'  Previous: "{result.prev_token}" ({result.prev_type})')
    print(f'  Current:  "{result.token}" ({result.token_type})')
    print(f'  Next:     "{result.next_token}" ({result.next_type})')
    print()

    # Show routing across all layers
    print("ROUTING ACROSS ALL LAYERS")
    print("-" * 70)

    sorted_experts = result.all_experts[:8]

    # Print header
    header = "Layer"
    for exp in sorted_experts:
        header += f"  E{exp:02d}"
    print(header)
    print("-" * len(header))

    # Print data
    for layer, exp_weights in result.layer_routing:
        row = f"L{layer:02d}  "
        exp_dict = dict(exp_weights)
        for exp in sorted_experts:
            weight = exp_dict.get(exp, 0)
            if weight > 0.1:
                bar = "#" * int(weight * 5)
                row += f" {bar:<5}"
            else:
                row += "   -  "
        print(row)

    print()

    if result.dominant_expert is not None:
        print(f"FINDING: E{result.dominant_expert} dominates for trigram {result.trigram}")
        print(
            f"         Active in {len([1 for _, ew in result.layer_routing for e, _ in ew if e == result.dominant_expert])}/{len(result.layer_routing)} layers"
        )
        if result.peak_layer is not None:
            print(f"         Peak around layer {result.peak_layer}")

    print()
