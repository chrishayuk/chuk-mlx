"""Handler for 'explore' action - interactive MoE expert explorer.

Provides an interactive REPL for exploring expert routing patterns
in real-time. Perfect for video demonstrations and understanding
how context affects expert selection.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header
from .full_taxonomy import classify_token


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
    default_layer = getattr(args, "layer", None) or 11

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
        _ = None  # Placeholder for future caching

        while True:
            try:
                # Show current context
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

            # Quit
            if cmd.lower() == "q":
                print("Goodbye!")
                break

            # Layer switch
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

            # Compare
            if cmd.lower().startswith("c "):
                compare_prompt = cmd[2:].strip().strip("\"'")
                if current_prompt:
                    await _compare_prompts(router, current_prompt, compare_prompt, current_layer)
                else:
                    print("No current prompt. Enter a prompt first.")
                continue

            # All layers
            if cmd.lower() == "a":
                if current_prompt:
                    await _show_all_layers(router, current_prompt, info.moe_layers)
                else:
                    print("No current prompt. Enter a prompt first.")
                continue

            # Deep dive
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

            # New prompt
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

    # Classify tokens and build trigrams
    tokens = [p.token for p in positions]
    sem_types = [classify_token(t) for t in tokens]

    # Print tokenization table
    print("TOKENIZATION")
    print("-" * 70)
    print(f"{'Pos':<4} {'Token':<15} {'Type':<8} {'Trigram':<24}")
    print("-" * 70)

    for i, (tok, sem_type) in enumerate(zip(tokens, sem_types)):
        prev_t = sem_types[i - 1] if i > 0 else "^"
        next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{sem_type}→{next_t}"

        tok_display = tok.strip()[:14] if tok else ""
        print(f"{i:<4} {tok_display:<15} {sem_type:<8} {trigram:<24}")

    print()

    # Print expert routing table
    print(f"EXPERT ROUTING (Layer {layer})")
    print("-" * 70)
    print(f"{'Pos':<4} {'Token':<12} {'Trigram':<22} {'Top-4 Experts':<30}")
    print("-" * 70)

    for i, pos in enumerate(positions):
        tok_display = tokens[i].strip()[:11] if tokens[i] else ""
        prev_t = sem_types[i - 1] if i > 0 else "^"
        curr_t = sem_types[i]
        next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"

        # Format expert weights if available
        experts_str = ""
        if hasattr(pos, "expert_weights") and pos.expert_weights:
            # Sort by weight
            exp_weights = sorted(zip(pos.expert_indices, pos.expert_weights), key=lambda x: -x[1])
            experts_str = " ".join(f"E{e}({w:.0%})" for e, w in exp_weights[:4])
        else:
            experts_str = " ".join(f"E{e}" for e in pos.expert_indices[:4])

        print(f"{i:<4} {tok_display:<12} {trigram:<22} {experts_str:<30}")

    print()

    # Pattern summary
    _show_pattern_summary(tokens, sem_types, positions)


def _show_pattern_summary(tokens: list, sem_types: list, positions: list) -> None:
    """Show a summary of interesting patterns found."""
    print("PATTERN SUMMARY")
    print("-" * 70)

    # Find notable patterns
    patterns_found = []

    for i, (tok, sem_type) in enumerate(zip(tokens, sem_types)):
        prev_t = sem_types[i - 1] if i > 0 else "^"
        next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{sem_type}→{next_t}"

        pos = positions[i]
        top_exp = pos.expert_indices[0] if pos.expert_indices else None

        # Identify interesting patterns
        pattern_type = None
        if "TO" in trigram and "AS" in sem_types:
            pattern_type = "analogy marker"
        elif "→AS→" in trigram:
            pattern_type = "analogy pivot"
        elif "→OP→" in trigram or "OP→" in trigram:
            pattern_type = "arithmetic operator"
        elif "NUM→OP" in trigram:
            pattern_type = "number before op"
        elif "^→KW" in trigram:
            pattern_type = "code start"
        elif "^→" in trigram:
            pattern_type = "sequence start"
        elif "→$" in trigram:
            pattern_type = "sequence end"
        elif "→SYN→" in trigram:
            pattern_type = "synonym relation"
        elif "→ANT→" in trigram:
            pattern_type = "antonym relation"

        if pattern_type:
            patterns_found.append((i, tok.strip(), trigram, top_exp, pattern_type))

    if patterns_found:
        for pos_idx, tok, trigram, exp, ptype in patterns_found[:5]:
            print(f'  Pos {pos_idx} "{tok}" ({trigram}): {ptype} -> E{exp}')
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

    # Get routing for both
    weights1 = await router.capture_router_weights(prompt1, layers=[layer])
    weights2 = await router.capture_router_weights(prompt2, layers=[layer])

    if not weights1 or not weights2:
        print("Could not capture routing for one or both prompts.")
        return

    # Display prompt 1
    print(f'"{prompt1}"')
    print("-" * 70)

    positions1 = weights1[0].positions
    tokens1 = [p.token for p in positions1]
    sem_types1 = [classify_token(t) for t in tokens1]

    for i, pos in enumerate(positions1):
        tok = tokens1[i].strip()[:10]
        prev_t = sem_types1[i - 1] if i > 0 else "^"
        curr_t = sem_types1[i]
        next_t = sem_types1[i + 1] if i < len(sem_types1) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"
        top_exp = pos.expert_indices[0] if pos.expert_indices else None
        print(f"  {i}: {tok:<10} {trigram:<20} -> E{top_exp}")

    print()

    # Display prompt 2
    print(f'"{prompt2}"')
    print("-" * 70)

    positions2 = weights2[0].positions
    tokens2 = [p.token for p in positions2]
    sem_types2 = [classify_token(t) for t in tokens2]

    for i, pos in enumerate(positions2):
        tok = tokens2[i].strip()[:10]
        prev_t = sem_types2[i - 1] if i > 0 else "^"
        curr_t = sem_types2[i]
        next_t = sem_types2[i + 1] if i < len(sem_types2) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"
        top_exp = pos.expert_indices[0] if pos.expert_indices else None
        print(f"  {i}: {tok:<10} {trigram:<20} -> E{top_exp}")

    print()

    # Compare experts used
    experts1 = set()
    for p in positions1:
        experts1.update(p.expert_indices)

    experts2 = set()
    for p in positions2:
        experts2.update(p.expert_indices)

    shared = experts1 & experts2
    only1 = experts1 - experts2
    only2 = experts2 - experts1

    print("EXPERT OVERLAP")
    print("-" * 70)
    print(f"  Shared experts: {sorted(shared)}")
    print(f"  Only in prompt 1: {sorted(only1)}")
    print(f"  Only in prompt 2: {sorted(only2)}")
    print(
        f"  Overlap: {len(shared)}/{len(experts1 | experts2)} ({100 * len(shared) / max(1, len(experts1 | experts2)):.0f}%)"
    )
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

    # Get all layers at once
    weights = await router.capture_router_weights(prompt)

    if not weights:
        print("No routing data captured.")
        return

    # Get tokens from first layer
    first_layer = weights[0]
    tokens = [p.token.strip() for p in first_layer.positions]
    sem_types = [classify_token(t) for t in tokens]

    # Find interesting positions to focus on
    interesting = _find_interesting_positions(tokens, sem_types)[:4]

    # Show focused view for interesting positions
    for pos_idx in interesting:
        tok = tokens[pos_idx][:12]
        prev_t = sem_types[pos_idx - 1] if pos_idx > 0 else "^"
        curr_t = sem_types[pos_idx]
        next_t = sem_types[pos_idx + 1] if pos_idx < len(sem_types) - 1 else "$"
        trigram = f"{prev_t}→{curr_t}→{next_t}"

        print(f'Position {pos_idx}: "{tok}" ({trigram})')
        print("-" * 60)

        # Collect experts across layers
        layer_experts: list[tuple[int, int]] = []
        for layer_weights in weights:
            if pos_idx < len(layer_weights.positions):
                pos = layer_weights.positions[pos_idx]
                top = pos.expert_indices[0] if pos.expert_indices else None
                if top is not None:
                    layer_experts.append((layer_weights.layer_idx, top))

        # Group by phase
        from collections import Counter

        early = [(layer, exp) for layer, exp in layer_experts if layer < 8]
        mid = [(layer, exp) for layer, exp in layer_experts if 8 <= layer < 16]
        late = [(layer, exp) for layer, exp in layer_experts if layer >= 16]

        # Show each phase on one line
        def format_phase(phase_data: list, name: str) -> str:
            if not phase_data:
                return f"  {name}: --"
            # Show layer:expert pairs
            pairs = [f"L{layer}:E{exp}" for layer, exp in phase_data[:4]]
            if len(phase_data) > 4:
                pairs.append("...")
            # Find dominant expert
            counts = Counter(e for _, e in phase_data)
            top_exp, top_count = counts.most_common(1)[0]
            return f"  {name}: {' '.join(pairs)}  (E{top_exp} dominates)"

        print(format_phase(early, "Early  (L0-7) "))
        print(format_phase(mid, "Middle (L8-15)"))
        print(format_phase(late, "Late   (L16+) "))
        print()

    # Summary: show expert changes
    print("EXPERT TRANSITIONS")
    print("-" * 60)
    print("Positions where top expert changes between phases:")
    print()

    for pos_idx in interesting:
        tok = tokens[pos_idx][:8]

        # Get phase dominants

        early_exps = []
        mid_exps = []
        late_exps = []

        for layer_weights in weights:
            if pos_idx < len(layer_weights.positions):
                pos = layer_weights.positions[pos_idx]
                top = pos.expert_indices[0] if pos.expert_indices else None
                if top is not None:
                    layer_idx = layer_weights.layer_idx
                    if layer_idx < 8:
                        early_exps.append(top)
                    elif layer_idx < 16:
                        mid_exps.append(top)
                    else:
                        late_exps.append(top)

        early_dom = Counter(early_exps).most_common(1)[0][0] if early_exps else None
        mid_dom = Counter(mid_exps).most_common(1)[0][0] if mid_exps else None
        late_dom = Counter(late_exps).most_common(1)[0][0] if late_exps else None

        # Check if there are transitions
        transitions = []
        if early_dom != mid_dom and early_dom is not None and mid_dom is not None:
            transitions.append(f"E{early_dom}→E{mid_dom}")
        if mid_dom != late_dom and mid_dom is not None and late_dom is not None:
            transitions.append(f"E{mid_dom}→E{late_dom}")

        if transitions:
            print(f'  "{tok}": {" then ".join(transitions)}')
        else:
            dom = early_dom or mid_dom or late_dom
            print(f'  "{tok}": E{dom} (stable)')

    print()


def _find_interesting_positions(tokens: list, sem_types: list) -> list[int]:
    """Find positions with interesting patterns."""
    interesting = []

    for i, (tok, sem_type) in enumerate(zip(tokens, sem_types)):
        score = 0
        prev_t = sem_types[i - 1] if i > 0 else "^"
        next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"

        # Position markers
        if prev_t == "^":
            score += 2
        if next_t == "$":
            score += 2

        # Semantic relations
        if sem_type in ["AS", "TO", "SYN", "ANT", "CAUSE", "THAN"]:
            score += 3

        # Operators
        if sem_type == "OP":
            score += 2

        # Content words in specific patterns
        if sem_type in ["NOUN", "ADJ", "VERB"] and prev_t in ["AS", "TO"]:
            score += 2

        if score > 0:
            interesting.append((score, i))

    interesting.sort(reverse=True)
    return [idx for _, idx in interesting]


async def _deep_dive(
    router: ExpertRouter, prompt: str, pos_idx: int, moe_layers: list[int]
) -> None:
    """Deep dive into a specific position."""
    print()
    print("=" * 70)
    print(f"DEEP DIVE: Position {pos_idx}")
    print("=" * 70)
    print()

    # Get all layers
    weights = await router.capture_router_weights(prompt)

    if not weights:
        print("No routing data captured.")
        return

    first_layer = weights[0]
    if pos_idx >= len(first_layer.positions):
        print(f"Invalid position. Valid range: 0-{len(first_layer.positions) - 1}")
        return

    tokens = [p.token for p in first_layer.positions]
    sem_types = [classify_token(t) for t in tokens]

    tok = tokens[pos_idx].strip()
    sem_type = sem_types[pos_idx]
    prev_t = sem_types[pos_idx - 1] if pos_idx > 0 else "^"
    next_t = sem_types[pos_idx + 1] if pos_idx < len(sem_types) - 1 else "$"
    trigram = f"{prev_t}→{sem_type}→{next_t}"

    prev_tok = tokens[pos_idx - 1].strip() if pos_idx > 0 else "^"
    next_tok = tokens[pos_idx + 1].strip() if pos_idx < len(tokens) - 1 else "$"

    print(f'Token: "{tok}"')
    print(f"Type: {sem_type}")
    print(f"Trigram: {trigram}")
    print()
    print("Context:")
    print(f'  Previous: "{prev_tok}" ({prev_t})')
    print(f'  Current:  "{tok}" ({sem_type})')
    print(f'  Next:     "{next_tok}" ({next_t})')
    print()

    # Show routing across all layers
    print("ROUTING ACROSS ALL LAYERS")
    print("-" * 70)

    # Collect all experts used
    all_experts: set[int] = set()
    layer_data: list[tuple[int, list[tuple[int, float]]]] = []

    for layer_weights in weights:
        if pos_idx < len(layer_weights.positions):
            pos = layer_weights.positions[pos_idx]
            all_experts.update(pos.expert_indices)

            if hasattr(pos, "expert_weights") and pos.expert_weights:
                exp_weights = list(zip(pos.expert_indices, pos.expert_weights))
            else:
                # Fake weights if not available
                n = len(pos.expert_indices)
                exp_weights = [(e, 1.0 / n) for e in pos.expert_indices]

            layer_data.append((layer_weights.layer_idx, exp_weights))

    # Sort experts
    sorted_experts = sorted(all_experts)

    # Print header
    header = "Layer"
    for exp in sorted_experts[:8]:  # Limit columns
        header += f"  E{exp:02d}"
    print(header)
    print("-" * len(header))

    # Print data
    for layer, exp_weights in layer_data:
        row = f"L{layer:02d}  "
        exp_dict = dict(exp_weights)
        for exp in sorted_experts[:8]:
            weight = exp_dict.get(exp, 0)
            if weight > 0.1:
                bar = "#" * int(weight * 5)
                row += f" {bar:<5}"
            else:
                row += "   -  "
        print(row)

    print()

    # Find peak layer/expert
    from collections import Counter

    exp_layer_counts: dict[int, list[int]] = {}
    for layer, exp_weights in layer_data:
        for exp, weight in exp_weights:
            if exp not in exp_layer_counts:
                exp_layer_counts[exp] = []
            exp_layer_counts[exp].append(layer)

    # Most common expert
    all_exp_counts = Counter()
    for layer, exp_weights in layer_data:
        for exp, weight in exp_weights:
            all_exp_counts[exp] += 1

    top_exp, top_count = all_exp_counts.most_common(1)[0] if all_exp_counts else (None, 0)

    if top_exp is not None:
        layers_active = exp_layer_counts.get(top_exp, [])
        peak_layer = layers_active[len(layers_active) // 2] if layers_active else 0
        print(f"FINDING: E{top_exp} dominates for trigram {trigram}")
        print(f"         Active in {top_count}/{len(layer_data)} layers")
        print(f"         Peak around layer {peak_layer}")

    print()
