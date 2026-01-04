"""Output formatters for MoE expert CLI commands.

Provides consistent, structured output formatting for all MoE expert actions.
Separates presentation logic from business logic.
"""

from __future__ import annotations

from .....introspection.moe import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertTaxonomy,
    LayerRouterWeights,
    MoEModelInfo,
    TopKVariationResult,
)


def format_header(title: str, width: int = 70) -> str:
    """Format a section header.

    Args:
        title: Header title.
        width: Total width of the header line.

    Returns:
        Formatted header string.
    """
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def format_subheader(title: str, width: int = 70) -> str:
    """Format a subsection header.

    Args:
        title: Header title.
        width: Total width of the header line.

    Returns:
        Formatted subheader string.
    """
    return f"\n{'-' * width}\n{title}\n{'-' * width}"


def format_model_info(info: MoEModelInfo, model_id: str) -> str:
    """Format model information for display.

    Args:
        info: MoE model information.
        model_id: Model identifier.

    Returns:
        Formatted model info string.
    """
    lines = [
        f"Model: {model_id}",
        f"  Architecture: {info.architecture.value}",
        f"  Total layers: {info.total_layers}",
        f"  MoE layers: {len(info.moe_layers)}",
        f"  Experts per layer: {info.num_experts}",
        f"  Experts per token: {info.num_experts_per_tok}",
    ]
    if info.has_shared_expert:
        lines.append("  Has shared expert: Yes")
    return "\n".join(lines)


def format_chat_result(
    result: ExpertChatResult,
    model_id: str,
    moe_type: str,
    *,
    verbose: bool = False,
) -> str:
    """Format chat result for display.

    Args:
        result: Chat result from ExpertRouter.
        model_id: Model identifier.
        moe_type: Type of MoE architecture.
        verbose: Whether to include detailed statistics.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"CHAT WITH EXPERT {result.expert_idx}"),
        f"Model: {model_id}",
        f"MoE type: {moe_type}",
        "",
        f"Prompt: {result.prompt}",
        "",
        "Response:",
        result.response,
    ]

    if verbose:
        lines.extend(
            [
                "",
                "Statistics:",
                f"  Tokens generated: {result.stats.tokens_generated}",
                f"  Layers modified: {result.stats.layers_modified}",
                f"  Prompt tokens: {result.stats.prompt_tokens}",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_comparison_result(
    result: ExpertComparisonResult,
    model_id: str,
    *,
    verbose: bool = False,
) -> str:
    """Format comparison result for display.

    Args:
        result: Comparison result from ExpertRouter.
        model_id: Model identifier.
        verbose: Whether to include detailed statistics.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("EXPERT COMPARISON"),
        f"Model: {model_id}",
        f"Prompt: {result.prompt}",
        "",
    ]

    for expert_result in result.expert_results:
        lines.append(f"--- Expert {expert_result.expert_idx} ---")
        lines.append(expert_result.response)
        if verbose:
            lines.append(f"  (tokens: {expert_result.stats.tokens_generated})")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_topk_result(result: TopKVariationResult, model_id: str) -> str:
    """Format top-k variation result for display.

    Args:
        result: Top-k variation result.
        model_id: Model identifier.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"TOP-K EXPERIMENT - Using k={result.k_value} (default: {result.default_k})"),
        f"Model: {model_id}",
        f"Prompt: {result.prompt}",
        "",
        f"Normal (k={result.default_k}): {result.normal_response}",
        f"Modified (k={result.k_value}): {result.response}",
        "",
    ]

    if result.response != result.normal_response:
        lines.append("** OUTPUTS DIFFER **")
    else:
        lines.append("Outputs are identical")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_router_weights(
    weights: list[LayerRouterWeights],
    model_id: str,
    prompt: str,
) -> str:
    """Format router weights for display.

    Args:
        weights: Router weights from capture.
        model_id: Model identifier.
        prompt: The analyzed prompt.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("ROUTER WEIGHTS"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
    ]

    for layer_weights in weights:
        lines.append(f"Layer {layer_weights.layer_idx}:")
        for pos in layer_weights.positions:
            experts_str = ", ".join(
                f"E{e}({w:.3f})" for e, w in zip(pos.expert_indices, pos.weights)
            )
            lines.append(f"  [{pos.position_idx}] '{pos.token}': {experts_str}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_coactivation(
    analysis: CoactivationAnalysis,
    model_id: str,
    layer_idx: int,
) -> str:
    """Format co-activation analysis for display.

    Args:
        analysis: Co-activation analysis result.
        model_id: Model identifier.
        layer_idx: Layer that was analyzed.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header(f"CO-ACTIVATION ANALYSIS - Layer {layer_idx}"),
        f"Model: {model_id}",
        f"Total activations: {analysis.total_activations}",
        "",
        "Top Expert Pairs:",
    ]

    for pair in analysis.top_pairs[:10]:
        lines.append(
            f"  E{pair.expert_a} + E{pair.expert_b}: "
            f"{pair.coactivation_count} times ({pair.coactivation_rate:.1%})"
        )

    if analysis.generalist_experts:
        lines.append("")
        lines.append(f"Generalist experts: {list(analysis.generalist_experts)}")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_taxonomy(taxonomy: ExpertTaxonomy, *, verbose: bool = False) -> str:
    """Format full expert taxonomy for display.

    Args:
        taxonomy: Expert taxonomy result.
        verbose: Whether to include all details.

    Returns:
        Formatted output string.
    """
    from collections import Counter

    lines = [
        format_header("EXPERT TAXONOMY"),
        f"Model: {taxonomy.model_id}",
        f"Layers: {taxonomy.num_layers}",
        f"Experts per layer: {taxonomy.num_experts}",
        f"Total experts analyzed: {len(taxonomy.expert_identities)}",
    ]

    # Group by layer
    by_layer: dict[int, list] = {}
    for identity in taxonomy.expert_identities:
        if identity.layer_idx not in by_layer:
            by_layer[identity.layer_idx] = []
        by_layer[identity.layer_idx].append(identity)

    # Overall statistics
    all_categories = Counter(
        identity.primary_category for identity in taxonomy.expert_identities
    )
    all_roles = Counter(
        identity.role.value for identity in taxonomy.expert_identities
    )

    # Group categories by type for summary
    code_cats = {k: v for k, v in all_categories.items() if k.startswith("code:")}
    structure_cats = {
        k: v for k, v in all_categories.items()
        if k in ("bracket", "operator", "punctuation", "identifier", "constant",
                 "variable", "short_identifier")
    }
    lang_cats = {
        k: v for k, v in all_categories.items()
        if k in ("function_word", "capitalized", "content", "whitespace", "number")
    }
    total_experts = len(taxonomy.expert_identities)

    lines.append("")
    lines.append("Category Summary:")

    # Code keywords summary
    if code_cats:
        code_total = sum(code_cats.values())
        code_pct = code_total / total_experts * 100
        code_details = ", ".join(f"{k.split(':')[1]}({v})" for k, v in
                                  sorted(code_cats.items(), key=lambda x: -x[1])[:5])
        lines.append(f"  Code Keywords:    {code_total:4d} ({code_pct:5.1f}%) [{code_details}]")

    # Structure tokens
    if structure_cats:
        struct_total = sum(structure_cats.values())
        struct_pct = struct_total / total_experts * 100
        struct_details = ", ".join(f"{k}({v})" for k, v in
                                    sorted(structure_cats.items(), key=lambda x: -x[1])[:4])
        lines.append(f"  Code Structure:   {struct_total:4d} ({struct_pct:5.1f}%) [{struct_details}]")

    # Language tokens
    if lang_cats:
        lang_total = sum(lang_cats.values())
        lang_pct = lang_total / total_experts * 100
        lang_details = ", ".join(f"{k}({v})" for k, v in
                                  sorted(lang_cats.items(), key=lambda x: -x[1])[:4])
        lines.append(f"  Language/Other:   {lang_total:4d} ({lang_pct:5.1f}%) [{lang_details}]")

    lines.append("")
    lines.append("Detailed Category Distribution:")
    for cat, count in all_categories.most_common():
        pct = count / total_experts * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {cat:<20} {count:4d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("Role Distribution: ")
    role_parts = [f"{role}: {count}" for role, count in all_roles.most_common()]
    lines[-1] += ", ".join(role_parts)

    # High-confidence specialists (notable experts)
    specialists = [
        e for e in taxonomy.expert_identities
        if e.role.value == "specialist" and e.confidence > 0.6
    ]
    if specialists:
        specialists.sort(key=lambda e: e.confidence, reverse=True)
        lines.append("")
        lines.append(format_subheader("HIGH-CONFIDENCE SPECIALISTS"))
        for exp in specialists[:20]:  # Show top 20
            tokens_str = ""
            if exp.top_tokens:
                tokens_str = f" tokens: {', '.join(repr(t) for t in exp.top_tokens[:3])}"
            lines.append(
                f"  L{exp.layer_idx:02d} E{exp.expert_idx:02d}: "
                f"{exp.primary_category:<15} "
                f"({exp.confidence:5.1%} conf, {exp.activation_rate:5.1%} act)"
                f"{tokens_str}"
            )
        if len(specialists) > 20:
            lines.append(f"  ... and {len(specialists) - 20} more specialists")

    # Per-layer summaries
    lines.append("")
    lines.append(format_subheader("LAYER SUMMARIES"))

    for layer_idx in sorted(by_layer.keys()):
        layer_experts = by_layer[layer_idx]
        layer_categories = Counter(e.primary_category for e in layer_experts)
        layer_specialists = sum(1 for e in layer_experts if e.role.value == "specialist")
        avg_confidence = sum(e.confidence for e in layer_experts) / len(layer_experts)

        # Top 2 categories for this layer
        top_cats = layer_categories.most_common(2)
        top_cats_str = ", ".join(f"{cat}({cnt})" for cat, cnt in top_cats)

        lines.append(
            f"  Layer {layer_idx:2d}: "
            f"{len(layer_experts):2d} experts, "
            f"{layer_specialists:2d} specialists, "
            f"avg conf {avg_confidence:.1%}, "
            f"top: {top_cats_str}"
        )

    # Detailed per-layer breakdown (verbose only)
    if verbose:
        lines.append("")
        lines.append(format_subheader("DETAILED LAYER BREAKDOWN"))

        for layer_idx in sorted(by_layer.keys()):
            layer_experts = by_layer[layer_idx]
            # Sort by confidence descending
            layer_experts.sort(key=lambda e: e.confidence, reverse=True)

            lines.append(f"\n  Layer {layer_idx}:")
            for exp in layer_experts:
                tokens_str = ""
                if exp.top_tokens:
                    tokens_str = f" [{', '.join(repr(t) for t in exp.top_tokens[:3])}]"
                role_marker = "★" if exp.role.value == "specialist" else "○"
                lines.append(
                    f"    {role_marker} E{exp.expert_idx:02d}: "
                    f"{exp.primary_category:<15} "
                    f"{exp.confidence:5.1%} conf, {exp.activation_rate:5.1%} act"
                    f"{tokens_str}"
                )

    if taxonomy.patterns:
        lines.append("")
        lines.append(format_subheader("DISCOVERED PATTERNS"))
        for pattern in taxonomy.patterns[:20]:
            tokens = ", ".join(f"'{t}'" for t in pattern.trigger_tokens[:3])
            lines.append(
                f"  E{pattern.expert_idx}@L{pattern.layer_idx}: {pattern.pattern_type} - {tokens}"
            )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_ablation_result(
    normal_output: str,
    ablated_output: str,
    expert_indices: list[int],
    prompt: str,
    model_id: str,
) -> str:
    """Format ablation result for display.

    Args:
        normal_output: Output without ablation.
        ablated_output: Output with ablation.
        expert_indices: Experts that were ablated.
        prompt: The input prompt.
        model_id: Model identifier.

    Returns:
        Formatted output string.
    """
    experts_str = ", ".join(str(e) for e in expert_indices)
    lines = [
        format_header(f"ABLATION - Expert(s) {experts_str}"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
        f"Normal:  {normal_output}",
        f"Ablated: {ablated_output}",
        "",
    ]

    if normal_output != ablated_output:
        lines.append("** OUTPUTS DIFFER - Expert(s) had an effect! **")
    else:
        lines.append("Outputs are identical - Expert(s) had no effect")

    lines.append("=" * 70)
    return "\n".join(lines)


def format_entropy_analysis(
    entropies: list[tuple[int, float, float]],
    model_id: str,
    prompt: str,
) -> str:
    """Format routing entropy analysis for display.

    Args:
        entropies: List of (layer_idx, mean_entropy, normalized_entropy) tuples.
        model_id: Model identifier.
        prompt: The analyzed prompt.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("ROUTING ENTROPY ANALYSIS"),
        f"Model: {model_id}",
        f"Prompt: {prompt}",
        "",
        "Layer  Mean Entropy  Normalized",
        "-" * 35,
    ]

    for layer_idx, mean_ent, norm_ent in entropies:
        bar = "#" * int(norm_ent * 20)
        lines.append(f"  {layer_idx:3d}    {mean_ent:6.3f}       {norm_ent:.3f} {bar}")

    lines.append("=" * 70)
    return "\n".join(lines)
