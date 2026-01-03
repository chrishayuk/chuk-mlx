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
    lines = [
        format_header("EXPERT TAXONOMY"),
        f"Model: {taxonomy.model_id}",
        f"Layers: {taxonomy.num_layers}",
        f"Experts per layer: {taxonomy.num_experts}",
        "",
        "Expert Identities:",
    ]

    for identity in taxonomy.expert_identities:
        lines.append(
            f"  Layer {identity.layer_idx} E{identity.expert_idx}: "
            f"{identity.role.value} - {identity.primary_category.value} "
            f"({identity.confidence:.1%} confidence)"
        )
        if verbose and identity.top_tokens:
            tokens = ", ".join(f"'{t}'" for t in identity.top_tokens[:5])
            lines.append(f"    Top tokens: {tokens}")

    if taxonomy.patterns:
        lines.append("")
        lines.append("Discovered Patterns:")
        for pattern in taxonomy.patterns[:20]:
            tokens = ", ".join(f"'{t}'" for t in pattern.trigger_tokens[:3])
            lines.append(
                f"  E{pattern.expert_idx}@L{pattern.layer_idx}: {pattern.pattern_type} - {tokens}"
            )

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
