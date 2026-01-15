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
    MoEType,
    MoETypeAnalysis,
    OverlayRepresentation,
    ReconstructionVerification,
    StorageEstimate,
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
    all_categories = Counter(identity.primary_category for identity in taxonomy.expert_identities)
    all_roles = Counter(identity.role.value for identity in taxonomy.expert_identities)

    # Group categories by type for summary
    code_cats = {k: v for k, v in all_categories.items() if k.startswith("code:")}
    structure_cats = {
        k: v
        for k, v in all_categories.items()
        if k
        in (
            "bracket",
            "operator",
            "punctuation",
            "identifier",
            "constant",
            "variable",
            "short_identifier",
        )
    }
    lang_cats = {
        k: v
        for k, v in all_categories.items()
        if k in ("function_word", "capitalized", "content", "whitespace", "number")
    }
    total_experts = len(taxonomy.expert_identities)

    lines.append("")
    lines.append("Category Summary:")

    # Code keywords summary
    if code_cats:
        code_total = sum(code_cats.values())
        code_pct = code_total / total_experts * 100
        code_details = ", ".join(
            f"{k.split(':')[1]}({v})" for k, v in sorted(code_cats.items(), key=lambda x: -x[1])[:5]
        )
        lines.append(f"  Code Keywords:    {code_total:4d} ({code_pct:5.1f}%) [{code_details}]")

    # Structure tokens
    if structure_cats:
        struct_total = sum(structure_cats.values())
        struct_pct = struct_total / total_experts * 100
        struct_details = ", ".join(
            f"{k}({v})" for k, v in sorted(structure_cats.items(), key=lambda x: -x[1])[:4]
        )
        lines.append(
            f"  Code Structure:   {struct_total:4d} ({struct_pct:5.1f}%) [{struct_details}]"
        )

    # Language tokens
    if lang_cats:
        lang_total = sum(lang_cats.values())
        lang_pct = lang_total / total_experts * 100
        lang_details = ", ".join(
            f"{k}({v})" for k, v in sorted(lang_cats.items(), key=lambda x: -x[1])[:4]
        )
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
        e for e in taxonomy.expert_identities if e.role.value == "specialist" and e.confidence > 0.6
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


def format_moe_type_result(result: MoETypeAnalysis) -> str:
    """Format MoE type analysis for display.

    Args:
        result: MoE type analysis result.

    Returns:
        Formatted output string.
    """
    type_labels = {
        MoEType.PSEUDO: "PSEUDO-MOE",
        MoEType.NATIVE: "NATIVE-MOE",
        MoEType.UNKNOWN: "UNKNOWN",
    }
    type_label = type_labels.get(result.moe_type, "UNKNOWN")

    compressible = "Yes" if result.is_compressible else "No"
    compression = f"{result.estimated_compression:.1f}x" if result.is_compressible else "N/A"

    lines = [
        format_header("MOE TYPE ANALYSIS"),
        f"Model:  {result.model_id}",
        f"Layer:  {result.layer_idx}",
        f"Type:   {type_label}",
        "",
        "Evidence:",
        f"  Gate Rank:         {result.gate.effective_rank_95:>4} / {result.gate.max_rank:<4} ({result.gate.rank_ratio * 100:>5.1f}%)",
        f"  Up Rank:           {result.up.effective_rank_95:>4} / {result.up.max_rank:<4} ({result.up.rank_ratio * 100:>5.1f}%)",
        f"  Down Rank:         {result.down.effective_rank_95:>4} / {result.down.max_rank:<4} ({result.down.rank_ratio * 100:>5.1f}%)",
        f"  Cosine Similarity: {result.mean_cosine_similarity:.3f} (+/- {result.std_cosine_similarity:.3f})",
        "",
        "Compression:",
        f"  Compressible:      {compressible}",
        f"  Estimated Ratio:   {compression}",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_moe_type_comparison(r1: MoETypeAnalysis, r2: MoETypeAnalysis) -> str:
    """Format side-by-side MoE type comparison.

    Args:
        r1: First model's analysis.
        r2: Second model's analysis.

    Returns:
        Formatted comparison table.
    """

    def _type_str(r: MoETypeAnalysis) -> str:
        return {
            MoEType.PSEUDO: "PSEUDO",
            MoEType.NATIVE: "NATIVE",
            MoEType.UNKNOWN: "UNKNOWN",
        }.get(r.moe_type, "UNKNOWN")

    def _compress_str(r: MoETypeAnalysis) -> str:
        return f"Yes ({r.estimated_compression:.1f}x)" if r.is_compressible else "No"

    # Truncate model names for table (use last path component)
    name1 = r1.model_id.split("/")[-1][:14]
    name2 = r2.model_id.split("/")[-1][:14]

    lines = [
        format_header("MOE TYPE COMPARISON"),
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
        f"| {'Metric':<21} | {name1:<14} | {name2:<14} |",
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
        f"| {'Type':<21} | {_type_str(r1):<14} | {_type_str(r2):<14} |",
        f"| {'Gate Rank':<21} | {r1.gate.effective_rank_95:>4}/{r1.gate.max_rank:<9} | {r2.gate.effective_rank_95:>4}/{r2.gate.max_rank:<9} |",
        f"| {'Gate Rank %':<21} | {r1.gate.rank_ratio * 100:>13.1f}% | {r2.gate.rank_ratio * 100:>13.1f}% |",
        f"| {'Cosine Similarity':<21} | {r1.mean_cosine_similarity:>14.3f} | {r2.mean_cosine_similarity:>14.3f} |",
        f"| {'Compressible':<21} | {_compress_str(r1):<14} | {_compress_str(r2):<14} |",
        f"+-----------------------+{'-' * 16}+{'-' * 16}+",
    ]
    return "\n".join(lines)


def format_overlay_result(result: OverlayRepresentation) -> str:
    """Format overlay computation result for display.

    Args:
        result: Overlay representation result.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("OVERLAY REPRESENTATION"),
        f"Model:   {result.model_id}",
        f"Layer:   {result.layer_idx}",
        f"Experts: {result.num_experts}",
        "",
        "Projection Analysis:",
        f"  Gate:  rank={result.gate_rank:<4} shape={result.gate.shape}",
        f"         compression: {result.gate.compression_ratio:.1f}x",
        f"  Up:    rank={result.up_rank:<4} shape={result.up.shape}",
        f"         compression: {result.up.compression_ratio:.1f}x",
        f"  Down:  rank={result.down_rank:<4} shape={result.down.shape}",
        f"         compression: {result.down.compression_ratio:.1f}x",
        "",
        "Storage:",
        f"  Original:   {result.total_original_bytes / (1024 * 1024):>8.1f} MB",
        f"  Compressed: {result.total_compressed_bytes / (1024 * 1024):>8.1f} MB",
        f"  Ratio:      {result.compression_ratio:>8.1f}x",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_verification_result(result: ReconstructionVerification) -> str:
    """Format reconstruction verification result for display.

    Args:
        result: Reconstruction verification result.

    Returns:
        Formatted output string.
    """
    status = "PASSED" if result.passed else "FAILED"
    status_marker = "✓" if result.passed else "✗"

    lines = [
        format_header("RECONSTRUCTION VERIFICATION"),
        f"Model:  {result.model_id}",
        f"Layer:  {result.layer_idx}",
        f"Status: {status_marker} {status}",
        "",
        f"Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}",
        "",
        "Weight Reconstruction Errors:",
        f"  Gate:  {result.gate.mean_relative_error:.6f} (max: {result.gate.max_relative_error:.6f})",
        f"  Up:    {result.up.mean_relative_error:.6f} (max: {result.up.max_relative_error:.6f})",
        f"  Down:  {result.down.mean_relative_error:.6f} (max: {result.down.max_relative_error:.6f})",
        "",
        "Output Reconstruction Errors:",
        f"  Mean:  {result.mean_output_error:.6f}",
        f"  Max:   {result.max_output_error:.6f}",
        "",
        f"Quality: {'<1% error - suitable for production' if result.passed else '>1% error - increase ranks'}",
        "=" * 70,
    ]
    return "\n".join(lines)


def format_storage_estimate(result: StorageEstimate) -> str:
    """Format storage estimate result for display.

    Args:
        result: Storage estimate result.

    Returns:
        Formatted output string.
    """
    lines = [
        format_header("STORAGE ESTIMATE"),
        f"Model:   {result.model_id}",
        f"Layers:  {result.num_layers} MoE layers",
        f"Experts: {result.num_experts} per layer",
        "",
        f"Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}",
        "",
        "Full Model Storage:",
        f"  Original:   {result.original_mb:>10.1f} MB",
        f"  Compressed: {result.compressed_mb:>10.1f} MB",
        f"  Savings:    {result.savings_mb:>10.1f} MB ({result.compression_ratio:.1f}x)",
        "",
        "Breakdown:",
        f"  Base experts (shared):     {result.compressed_mb / result.compression_ratio:>6.1f} MB",
        f"  Low-rank deltas:           {result.compressed_mb - result.compressed_mb / result.compression_ratio:>6.1f} MB",
        "=" * 70,
    ]
    return "\n".join(lines)


def _compute_2d_embedding(
    similarity_matrix: tuple[tuple[float, ...], ...],
    num_experts_to_show: int = 8,
) -> list[tuple[float, float, int]]:
    """Compute 2D embedding of experts from similarity matrix using classical MDS.

    Args:
        similarity_matrix: Pairwise cosine similarities (num_experts x num_experts)
        num_experts_to_show: Number of experts to embed (default 8 for clarity)

    Returns:
        List of (x, y, expert_idx) tuples for 2D positions
    """
    import math

    import numpy as np

    n = min(len(similarity_matrix), num_experts_to_show)

    # Convert similarity to distance (angle-based)
    # cos(theta) = similarity, so theta = arccos(similarity)
    # Use distance = 1 - similarity for simplicity (bounded [0, 2])
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim = similarity_matrix[i][j]
            # Clamp to valid range for arccos
            sim = max(-1.0, min(1.0, sim))
            # Use angle as distance
            dist_matrix[i, j] = math.acos(sim)

    # Classical MDS: convert distance matrix to coordinates
    # 1. Square the distances
    D_sq = dist_matrix**2

    # 2. Double-center the matrix
    n_pts = n
    H = np.eye(n_pts) - np.ones((n_pts, n_pts)) / n_pts
    B = -0.5 * H @ D_sq @ H

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 2 dimensions
    # Handle negative eigenvalues (set to small positive)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    coords_2d = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

    # Normalize to [-1, 1] range
    max_abs = np.max(np.abs(coords_2d)) + 1e-10
    coords_2d = coords_2d / max_abs

    return [(float(coords_2d[i, 0]), float(coords_2d[i, 1]), i) for i in range(n)]


def _draw_direction_diagram(
    coords: list[tuple[float, float, int]],
    width: int = 61,
    height: int = 25,
) -> list[str]:
    """Draw ASCII diagram with arrows showing expert directions.

    Args:
        coords: List of (x, y, expert_idx) in [-1, 1] range
        width: Diagram width in characters
        height: Diagram height in characters

    Returns:
        List of strings representing the ASCII diagram
    """
    # Create empty grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Center point
    cx, cy = width // 2, height // 2

    # Draw border
    for x in range(width):
        grid[0][x] = "─"
        grid[height - 1][x] = "─"
    for y in range(height):
        grid[y][0] = "│"
        grid[y][width - 1] = "│"
    grid[0][0] = "┌"
    grid[0][width - 1] = "┐"
    grid[height - 1][0] = "└"
    grid[height - 1][width - 1] = "┘"

    # Draw axes
    for x in range(2, width - 2):
        grid[cy][x] = "─"
    for y in range(2, height - 2):
        grid[y][cx] = "│"
    grid[cy][cx] = "┼"

    # Arrow characters based on direction (8 directions)
    def get_arrow_char(dx: float, dy: float) -> str:
        """Get arrow character based on direction."""
        import math

        angle = math.atan2(-dy, dx)  # Negative dy because y increases downward
        # Normalize to [0, 2pi)
        if angle < 0:
            angle += 2 * math.pi

        # 8 directions: →, ↗, ↑, ↖, ←, ↙, ↓, ↘
        arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
        idx = int((angle + math.pi / 8) / (math.pi / 4)) % 8
        return arrows[idx]

    # Draw expert arrows
    for x, y, expert_idx in coords:
        # Convert [-1, 1] to grid coordinates
        # Leave margin for border and labels
        margin_x = 4
        margin_y = 2
        gx = int(cx + x * (width // 2 - margin_x))
        gy = int(cy - y * (height // 2 - margin_y))  # Flip y

        # Clamp to valid range
        gx = max(2, min(width - 3, gx))
        gy = max(2, min(height - 3, gy))

        # Draw arrow from center towards this point
        # Draw the expert label at the endpoint
        arrow = get_arrow_char(x, y)

        # Draw a line from center to the point
        steps = max(abs(gx - cx), abs(gy - cy))
        if steps > 0:
            for step in range(1, steps + 1):
                px = cx + int((gx - cx) * step / steps)
                py = cy + int((gy - cy) * step / steps)
                if 2 <= px < width - 2 and 2 <= py < height - 2:
                    if step == steps:
                        # Endpoint: show arrow and label
                        if px + 2 < width - 1:
                            label = f"{arrow}E{expert_idx}"
                            for i, c in enumerate(label):
                                if px + i < width - 1:
                                    grid[py][px + i] = c
                        else:
                            grid[py][px] = arrow
                    elif grid[py][px] in [" ", "─", "│"]:
                        # Path: show direction
                        if abs(gx - cx) > abs(gy - cy) * 2:
                            grid[py][px] = "─"
                        elif abs(gy - cy) > abs(gx - cx) * 2:
                            grid[py][px] = "│"
                        else:
                            grid[py][px] = "·"

    return ["".join(row) for row in grid]


def format_orthogonality_ascii(result: MoETypeAnalysis, *, max_display: int = 16) -> str:
    """Format ASCII visualization of expert orthogonality from actual similarity data.

    Creates a data-driven heatmap and directional diagram showing expert relationships.

    Args:
        result: MoE type analysis result with similarity_matrix.
        max_display: Maximum number of experts to display (default 16 for readability).

    Returns:
        ASCII art visualization string with heatmap and direction diagram.
    """
    sim = result.mean_cosine_similarity
    is_orthogonal = sim < 0.10
    is_clustered = sim > 0.25

    lines = [
        format_header("EXPERT ORTHOGONALITY VISUALIZATION"),
        f"Model:   {result.model_id}",
        f"Layer:   {result.layer_idx}",
        f"Experts: {result.num_experts}",
        f"Type:    {result.moe_type.value.upper()}",
        "",
    ]

    # Add directional diagram if we have the matrix
    if result.similarity_matrix:
        lines.append("Expert Direction Diagram (2D MDS projection):")
        lines.append("")

        # Compute 2D embedding and draw diagram
        num_to_show = min(8, len(result.similarity_matrix))
        coords = _compute_2d_embedding(result.similarity_matrix, num_to_show)
        diagram_lines = _draw_direction_diagram(coords)
        lines.extend(diagram_lines)

        # Add explanation
        lines.append("")
        if is_orthogonal:
            lines.append("  Arrows point in different directions → Experts are ORTHOGONAL")
        elif is_clustered:
            lines.append("  Arrows cluster together → Experts SHARE a common base")
        else:
            lines.append("  Mixed directions → Expert structure is ambiguous")

        lines.append("")

    # Add similarity heatmap if we have the matrix
    if result.similarity_matrix:
        lines.append("Expert Similarity Heatmap (cosine similarity):")
        lines.append("")

        # Determine how many experts to show
        num_experts = len(result.similarity_matrix)
        display_count = min(num_experts, max_display)

        # Heatmap characters from low to high similarity
        # Using intensity blocks: ░ ▒ ▓ █ for different similarity ranges
        def sim_to_char(s: float) -> str:
            """Convert similarity value to heatmap character."""
            if s >= 0.99:  # Diagonal (self-similarity)
                return "■"
            elif s >= 0.5:
                return "█"
            elif s >= 0.3:
                return "▓"
            elif s >= 0.15:
                return "▒"
            elif s >= 0.05:
                return "░"
            else:
                return "·"

        # Create header row with expert indices
        if display_count <= 10:
            header = "     " + " ".join(f"{i:2d}" for i in range(display_count))
        else:
            # Compact header for many experts
            header = "   " + "".join(f"{i % 10}" for i in range(display_count))

        lines.append(header)

        # Create heatmap rows
        for i in range(display_count):
            row_chars = []
            for j in range(display_count):
                similarity = result.similarity_matrix[i][j]
                row_chars.append(sim_to_char(similarity))

            if display_count <= 10:
                row = f" {i:2d}  " + "  ".join(row_chars)
            else:
                row = f"{i:2d} " + "".join(row_chars)

            # Add row summary
            row_sims = [result.similarity_matrix[i][j] for j in range(display_count) if i != j]
            if row_sims:
                avg_sim = sum(row_sims) / len(row_sims)
                row += f"  avg:{avg_sim:.2f}"

            lines.append(row)

        if num_experts > max_display:
            lines.append(f"     ... ({num_experts - max_display} more experts not shown)")

        # Add legend
        lines.append("")
        lines.append(
            "Legend: · (<0.05)  ░ (0.05-0.15)  ▒ (0.15-0.3)  ▓ (0.3-0.5)  █ (>0.5)  ■ (self)"
        )

        # Add similarity distribution summary
        lines.append("")
        lines.append("Similarity Distribution:")

        # Count similarities in each range
        all_sims = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                all_sims.append(result.similarity_matrix[i][j])

        if all_sims:
            ranges = [
                ("≈0 (orthogonal)", lambda s: s < 0.05),
                ("0.05-0.15", lambda s: 0.05 <= s < 0.15),
                ("0.15-0.30", lambda s: 0.15 <= s < 0.30),
                ("0.30-0.50", lambda s: 0.30 <= s < 0.50),
                (">0.50 (similar)", lambda s: s >= 0.50),
            ]

            for label, condition in ranges:
                count = sum(1 for s in all_sims if condition(s))
                pct = count / len(all_sims) * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  {label:<18} {count:4d} ({pct:5.1f}%) {bar}")

    lines.append("")
    lines.append("Summary Statistics:")
    lines.append(f"  Mean Similarity:   {result.mean_cosine_similarity:.3f}")
    lines.append(f"  Std Deviation:     {result.std_cosine_similarity:.3f}")
    lines.append(f"  Gate Rank Ratio:   {result.gate.rank_ratio * 100:.1f}%")
    lines.append("")

    # Add interpretation based on actual data
    if is_clustered:
        lines.extend(
            [
                "Interpretation: PSEUDO-MoE (Clustered Experts)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  High similarity indicates experts share a common BASE.       ║",
                "  ║  Model was likely converted from dense → MoE (upcycling).     ║",
                "  ║                                                               ║",
                "  ║     Expert[i] = BASE + low_rank_delta[i]                      ║",
                "  ║                                                               ║",
                "  ║  ✓ COMPRESSIBLE via SVD overlay representation                ║",
                f"  ║  Estimated compression: {result.estimated_compression:.1f}x"
                + " " * (37 - len(f"{result.estimated_compression:.1f}"))
                + "║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )
    elif is_orthogonal:
        lines.extend(
            [
                "Interpretation: NATIVE-MoE (Orthogonal Experts)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  Low similarity indicates experts are genuinely different.    ║",
                "  ║  Model was trained natively as MoE from scratch.              ║",
                "  ║                                                               ║",
                "  ║     Expert[i] ⟂ Expert[j]  (orthogonal)                       ║",
                "  ║                                                               ║",
                "  ║  ✗ NOT compressible via SVD overlay                           ║",
                "  ║  Use quantization/pruning instead                             ║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )
    else:
        lines.extend(
            [
                "Interpretation: UNKNOWN (Ambiguous Structure)",
                "  ╔═══════════════════════════════════════════════════════════════╗",
                "  ║  Mixed similarity pattern - neither fully clustered           ║",
                "  ║  nor fully orthogonal.                                        ║",
                "  ║                                                               ║",
                "  ║  Possible causes:                                             ║",
                "  ║  - Partial MoE training / fine-tuning                         ║",
                "  ║  - Hybrid architecture                                        ║",
                "  ║  - Model in transition state                                  ║",
                "  ║                                                               ║",
                "  ║  ? Compression potential unclear - test empirically           ║",
                "  ╚═══════════════════════════════════════════════════════════════╝",
            ]
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
