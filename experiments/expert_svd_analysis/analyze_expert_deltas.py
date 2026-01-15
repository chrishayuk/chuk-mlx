"""
SVD Analysis of Expert Weight Deltas

This experiment tests whether MoE experts can be represented as:
    expert_i = base_expert + low_rank_delta_i

If the effective rank of deltas is low (e.g., 64-128 vs hidden_dim 2880),
then full experts are massive overkill and LoRA-style experts would work.

Key insight: If SVD shows 95% variance captured at rank ~64,
we could replace 32 full experts with 1 base + 32 small LoRA deltas.

Parameter savings:
- Full MoE: 32 x (3 * hidden * intermediate) parameters
- Overlay MoE: 1 x (3 * hidden * intermediate) + 32 x (2 * rank * dim) parameters
- At rank=64 with hidden=2880, intermediate=7680:
  - Full: 32 * 66M = 2.1B params
  - Overlay: 66M + 32 * 0.4M = ~79M params (27x smaller!)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
from tqdm import tqdm


@dataclass
class ExpertDeltaSVD:
    """SVD analysis results for one expert's delta from base."""

    expert_idx: int
    layer_idx: int
    proj_name: str  # 'gate', 'up', or 'down'

    # SVD results
    singular_values: np.ndarray
    total_variance: float

    # Effective rank at various thresholds
    rank_90: int  # Rank to capture 90% variance
    rank_95: int  # Rank to capture 95% variance
    rank_99: int  # Rank to capture 99% variance

    # Original dimensions
    original_shape: tuple[int, int]

    @property
    def compression_ratio_95(self) -> float:
        """Compression ratio at 95% variance threshold."""
        m, n = self.original_shape
        original_params = m * n
        lora_params = self.rank_95 * (m + n)
        return original_params / lora_params if lora_params > 0 else float("inf")


@dataclass
class LayerSVDAnalysis:
    """SVD analysis for all experts in one layer."""

    layer_idx: int
    num_experts: int

    # Per-expert, per-projection SVD results
    expert_svds: dict[tuple[int, str], ExpertDeltaSVD] = field(default_factory=dict)

    # Summary statistics
    mean_rank_95_gate: float = 0.0
    mean_rank_95_up: float = 0.0
    mean_rank_95_down: float = 0.0

    def add_svd(self, svd: ExpertDeltaSVD):
        """Add SVD result for an expert projection."""
        self.expert_svds[(svd.expert_idx, svd.proj_name)] = svd

    def compute_summaries(self):
        """Compute summary statistics."""
        gate_ranks = [s.rank_95 for (_, p), s in self.expert_svds.items() if p == "gate"]
        up_ranks = [s.rank_95 for (_, p), s in self.expert_svds.items() if p == "up"]
        down_ranks = [s.rank_95 for (_, p), s in self.expert_svds.items() if p == "down"]

        self.mean_rank_95_gate = np.mean(gate_ranks) if gate_ranks else 0.0
        self.mean_rank_95_up = np.mean(up_ranks) if up_ranks else 0.0
        self.mean_rank_95_down = np.mean(down_ranks) if down_ranks else 0.0


def compute_effective_rank(singular_values: np.ndarray, threshold: float) -> int:
    """
    Compute effective rank to capture given fraction of variance.

    Args:
        singular_values: Sorted singular values (descending)
        threshold: Fraction of variance to capture (e.g., 0.95)

    Returns:
        Minimum rank to capture threshold fraction of total variance
    """
    # Total variance is sum of squared singular values
    total_variance = np.sum(singular_values**2)

    if total_variance == 0:
        return 0

    # Cumulative variance
    cumsum = np.cumsum(singular_values**2)
    cumsum_ratio = cumsum / total_variance

    # Find first index where we exceed threshold
    rank = np.searchsorted(cumsum_ratio, threshold) + 1
    return int(min(rank, len(singular_values)))


def analyze_expert_delta(
    delta_weight: mx.array,
    expert_idx: int,
    layer_idx: int,
    proj_name: str,
) -> ExpertDeltaSVD:
    """
    Perform SVD analysis on an expert's delta weight matrix.

    Args:
        delta_weight: Weight delta (expert - base), shape (out, in)
        expert_idx: Expert index
        layer_idx: Layer index
        proj_name: Projection name ('gate', 'up', 'down')

    Returns:
        ExpertDeltaSVD with analysis results
    """
    # Convert to float32 in MLX first (handles bfloat16), then evaluate and convert to numpy
    delta_f32 = delta_weight.astype(mx.float32)
    mx.eval(delta_f32)
    delta_np = np.array(delta_f32)

    # Compute SVD
    try:
        U, S, Vh = np.linalg.svd(delta_np, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD failed, return degenerate result
        return ExpertDeltaSVD(
            expert_idx=expert_idx,
            layer_idx=layer_idx,
            proj_name=proj_name,
            singular_values=np.array([0.0]),
            total_variance=0.0,
            rank_90=0,
            rank_95=0,
            rank_99=0,
            original_shape=delta_np.shape,
        )

    total_variance = float(np.sum(S**2))

    return ExpertDeltaSVD(
        expert_idx=expert_idx,
        layer_idx=layer_idx,
        proj_name=proj_name,
        singular_values=S,
        total_variance=total_variance,
        rank_90=compute_effective_rank(S, 0.90),
        rank_95=compute_effective_rank(S, 0.95),
        rank_99=compute_effective_rank(S, 0.99),
        original_shape=delta_np.shape,
    )


def analyze_layer_experts(
    model,
    layer_idx: int,
) -> LayerSVDAnalysis | None:
    """
    Analyze all experts in a layer.

    Args:
        model: Model with MoE layers
        layer_idx: Layer index to analyze

    Returns:
        LayerSVDAnalysis or None if layer is not MoE
    """
    # Get layer
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    # Check for experts
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return None

    # Handle list-based experts (OLMoE, Mixtral)
    if isinstance(experts, list):
        return _analyze_list_experts(experts, layer_idx)

    # Handle batched experts (GPT-OSS) - dequantize MXFP4 weights
    if hasattr(experts, "gate_up_proj_blocks"):
        return _analyze_batched_experts(experts, layer_idx)

    return None


def _dequantize_mxfp4(
    blocks: mx.array,
    scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Dequantize MXFP4 weights to float32.

    Args:
        blocks: Quantized weight blocks shape (num_experts, out_features, packed_in)
        scales: Scale factors shape (num_experts, out_features, num_groups)
        bias: Optional bias shape (num_experts, out_features)

    Returns:
        Dequantized weights as float32 shape (num_experts, out_features, in_features)
    """
    # Use MLX's built-in dequantize with mxfp4 mode
    dequant = mx.dequantize(
        blocks,
        scales,
        biases=None,  # MXFP4 doesn't use per-group biases in MLX
        group_size=32,
        bits=4,
        mode="mxfp4",
    )

    # Add the per-output bias if present
    # bias is (num_experts, out_features), dequant is (num_experts, out_features, in_features)
    # Need to expand bias to broadcast: (num_experts, out_features, 1)
    if bias is not None:
        dequant = dequant + bias[:, :, None]

    return dequant.astype(mx.float32)


def _analyze_batched_experts(
    experts,
    layer_idx: int,
) -> LayerSVDAnalysis:
    """Analyze GPT-OSS batched/quantized experts."""
    num_experts = experts.num_experts
    analysis = LayerSVDAnalysis(layer_idx=layer_idx, num_experts=num_experts)

    print(f"  Layer {layer_idx}: Dequantizing {num_experts} MXFP4 experts...", flush=True)

    # Dequantize gate_up_proj for all experts
    # Shape after dequant: (num_experts, 2*intermediate, hidden)
    print("    Dequantizing gate_up_proj...", flush=True)
    gate_up_weights = _dequantize_mxfp4(
        experts.gate_up_proj_blocks,
        experts.gate_up_proj_scales,
        experts.gate_up_proj_bias,
    )
    mx.eval(gate_up_weights)
    print(f"    gate_up shape: {gate_up_weights.shape}", flush=True)

    # Dequantize down_proj for all experts
    # Shape after dequant: (num_experts, hidden, intermediate)
    print("    Dequantizing down_proj...", flush=True)
    down_weights = _dequantize_mxfp4(
        experts.down_proj_blocks,
        experts.down_proj_scales,
        experts.down_proj_bias,
    )
    mx.eval(down_weights)
    print(f"    down shape: {down_weights.shape}", flush=True)

    # gate_up is interleaved: gate at even indices, up at odd
    # Split them: gate_up_weights is (num_experts, 2*intermediate, hidden)
    gate_weights = gate_up_weights[:, 0::2, :]  # (num_experts, intermediate, hidden)
    up_weights = gate_up_weights[:, 1::2, :]  # (num_experts, intermediate, hidden)
    print(f"    Split gate: {gate_weights.shape}, up: {up_weights.shape}", flush=True)

    # Analyze each projection type
    for proj_name, weights in [
        ("gate", gate_weights),
        ("up", up_weights),
        ("down", down_weights),
    ]:
        print(f"    Analyzing {proj_name} projection ({weights.shape})...", flush=True)

        # Compute base (mean) expert
        base = mx.mean(weights, axis=0)  # (out, in)
        mx.eval(base)

        # Analyze each expert's delta
        for expert_idx in range(num_experts):
            if expert_idx % 8 == 0:
                print(f"      Expert {expert_idx}/{num_experts}...", flush=True)
            delta = weights[expert_idx] - base
            svd_result = analyze_expert_delta(delta, expert_idx, layer_idx, proj_name)
            analysis.add_svd(svd_result)

    analysis.compute_summaries()
    return analysis


def _analyze_list_experts(
    experts: list,
    layer_idx: int,
) -> LayerSVDAnalysis:
    """Analyze list-based experts (OLMoE, Mixtral style)."""
    num_experts = len(experts)
    analysis = LayerSVDAnalysis(layer_idx=layer_idx, num_experts=num_experts)

    # Collect all weights for each projection type
    gate_weights = []
    up_weights = []
    down_weights = []

    for expert in experts:
        if hasattr(expert, "gate_proj"):
            gate_weights.append(expert.gate_proj.weight)
        if hasattr(expert, "up_proj"):
            up_weights.append(expert.up_proj.weight)
        if hasattr(expert, "down_proj"):
            down_weights.append(expert.down_proj.weight)

    # Analyze each projection type
    for proj_name, weights in [
        ("gate", gate_weights),
        ("up", up_weights),
        ("down", down_weights),
    ]:
        if not weights:
            continue

        # Compute base (mean) expert - convert to float32 for precision
        stacked = mx.stack(
            [w.astype(mx.float32) for w in weights], axis=0
        )  # (num_experts, out, in)
        base = mx.mean(stacked, axis=0)  # (out, in)
        mx.eval(base)

        # Analyze each expert's delta
        for expert_idx, weight in enumerate(weights):
            delta = weight.astype(mx.float32) - base
            svd_result = analyze_expert_delta(delta, expert_idx, layer_idx, proj_name)
            analysis.add_svd(svd_result)

    analysis.compute_summaries()
    return analysis


def _get_model_layers(model):
    """Get transformer layers from model."""
    for attr in ["model", "transformer", "decoder"]:
        submodel = getattr(model, attr, None)
        if submodel is not None:
            layers = getattr(submodel, "layers", None)
            if layers is not None:
                return list(layers)
    return list(getattr(model, "layers", []))


def find_moe_layers(model) -> list[int]:
    """Find indices of layers with MoE."""
    moe_layers = []
    layers = _get_model_layers(model)

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Check for expert indicators
        if hasattr(mlp, "experts") or hasattr(mlp, "router"):
            moe_layers.append(i)

    return moe_layers


def compute_expert_similarities(
    model,
    layer_idx: int,
) -> np.ndarray | None:
    """Compute pairwise cosine similarities between experts."""
    layers = _get_model_layers(model)
    if layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None

    experts = getattr(mlp, "experts", None)
    if experts is None:
        return None

    weights = []

    # Handle list-based experts (OLMoE, Mixtral)
    if isinstance(experts, list):
        for expert in experts:
            if hasattr(expert, "down_proj"):
                w = expert.down_proj.weight.astype(mx.float32)
                mx.eval(w)
                weights.append(np.array(w).flatten())

    # Handle batched experts (GPT-OSS)
    elif hasattr(experts, "down_proj_blocks"):
        print("  Dequantizing experts for similarity computation...")
        down_weights = _dequantize_mxfp4(
            experts.down_proj_blocks,
            experts.down_proj_scales,
            experts.down_proj_bias,
        )
        mx.eval(down_weights)

        for i in range(experts.num_experts):
            w = down_weights[i]
            weights.append(np.array(w).flatten())

    if not weights:
        return None

    n = len(weights)
    similarities = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                similarities[i, j] = 1.0
            else:
                dot = np.dot(weights[i], weights[j])
                norm_i = np.linalg.norm(weights[i])
                norm_j = np.linalg.norm(weights[j])
                similarities[i, j] = dot / (norm_i * norm_j + 1e-10)

    return similarities


def print_analysis_summary(analyses: list[LayerSVDAnalysis]):
    """Print summary of SVD analysis across all layers."""
    print("\n" + "=" * 80)
    print("EXPERT DELTA SVD ANALYSIS SUMMARY")
    print("=" * 80)

    if not analyses:
        print("No MoE layers analyzed.")
        return

    # Get first expert's SVD for dimension info
    first_analysis = analyses[0]
    first_svd = next(iter(first_analysis.expert_svds.values()), None)

    if first_svd:
        m, n = first_svd.original_shape
        print(f"\nWeight dimensions: ({m}, {n})")
        print(f"Max possible rank: {min(m, n)}")

    print("\n" + "-" * 80)
    print("Effective Rank at 95% Variance (lower = more compressible)")
    print("-" * 80)
    print(f"{'Layer':>6} | {'Gate':>8} | {'Up':>8} | {'Down':>8} | {'Avg':>8}")
    print("-" * 80)

    all_gate = []
    all_up = []
    all_down = []

    for analysis in analyses:
        avg = (
            analysis.mean_rank_95_gate + analysis.mean_rank_95_up + analysis.mean_rank_95_down
        ) / 3
        print(
            f"{analysis.layer_idx:>6} | {analysis.mean_rank_95_gate:>8.1f} | "
            f"{analysis.mean_rank_95_up:>8.1f} | {analysis.mean_rank_95_down:>8.1f} | {avg:>8.1f}"
        )

        all_gate.append(analysis.mean_rank_95_gate)
        all_up.append(analysis.mean_rank_95_up)
        all_down.append(analysis.mean_rank_95_down)

    print("-" * 80)
    avg_all = (np.mean(all_gate) + np.mean(all_up) + np.mean(all_down)) / 3
    print(
        f"{'Mean':>6} | {np.mean(all_gate):>8.1f} | {np.mean(all_up):>8.1f} | "
        f"{np.mean(all_down):>8.1f} | {avg_all:>8.1f}"
    )

    # Compression ratio estimate
    print("\n" + "-" * 80)
    print("COMPRESSION POTENTIAL")
    print("-" * 80)

    if first_svd:
        m, n = first_svd.original_shape
        original_params = m * n
        lora_rank = int(avg_all)
        lora_params = lora_rank * (m + n)

        # For full MoE vs overlay
        num_experts = first_analysis.num_experts
        full_moe_params = num_experts * original_params * 3  # 3 projections
        overlay_params = original_params * 3 + num_experts * lora_params * 3  # base + deltas

        print(f"Average effective rank at 95%: {avg_all:.1f}")
        print(f"Original weight shape: ({m}, {n}) = {original_params:,} params per projection")
        print(f"LoRA params at rank {lora_rank}: {lora_params:,} params per projection")
        print()
        print(f"Full MoE ({num_experts} experts x 3 projections): {full_moe_params:,} params")
        print(f"Overlay MoE (1 base + {num_experts} LoRA deltas): {overlay_params:,} params")
        print(f"Compression ratio: {full_moe_params / overlay_params:.1f}x smaller")

    # Per-expert variation
    print("\n" + "-" * 80)
    print("PER-EXPERT RANK DISTRIBUTION (Layer 0, Down projection)")
    print("-" * 80)

    if analyses:
        first = analyses[0]
        down_svds = [(idx, svd) for (idx, proj), svd in first.expert_svds.items() if proj == "down"]
        down_svds.sort(key=lambda x: x[0])

        ranks = [svd.rank_95 for _, svd in down_svds]
        if ranks:
            print(f"Ranks: {ranks}")
            print(f"Min: {min(ranks)}, Max: {max(ranks)}, Std: {np.std(ranks):.1f}")


def visualize_singular_values(analyses: list[LayerSVDAnalysis], output_path: str | None = None):
    """Create visualization of singular value decay."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    if not analyses:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, proj_name in zip(axes, ["gate", "up", "down"]):
        # Plot singular values for first 3 experts in first layer
        first = analyses[0]
        for expert_idx in range(min(3, first.num_experts)):
            key = (expert_idx, proj_name)
            if key in first.expert_svds:
                svd = first.expert_svds[key]
                # Normalize by largest singular value
                normalized = svd.singular_values / (svd.singular_values[0] + 1e-10)
                ax.semilogy(normalized[:100], label=f"Expert {expert_idx}", alpha=0.7)

        ax.set_xlabel("Rank")
        ax.set_ylabel("Normalized Singular Value")
        ax.set_title(f"{proj_name.capitalize()} Projection")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Singular Value Decay in Expert Deltas", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze expert weight deltas via SVD")
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924",
        help="Model ID or path",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,5,10,15",
        help="Comma-separated layer indices to analyze",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Path to save singular value plot",
    )
    args = parser.parse_args()

    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Loading model: {args.model}")

    # Load model using the framework's loader (required for GPT-OSS)
    import sys

    sys.path.insert(0, "/Users/christopherhay/chris-source/chuk-mlx/src")

    try:
        from chuk_lazarus.models_v2.loader import load_model

        loaded = load_model(args.model)
        model = loaded.model
        print(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        print(f"Could not import chuk_lazarus loader ({e}), trying direct MLX load...")

        # Fallback to direct HuggingFace/MLX load (won't work for GPT-OSS)
        from mlx_lm import load

        model, tokenizer = load(args.model)
        print(f"Loaded via mlx_lm: {type(model).__name__}")

    # Find MoE layers
    moe_layers = find_moe_layers(model)
    print(f"Found MoE layers: {moe_layers}")

    # Filter to requested layers
    layer_indices = [i for i in layer_indices if i in moe_layers]
    if not layer_indices:
        print("No MoE layers found in requested indices. Using all MoE layers.")
        layer_indices = moe_layers[:4]  # First 4 MoE layers

    print(f"Analyzing layers: {layer_indices}")

    # Analyze each layer
    analyses = []
    for layer_idx in tqdm(layer_indices, desc="Analyzing layers"):
        analysis = analyze_layer_experts(model, layer_idx)
        if analysis:
            analyses.append(analysis)

    # Print summary
    print_analysis_summary(analyses)

    # Compute pairwise similarities for first layer
    if layer_indices:
        print("\n" + "-" * 80)
        print(f"PAIRWISE EXPERT SIMILARITIES (Layer {layer_indices[0]}, down_proj)")
        print("-" * 80)
        sims = compute_expert_similarities(model, layer_indices[0])
        if sims is not None:
            # Get off-diagonal elements
            n = sims.shape[0]
            off_diag = sims[np.triu_indices(n, k=1)]
            print(f"Mean similarity: {np.mean(off_diag):.4f}")
            print(f"Std similarity: {np.std(off_diag):.4f}")
            print(f"Min similarity: {np.min(off_diag):.4f}")
            print(f"Max similarity: {np.max(off_diag):.4f}")

            # Count pairs above threshold
            high_sim = np.sum(off_diag > 0.9)
            med_sim = np.sum((off_diag > 0.7) & (off_diag <= 0.9))
            low_sim = np.sum(off_diag <= 0.7)
            total = len(off_diag)
            print(f"\nPairs with similarity > 0.9: {high_sim} ({100 * high_sim / total:.1f}%)")
            print(f"Pairs with similarity 0.7-0.9: {med_sim} ({100 * med_sim / total:.1f}%)")
            print(f"Pairs with similarity < 0.7: {low_sim} ({100 * low_sim / total:.1f}%)")

    # Visualize
    if args.output_plot:
        visualize_singular_values(analyses, args.output_plot)


if __name__ == "__main__":
    main()
