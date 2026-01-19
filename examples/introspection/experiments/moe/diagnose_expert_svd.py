"""Diagnose SVD structure of GPT-OSS experts.

Investigate why compression isn't working by examining:
1. Raw expert weights (are they similar?)
2. Delta singular value spectrum (is it low-rank?)
3. Different approaches to compression
"""

import mlx.core as mx
import numpy as np

from chuk_lazarus.introspection.moe.detector import detect_moe_architecture, get_moe_layers
from chuk_lazarus.introspection.moe.moe_type import MoETypeService


def analyze_expert_structure(model, layer_idx: int, proj_name: str = "down"):
    """Deep analysis of expert weight structure."""

    experts = MoETypeService._get_experts(model, layer_idx)
    architecture = detect_moe_architecture(model)
    gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

    weights = {"gate": gate_w, "up": up_w, "down": down_w}[proj_name]

    print(f"\n{'='*70}")
    print(f"ANALYZING {proj_name.upper()} PROJECTION - Layer {layer_idx}")
    print(f"{'='*70}")
    print(f"Shape: {weights.shape} (num_experts, out_dim, in_dim)")

    # 1. Check raw expert similarity
    print("\n1. EXPERT SIMILARITY")
    print("-" * 50)

    weights_np = np.array(weights.astype(mx.float32))

    # Flatten experts and compute cosine similarity
    flat = weights_np.reshape(num_experts, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normalized = flat / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T

    # Get off-diagonal similarities
    mask = np.ones(sim_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diag = sim_matrix[mask]

    print("   Pairwise cosine similarity:")
    print(f"   Min:  {np.min(off_diag):.4f}")
    print(f"   Mean: {np.mean(off_diag):.4f}")
    print(f"   Max:  {np.max(off_diag):.4f}")

    # 2. Analyze base and deltas
    print("\n2. BASE AND DELTA ANALYSIS")
    print("-" * 50)

    base = np.mean(weights_np, axis=0)
    print("   Base (mean expert) stats:")
    print(f"   Min: {np.min(base):.6f}, Max: {np.max(base):.6f}")
    print(f"   Mean: {np.mean(base):.6f}, Std: {np.std(base):.6f}")
    print(f"   Norm: {np.linalg.norm(base):.4f}")

    # Check delta magnitudes
    deltas = weights_np - base[None, :, :]
    print("\n   Delta (expert - base) stats:")
    print(f"   Min: {np.min(deltas):.6f}, Max: {np.max(deltas):.6f}")
    print(f"   Mean: {np.mean(deltas):.6f}, Std: {np.std(deltas):.6f}")

    # Delta to base ratio
    base_norm = np.linalg.norm(base)
    delta_norms = [np.linalg.norm(deltas[i]) for i in range(num_experts)]
    print(f"   Delta/Base norm ratio: {np.mean(delta_norms)/base_norm:.4f}")

    # 3. SVD analysis of deltas
    print("\n3. SVD ANALYSIS OF DELTAS")
    print("-" * 50)

    # Analyze first expert's delta
    delta_0 = deltas[0]
    U, S, Vh = np.linalg.svd(delta_0, full_matrices=False)

    print("   Expert 0 delta SVD:")
    print(f"   Singular values shape: {S.shape}")
    print(f"   Top 10 singular values: {S[:10].round(4)}")

    # Variance explained
    total_var = np.sum(S**2)
    cumsum = np.cumsum(S**2) / total_var

    print("\n   Variance explained:")
    for rank in [1, 2, 5, 10, 32, 64, 128, 256, 512]:
        if rank <= len(cumsum):
            print(f"   Rank {rank:4d}: {cumsum[rank-1]*100:6.2f}%")

    # 4. Check if experts are near-copies of base with additive noise
    print("\n4. RECONSTRUCTION QUALITY AT DIFFERENT RANKS")
    print("-" * 50)

    for rank in [1, 2, 8, 32, 64, 128, 256, 512, 1024]:
        if rank > min(delta_0.shape):
            break

        # Truncated reconstruction
        U_t = U[:, :rank]
        S_t = S[:rank]
        Vh_t = Vh[:rank, :]
        delta_recon = U_t @ np.diag(S_t) @ Vh_t

        # Reconstruction error
        recon_error = np.linalg.norm(delta_0 - delta_recon) / np.linalg.norm(delta_0)
        expert_recon = base + delta_recon
        expert_error = np.linalg.norm(weights_np[0] - expert_recon) / np.linalg.norm(weights_np[0])

        print(f"   Rank {rank:4d}: delta_error={recon_error:.4f}, expert_error={expert_error:.4f}")

    # 5. Alternative: Check if expert weights themselves are low-rank
    print("\n5. RAW EXPERT SVD (not delta)")
    print("-" * 50)

    expert_0 = weights_np[0]
    _, S_raw, _ = np.linalg.svd(expert_0, full_matrices=False)
    total_var_raw = np.sum(S_raw**2)
    cumsum_raw = np.cumsum(S_raw**2) / total_var_raw

    print(f"   Expert 0 raw singular values (top 10): {S_raw[:10].round(4)}")
    print("\n   Variance explained (raw expert):")
    for rank in [1, 2, 5, 10, 32, 64, 128, 256, 512]:
        if rank <= len(cumsum_raw):
            print(f"   Rank {rank:4d}: {cumsum_raw[rank-1]*100:6.2f}%")

    # 6. Try different base choices
    print("\n6. DIFFERENT BASE STRATEGIES")
    print("-" * 50)

    # Strategy A: Mean base (current approach)
    base_mean = np.mean(weights_np, axis=0)
    delta_mean = weights_np[0] - base_mean
    _, S_mean, _ = np.linalg.svd(delta_mean, full_matrices=False)
    var_mean = np.cumsum(S_mean**2) / np.sum(S_mean**2)

    # Strategy B: Use expert 0 as base (then deltas to other experts are relative)
    # Not applicable for single-expert test

    # Strategy C: Median base
    base_median = np.median(weights_np, axis=0)
    delta_median = weights_np[0] - base_median
    _, S_median, _ = np.linalg.svd(delta_median, full_matrices=False)
    var_median = np.cumsum(S_median**2) / np.sum(S_median**2)

    print(f"   Mean base - rank for 95% variance: {np.searchsorted(var_mean, 0.95) + 1}")
    print(f"   Median base - rank for 95% variance: {np.searchsorted(var_median, 0.95) + 1}")

    # 7. Spectral analysis of ALL experts
    print("\n7. SPECTRAL ANALYSIS ACROSS ALL EXPERTS")
    print("-" * 50)

    ranks_95 = []
    delta_norms_list = []
    for i in range(min(8, num_experts)):  # Sample 8 experts
        delta_i = weights_np[i] - base_mean
        _, S_i, _ = np.linalg.svd(delta_i, full_matrices=False)
        total_i = np.sum(S_i**2)
        cumsum_i = np.cumsum(S_i**2) / total_i
        rank_95 = np.searchsorted(cumsum_i, 0.95) + 1
        ranks_95.append(rank_95)
        delta_norms_list.append(np.linalg.norm(delta_i))
        print(f"   Expert {i}: rank_95={rank_95}, delta_norm={np.linalg.norm(delta_i):.4f}")

    print(f"\n   Mean rank for 95% variance: {np.mean(ranks_95):.1f}")
    print(f"   Min/Max: {min(ranks_95)} / {max(ranks_95)}")

    return {
        "similarity_mean": np.mean(off_diag),
        "delta_to_base_ratio": np.mean(delta_norms) / base_norm,
        "mean_rank_95": np.mean(ranks_95),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose expert SVD structure")
    parser.add_argument("--model", "-m", default="openai/gpt-oss-20b")
    parser.add_argument("--layer", "-l", type=int, default=0)
    parser.add_argument("--projection", "-p", default="down", choices=["gate", "up", "down"])

    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    model = MoETypeService._load_model(args.model)

    moe_layers = get_moe_layers(model)
    print(f"MoE layers: {moe_layers}")

    # Analyze all projections
    for proj in ["gate", "up", "down"]:
        analyze_expert_structure(model, args.layer, proj)


if __name__ == "__main__":
    main()
