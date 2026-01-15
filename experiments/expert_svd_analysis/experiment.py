"""
Expert SVD Compression Experiment.

Phase 1: Validate SVD analysis across all layers
Phase 2: Compute overlay representation (base + low-rank deltas)
Phase 3: Verify reconstruction accuracy
Phase 4: End-to-end quality evaluation (perplexity)

Key hypothesis: MoE experts are low-rank perturbations of a shared base,
enabling 8x compression with <1% quality loss.
"""

import logging
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class ProjectionSVD:
    """SVD analysis for one projection type across all experts."""

    proj_name: str  # 'gate', 'up', 'down'
    shape: tuple[int, int]  # (out_features, in_features)
    max_rank: int

    # Per-expert effective ranks
    ranks_90: list[int] = field(default_factory=list)
    ranks_95: list[int] = field(default_factory=list)
    ranks_99: list[int] = field(default_factory=list)

    # Aggregates
    mean_rank_95: float = 0.0
    std_rank_95: float = 0.0
    compression_ratio: float = 0.0

    def compute_stats(self):
        if self.ranks_95:
            self.mean_rank_95 = float(np.mean(self.ranks_95))
            self.std_rank_95 = float(np.std(self.ranks_95))
            m, n = self.shape
            original = m * n
            lora = int(self.mean_rank_95) * (m + n)
            self.compression_ratio = original / lora if lora > 0 else 0


@dataclass
class LayerAnalysis:
    """Complete analysis for one layer."""

    layer_idx: int
    num_experts: int

    gate: ProjectionSVD | None = None
    up: ProjectionSVD | None = None
    down: ProjectionSVD | None = None

    # Pairwise similarities
    mean_similarity: float = 0.0
    std_similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "num_experts": self.num_experts,
            "gate": asdict(self.gate) if self.gate else None,
            "up": asdict(self.up) if self.up else None,
            "down": asdict(self.down) if self.down else None,
            "mean_similarity": self.mean_similarity,
            "std_similarity": self.std_similarity,
        }


@dataclass
class ReconstructionResult:
    """Verification results for overlay reconstruction."""

    layer_idx: int
    expert_idx: int

    # Weight-level errors
    gate_mse: float = 0.0
    up_mse: float = 0.0
    down_mse: float = 0.0
    gate_relative_error: float = 0.0
    up_relative_error: float = 0.0
    down_relative_error: float = 0.0

    # Output-level errors
    output_mse: float = 0.0
    output_relative_error: float = 0.0


class ExpertSVDExperiment(ExperimentBase):
    """
    Validate and implement MoE expert compression via SVD.

    Tests hypothesis: experts = base + low_rank_delta
    """

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up Expert SVD compression experiment...")
        self.params = self.config.parameters
        self.layer_analyses: list[LayerAnalysis] = []
        self.reconstruction_results: list[ReconstructionResult] = []

    def run(self) -> dict:
        """Run all experiment phases."""
        self.log("=" * 70)
        self.log("EXPERT SVD COMPRESSION EXPERIMENT")
        self.log("Testing: experts = base + low_rank_delta")
        self.log("=" * 70)

        # Load model
        loaded = self.load_model()
        model = loaded.model
        tokenizer = loaded.tokenizer

        # Find MoE layers
        moe_layers = self._find_moe_layers(model)
        self.log(f"Found {len(moe_layers)} MoE layers: {moe_layers}")

        if not moe_layers:
            return {"error": "No MoE layers found"}

        # Determine which layers to analyze
        layers_to_analyze = self._select_layers(moe_layers)
        self.log(f"Analyzing layers: {layers_to_analyze}")

        # Phase 1: SVD analysis across all layers
        self.log("\n" + "=" * 70)
        self.log("PHASE 1: SVD Analysis")
        self.log("=" * 70)
        phase1_results = self._phase1_svd_analysis(model, layers_to_analyze)

        # Phase 2: Compute overlay representation
        self.log("\n" + "=" * 70)
        self.log("PHASE 2: Overlay Representation")
        self.log("=" * 70)
        phase2_results = self._phase2_overlay_representation(model, layers_to_analyze)

        # Phase 3: Verify reconstruction
        self.log("\n" + "=" * 70)
        self.log("PHASE 3: Reconstruction Verification")
        self.log("=" * 70)
        phase3_results = self._phase3_verify_reconstruction(model, tokenizer, layers_to_analyze)

        # Phase 4: Perplexity evaluation (if enabled)
        phase4_results = {}
        if self.params.get("run_perplexity", False):
            self.log("\n" + "=" * 70)
            self.log("PHASE 4: Perplexity Evaluation")
            self.log("=" * 70)
            phase4_results = self._phase4_perplexity(model, tokenizer)

        # Build summary
        summary = self._build_summary(
            phase1_results, phase2_results, phase3_results, phase4_results
        )

        return {
            "phase1_svd": phase1_results,
            "phase2_overlay": phase2_results,
            "phase3_reconstruction": phase3_results,
            "phase4_perplexity": phase4_results,
            "summary": summary,
        }

    def _phase1_svd_analysis(self, model, layers: list[int]) -> dict:
        """Analyze effective rank of expert deltas via SVD."""
        results = {"layers": {}, "summary": {}}

        for layer_idx in tqdm(layers, desc="SVD Analysis"):
            self.log(f"  Layer {layer_idx}...")
            analysis = self._analyze_layer_svd(model, layer_idx)
            if analysis:
                self.layer_analyses.append(analysis)
                results["layers"][layer_idx] = analysis.to_dict()

                # Log summary
                if analysis.gate:
                    self.log(
                        f"    Gate: rank={analysis.gate.mean_rank_95:.0f}, "
                        f"compression={analysis.gate.compression_ratio:.1f}x"
                    )
                if analysis.up:
                    self.log(
                        f"    Up: rank={analysis.up.mean_rank_95:.0f}, "
                        f"compression={analysis.up.compression_ratio:.1f}x"
                    )
                if analysis.down:
                    self.log(
                        f"    Down: rank={analysis.down.mean_rank_95:.0f}, "
                        f"compression={analysis.down.compression_ratio:.1f}x"
                    )
                self.log(f"    Similarity: {analysis.mean_similarity:.3f}")

        # Compute overall summary
        if self.layer_analyses:
            gate_ranks = [a.gate.mean_rank_95 for a in self.layer_analyses if a.gate]
            up_ranks = [a.up.mean_rank_95 for a in self.layer_analyses if a.up]
            down_ranks = [a.down.mean_rank_95 for a in self.layer_analyses if a.down]
            similarities = [a.mean_similarity for a in self.layer_analyses]

            results["summary"] = {
                "mean_gate_rank": float(np.mean(gate_ranks)) if gate_ranks else 0,
                "mean_up_rank": float(np.mean(up_ranks)) if up_ranks else 0,
                "mean_down_rank": float(np.mean(down_ranks)) if down_ranks else 0,
                "mean_similarity": float(np.mean(similarities)) if similarities else 0,
                "overall_compression": self._compute_compression_ratio(
                    gate_ranks, up_ranks, down_ranks
                ),
            }

            self.log("\n--- Phase 1 Summary ---")
            self.log(f"Mean gate rank (95%): {results['summary']['mean_gate_rank']:.1f}")
            self.log(f"Mean up rank (95%): {results['summary']['mean_up_rank']:.1f}")
            self.log(f"Mean down rank (95%): {results['summary']['mean_down_rank']:.1f}")
            self.log(f"Mean pairwise similarity: {results['summary']['mean_similarity']:.3f}")
            self.log(f"Overall compression ratio: {results['summary']['overall_compression']:.1f}x")

        return results

    def _phase2_overlay_representation(self, model, layers: list[int]) -> dict:
        """Compute base + delta overlay representation."""
        results = {"layers": {}, "storage_estimate": {}}

        # Get target ranks from config or use defaults
        gate_rank = self.params.get("gate_rank", 2)
        up_rank = self.params.get("up_rank", 128)
        down_rank = self.params.get("down_rank", 64)

        self.log(f"Target ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")

        total_original_bytes = 0
        total_compressed_bytes = 0

        for layer_idx in tqdm(layers, desc="Computing Overlay"):
            layer_result = self._compute_overlay_layer(
                model, layer_idx, gate_rank, up_rank, down_rank
            )
            if layer_result:
                results["layers"][layer_idx] = layer_result
                total_original_bytes += layer_result["original_bytes"]
                total_compressed_bytes += layer_result["compressed_bytes"]

        results["storage_estimate"] = {
            "total_original_mb": total_original_bytes / (1024 * 1024),
            "total_compressed_mb": total_compressed_bytes / (1024 * 1024),
            "compression_ratio": (
                total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
            ),
        }

        self.log("\n--- Phase 2 Summary ---")
        self.log(f"Original size: {results['storage_estimate']['total_original_mb']:.1f} MB")
        self.log(f"Compressed size: {results['storage_estimate']['total_compressed_mb']:.1f} MB")
        self.log(f"Compression ratio: {results['storage_estimate']['compression_ratio']:.1f}x")

        return results

    def _phase3_verify_reconstruction(self, model, tokenizer, layers: list[int]) -> dict:
        """Verify that overlay representation reconstructs accurately."""
        results = {"layers": {}, "summary": {}}

        gate_rank = self.params.get("gate_rank", 2)
        up_rank = self.params.get("up_rank", 128)
        down_rank = self.params.get("down_rank", 64)

        all_weight_errors = []
        all_output_errors = []

        for layer_idx in tqdm(layers, desc="Verification"):
            layer_result = self._verify_layer_reconstruction(
                model, tokenizer, layer_idx, gate_rank, up_rank, down_rank
            )
            if layer_result:
                results["layers"][layer_idx] = layer_result
                all_weight_errors.append(layer_result["mean_weight_relative_error"])
                all_output_errors.append(layer_result["mean_output_relative_error"])

                self.log(
                    f"  Layer {layer_idx}: weight_err={layer_result['mean_weight_relative_error']:.4f}, "
                    f"output_err={layer_result['mean_output_relative_error']:.4f}"
                )

        if all_weight_errors:
            results["summary"] = {
                "mean_weight_error": float(np.mean(all_weight_errors)),
                "max_weight_error": float(np.max(all_weight_errors)),
                "mean_output_error": float(np.mean(all_output_errors)),
                "max_output_error": float(np.max(all_output_errors)),
                "passed": float(np.max(all_weight_errors)) < 0.01,  # <1% threshold
            }

            self.log("\n--- Phase 3 Summary ---")
            self.log(f"Mean weight error: {results['summary']['mean_weight_error']:.4f}")
            self.log(f"Max weight error: {results['summary']['max_weight_error']:.4f}")
            self.log(f"Mean output error: {results['summary']['mean_output_error']:.4f}")
            self.log(f"Verification: {'PASSED' if results['summary']['passed'] else 'FAILED'}")

        return results

    def _phase4_perplexity(self, model, tokenizer) -> dict:
        """Evaluate perplexity with overlay representation."""
        # This is a placeholder - full implementation would:
        # 1. Compute baseline perplexity on original model
        # 2. Replace experts with overlay representation
        # 3. Compute perplexity on overlay model
        # 4. Compare

        self.log("Perplexity evaluation not yet implemented")
        return {"status": "not_implemented"}

    def _analyze_layer_svd(self, model, layer_idx: int) -> LayerAnalysis | None:
        """Perform SVD analysis on one layer's experts."""
        layers = self._get_model_layers(model)
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        experts = getattr(mlp, "experts", None)
        if experts is None:
            return None

        # Get weights based on expert structure
        if hasattr(experts, "gate_up_proj_blocks"):
            # GPT-OSS batched/quantized experts
            return self._analyze_batched_experts(experts, layer_idx)
        elif isinstance(experts, list):
            # List-based experts (OLMoE, Mixtral)
            return self._analyze_list_experts(experts, layer_idx)

        return None

    def _analyze_batched_experts(self, experts, layer_idx: int) -> LayerAnalysis:
        """Analyze GPT-OSS batched/quantized experts."""
        num_experts = experts.num_experts

        # Dequantize weights
        gate_up = self._dequantize_mxfp4(
            experts.gate_up_proj_blocks,
            experts.gate_up_proj_scales,
            experts.gate_up_proj_bias,
        )
        down = self._dequantize_mxfp4(
            experts.down_proj_blocks,
            experts.down_proj_scales,
            experts.down_proj_bias,
        )

        # Split gate and up from interleaved
        gate = gate_up[:, 0::2, :]
        up = gate_up[:, 1::2, :]

        analysis = LayerAnalysis(layer_idx=layer_idx, num_experts=num_experts)

        # Analyze each projection
        analysis.gate = self._svd_projection(gate, "gate")
        analysis.up = self._svd_projection(up, "up")
        analysis.down = self._svd_projection(down, "down")

        # Compute pairwise similarities using down projection
        sims = self._compute_similarities(down)
        analysis.mean_similarity = float(np.mean(sims))
        analysis.std_similarity = float(np.std(sims))

        return analysis

    def _analyze_list_experts(self, experts: list, layer_idx: int) -> LayerAnalysis:
        """Analyze list-based experts."""
        num_experts = len(experts)

        # Collect weights
        gate_weights = []
        up_weights = []
        down_weights = []

        for expert in experts:
            if hasattr(expert, "gate_proj"):
                gate_weights.append(expert.gate_proj.weight.astype(mx.float32))
            if hasattr(expert, "up_proj"):
                up_weights.append(expert.up_proj.weight.astype(mx.float32))
            if hasattr(expert, "down_proj"):
                down_weights.append(expert.down_proj.weight.astype(mx.float32))

        analysis = LayerAnalysis(layer_idx=layer_idx, num_experts=num_experts)

        if gate_weights:
            gate = mx.stack(gate_weights, axis=0)
            mx.eval(gate)
            analysis.gate = self._svd_projection(gate, "gate")

        if up_weights:
            up = mx.stack(up_weights, axis=0)
            mx.eval(up)
            analysis.up = self._svd_projection(up, "up")

        if down_weights:
            down = mx.stack(down_weights, axis=0)
            mx.eval(down)
            analysis.down = self._svd_projection(down, "down")

            # Compute similarities
            sims = self._compute_similarities(down)
            analysis.mean_similarity = float(np.mean(sims))
            analysis.std_similarity = float(np.std(sims))

        return analysis

    def _svd_projection(self, weights: mx.array, proj_name: str) -> ProjectionSVD:
        """Compute SVD analysis for one projection across all experts."""
        num_experts, out_dim, in_dim = weights.shape
        max_rank = min(out_dim, in_dim)

        result = ProjectionSVD(
            proj_name=proj_name,
            shape=(out_dim, in_dim),
            max_rank=max_rank,
        )

        # Compute base (mean) expert
        base = mx.mean(weights, axis=0)
        mx.eval(base)

        # Analyze each expert's delta
        for i in range(num_experts):
            delta = weights[i] - base
            mx.eval(delta)

            # SVD
            delta_np = np.array(delta.astype(mx.float32))
            try:
                _, S, _ = np.linalg.svd(delta_np, full_matrices=False)
            except np.linalg.LinAlgError:
                result.ranks_90.append(0)
                result.ranks_95.append(0)
                result.ranks_99.append(0)
                continue

            result.ranks_90.append(self._effective_rank(S, 0.90))
            result.ranks_95.append(self._effective_rank(S, 0.95))
            result.ranks_99.append(self._effective_rank(S, 0.99))

        result.compute_stats()
        return result

    def _effective_rank(self, S: np.ndarray, threshold: float) -> int:
        """Compute effective rank to capture threshold variance."""
        total = np.sum(S**2)
        if total == 0:
            return 0
        cumsum = np.cumsum(S**2) / total
        return int(np.searchsorted(cumsum, threshold) + 1)

    def _compute_similarities(self, weights: mx.array) -> list[float]:
        """Compute pairwise cosine similarities."""
        num_experts = weights.shape[0]
        sims = []

        # Flatten each expert's weights
        flat = weights.reshape(num_experts, -1)
        mx.eval(flat)
        flat_np = np.array(flat.astype(mx.float32))

        # Normalize
        norms = np.linalg.norm(flat_np, axis=1, keepdims=True)
        normalized = flat_np / (norms + 1e-10)

        # Compute pairwise cosine similarities
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                sim = np.dot(normalized[i], normalized[j])
                sims.append(float(sim))

        return sims

    def _compute_overlay_layer(
        self,
        model,
        layer_idx: int,
        gate_rank: int,
        up_rank: int,
        down_rank: int,
    ) -> dict | None:
        """Compute overlay representation for one layer."""
        layers = self._get_model_layers(model)
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        experts = getattr(mlp, "experts", None)
        if experts is None:
            return None

        # Get weights
        if hasattr(experts, "gate_up_proj_blocks"):
            gate_up = self._dequantize_mxfp4(
                experts.gate_up_proj_blocks,
                experts.gate_up_proj_scales,
                experts.gate_up_proj_bias,
            )
            down = self._dequantize_mxfp4(
                experts.down_proj_blocks,
                experts.down_proj_scales,
                experts.down_proj_bias,
            )
            gate = gate_up[:, 0::2, :]
            up = gate_up[:, 1::2, :]
            num_experts = experts.num_experts
        elif isinstance(experts, list):
            gate = mx.stack([e.gate_proj.weight.astype(mx.float32) for e in experts])
            up = mx.stack([e.up_proj.weight.astype(mx.float32) for e in experts])
            down = mx.stack([e.down_proj.weight.astype(mx.float32) for e in experts])
            mx.eval(gate, up, down)
            num_experts = len(experts)
        else:
            return None

        # Compute storage estimates
        _, gate_out, gate_in = gate.shape
        _, up_out, up_in = up.shape
        _, down_out, down_in = down.shape

        # Original: all full matrices
        original_bytes = (
            num_experts * gate_out * gate_in * 2
            + num_experts * up_out * up_in * 2
            + num_experts * down_out * down_in * 2
        )

        # Compressed: base + low-rank deltas
        # Base: 3 full matrices
        base_bytes = (gate_out * gate_in + up_out * up_in + down_out * down_in) * 2

        # Deltas: U, S, V for each expert (we store U*S, V separately)
        delta_bytes = num_experts * (
            gate_rank * (gate_out + gate_in) * 2
            + up_rank * (up_out + up_in) * 2
            + down_rank * (down_out + down_in) * 2
        )

        compressed_bytes = base_bytes + delta_bytes

        return {
            "num_experts": num_experts,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 0,
            "ranks": {"gate": gate_rank, "up": up_rank, "down": down_rank},
        }

    def _verify_layer_reconstruction(
        self,
        model,
        tokenizer,
        layer_idx: int,
        gate_rank: int,
        up_rank: int,
        down_rank: int,
    ) -> dict | None:
        """Verify reconstruction accuracy for one layer."""
        layers = self._get_model_layers(model)
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        experts = getattr(mlp, "experts", None)
        if experts is None:
            return None

        # Get weights
        if hasattr(experts, "gate_up_proj_blocks"):
            gate_up = self._dequantize_mxfp4(
                experts.gate_up_proj_blocks,
                experts.gate_up_proj_scales,
                experts.gate_up_proj_bias,
            )
            down = self._dequantize_mxfp4(
                experts.down_proj_blocks,
                experts.down_proj_scales,
                experts.down_proj_bias,
            )
            gate = gate_up[:, 0::2, :]
            up = gate_up[:, 1::2, :]
            num_experts = experts.num_experts
        elif isinstance(experts, list):
            gate = mx.stack([e.gate_proj.weight.astype(mx.float32) for e in experts])
            up = mx.stack([e.up_proj.weight.astype(mx.float32) for e in experts])
            down = mx.stack([e.down_proj.weight.astype(mx.float32) for e in experts])
            mx.eval(gate, up, down)
            num_experts = len(experts)
        else:
            return None

        # Verify reconstruction for each projection
        weight_errors = []
        output_errors = []

        for proj_name, weights, rank in [
            ("gate", gate, gate_rank),
            ("up", up, up_rank),
            ("down", down, down_rank),
        ]:
            error = self._verify_projection_reconstruction(weights, rank)
            weight_errors.append(error)

        # Test output reconstruction using down projection (simpler, consistent shapes)
        # down: (num_experts, hidden, intermediate)
        hidden_dim = down.shape[1]
        test_input = mx.random.normal((1, 10, down.shape[2]))  # (batch, seq, intermediate)
        mx.eval(test_input)

        for expert_idx in range(min(4, num_experts)):  # Test first 4 experts
            # Original output through down projection
            orig_down = down[expert_idx]  # (hidden, intermediate)
            orig_out = test_input @ orig_down.T  # (1, 10, hidden)

            # Reconstructed
            recon_down = self._reconstruct_with_svd(down, expert_idx, down_rank)
            recon_out = test_input @ recon_down.T

            mx.eval(orig_out, recon_out)
            diff = orig_out - recon_out
            mse = float(mx.mean(diff * diff))
            orig_norm = float(mx.mean(orig_out * orig_out))
            rel_error = mse / (orig_norm + 1e-10)
            output_errors.append(rel_error)

        return {
            "mean_weight_relative_error": float(np.mean(weight_errors)),
            "max_weight_relative_error": float(np.max(weight_errors)),
            "mean_output_relative_error": float(np.mean(output_errors)),
            "max_output_relative_error": float(np.max(output_errors)),
        }

    def _verify_projection_reconstruction(self, weights: mx.array, rank: int) -> float:
        """Verify reconstruction error for one projection."""
        num_experts = weights.shape[0]
        base = mx.mean(weights, axis=0)
        mx.eval(base)

        errors = []
        for i in range(num_experts):
            original = weights[i]
            reconstructed = self._reconstruct_with_svd(weights, i, rank)

            mx.eval(original, reconstructed)
            diff = original - reconstructed
            mse = float(mx.mean(diff * diff))
            orig_norm = float(mx.mean(original * original))
            rel_error = mse / (orig_norm + 1e-10)
            errors.append(rel_error)

        return float(np.mean(errors))

    def _reconstruct_with_svd(self, weights: mx.array, expert_idx: int, rank: int) -> mx.array:
        """Reconstruct expert weight using truncated SVD."""
        base = mx.mean(weights, axis=0)
        delta = weights[expert_idx] - base

        # SVD truncation
        delta_np = np.array(delta.astype(mx.float32))
        U, S, Vh = np.linalg.svd(delta_np, full_matrices=False)

        # Truncate
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]

        # Reconstruct delta
        delta_recon = U_trunc @ np.diag(S_trunc) @ Vh_trunc

        # Add base
        result = mx.array(delta_recon) + base
        return result

    def _compute_compression_ratio(
        self,
        gate_ranks: list[float],
        up_ranks: list[float],
        down_ranks: list[float],
    ) -> float:
        """Compute overall compression ratio."""
        if not gate_ranks or not up_ranks or not down_ranks:
            return 0.0

        # Use first layer's analysis for dimensions
        if not self.layer_analyses:
            return 0.0

        first = self.layer_analyses[0]
        if not first.gate or not first.up or not first.down:
            return 0.0

        num_experts = first.num_experts
        gate_shape = first.gate.shape
        up_shape = first.up.shape
        down_shape = first.down.shape

        # Original params
        original = num_experts * (
            gate_shape[0] * gate_shape[1]
            + up_shape[0] * up_shape[1]
            + down_shape[0] * down_shape[1]
        )

        # Compressed params (base + deltas)
        avg_gate_rank = int(np.mean(gate_ranks))
        avg_up_rank = int(np.mean(up_ranks))
        avg_down_rank = int(np.mean(down_ranks))

        base_params = (
            gate_shape[0] * gate_shape[1]
            + up_shape[0] * up_shape[1]
            + down_shape[0] * down_shape[1]
        )

        delta_params = num_experts * (
            avg_gate_rank * (gate_shape[0] + gate_shape[1])
            + avg_up_rank * (up_shape[0] + up_shape[1])
            + avg_down_rank * (down_shape[0] + down_shape[1])
        )

        compressed = base_params + delta_params

        return original / compressed if compressed > 0 else 0.0

    def _build_summary(
        self,
        phase1: dict,
        phase2: dict,
        phase3: dict,
        phase4: dict,
    ) -> dict:
        """Build experiment summary."""
        summary = {
            "hypothesis_confirmed": False,
            "compression_achieved": 0.0,
            "quality_preserved": False,
        }

        # Check Phase 1: Low effective rank?
        if "summary" in phase1:
            avg_rank = (
                phase1["summary"].get("mean_gate_rank", 0)
                + phase1["summary"].get("mean_up_rank", 0)
                + phase1["summary"].get("mean_down_rank", 0)
            ) / 3
            # Hypothesis confirmed if rank < 200 (vs 2880 max)
            summary["hypothesis_confirmed"] = avg_rank < 200

        # Check Phase 2: Compression ratio
        if "storage_estimate" in phase2:
            summary["compression_achieved"] = phase2["storage_estimate"].get("compression_ratio", 0)

        # Check Phase 3: Quality preserved
        if "summary" in phase3:
            summary["quality_preserved"] = phase3["summary"].get("passed", False)

        return summary

    def _dequantize_mxfp4(
        self,
        blocks: mx.array,
        scales: mx.array,
        bias: mx.array | None = None,
    ) -> mx.array:
        """Dequantize MXFP4 weights."""
        dequant = mx.dequantize(blocks, scales, biases=None, group_size=32, bits=4, mode="mxfp4")
        if bias is not None:
            dequant = dequant + bias[:, :, None]
        mx.eval(dequant)
        return dequant.astype(mx.float32)

    def _find_moe_layers(self, model) -> list[int]:
        """Find indices of MoE layers."""
        moe_layers = []
        layers = self._get_model_layers(model)

        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp and (hasattr(mlp, "experts") or hasattr(mlp, "router")):
                moe_layers.append(i)

        return moe_layers

    def _select_layers(self, moe_layers: list[int]) -> list[int]:
        """Select which layers to analyze."""
        explicit = self.params.get("layers")
        if explicit:
            return [l for l in explicit if l in moe_layers]

        # Analyze all if requested
        if self.params.get("analyze_all_layers", False):
            return moe_layers

        # Default: sample 4 layers
        if len(moe_layers) <= 4:
            return moe_layers

        indices = [0, len(moe_layers) // 3, 2 * len(moe_layers) // 3, len(moe_layers) - 1]
        return [moe_layers[i] for i in indices]

    def _get_model_layers(self, model) -> list:
        """Get transformer layers from model."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(model, attr, None)
            if submodel:
                layers = getattr(submodel, "layers", None)
                if layers:
                    return list(layers)
        return list(getattr(model, "layers", []))

    def evaluate(self) -> dict:
        """Return summary metrics."""
        latest = self.load_latest_results("results")
        if not latest:
            return {"error": "No results"}

        return latest.get("run_results", {}).get("summary", {})

    def cleanup(self) -> None:
        """Cleanup."""
        self.layer_analyses = []
        self.reconstruction_results = []
