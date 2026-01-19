"""
Expert SVD Compression Experiment.

Phase 1: Validate SVD analysis across all layers (using MoETypeService)
Phase 2: Compute overlay representation (using MoECompressionService)
Phase 3: Verify reconstruction accuracy (using MoECompressionService)
Phase 4: Estimate storage savings (using MoECompressionService)

Key hypothesis: MoE experts are low-rank perturbations of a shared base,
enabling 8x compression with <1% quality loss.

This experiment now uses the consolidated introspection services instead
of duplicating SVD analysis logic.
"""

import asyncio
import logging

from chuk_lazarus.experiments import ExperimentBase
from chuk_lazarus.introspection.moe import (
    MoECompressionService,
    MoEType,
    MoETypeService,
)

logger = logging.getLogger(__name__)


class ExpertSVDExperiment(ExperimentBase):
    """
    Validate and implement MoE expert compression via SVD.

    Tests hypothesis: experts = base + low_rank_delta

    Uses the consolidated MoETypeService and MoECompressionService
    for all SVD analysis and compression operations.
    """

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up Expert SVD compression experiment...")
        self.params = self.config.parameters

    def run(self) -> dict:
        """Run all experiment phases."""
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict:
        """Async implementation of run."""
        self.log("=" * 70)
        self.log("EXPERT SVD COMPRESSION EXPERIMENT")
        self.log("Testing: experts = base + low_rank_delta")
        self.log("=" * 70)

        model_id = self.config.model

        # Get compression ranks from config or use defaults
        gate_rank = self.params.get("gate_rank", None)
        up_rank = self.params.get("up_rank", None)
        down_rank = self.params.get("down_rank", None)

        # Phase 1: MoE Type Analysis (SVD-based classification)
        self.log("\n" + "=" * 70)
        self.log("PHASE 1: MoE Type Analysis")
        self.log("=" * 70)
        phase1_results = await self._phase1_type_analysis(model_id)

        # Phase 2: Compute overlay representation
        self.log("\n" + "=" * 70)
        self.log("PHASE 2: Overlay Representation")
        self.log("=" * 70)
        phase2_results = await self._phase2_overlay_representation(
            model_id, gate_rank, up_rank, down_rank
        )

        # Phase 3: Verify reconstruction
        self.log("\n" + "=" * 70)
        self.log("PHASE 3: Reconstruction Verification")
        self.log("=" * 70)
        phase3_results = await self._phase3_verify_reconstruction(
            model_id, gate_rank, up_rank, down_rank
        )

        # Phase 4: Storage savings estimate
        self.log("\n" + "=" * 70)
        self.log("PHASE 4: Storage Savings Estimate")
        self.log("=" * 70)
        phase4_results = await self._phase4_estimate_savings(
            model_id, gate_rank, up_rank, down_rank
        )

        # Build summary
        summary = self._build_summary(
            phase1_results, phase2_results, phase3_results, phase4_results
        )

        return {
            "phase1_type_analysis": phase1_results,
            "phase2_overlay": phase2_results,
            "phase3_reconstruction": phase3_results,
            "phase4_storage": phase4_results,
            "summary": summary,
        }

    async def _phase1_type_analysis(self, model_id: str) -> dict:
        """Analyze MoE type using MoETypeService."""
        self.log(f"Analyzing MoE type for: {model_id}")

        try:
            result = await MoETypeService.analyze(model_id)

            self.log(f"  Model: {result.model_id}")
            self.log(f"  Layer: {result.layer_idx}")
            self.log(f"  Experts: {result.num_experts}")
            self.log(f"  Type: {result.moe_type.value}")
            self.log(f"  Confidence: {result.confidence:.1%} ({result.confidence_label})")

            self.log("\n  Projection Analysis (95% variance rank):")
            self.log(f"    Gate: rank={result.gate.effective_rank_95}, "
                     f"ratio={result.gate.rank_ratio:.1%}, "
                     f"compression={result.gate.compression_ratio:.1f}x")
            self.log(f"    Up:   rank={result.up.effective_rank_95}, "
                     f"ratio={result.up.rank_ratio:.1%}, "
                     f"compression={result.up.compression_ratio:.1f}x")
            self.log(f"    Down: rank={result.down.effective_rank_95}, "
                     f"ratio={result.down.rank_ratio:.1%}, "
                     f"compression={result.down.compression_ratio:.1f}x")

            self.log(f"\n  Expert Similarity: {result.mean_cosine_similarity:.3f} ± {result.std_cosine_similarity:.3f}")
            self.log(f"  Training Origin: {result.training_origin}")
            self.log(f"  Compressible: {result.is_compressible}")
            self.log(f"  Estimated Compression: {result.estimated_compression:.1f}x")

            return {
                "model_id": result.model_id,
                "layer_idx": result.layer_idx,
                "num_experts": result.num_experts,
                "moe_type": result.moe_type.value,
                "confidence": result.confidence,
                "gate_rank": result.gate.effective_rank_95,
                "up_rank": result.up.effective_rank_95,
                "down_rank": result.down.effective_rank_95,
                "mean_similarity": result.mean_cosine_similarity,
                "is_compressible": result.is_compressible,
                "estimated_compression": result.estimated_compression,
            }
        except Exception as e:
            self.log(f"Error in type analysis: {e}")
            return {"error": str(e)}

    async def _phase2_overlay_representation(
        self,
        model_id: str,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
    ) -> dict:
        """Compute overlay representation using MoECompressionService."""
        self.log(f"Computing overlay representation...")

        try:
            result = await MoECompressionService.compute_overlay(
                model_id,
                gate_rank=gate_rank,
                up_rank=up_rank,
                down_rank=down_rank,
            )

            self.log(f"  Layer: {result.layer_idx}")
            self.log(f"  Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}")
            self.log(f"\n  Per-projection compression:")
            self.log(f"    Gate: {result.gate.compression_ratio:.1f}x")
            self.log(f"    Up:   {result.up.compression_ratio:.1f}x")
            self.log(f"    Down: {result.down.compression_ratio:.1f}x")
            self.log(f"\n  Total compression: {result.compression_ratio:.1f}x")
            self.log(f"  Original: {result.total_original_bytes / (1024*1024):.1f} MB")
            self.log(f"  Compressed: {result.total_compressed_bytes / (1024*1024):.1f} MB")

            return {
                "layer_idx": result.layer_idx,
                "num_experts": result.num_experts,
                "gate_rank": result.gate_rank,
                "up_rank": result.up_rank,
                "down_rank": result.down_rank,
                "total_original_mb": result.total_original_bytes / (1024 * 1024),
                "total_compressed_mb": result.total_compressed_bytes / (1024 * 1024),
                "compression_ratio": result.compression_ratio,
            }
        except Exception as e:
            self.log(f"Error in overlay computation: {e}")
            return {"error": str(e)}

    async def _phase3_verify_reconstruction(
        self,
        model_id: str,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
    ) -> dict:
        """Verify reconstruction using MoECompressionService."""
        self.log(f"Verifying reconstruction accuracy...")

        try:
            result = await MoECompressionService.verify_reconstruction(
                model_id,
                gate_rank=gate_rank,
                up_rank=up_rank,
                down_rank=down_rank,
            )

            self.log(f"  Layer: {result.layer_idx}")
            self.log(f"  Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}")
            self.log(f"\n  Weight-level errors:")
            self.log(f"    Gate: mean={result.gate.mean_relative_error:.4f}, max={result.gate.max_relative_error:.4f}")
            self.log(f"    Up:   mean={result.up.mean_relative_error:.4f}, max={result.up.max_relative_error:.4f}")
            self.log(f"    Down: mean={result.down.mean_relative_error:.4f}, max={result.down.max_relative_error:.4f}")
            self.log(f"\n  Output-level errors:")
            self.log(f"    Mean: {result.mean_output_error:.4f}")
            self.log(f"    Max:  {result.max_output_error:.4f}")
            self.log(f"\n  Verification: {'PASSED' if result.passed else 'FAILED'}")

            return {
                "layer_idx": result.layer_idx,
                "gate_error": result.gate.mean_relative_error,
                "up_error": result.up.mean_relative_error,
                "down_error": result.down.mean_relative_error,
                "mean_output_error": result.mean_output_error,
                "max_output_error": result.max_output_error,
                "overall_weight_error": result.overall_weight_error,
                "passed": result.passed,
            }
        except Exception as e:
            self.log(f"Error in reconstruction verification: {e}")
            return {"error": str(e)}

    async def _phase4_estimate_savings(
        self,
        model_id: str,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
    ) -> dict:
        """Estimate storage savings using MoECompressionService."""
        self.log(f"Estimating storage savings...")

        try:
            result = await MoECompressionService.estimate_savings(
                model_id,
                gate_rank=gate_rank,
                up_rank=up_rank,
                down_rank=down_rank,
            )

            self.log(f"  Model: {result.model_id}")
            self.log(f"  MoE Layers: {result.num_layers}")
            self.log(f"  Experts per layer: {result.num_experts}")
            self.log(f"  Ranks: gate={result.gate_rank}, up={result.up_rank}, down={result.down_rank}")
            self.log(f"\n  Storage:")
            self.log(f"    Original:   {result.original_mb:.1f} MB")
            self.log(f"    Compressed: {result.compressed_mb:.1f} MB")
            self.log(f"    Savings:    {result.savings_mb:.1f} MB ({result.compression_ratio:.1f}x)")

            return {
                "num_layers": result.num_layers,
                "num_experts": result.num_experts,
                "original_mb": result.original_mb,
                "compressed_mb": result.compressed_mb,
                "savings_mb": result.savings_mb,
                "compression_ratio": result.compression_ratio,
            }
        except Exception as e:
            self.log(f"Error in savings estimation: {e}")
            return {"error": str(e)}

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
            "moe_type": None,
            "compression_achieved": 0.0,
            "quality_preserved": False,
        }

        # Check Phase 1: MoE type analysis
        if "error" not in phase1:
            summary["moe_type"] = phase1.get("moe_type")
            summary["hypothesis_confirmed"] = phase1.get("is_compressible", False)

        # Check Phase 2/4: Compression ratio
        if "error" not in phase2:
            summary["compression_achieved"] = phase2.get("compression_ratio", 0)
        elif "error" not in phase4:
            summary["compression_achieved"] = phase4.get("compression_ratio", 0)

        # Check Phase 3: Quality preserved
        if "error" not in phase3:
            summary["quality_preserved"] = phase3.get("passed", False)

        # Log summary
        self.log("\n" + "=" * 70)
        self.log("EXPERIMENT SUMMARY")
        self.log("=" * 70)
        self.log(f"  MoE Type: {summary['moe_type']}")
        self.log(f"  Hypothesis Confirmed: {summary['hypothesis_confirmed']}")
        self.log(f"  Compression Achieved: {summary['compression_achieved']:.1f}x")
        self.log(f"  Quality Preserved: {summary['quality_preserved']}")

        return summary

    def evaluate(self) -> dict:
        """Return summary metrics."""
        latest = self.load_latest_results("results")
        if not latest:
            return {"error": "No results"}

        return latest.get("run_results", {}).get("summary", {})

    def cleanup(self) -> None:
        """Cleanup."""
        pass
