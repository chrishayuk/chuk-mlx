"""MoE compression via SVD overlay representation.

Implements the overlay compression strategy for pseudo-MoE models:
    expert_i = base + U_i @ S_i @ V_i^T

Where base is the mean expert and U_i, S_i, V_i are truncated SVD factors
of the low-rank delta (expert_i - base).

This provides significant compression for pseudo-MoE models (8x typical)
while preserving quality (<1% reconstruction error).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .detector import get_moe_layers
from .enums import MoEArchitecture
from .moe_type import MoETypeService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import mlx.core as mx


# =============================================================================
# Pydantic Models
# =============================================================================


class ProjectionOverlay(BaseModel):
    """Overlay representation for a single projection type."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(description="Projection name: gate, up, or down")
    shape: tuple[int, int] = Field(description="(out_features, in_features)")
    rank: int = Field(ge=1, description="Truncation rank used")
    num_experts: int = Field(ge=1, description="Number of experts")

    # Storage metrics
    original_bytes: int = Field(ge=0, description="Original storage in bytes")
    compressed_bytes: int = Field(ge=0, description="Compressed storage in bytes")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 1.0


class OverlayRepresentation(BaseModel):
    """Complete overlay representation for a layer's experts."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    layer_idx: int = Field(ge=0, description="Layer index")
    num_experts: int = Field(ge=1, description="Number of experts")

    # Per-projection overlays
    gate: ProjectionOverlay = Field(description="Gate projection overlay")
    up: ProjectionOverlay = Field(description="Up projection overlay")
    down: ProjectionOverlay = Field(description="Down projection overlay")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank used for gate projection")
    up_rank: int = Field(ge=1, description="Rank used for up projection")
    down_rank: int = Field(ge=1, description="Rank used for down projection")

    @property
    def total_original_bytes(self) -> int:
        """Total original storage in bytes."""
        return self.gate.original_bytes + self.up.original_bytes + self.down.original_bytes

    @property
    def total_compressed_bytes(self) -> int:
        """Total compressed storage in bytes."""
        return self.gate.compressed_bytes + self.up.compressed_bytes + self.down.compressed_bytes

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        return (
            self.total_original_bytes / self.total_compressed_bytes
            if self.total_compressed_bytes > 0
            else 1.0
        )


class ReconstructionError(BaseModel):
    """Reconstruction error metrics for a projection."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Projection name")
    mean_relative_error: float = Field(ge=0.0, description="Mean relative error across experts")
    max_relative_error: float = Field(ge=0.0, description="Max relative error across experts")
    mean_mse: float = Field(ge=0.0, description="Mean squared error")


class ReconstructionVerification(BaseModel):
    """Verification results for overlay reconstruction."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    layer_idx: int = Field(ge=0, description="Layer index")

    # Per-projection errors
    gate: ReconstructionError = Field(description="Gate projection errors")
    up: ReconstructionError = Field(description="Up projection errors")
    down: ReconstructionError = Field(description="Down projection errors")

    # Output-level verification
    mean_output_error: float = Field(ge=0.0, description="Mean output relative error")
    max_output_error: float = Field(ge=0.0, description="Max output relative error")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank used for gate")
    up_rank: int = Field(ge=1, description="Rank used for up")
    down_rank: int = Field(ge=1, description="Rank used for down")

    @property
    def passed(self) -> bool:
        """Whether reconstruction quality is acceptable (<1% error)."""
        return self.max_output_error < 0.01

    @property
    def overall_weight_error(self) -> float:
        """Mean weight error across all projections."""
        return (
            self.gate.mean_relative_error
            + self.up.mean_relative_error
            + self.down.mean_relative_error
        ) / 3


class StorageEstimate(BaseModel):
    """Storage estimate for overlay compression."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(ge=1, description="Number of MoE layers")
    num_experts: int = Field(ge=1, description="Number of experts per layer")

    # Storage in MB
    original_mb: float = Field(ge=0.0, description="Original storage in MB")
    compressed_mb: float = Field(ge=0.0, description="Compressed storage in MB")

    # Ranks used
    gate_rank: int = Field(ge=1, description="Rank for gate projection")
    up_rank: int = Field(ge=1, description="Rank for up projection")
    down_rank: int = Field(ge=1, description="Rank for down projection")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_mb / self.compressed_mb if self.compressed_mb > 0 else 1.0

    @property
    def savings_mb(self) -> float:
        """Storage savings in MB."""
        return self.original_mb - self.compressed_mb


class CompressionConfig(BaseModel):
    """Configuration for compressed model format."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Original model identifier")
    num_layers: int = Field(ge=1, description="Number of MoE layers")
    num_experts: int = Field(ge=1, description="Number of experts per layer")
    moe_layer_indices: list[int] = Field(description="Indices of MoE layers in original model")

    # Projection dimensions
    gate_shape: tuple[int, int] = Field(description="(out_features, in_features) for gate")
    up_shape: tuple[int, int] = Field(description="(out_features, in_features) for up")
    down_shape: tuple[int, int] = Field(description="(out_features, in_features) for down")

    # Compression ranks
    gate_rank: int = Field(ge=1, description="Rank for gate projection")
    up_rank: int = Field(ge=1, description="Rank for up projection")
    down_rank: int = Field(ge=1, description="Rank for down projection")

    # Bias info
    has_biases: bool = Field(default=False, description="Whether biases are stored separately")

    # Storage stats
    original_bytes: int = Field(ge=0, description="Original expert storage in bytes")
    compressed_bytes: int = Field(ge=0, description="Compressed storage in bytes")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else 1.0


class CompressionResult(BaseModel):
    """Result of model compression."""

    model_config = ConfigDict(frozen=True)

    output_path: str = Field(description="Path to compressed model directory")
    config: CompressionConfig = Field(description="Compression configuration")
    mean_reconstruction_error: float = Field(ge=0.0, description="Mean reconstruction error")
    max_reconstruction_error: float = Field(ge=0.0, description="Max reconstruction error")

    @property
    def compression_ratio(self) -> float:
        """Compression ratio achieved."""
        return self.config.compression_ratio

    @property
    def original_mb(self) -> float:
        """Original size in MB."""
        return self.config.original_bytes / (1024 * 1024)

    @property
    def compressed_mb(self) -> float:
        """Compressed size in MB."""
        return self.config.compressed_bytes / (1024 * 1024)


# =============================================================================
# Service Class
# =============================================================================


class MoECompressionService:
    """Service for MoE compression via SVD overlay.

    Usage:
        # Compute overlay representation
        overlay = await MoECompressionService.compute_overlay(
            "openai/gpt-oss-20b",
            gate_rank=2, up_rank=128, down_rank=64
        )

        # Verify reconstruction accuracy
        verification = await MoECompressionService.verify_reconstruction(
            "openai/gpt-oss-20b",
            gate_rank=2, up_rank=128, down_rank=64
        )

        # Estimate storage savings
        estimate = await MoECompressionService.estimate_savings(
            "openai/gpt-oss-20b",
            gate_rank=2, up_rank=128, down_rank=64
        )
    """

    # Default ranks (based on GPT-OSS analysis)
    DEFAULT_GATE_RANK: int = 2
    DEFAULT_UP_RANK: int = 128
    DEFAULT_DOWN_RANK: int = 64

    # Variance threshold for auto-rank selection
    VARIANCE_THRESHOLD: float = 0.95

    @classmethod
    async def compute_overlay(
        cls,
        model_id: str,
        *,
        layer: int | None = None,
        gate_rank: int | None = None,
        up_rank: int | None = None,
        down_rank: int | None = None,
    ) -> OverlayRepresentation:
        """Compute overlay representation for a model layer.

        Args:
            model_id: HuggingFace model ID or local path
            layer: Layer to analyze (default: first MoE layer)
            gate_rank: Rank for gate projection (default: auto from SVD)
            up_rank: Rank for up projection (default: auto from SVD)
            down_rank: Rank for down projection (default: auto from SVD)

        Returns:
            OverlayRepresentation with compression metrics
        """
        return await asyncio.to_thread(
            cls._compute_overlay_sync, model_id, layer, gate_rank, up_rank, down_rank
        )

    @classmethod
    async def verify_reconstruction(
        cls,
        model_id: str,
        *,
        layer: int | None = None,
        gate_rank: int | None = None,
        up_rank: int | None = None,
        down_rank: int | None = None,
        num_experts_to_verify: int = 4,
    ) -> ReconstructionVerification:
        """Verify reconstruction accuracy of overlay representation.

        Args:
            model_id: HuggingFace model ID or local path
            layer: Layer to verify (default: first MoE layer)
            gate_rank: Rank for gate projection
            up_rank: Rank for up projection
            down_rank: Rank for down projection
            num_experts_to_verify: Number of experts to test (default: 4)

        Returns:
            ReconstructionVerification with error metrics
        """
        return await asyncio.to_thread(
            cls._verify_reconstruction_sync,
            model_id,
            layer,
            gate_rank,
            up_rank,
            down_rank,
            num_experts_to_verify,
        )

    @classmethod
    async def estimate_savings(
        cls,
        model_id: str,
        *,
        gate_rank: int | None = None,
        up_rank: int | None = None,
        down_rank: int | None = None,
    ) -> StorageEstimate:
        """Estimate storage savings for full model compression.

        Args:
            model_id: HuggingFace model ID or local path
            gate_rank: Rank for gate projection
            up_rank: Rank for up projection
            down_rank: Rank for down projection

        Returns:
            StorageEstimate with compression metrics
        """
        return await asyncio.to_thread(
            cls._estimate_savings_sync, model_id, gate_rank, up_rank, down_rank
        )

    @classmethod
    async def compress_model(
        cls,
        model_id: str,
        output_path: str | Path,
        *,
        gate_rank: int | None = None,
        up_rank: int | None = None,
        down_rank: int | None = None,
        dtype: str = "bfloat16",
        resume: bool = True,
    ) -> CompressionResult:
        """Compress MoE model to overlay format.

        Saves the model in a compressed format:
            output_path/
            ├── config.json           # CompressionConfig
            ├── base_weights.safetensors  # Mean expert per layer/projection
            └── deltas.safetensors    # Low-rank U, V factors per expert

        Reconstruction: expert_i = base + U_i @ V_i

        Args:
            model_id: HuggingFace model ID or local path
            output_path: Directory to save compressed model
            gate_rank: Rank for gate projection (default: auto from SVD)
            up_rank: Rank for up projection (default: auto from SVD)
            down_rank: Rank for down projection (default: auto from SVD)
            dtype: Weight dtype for saved tensors (default: bfloat16)
            resume: If True, resume from checkpoint if available (default: True)

        Returns:
            CompressionResult with paths and metrics
        """
        return await asyncio.to_thread(
            cls._compress_model_sync,
            model_id,
            Path(output_path),
            gate_rank,
            up_rank,
            down_rank,
            dtype,
            resume,
        )

    # =========================================================================
    # Synchronous implementations
    # =========================================================================

    @classmethod
    def _compute_overlay_sync(
        cls,
        model_id: str,
        layer: int | None,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
    ) -> OverlayRepresentation:
        """Synchronous implementation of compute_overlay."""

        model = MoETypeService._load_model(model_id)
        moe_layers = get_moe_layers(model)

        if not moe_layers:
            raise ValueError(f"No MoE layers found in {model_id}")

        layer_idx = layer if layer is not None else moe_layers[0]

        # Get experts
        experts = MoETypeService._get_experts(model, layer_idx)
        if experts is None:
            raise ValueError(f"Could not extract experts from layer {layer_idx}")

        # Detect architecture
        from .detector import detect_moe_architecture

        architecture = detect_moe_architecture(model)

        # Extract weights
        gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

        # Auto-select ranks if not provided
        if gate_rank is None or up_rank is None or down_rank is None:
            logger.info("Auto-selecting ranks from SVD analysis...")
            auto_gate, auto_up, auto_down = cls._auto_select_ranks(gate_w, up_w, down_w)
            gate_rank = gate_rank or auto_gate
            up_rank = up_rank or auto_up
            down_rank = down_rank or auto_down

        logger.info(f"Using ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")

        # Compute overlay for each projection
        gate_overlay = cls._compute_projection_overlay(gate_w, "gate", gate_rank)
        up_overlay = cls._compute_projection_overlay(up_w, "up", up_rank)
        down_overlay = cls._compute_projection_overlay(down_w, "down", down_rank)

        return OverlayRepresentation(
            model_id=model_id,
            layer_idx=layer_idx,
            num_experts=num_experts,
            gate=gate_overlay,
            up=up_overlay,
            down=down_overlay,
            gate_rank=gate_rank,
            up_rank=up_rank,
            down_rank=down_rank,
        )

    @classmethod
    def _verify_reconstruction_sync(
        cls,
        model_id: str,
        layer: int | None,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
        num_experts_to_verify: int,
    ) -> ReconstructionVerification:
        """Synchronous implementation of verify_reconstruction."""

        model = MoETypeService._load_model(model_id)
        moe_layers = get_moe_layers(model)

        if not moe_layers:
            raise ValueError(f"No MoE layers found in {model_id}")

        layer_idx = layer if layer is not None else moe_layers[0]

        # Get experts
        experts = MoETypeService._get_experts(model, layer_idx)
        if experts is None:
            raise ValueError(f"Could not extract experts from layer {layer_idx}")

        # Detect architecture
        from .detector import detect_moe_architecture

        architecture = detect_moe_architecture(model)

        # Extract weights
        gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

        # Auto-select ranks if not provided
        if gate_rank is None or up_rank is None or down_rank is None:
            auto_gate, auto_up, auto_down = cls._auto_select_ranks(gate_w, up_w, down_w)
            gate_rank = gate_rank or auto_gate
            up_rank = up_rank or auto_up
            down_rank = down_rank or auto_down

        logger.info(f"Verifying with ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")

        # Verify each projection
        gate_error = cls._verify_projection(gate_w, gate_rank, "gate")
        up_error = cls._verify_projection(up_w, up_rank, "up")
        down_error = cls._verify_projection(down_w, down_rank, "down")

        # Verify output-level reconstruction
        output_errors = cls._verify_output_reconstruction(
            down_w, down_rank, min(num_experts_to_verify, num_experts)
        )

        return ReconstructionVerification(
            model_id=model_id,
            layer_idx=layer_idx,
            gate=gate_error,
            up=up_error,
            down=down_error,
            mean_output_error=float(np.mean(output_errors)),
            max_output_error=float(np.max(output_errors)),
            gate_rank=gate_rank,
            up_rank=up_rank,
            down_rank=down_rank,
        )

    @classmethod
    def _estimate_savings_sync(
        cls,
        model_id: str,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
    ) -> StorageEstimate:
        """Synchronous implementation of estimate_savings."""

        model = MoETypeService._load_model(model_id)
        moe_layers = get_moe_layers(model)

        if not moe_layers:
            raise ValueError(f"No MoE layers found in {model_id}")

        # Get first layer for dimensions
        experts = MoETypeService._get_experts(model, moe_layers[0])
        if experts is None:
            raise ValueError("Could not extract experts")

        from .detector import detect_moe_architecture

        architecture = detect_moe_architecture(model)
        gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

        # Auto-select ranks if not provided
        if gate_rank is None or up_rank is None or down_rank is None:
            auto_gate, auto_up, auto_down = cls._auto_select_ranks(gate_w, up_w, down_w)
            gate_rank = gate_rank or auto_gate
            up_rank = up_rank or auto_up
            down_rank = down_rank or auto_down

        # Calculate storage
        _, gate_out, gate_in = gate_w.shape
        _, up_out, up_in = up_w.shape
        _, down_out, down_in = down_w.shape

        bytes_per_param = 2  # bfloat16

        # Original: num_experts * all projections * all layers
        original_per_layer = (
            num_experts
            * (gate_out * gate_in + up_out * up_in + down_out * down_in)
            * bytes_per_param
        )

        # Compressed: base + low-rank deltas
        base_per_layer = (
            gate_out * gate_in + up_out * up_in + down_out * down_in
        ) * bytes_per_param

        deltas_per_layer = (
            num_experts
            * (
                gate_rank * (gate_out + gate_in)
                + up_rank * (up_out + up_in)
                + down_rank * (down_out + down_in)
            )
            * bytes_per_param
        )

        compressed_per_layer = base_per_layer + deltas_per_layer

        num_layers = len(moe_layers)
        original_bytes = original_per_layer * num_layers
        compressed_bytes = compressed_per_layer * num_layers

        return StorageEstimate(
            model_id=model_id,
            num_layers=num_layers,
            num_experts=num_experts,
            original_mb=original_bytes / (1024 * 1024),
            compressed_mb=compressed_bytes / (1024 * 1024),
            gate_rank=gate_rank,
            up_rank=up_rank,
            down_rank=down_rank,
        )

    @classmethod
    def _compress_model_sync(
        cls,
        model_id: str,
        output_path: Path,
        gate_rank: int | None,
        up_rank: int | None,
        down_rank: int | None,
        dtype: str,
        resume: bool = True,
    ) -> CompressionResult:
        """Synchronous implementation of compress_model."""
        import mlx.core as mx
        from safetensors.numpy import save_file as save_safetensors

        model = MoETypeService._load_model(model_id)
        moe_layers = get_moe_layers(model)

        if not moe_layers:
            raise ValueError(f"No MoE layers found in {model_id}")

        from .detector import detect_moe_architecture

        architecture = detect_moe_architecture(model)

        # Get first layer to determine dimensions and auto-ranks
        experts = MoETypeService._get_experts(model, moe_layers[0])
        if experts is None:
            raise ValueError("Could not extract experts")

        gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

        # Auto-select ranks if not provided
        if gate_rank is None or up_rank is None or down_rank is None:
            logger.info("Auto-selecting ranks from SVD analysis...")
            auto_gate, auto_up, auto_down = cls._auto_select_ranks(gate_w, up_w, down_w)
            gate_rank = gate_rank or auto_gate
            up_rank = up_rank or auto_up
            down_rank = down_rank or auto_down

        logger.info(f"Compressing with ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Checkpoint paths
        checkpoint_path = output_path / "checkpoint.json"
        base_checkpoint_path = output_path / "base_weights_checkpoint.safetensors"
        delta_checkpoint_path = output_path / "deltas_checkpoint.safetensors"

        # Load checkpoint if resuming
        completed_layers: set[int] = set()
        base_weights: dict[str, mx.array] = {}
        delta_weights: dict[str, mx.array] = {}
        all_errors: list[float] = []

        if resume and checkpoint_path.exists():
            checkpoint_data = json.loads(checkpoint_path.read_text())
            completed_layers = set(checkpoint_data.get("completed_layers", []))
            all_errors = checkpoint_data.get("errors", [])

            if base_checkpoint_path.exists():
                base_weights = {
                    k: mx.array(v) for k, v in mx.load(str(base_checkpoint_path)).items()
                }
            if delta_checkpoint_path.exists():
                delta_weights = {
                    k: mx.array(v) for k, v in mx.load(str(delta_checkpoint_path)).items()
                }

            logger.info(
                f"Resuming from checkpoint: {len(completed_layers)}/{len(moe_layers)} layers complete"
            )

        # Map dtype string to mlx dtype
        dtype_map = {
            "bfloat16": mx.bfloat16,
            "float16": mx.float16,
            "float32": mx.float32,
        }
        target_dtype = dtype_map.get(dtype, mx.bfloat16)

        # Helper to save checkpoint
        def save_checkpoint(completed: set[int], errors: list[float]) -> None:
            """Save checkpoint after each layer."""

            def to_numpy_ckpt(arr):
                if arr.dtype == mx.bfloat16:
                    return np.array(arr.astype(mx.float16))
                return np.array(arr)

            # Save weights checkpoint
            if base_weights:
                base_np = {k: to_numpy_ckpt(v) for k, v in base_weights.items()}
                save_safetensors(base_np, str(base_checkpoint_path))
            if delta_weights:
                delta_np = {k: to_numpy_ckpt(v) for k, v in delta_weights.items()}
                save_safetensors(delta_np, str(delta_checkpoint_path))

            # Save progress
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "completed_layers": list(completed),
                        "errors": errors,
                        "gate_rank": gate_rank,
                        "up_rank": up_rank,
                        "down_rank": down_rank,
                    },
                    indent=2,
                )
            )

        # Process each MoE layer with progress bar
        # Determine number of workers (use CPU count, but cap at 8 to avoid memory issues)
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from sklearn.utils.extmath import randomized_svd
        from tqdm import tqdm

        num_workers = min(8, os.cpu_count() or 4)

        def process_expert(args: tuple) -> tuple:
            """Process single expert SVD (runs in thread pool)."""
            expert_idx, delta_np, rank, seed = args
            U_trunc, S_trunc, Vh_trunc = randomized_svd(
                delta_np,
                n_components=rank,
                n_iter=2,
                random_state=seed + expert_idx,  # Vary seed per expert
            )
            U_scaled = U_trunc @ np.diag(S_trunc)
            reconstructed = U_scaled @ Vh_trunc
            error = float(np.mean((delta_np - reconstructed) ** 2))
            return expert_idx, U_scaled, Vh_trunc, error

        remaining_layers = [layer for layer in moe_layers if layer not in completed_layers]
        total_experts = len(remaining_layers) * 3 * num_experts

        with tqdm(total=total_experts, desc="Compressing", unit="expert") as pbar:
            for layer_idx in moe_layers:
                # Skip already completed layers
                if layer_idx in completed_layers:
                    continue

                experts = MoETypeService._get_experts(model, layer_idx)
                if experts is None:
                    raise ValueError(f"Could not extract experts from layer {layer_idx}")

                gate_w, up_w, down_w, _ = MoETypeService._extract_weights(experts, architecture)

                # Process each projection
                for proj_name, weights, rank in [
                    ("gate", gate_w, gate_rank),
                    ("up", up_w, up_rank),
                    ("down", down_w, down_rank),
                ]:
                    pbar.set_description(f"Layer {layer_idx}/{moe_layers[-1]} {proj_name}")

                    # Compute base (mean expert)
                    base = mx.mean(weights, axis=0)
                    mx.eval(base)
                    base_key = f"layer_{layer_idx}_{proj_name}_base"
                    base_weights[base_key] = base.astype(target_dtype)

                    # Prepare all expert deltas
                    expert_deltas = []
                    for expert_idx in range(num_experts):
                        delta = weights[expert_idx] - base
                        mx.eval(delta)
                        delta_np = np.array(delta.astype(mx.float32))
                        expert_deltas.append((expert_idx, delta_np, rank, 42))

                    # Process all experts in parallel with progress updates
                    results_dict = {}
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {
                            executor.submit(process_expert, args): args[0] for args in expert_deltas
                        }
                        for future in as_completed(futures):
                            expert_idx, U_scaled, Vh_trunc, error = future.result()
                            results_dict[expert_idx] = (U_scaled, Vh_trunc, error)
                            pbar.update(1)

                    # Store results in order
                    for expert_idx in range(num_experts):
                        U_scaled, Vh_trunc, error = results_dict[expert_idx]
                        u_key = f"layer_{layer_idx}_{proj_name}_expert_{expert_idx}_U"
                        v_key = f"layer_{layer_idx}_{proj_name}_expert_{expert_idx}_V"
                        delta_weights[u_key] = mx.array(U_scaled).astype(target_dtype)
                        delta_weights[v_key] = mx.array(Vh_trunc).astype(target_dtype)
                        all_errors.append(error)

                # Mark layer complete and save checkpoint
                completed_layers.add(layer_idx)
                save_checkpoint(completed_layers, all_errors)
                logger.info(
                    f"Checkpoint saved: layer {layer_idx} complete ({len(completed_layers)}/{len(moe_layers)})"
                )

        # Save final weights (convert mlx arrays to numpy for safetensors)
        # Note: numpy doesn't support bfloat16, so convert to float16 for saving
        logger.info("Saving compressed weights...")

        def to_numpy(arr):
            """Convert mlx array to numpy, handling bfloat16."""
            if arr.dtype == mx.bfloat16:
                return np.array(arr.astype(mx.float16))
            return np.array(arr)

        base_np = {k: to_numpy(v) for k, v in base_weights.items()}
        delta_np = {k: to_numpy(v) for k, v in delta_weights.items()}
        save_safetensors(base_np, str(output_path / "base_weights.safetensors"))
        save_safetensors(delta_np, str(output_path / "deltas.safetensors"))

        # Extract and save biases (same for all layers, stored per-expert)
        # Get experts from first MoE layer
        first_experts = MoETypeService._get_experts(model, moe_layers[0])
        has_biases = False
        if first_experts is not None and architecture == MoEArchitecture.GPT_OSS:
            gate_bias, up_bias, down_bias = MoETypeService._extract_biases(first_experts)
            if gate_bias is not None or up_bias is not None or down_bias is not None:
                has_biases = True
                bias_weights = {}
                if gate_bias is not None:
                    bias_weights["gate_bias"] = to_numpy(gate_bias)
                if up_bias is not None:
                    bias_weights["up_bias"] = to_numpy(up_bias)
                if down_bias is not None:
                    bias_weights["down_bias"] = to_numpy(down_bias)
                save_safetensors(bias_weights, str(output_path / "biases.safetensors"))
                logger.info(
                    f"Saved biases: gate={gate_bias is not None}, up={up_bias is not None}, down={down_bias is not None}"
                )

        # Calculate storage
        _, gate_out, gate_in = gate_w.shape
        _, up_out, up_in = up_w.shape
        _, down_out, down_in = down_w.shape
        bytes_per_param = 2  # bfloat16

        original_bytes = (
            len(moe_layers)
            * num_experts
            * (gate_out * gate_in + up_out * up_in + down_out * down_in)
            * bytes_per_param
        )

        compressed_bytes = (
            len(moe_layers)
            * (
                # Base weights
                (gate_out * gate_in + up_out * up_in + down_out * down_in)
                # Delta factors
                + num_experts
                * (
                    gate_rank * (gate_out + gate_in)
                    + up_rank * (up_out + up_in)
                    + down_rank * (down_out + down_in)
                )
            )
            * bytes_per_param
        )

        # Save config
        config = CompressionConfig(
            model_id=model_id,
            num_layers=len(moe_layers),
            num_experts=num_experts,
            moe_layer_indices=moe_layers,
            gate_shape=(gate_out, gate_in),
            up_shape=(up_out, up_in),
            down_shape=(down_out, down_in),
            gate_rank=gate_rank,
            up_rank=up_rank,
            down_rank=down_rank,
            has_biases=has_biases,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )

        config_path = output_path / "config.json"
        config_path.write_text(json.dumps(config.model_dump(), indent=2))

        # Clean up checkpoint files on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if base_checkpoint_path.exists():
            base_checkpoint_path.unlink()
        if delta_checkpoint_path.exists():
            delta_checkpoint_path.unlink()

        logger.info(
            f"Compression complete: {original_bytes / 1e9:.2f}GB -> "
            f"{compressed_bytes / 1e9:.2f}GB ({config.compression_ratio:.1f}x)"
        )

        return CompressionResult(
            output_path=str(output_path),
            config=config,
            mean_reconstruction_error=float(np.mean(all_errors)),
            max_reconstruction_error=float(np.max(all_errors)),
        )

    # =========================================================================
    # Helper methods
    # =========================================================================

    @classmethod
    def _auto_select_ranks(
        cls,
        gate_w: mx.array,
        up_w: mx.array,
        down_w: mx.array,
    ) -> tuple[int, int, int]:
        """Auto-select ranks based on SVD analysis at 95% variance."""
        import mlx.core as mx

        def get_rank(weights: mx.array) -> int:
            base = mx.mean(weights, axis=0)
            mx.eval(base)
            delta = weights[0] - base
            mx.eval(delta)
            delta_np = np.array(delta.astype(mx.float32))

            try:
                _, S, _ = np.linalg.svd(delta_np, full_matrices=False)
                total = np.sum(S**2)
                if total == 0:
                    return 1
                cumsum = np.cumsum(S**2) / total
                return max(1, int(np.searchsorted(cumsum, cls.VARIANCE_THRESHOLD) + 1))
            except np.linalg.LinAlgError:
                return 1

        return get_rank(gate_w), get_rank(up_w), get_rank(down_w)

    @classmethod
    def _compute_projection_overlay(
        cls,
        weights: mx.array,
        name: str,
        rank: int,
    ) -> ProjectionOverlay:
        """Compute overlay for a single projection."""
        num_experts, out_dim, in_dim = weights.shape
        bytes_per_param = 2  # bfloat16

        # Original storage
        original_bytes = num_experts * out_dim * in_dim * bytes_per_param

        # Compressed: 1 base + num_experts * (U @ S, V) low-rank factors
        base_bytes = out_dim * in_dim * bytes_per_param
        delta_bytes = num_experts * rank * (out_dim + in_dim) * bytes_per_param
        compressed_bytes = base_bytes + delta_bytes

        return ProjectionOverlay(
            name=name,
            shape=(out_dim, in_dim),
            rank=rank,
            num_experts=num_experts,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )

    @classmethod
    def _verify_projection(
        cls,
        weights: mx.array,
        rank: int,
        name: str,
    ) -> ReconstructionError:
        """Verify reconstruction error for a projection."""
        import mlx.core as mx

        num_experts = weights.shape[0]
        base = mx.mean(weights, axis=0)
        mx.eval(base)

        errors = []
        mses = []

        for i in range(num_experts):
            original = weights[i]
            reconstructed = cls._reconstruct_expert(weights, i, rank)

            mx.eval(original, reconstructed)
            diff = original - reconstructed
            mse = float(mx.mean(diff * diff))
            orig_norm = float(mx.mean(original * original))
            rel_error = mse / (orig_norm + 1e-10)

            errors.append(rel_error)
            mses.append(mse)

        return ReconstructionError(
            name=name,
            mean_relative_error=float(np.mean(errors)),
            max_relative_error=float(np.max(errors)),
            mean_mse=float(np.mean(mses)),
        )

    @classmethod
    def _verify_output_reconstruction(
        cls,
        weights: mx.array,
        rank: int,
        num_experts: int,
    ) -> list[float]:
        """Verify output-level reconstruction using down projection."""
        import mlx.core as mx

        # Create test input
        test_input = mx.random.normal((1, 10, weights.shape[2]))
        mx.eval(test_input)

        errors = []
        for i in range(num_experts):
            # Original output
            orig_weight = weights[i]
            orig_out = test_input @ orig_weight.T

            # Reconstructed output
            recon_weight = cls._reconstruct_expert(weights, i, rank)
            recon_out = test_input @ recon_weight.T

            mx.eval(orig_out, recon_out)
            diff = orig_out - recon_out
            mse = float(mx.mean(diff * diff))
            orig_norm = float(mx.mean(orig_out * orig_out))
            rel_error = mse / (orig_norm + 1e-10)
            errors.append(rel_error)

        return errors

    @classmethod
    def _reconstruct_expert(
        cls,
        weights: mx.array,
        expert_idx: int,
        rank: int,
    ) -> mx.array:
        """Reconstruct expert weight using truncated SVD."""
        import mlx.core as mx

        base = mx.mean(weights, axis=0)
        delta = weights[expert_idx] - base

        # SVD truncation
        delta_np = np.array(delta.astype(mx.float32))
        U, S, Vh = np.linalg.svd(delta_np, full_matrices=False)

        # Truncate to rank
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]

        # Reconstruct
        delta_recon = U_trunc @ np.diag(S_trunc) @ Vh_trunc

        return mx.array(delta_recon) + base

    @classmethod
    def load_compressed(cls, compressed_path: str | Path) -> OverlayExperts:
        """Load compressed model for inference.

        Args:
            compressed_path: Path to compressed model directory

        Returns:
            OverlayExperts instance for efficient inference
        """
        return OverlayExperts.load(compressed_path)


# =============================================================================
# Overlay Inference
# =============================================================================


class OverlayExperts:
    """Efficient expert computation using overlay representation.

    Instead of storing full expert weights, stores:
    - base: Mean expert weight (shared)
    - U, V: Low-rank factors per expert
    - biases: Per-expert biases (if model has them)

    Reconstruction: expert_i = base + U_i @ V_i

    Usage:
        experts = OverlayExperts.load("gpt-oss-20b-overlay")
        weight = experts.get_expert_weight(layer=0, projection="gate", expert=5)
        # or for efficient inference:
        output = experts.apply_expert(layer=0, projection="gate", expert=5, x=hidden)
    """

    def __init__(
        self,
        config: CompressionConfig,
        base_weights: dict,
        delta_weights: dict,
        biases: dict | None = None,
    ) -> None:
        """Initialize from loaded weights."""
        self.config = config
        self._base = base_weights
        self._deltas = delta_weights
        self._biases = biases or {}

    @classmethod
    def load(cls, path: str | Path) -> OverlayExperts:
        """Load compressed model from disk."""
        import mlx.core as mx

        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = CompressionConfig.model_validate_json(config_path.read_text())

        # Load weights
        base_path = path / "base_weights.safetensors"
        deltas_path = path / "deltas.safetensors"

        if not base_path.exists():
            raise FileNotFoundError(f"Base weights not found: {base_path}")
        if not deltas_path.exists():
            raise FileNotFoundError(f"Deltas not found: {deltas_path}")

        base_weights = mx.load(str(base_path))
        delta_weights = mx.load(str(deltas_path))

        # Load biases if available
        biases = None
        biases_path = path / "biases.safetensors"
        if biases_path.exists():
            biases = mx.load(str(biases_path))
            logger.info(f"Loaded biases: {list(biases.keys())}")

        logger.info(
            f"Loaded compressed model: {config.num_layers} layers, "
            f"{config.num_experts} experts, {config.compression_ratio:.1f}x compression"
        )

        return cls(config, base_weights, delta_weights, biases)

    def get_expert_weight(
        self,
        layer: int,
        projection: str,
        expert: int,
    ):
        """Reconstruct full expert weight matrix.

        Args:
            layer: Layer index (from moe_layer_indices)
            projection: "gate", "up", or "down"
            expert: Expert index

        Returns:
            Reconstructed weight matrix: base + U @ V
        """
        import mlx.core as mx

        base_key = f"layer_{layer}_{projection}_base"
        u_key = f"layer_{layer}_{projection}_expert_{expert}_U"
        v_key = f"layer_{layer}_{projection}_expert_{expert}_V"

        if base_key not in self._base:
            raise KeyError(f"Base weight not found: {base_key}")
        if u_key not in self._deltas:
            raise KeyError(f"Delta U not found: {u_key}")

        base = self._base[base_key]
        U = self._deltas[u_key]
        V = self._deltas[v_key]

        # Reconstruct: base + U @ V
        weight = base + U @ V
        mx.eval(weight)

        return weight

    def apply_expert(
        self,
        layer: int,
        projection: str,
        expert: int,
        x,
    ):
        """Apply expert to input efficiently using low-rank factorization.

        Instead of: y = x @ (base + U @ V).T + bias
        Computes:   y = x @ base.T + (x @ V.T) @ U.T + bias

        This is more efficient when rank << min(in_dim, out_dim).

        Args:
            layer: Layer index
            projection: "gate", "up", or "down"
            expert: Expert index
            x: Input tensor of shape (..., in_dim)

        Returns:
            Output tensor of shape (..., out_dim)
        """

        base_key = f"layer_{layer}_{projection}_base"
        u_key = f"layer_{layer}_{projection}_expert_{expert}_U"
        v_key = f"layer_{layer}_{projection}_expert_{expert}_V"

        base = self._base[base_key]
        U = self._deltas[u_key]  # (out_dim, rank)
        V = self._deltas[v_key]  # (rank, in_dim)

        # Efficient low-rank application
        # y = x @ base.T + x @ V.T @ U.T
        base_out = x @ base.T
        delta_out = (x @ V.T) @ U.T
        out = base_out + delta_out

        # Apply bias if available
        # Biases are stored per-expert: (num_experts, out_dim)
        bias_key = f"{projection}_bias"
        if bias_key in self._biases:
            bias = self._biases[bias_key][expert]  # (out_dim,)
            out = out + bias

        return out

    @property
    def num_layers(self) -> int:
        """Number of MoE layers."""
        return self.config.num_layers

    @property
    def num_experts(self) -> int:
        """Number of experts per layer."""
        return self.config.num_experts

    @property
    def moe_layer_indices(self) -> list[int]:
        """Original layer indices that are MoE layers."""
        return self.config.moe_layer_indices

    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        total = 0
        for w in self._base.values():
            total += w.nbytes
        for w in self._deltas.values():
            total += w.nbytes
        return total / (1024 * 1024)
