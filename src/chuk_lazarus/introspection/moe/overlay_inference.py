"""Inference using compressed MoE overlay model.

Patches an MoE model to use compressed expert weights instead of full weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from .moe_compression import OverlayExperts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OverlayMoEModel:
    """Wrapper that patches MoE model to use compressed experts.

    Usage:
        # Load compressed model
        overlay_model = OverlayMoEModel.load(
            compressed_path="gpt-oss-20b-overlay",
            original_model_id="openai/gpt-oss-20b",
        )

        # Generate text
        result = overlay_model.generate("The capital of France is", max_tokens=50)
    """

    def __init__(
        self,
        model,
        overlay: OverlayExperts,
        tokenizer,
    ):
        self._model = model
        self.overlay = overlay
        self.tokenizer = tokenizer
        self._patched = False

    @property
    def model(self):
        """Return the underlying model for generation."""
        return self._model

    @classmethod
    def load(
        cls,
        compressed_path: str | Path,
        original_model_id: str,
    ) -> OverlayMoEModel:
        """Load compressed overlay model for inference.

        Args:
            compressed_path: Path to compressed overlay directory
            original_model_id: Original model ID for non-expert weights

        Returns:
            OverlayMoEModel ready for inference
        """
        from .moe_type import MoETypeService

        logger.info(f"Loading compressed model from: {compressed_path}")
        overlay = OverlayExperts.load(compressed_path)

        logger.info(f"Loading base model: {original_model_id}")
        model = MoETypeService._load_model(original_model_id)

        # Load tokenizer
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(original_model_id)

        wrapper = cls(model, overlay, tokenizer)
        wrapper._patch_experts()

        logger.info(
            f"Loaded overlay model: {overlay.num_layers} layers, "
            f"{overlay.num_experts} experts, {overlay.config.compression_ratio:.1f}x compression"
        )

        return wrapper

    def _patch_experts(self) -> None:
        """Patch model to use compressed expert weights."""
        if self._patched:
            return

        # Find MoE layers and patch them
        for layer_idx in self.overlay.moe_layer_indices:
            layer = self._get_layer(layer_idx)
            if layer is None:
                continue

            # Get the MoE block
            moe_block = self._get_moe_block(layer)
            if moe_block is None:
                continue

            # Patch expert forward
            self._patch_moe_block(moe_block, layer_idx)

        self._patched = True
        logger.info(f"Patched {len(self.overlay.moe_layer_indices)} MoE layers")

    def _get_layer(self, layer_idx: int):
        """Get transformer layer by index."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        return None

    def _get_moe_block(self, layer):
        """Get MoE block from layer."""
        # Try common attribute names
        for attr in ["block_sparse_moe", "moe", "mlp", "feed_forward"]:
            if hasattr(layer, attr):
                block = getattr(layer, attr)
                if hasattr(block, "experts"):
                    return block
        return None

    def _patch_moe_block(self, moe_block, layer_idx: int) -> None:
        """Patch a single MoE block to use overlay experts."""
        # Store original expert weights reference for comparison
        original_experts = moe_block.experts

        # Create patched expert that uses overlay
        overlay = self.overlay

        class OverlayExpert(nn.Module):
            """Expert that reconstructs weights from overlay."""

            def __init__(self, expert_idx: int):
                super().__init__()
                self.expert_idx = expert_idx
                self.layer_idx = layer_idx

            def __call__(self, x: mx.array) -> mx.array:
                # Use efficient low-rank application
                # gate
                gate_out = overlay.apply_expert(self.layer_idx, "gate", self.expert_idx, x)
                # up
                up_out = overlay.apply_expert(self.layer_idx, "up", self.expert_idx, x)
                # activation (assume SiLU/swish)
                hidden = nn.silu(gate_out) * up_out
                # down
                out = overlay.apply_expert(self.layer_idx, "down", self.expert_idx, hidden)
                return out

        # Replace experts list with overlay versions
        num_experts = len(original_experts)
        moe_block.experts = [OverlayExpert(i) for i in range(num_experts)]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using compressed model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated text
        """
        from mlx_lm import generate

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def memory_usage(self) -> dict:
        """Get memory usage statistics."""
        overlay_mb = self.overlay.memory_usage_mb()
        original_mb = self.overlay.config.original_bytes / (1024 * 1024)

        return {
            "overlay_mb": overlay_mb,
            "original_expert_mb": original_mb,
            "savings_mb": original_mb - overlay_mb,
            "compression_ratio": self.overlay.config.compression_ratio,
        }
