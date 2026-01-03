"""MoE-aware hooks that compose ModelHooks."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..hooks import CaptureConfig, CapturedState, ModelHooks
from .config import MoECaptureConfig
from .detector import detect_moe_architecture, get_moe_layer_info, get_moe_layers
from .enums import MoEArchitecture
from .models import ExpertUtilization, MoELayerInfo, RouterEntropy


class MoECapturedState:
    """State captured from MoE forward pass."""

    def __init__(self) -> None:
        self.router_logits: dict[int, mx.array] = {}
        self.router_weights: dict[int, mx.array] = {}
        self.selected_experts: dict[int, mx.array] = {}
        self.expert_outputs: dict[int, dict[int, mx.array]] = {}

    def clear(self) -> None:
        """Clear all captured state."""
        self.router_logits.clear()
        self.router_weights.clear()
        self.selected_experts.clear()
        self.expert_outputs.clear()


class MoEHooks:
    """MoE-aware hooks that compose ModelHooks.

    Example:
        >>> hooks = MoEHooks(model)
        >>> hooks.configure(MoECaptureConfig(capture_router_logits=True))
        >>> output = hooks.forward(input_ids)
        >>> routing = hooks.moe_state.router_logits[4]
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.architecture = detect_moe_architecture(model)
        self.moe_layers = get_moe_layers(model)

        # Compose ModelHooks for standard captures
        self._hooks = ModelHooks(model)

        # MoE-specific state
        self.moe_state = MoECapturedState()
        self.config: MoECaptureConfig | None = None

        # Cache layer info
        self._layer_info: dict[int, MoELayerInfo] = {}

    def configure(self, config: MoECaptureConfig) -> "MoEHooks":
        """Configure what to capture."""
        self.config = config

        # Determine which layers to capture
        layers = config.layers if config.layers else self.moe_layers

        # Configure underlying hooks for hidden states
        self._hooks.configure(CaptureConfig(
            layers=layers,
            capture_hidden_states=True,
        ))

        return self

    def forward(self, input_ids: mx.array) -> mx.array:
        """Forward pass with MoE capture.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Model output logits
        """
        self.moe_state.clear()

        if self.config is None:
            self.configure(MoECaptureConfig())

        # Get layers to capture
        layers = self.config.layers if self.config.layers else self.moe_layers

        # Install capture hooks on MoE layers
        original_forwards = {}
        model_layers = self._get_model_layers()

        for layer_idx in layers:
            if layer_idx >= len(model_layers):
                continue

            layer = model_layers[layer_idx]
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "router"):
                continue

            # Store original and wrap
            original_forwards[layer_idx] = mlp.__call__

            def make_capture_fn(idx: int, orig_fn):
                def capture_fn(x):
                    self._capture_moe_routing(idx, x, mlp)
                    return orig_fn(x)
                return capture_fn

            mlp.__call__ = make_capture_fn(layer_idx, mlp.__call__)

        try:
            # Run forward pass
            output = self.model(input_ids)
            if hasattr(output, "logits"):
                output = output.logits
            return output
        finally:
            # Restore original forwards
            for layer_idx, orig_fn in original_forwards.items():
                layer = model_layers[layer_idx]
                layer.mlp.__call__ = orig_fn

    def _capture_moe_routing(
        self,
        layer_idx: int,
        x: mx.array,
        moe: nn.Module,
    ) -> None:
        """Capture routing decisions for an MoE layer."""
        if self.config is None:
            return

        router = moe.router
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)

        # Compute router logits
        router_logits = x_flat @ router.weight.T
        if hasattr(router, "bias") and router.bias is not None:
            router_logits = router_logits + router.bias

        if self.config.capture_router_logits:
            self.moe_state.router_logits[layer_idx] = router_logits

        # Get routing weights and selections
        k = router.num_experts_per_tok
        topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
        weights = mx.softmax(topk_logits, axis=-1)

        if self.config.capture_router_weights:
            self.moe_state.router_weights[layer_idx] = weights

        if self.config.capture_selected_experts:
            self.moe_state.selected_experts[layer_idx] = topk_indices.reshape(
                batch_size, seq_len, k
            )

    def _get_model_layers(self) -> list[nn.Module]:
        """Get model layers."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(self.model, attr, None)
            if submodel is not None:
                layers = getattr(submodel, "layers", None)
                if layers is not None:
                    return list(layers)
        return list(getattr(self.model, "layers", []))

    def get_layer_info(self, layer_idx: int) -> MoELayerInfo | None:
        """Get cached layer info."""
        if layer_idx not in self._layer_info:
            info = get_moe_layer_info(self.model, layer_idx)
            if info:
                self._layer_info[layer_idx] = info
        return self._layer_info.get(layer_idx)

    def get_expert_utilization(self, layer_idx: int) -> ExpertUtilization | None:
        """Compute expert utilization for a layer."""
        if layer_idx not in self.moe_state.selected_experts:
            return None

        info = self.get_layer_info(layer_idx)
        if info is None:
            return None

        selected = self.moe_state.selected_experts[layer_idx]
        flat_selected = selected.reshape(-1).tolist()

        # Count activations per expert
        counts = [0] * info.num_experts
        for exp_idx in flat_selected:
            counts[exp_idx] += 1

        total = len(flat_selected)
        frequencies = [c / total if total > 0 else 0 for c in counts]

        # Compute load balance (1.0 = perfectly balanced)
        expected = total / info.num_experts if info.num_experts > 0 else 0
        if expected > 0:
            balance = 1.0 - sum(abs(c - expected) for c in counts) / (2 * total)
        else:
            balance = 1.0

        return ExpertUtilization(
            layer_idx=layer_idx,
            num_experts=info.num_experts,
            total_activations=total,
            expert_counts=tuple(counts),
            expert_frequencies=tuple(frequencies),
            load_balance_score=max(0, min(1, balance)),
            most_used_expert=counts.index(max(counts)),
            least_used_expert=counts.index(min(counts)),
        )

    def get_router_entropy(self, layer_idx: int) -> RouterEntropy | None:
        """Compute router entropy for a layer."""
        if layer_idx not in self.moe_state.router_logits:
            return None

        info = self.get_layer_info(layer_idx)
        if info is None:
            return None

        logits = self.moe_state.router_logits[layer_idx]
        probs = mx.softmax(logits, axis=-1)

        # Entropy per position: -sum(p * log(p))
        log_probs = mx.log(probs + 1e-10)
        entropy = -mx.sum(probs * log_probs, axis=-1)

        mean_entropy = float(mx.mean(entropy))
        max_entropy = float(mx.log(mx.array(info.num_experts)))
        normalized = mean_entropy / max_entropy if max_entropy > 0 else 0

        return RouterEntropy(
            layer_idx=layer_idx,
            mean_entropy=mean_entropy,
            max_entropy=max_entropy,
            normalized_entropy=normalized,
            per_position_entropy=tuple(entropy.tolist()),
        )

    @property
    def state(self) -> CapturedState:
        """Access underlying ModelHooks state."""
        return self._hooks.state
