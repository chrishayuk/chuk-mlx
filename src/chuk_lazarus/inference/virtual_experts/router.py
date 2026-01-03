"""
Virtual Router for MoE models.

The VirtualRouter wraps an existing MoE router and adds virtual expert slots
that can route to external plugins.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class VirtualRouter(nn.Module):
    """
    Router wrapper that adds virtual expert slots.

    This wraps an existing MoE router and:
    1. Computes the original router logits for real experts
    2. Adds virtual expert logits based on learned directions
    3. Includes virtual experts in top-k selection
    4. Tracks when virtual experts are selected

    Each registered plugin gets its own virtual expert slot with a learned
    direction in activation space that separates positive from negative examples.

    Architecture:
        Original router: input → logits for experts 0..N-1
        Virtual router:  input → logits for experts 0..N-1, V0, V1, ...

        Where V0, V1, ... are virtual expert logits computed as:
            logit_i = (input · direction_i) * scale_i + bias_i
    """

    def __init__(
        self,
        original_router: nn.Module,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_virtual_experts: int = 1,
    ):
        super().__init__()

        self._original_router = original_router
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.num_virtual_experts = num_virtual_experts

        # Virtual experts get indices starting at num_experts
        self.virtual_expert_start_idx = num_experts

        # Learned parameters for each virtual expert
        self.directions: list[mx.array] = [
            mx.zeros((hidden_size,)) for _ in range(num_virtual_experts)
        ]
        self.scales: list[float] = [1.0] * num_virtual_experts
        self.biases: list[float] = [0.0] * num_virtual_experts
        self.thresholds: list[float] = [0.0] * num_virtual_experts

        # Calibration state
        self._calibrated: list[bool] = [False] * num_virtual_experts

        # Tracking for analysis
        self.virtual_selected_positions: dict[int, list[int]] = {}
        self._last_virtual_logits: mx.array | None = None

    def calibrate_expert(
        self,
        expert_idx: int,
        positive_activations: list[mx.array],
        negative_activations: list[mx.array],
    ) -> None:
        """
        Calibrate a specific virtual expert's routing.

        Learns a direction in activation space that separates positive
        (should route to this expert) from negative (should not route)
        examples.

        Args:
            expert_idx: Index of the virtual expert (0-based)
            positive_activations: Hidden states that SHOULD route here
            negative_activations: Hidden states that should NOT route here
        """
        if expert_idx >= self.num_virtual_experts:
            raise ValueError(f"Expert index {expert_idx} >= {self.num_virtual_experts}")

        # Stack activations
        pos_stack = mx.stack(positive_activations)
        neg_stack = mx.stack(negative_activations)

        # Compute means
        pos_mean = mx.mean(pos_stack, axis=0)
        neg_mean = mx.mean(neg_stack, axis=0)

        # Direction from negative to positive
        direction = pos_mean - neg_mean
        norm = mx.linalg.norm(direction)
        direction = direction / (norm + 1e-10)

        mx.eval(direction)
        self.directions[expert_idx] = direction

        # Compute projections for threshold
        pos_projs = [float(mx.sum(h * direction)) for h in positive_activations]
        neg_projs = [float(mx.sum(h * direction)) for h in negative_activations]

        # Threshold at midpoint between means
        self.thresholds[expert_idx] = (np.mean(pos_projs) + np.mean(neg_projs)) / 2

        # Scale so positive inputs get high logits (~5)
        avg_pos_proj = np.mean(pos_projs)
        threshold = self.thresholds[expert_idx]
        if abs(avg_pos_proj - threshold) > 0.01:
            self.scales[expert_idx] = 5.0 / (avg_pos_proj - threshold)
        else:
            self.scales[expert_idx] = 1.0

        self.biases[expert_idx] = -threshold * self.scales[expert_idx]
        self._calibrated[expert_idx] = True

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, dict[int, mx.array]]:
        """
        Compute routing with virtual experts included.

        Args:
            x: Input tensor, shape (batch*seq, hidden) or (batch, seq, hidden)

        Returns:
            weights: Routing weights for selected experts
            indices: Expert indices (may include virtual expert indices)
            virtual_masks: Dict mapping virtual expert idx to boolean mask
                           indicating which positions selected that expert
        """
        # Handle 3D input
        if x.ndim == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.reshape(-1, hidden_size)

        num_tokens = x.shape[0]

        # Get original router logits
        original_logits = x @ self._original_router.weight.T + self._original_router.bias

        # Compute virtual expert logits
        virtual_logits_list = []
        for i in range(self.num_virtual_experts):
            if self._calibrated[i]:
                v_logits = mx.sum(x * self.directions[i], axis=-1)
                v_logits = v_logits * self.scales[i] + self.biases[i]
            else:
                # Uncalibrated experts get very negative logits (never selected)
                v_logits = mx.full((num_tokens,), -100.0)
            virtual_logits_list.append(v_logits[:, None])

        # Concatenate all logits
        if virtual_logits_list:
            virtual_logits = mx.concatenate(virtual_logits_list, axis=-1)
            all_logits = mx.concatenate([original_logits, virtual_logits], axis=-1)
        else:
            all_logits = original_logits

        self._last_virtual_logits = virtual_logits if virtual_logits_list else None

        # Top-k selection
        k = self.num_experts_per_tok
        partitioned_indices = mx.argpartition(all_logits, kth=-k, axis=-1)
        top_k_indices = partitioned_indices[..., -k:]
        top_k_logits = mx.take_along_axis(all_logits, top_k_indices, axis=-1)

        # Softmax weights
        expert_weights = mx.softmax(top_k_logits, axis=-1)

        # Create masks for each virtual expert
        virtual_masks = {}
        for i in range(self.num_virtual_experts):
            virtual_idx = self.virtual_expert_start_idx + i
            mask = mx.any(top_k_indices == virtual_idx, axis=-1)
            virtual_masks[i] = mask

            # Track positions for analysis
            mx.eval(mask)
            self.virtual_selected_positions[i] = [
                j for j, selected in enumerate(mask.tolist()) if selected
            ]

        mx.eval(expert_weights, top_k_indices)

        return expert_weights, top_k_indices, virtual_masks

    def get_routing_score(self, x: mx.array, expert_idx: int = 0) -> float:
        """
        Get a virtual expert's routing score for input.

        The score is normalized to 0-1 range where:
        - 0.0 = definitely not this expert
        - 0.5 = at threshold
        - 1.0 = definitely this expert

        Args:
            x: Input tensor
            expert_idx: Which virtual expert to score

        Returns:
            Normalized routing score
        """
        if not self._calibrated[expert_idx]:
            return 0.0

        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        x_last = x[-1]
        proj = float(mx.sum(x_last * self.directions[expert_idx]))

        # Normalize to 0-1
        threshold = self.thresholds[expert_idx]
        score = (proj - threshold) / (abs(threshold) + 1.0)
        score = max(0.0, min(1.0, (score + 1) / 2))

        return score

    def is_calibrated(self, expert_idx: int = 0) -> bool:
        """Check if a virtual expert has been calibrated."""
        return self._calibrated[expert_idx] if expert_idx < len(self._calibrated) else False
