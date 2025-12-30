"""
Steering hook for modifying activations during forward pass.

This module contains the SteeringHook class that applies direction
vectors to model activations during inference.
"""

from __future__ import annotations

import mlx.core as mx


class SteeringHook:
    """
    Hook that modifies activations during forward pass.

    Generic - applies any direction vector to steer activations.
    """

    def __init__(
        self,
        direction: mx.array,
        coefficient: float = 1.0,
        position: int | None = None,
        normalize: bool = True,
        scale_by_norm: bool = False,
    ):
        """
        Initialize steering hook.

        Args:
            direction: Direction vector to add [hidden_size]
            coefficient: Scaling factor (positive = toward direction)
            position: Position to steer (None = all)
            normalize: Normalize direction before applying
            scale_by_norm: Scale by activation norm
        """
        self.direction = direction
        self.coefficient = coefficient
        self.position = position
        self.normalize = normalize
        self.scale_by_norm = scale_by_norm

        if self.normalize:
            norm = mx.sqrt(mx.sum(direction * direction))
            self.direction = direction / (norm + 1e-8)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Apply steering to hidden states.

        Args:
            h: Hidden states [batch, seq, hidden]

        Returns:
            Steered hidden states
        """
        steering = self.direction * self.coefficient

        if self.scale_by_norm:
            # Scale by mean activation norm
            h_norm = mx.sqrt(mx.mean(h * h))
            steering = steering * h_norm

        if self.position is not None:
            # Only steer at specific position
            steered = h.at[:, self.position, :].add(steering)
            return steered
        else:
            # Steer all positions
            return h + steering
