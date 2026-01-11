"""Service layer for steering CLI commands.

This module provides the SteeringService class that wraps ActivationSteering
to provide a simple interface for CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..hooks import CaptureConfig, ModelHooks, PositionSelection
from .config import SteeringConfig
from .core import ActivationSteering


class SteeringServiceConfig(BaseModel):
    """Configuration for SteeringService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    layer: int | None = Field(default=None, description="Layer for steering")
    coefficient: float = Field(default=1.0, description="Steering coefficient")
    max_tokens: int = Field(default=100, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Generation temperature")


class DirectionExtractionResult(BaseModel):
    """Result of direction extraction."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    direction: Any = Field(..., description="Direction vector")
    layer: int = Field(..., description="Layer index")
    norm: float = Field(..., description="Direction norm")
    cosine_similarity: float = Field(
        ..., description="Cosine similarity between positive and negative"
    )
    separation: float = Field(..., description="1 - cosine similarity")
    positive_prompt: str = Field(..., description="Positive prompt")
    negative_prompt: str = Field(..., description="Negative prompt")


class SteeringGenerationResult(BaseModel):
    """Result of steering generation."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    output: str = Field(..., description="Generated output")
    layer: int = Field(..., description="Steering layer")
    coefficient: float = Field(..., description="Steering coefficient")


class SteeringComparisonResult(BaseModel):
    """Result of steering comparison."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    results: dict[float, str] = Field(..., description="Coefficient -> output mapping")


class SteeringService:
    """Service class for steering operations.

    Provides a high-level interface for CLI commands to run steering
    without needing to understand the internal architecture.
    """

    Config = SteeringServiceConfig

    @classmethod
    async def extract_direction(
        cls,
        model: str,
        positive_prompt: str,
        negative_prompt: str,
        layer: int | None = None,
    ) -> DirectionExtractionResult:
        """Extract steering direction from contrastive prompts.

        Args:
            model: Model path or name.
            positive_prompt: Prompt for positive direction.
            negative_prompt: Prompt for negative direction.
            layer: Layer to extract from (default: middle layer).

        Returns:
            DirectionExtractionResult with direction and metadata.
        """
        steerer = ActivationSteering.from_pretrained(model)

        # Determine layer
        target_layer = layer if layer is not None else steerer.num_layers // 2

        # Get positive activation
        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[target_layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(positive_prompt))[None, :]
        hooks.forward(input_ids)
        h_positive = hooks.state.hidden_states[target_layer][0, -1, :]

        # Get negative activation
        hooks = ModelHooks(steerer.model)
        hooks.configure(
            CaptureConfig(
                layers=[target_layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = mx.array(steerer.tokenizer.encode(negative_prompt))[None, :]
        hooks.forward(input_ids)
        h_negative = hooks.state.hidden_states[target_layer][0, -1, :]

        # Compute direction: positive - negative
        direction = h_positive - h_negative
        direction_np = np.array(direction.tolist(), dtype=np.float32)

        # Compute statistics
        norm = float(mx.sqrt(mx.sum(direction * direction)))
        cos_sim = float(
            mx.sum(h_positive * h_negative)
            / (
                mx.sqrt(mx.sum(h_positive * h_positive)) * mx.sqrt(mx.sum(h_negative * h_negative))
                + 1e-8
            )
        )

        return DirectionExtractionResult(
            direction=direction_np,
            layer=target_layer,
            norm=norm,
            cosine_similarity=cos_sim,
            separation=1 - cos_sim,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )

    @classmethod
    def save_direction(
        cls,
        result: DirectionExtractionResult,
        output_path: str | Path,
        model_id: str,
    ) -> None:
        """Save extracted direction to file.

        Args:
            result: Direction extraction result.
            output_path: Path to save to.
            model_id: Model identifier.
        """
        np.savez(
            output_path,
            direction=result.direction,
            layer=result.layer,
            positive_prompt=result.positive_prompt,
            negative_prompt=result.negative_prompt,
            model_id=model_id,
            norm=result.norm,
            cosine_similarity=result.cosine_similarity,
        )

    @classmethod
    def load_direction(cls, path: str | Path) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Load direction from file.

        Args:
            path: Path to direction file.

        Returns:
            Tuple of (direction, layer, metadata).
        """
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            direction = data["direction"]
            layer = int(data["layer"]) if "layer" in data else None

            metadata = {}
            if "positive_prompt" in data:
                metadata["positive_prompt"] = str(data["positive_prompt"])
            if "negative_prompt" in data:
                metadata["negative_prompt"] = str(data["negative_prompt"])
            if "norm" in data:
                metadata["norm"] = float(data["norm"])
            if "cosine_similarity" in data:
                metadata["cosine_similarity"] = float(data["cosine_similarity"])

            return direction, layer, metadata

        elif path.suffix == ".json":
            import json

            with open(path) as f:
                data = json.load(f)
            direction = np.array(data["direction"], dtype=np.float32)
            layer = data.get("layer")
            return direction, layer, data

        else:
            raise ValueError(f"Unsupported direction format: {path.suffix}")

    @classmethod
    async def generate_with_steering(
        cls,
        model: str,
        prompts: list[str],
        direction: np.ndarray,
        layer: int,
        coefficient: float = 1.0,
        max_tokens: int = 100,
        temperature: float = 0.0,
        name: str = "custom",
        positive_label: str = "positive",
        negative_label: str = "negative",
    ) -> list[SteeringGenerationResult]:
        """Generate text with steering applied.

        Args:
            model: Model path or name.
            prompts: Prompts to generate from.
            direction: Steering direction vector.
            layer: Layer to apply steering.
            coefficient: Steering coefficient.
            max_tokens: Max tokens to generate.
            temperature: Generation temperature.
            name: Direction name.
            positive_label: Positive direction label.
            negative_label: Negative direction label.

        Returns:
            List of generation results.
        """
        steerer = ActivationSteering.from_pretrained(model)

        # Add direction
        steerer.add_direction(
            layer=layer,
            direction=direction,
            name=name,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        config = SteeringConfig(
            layers=[layer],
            coefficient=coefficient,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        results = []
        for prompt in prompts:
            output = steerer.generate(prompt, config)
            results.append(
                SteeringGenerationResult(
                    prompt=prompt,
                    output=output,
                    layer=layer,
                    coefficient=coefficient,
                )
            )

        return results

    @classmethod
    async def compare_coefficients(
        cls,
        model: str,
        prompt: str,
        direction: np.ndarray,
        layer: int,
        coefficients: list[float],
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> SteeringComparisonResult:
        """Compare steering at different coefficients.

        Args:
            model: Model path or name.
            prompt: Prompt to generate from.
            direction: Steering direction vector.
            layer: Layer to apply steering.
            coefficients: Coefficients to compare.
            max_tokens: Max tokens to generate.
            temperature: Generation temperature.

        Returns:
            Comparison result with outputs for each coefficient.
        """
        steerer = ActivationSteering.from_pretrained(model)

        # Add direction
        steerer.add_direction(layer=layer, direction=direction)

        results = {}
        for coef in coefficients:
            config = SteeringConfig(
                layers=[layer],
                coefficient=coef,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            results[coef] = steerer.generate(prompt, config, coefficient=coef)

        return SteeringComparisonResult(prompt=prompt, results=results)

    @classmethod
    def create_neuron_direction(
        cls,
        hidden_size: int,
        neuron_idx: int,
    ) -> np.ndarray:
        """Create a one-hot direction for single neuron steering.

        Args:
            hidden_size: Model hidden size.
            neuron_idx: Neuron index to steer.

        Returns:
            One-hot direction vector.
        """
        direction = np.zeros(hidden_size, dtype=np.float32)
        direction[neuron_idx] = 1.0
        return direction


__all__ = [
    "SteeringService",
    "SteeringServiceConfig",
    "DirectionExtractionResult",
    "SteeringGenerationResult",
    "SteeringComparisonResult",
]
