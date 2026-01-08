"""Service layer for analyzer CLI commands.

This module provides the AnalyzerService class that wraps ModelAnalyzer
to provide a simple interface for CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import AnalysisConfig, LayerStrategy
from .core import ModelAnalyzer
from .models import AnalysisResult


class AnalyzerServiceConfig(BaseModel):
    """Configuration for AnalyzerService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    adapter_path: str | None = Field(default=None, description="Path to LoRA adapter")
    embedding_scale: float | None = Field(default=None, description="Embedding scale factor")
    use_raw: bool = Field(default=False, description="Use raw mode (no chat template)")
    use_prefix_mode: bool = Field(default=False, description="Use prefix mode")

    # Steering config
    steer_file: str | None = Field(default=None, description="Steering vector file")
    steer_neuron: int | None = Field(default=None, description="Neuron to steer")
    steer_layer: int | None = Field(default=None, description="Layer to steer at")
    steer_strength: float | None = Field(default=None, description="Steering strength")

    # Injection config
    inject_layer: int | None = Field(default=None, description="Layer for injection")
    inject_token: str | None = Field(default=None, description="Token to inject")
    inject_blend: float = Field(default=1.0, description="Injection blend factor")

    # Compute override
    compute_override: str = Field(default="none", description="Compute override mode")
    compute_layer: int | None = Field(default=None, description="Compute override layer")

    # Answer finding
    find_answer: str | None = Field(default=None, description="Pattern to find in answer")
    no_find_answer: bool = Field(default=False, description="Disable answer finding")
    gen_tokens: int = Field(default=30, description="Tokens to generate for answer finding")
    expected: str | None = Field(default=None, description="Expected answer")


class AnalyzerService:
    """Service class for analyzer operations.

    Provides a high-level interface for CLI commands to run analysis
    without needing to understand the internal architecture.
    """

    # Alias for CLI access
    Config = AnalyzerServiceConfig

    @classmethod
    async def analyze(
        cls,
        prompt: str,
        analysis_config: AnalysisConfig,
        service_config: AnalyzerServiceConfig,
    ) -> AnalysisResult:
        """Run analysis on a prompt.

        Args:
            prompt: The prompt to analyze.
            analysis_config: Analysis configuration (layers, top_k, etc.).
            service_config: Service configuration (model, steering, etc.).

        Returns:
            AnalysisResult with layer predictions and token evolutions.
        """
        async with ModelAnalyzer.from_pretrained(
            service_config.model,
            embedding_scale=service_config.embedding_scale,
            adapter_path=service_config.adapter_path,
        ) as analyzer:
            # TODO: Add steering/injection support when needed
            result = await analyzer.analyze(prompt, analysis_config)
            return result

    @classmethod
    async def compare_models(
        cls,
        model1: str,
        model2: str,
        prompt: str,
        top_k: int = 10,
        track_tokens: list[str] | None = None,
    ) -> Any:
        """Compare two models' predictions.

        Args:
            model1: First model path/name.
            model2: Second model path/name.
            prompt: Prompt to analyze.
            top_k: Number of top predictions.
            track_tokens: Tokens to track.

        Returns:
            Comparison result.
        """
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.EVENLY_SPACED,
            top_k=top_k,
            track_tokens=track_tokens or [],
        )

        async with ModelAnalyzer.from_pretrained(model1) as analyzer1:
            result1 = await analyzer1.analyze(prompt, config)

        async with ModelAnalyzer.from_pretrained(model2) as analyzer2:
            result2 = await analyzer2.analyze(prompt, config)

        # Return simple comparison (could be enhanced)
        return ComparisonResult(
            model1=model1,
            model2=model2,
            prompt=prompt,
            result1=result1,
            result2=result2,
        )

    @classmethod
    async def demonstrate_hooks(
        cls,
        model: str,
        prompt: str,
        layers: list[int],
        capture_attention: bool = False,
        last_only: bool = False,
        no_logit_lens: bool = False,
    ) -> Any:
        """Demonstrate low-level hooks.

        Args:
            model: Model path/name.
            prompt: Prompt to analyze.
            layers: Layers to capture.
            capture_attention: Whether to capture attention.
            last_only: Capture last position only.
            no_logit_lens: Skip logit lens.

        Returns:
            Hook demonstration result.
        """
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.CUSTOM,
            custom_layers=layers,
        )

        async with ModelAnalyzer.from_pretrained(model) as analyzer:
            result = await analyzer.analyze(prompt, config)
            return result


class ComparisonResult(BaseModel):
    """Result of comparing two models."""

    model_config = ConfigDict(frozen=True)

    model1: str
    model2: str
    prompt: str
    result1: AnalysisResult
    result2: AnalysisResult

    def to_display(self) -> str:
        """Format comparison for display."""
        lines = [
            f"\n{'=' * 60}",
            f"Model Comparison: {self.prompt[:50]}...",
            f"{'=' * 60}",
            f"\nModel 1: {self.model1}",
            f"  Predicted: {self.result1.predicted_token}",
            f"  Confidence: {self.result1.confidence:.4f}",
            f"\nModel 2: {self.model2}",
            f"  Predicted: {self.result2.predicted_token}",
            f"  Confidence: {self.result2.confidence:.4f}",
        ]
        return "\n".join(lines)


__all__ = [
    "AnalyzerService",
    "AnalyzerServiceConfig",
    "ComparisonResult",
]
