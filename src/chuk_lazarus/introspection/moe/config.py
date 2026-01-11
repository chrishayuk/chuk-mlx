"""MoE capture configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MoECaptureConfig(BaseModel):
    """Configuration for MoE hook capture."""

    model_config = ConfigDict(frozen=True)

    # Layer selection
    layers: list[int] | None = Field(
        default=None,
        description="Specific layers to capture. None = all MoE layers.",
    )

    # What to capture
    capture_router_logits: bool = Field(
        default=True,
        description="Capture raw router logits before softmax.",
    )
    capture_router_weights: bool = Field(
        default=True,
        description="Capture router weights after softmax.",
    )
    capture_selected_experts: bool = Field(
        default=True,
        description="Capture indices of selected experts.",
    )
    capture_expert_outputs: bool = Field(
        default=False,
        description="Capture individual expert outputs (memory intensive).",
    )

    # Analysis options
    compute_entropy: bool = Field(
        default=False,
        description="Compute router entropy (routing confidence).",
    )
    compute_utilization: bool = Field(
        default=False,
        description="Compute expert utilization statistics.",
    )


class MoEAblationConfig(BaseModel):
    """Configuration for MoE ablation studies."""

    model_config = ConfigDict(frozen=True)

    target_layers: list[int] | None = Field(
        default=None,
        description="Layers to ablate. None = all MoE layers.",
    )
    ablation_method: str = Field(
        default="zero",
        description="How to ablate: 'zero', 'mean', 'random'.",
    )
    preserve_scale: bool = Field(
        default=True,
        description="Preserve output scale after ablation.",
    )
    max_new_tokens: int = Field(
        default=10,
        description="Maximum new tokens to generate during ablation testing.",
    )
