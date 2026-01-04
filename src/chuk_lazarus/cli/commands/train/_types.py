"""Type definitions for training CLI commands.

This module contains Pydantic models and enums for training commands.
"""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin


class TrainMode(str, Enum):
    """Training mode."""

    SFT = "sft"
    DPO = "dpo"


class DataGenType(str, Enum):
    """Type of data to generate."""

    MATH = "math"
    TOOL_CALL = "tool_call"


class SFTConfig(CommandConfig):
    """Configuration for SFT training command.

    Attributes:
        model: Path or HuggingFace name of the model
        data: Path to training data
        eval_data: Path to evaluation data (optional)
        output: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
        mask_prompt: Whether to mask prompt in loss
        log_interval: Logging interval
    """

    model: str = Field(..., description="Model path or name")
    data: Path = Field(..., description="Path to training data")
    eval_data: Path | None = Field(default=None, description="Path to eval data")
    output: Path = Field(default=Path("./checkpoints/sft"), description="Output dir")
    epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")
    use_lora: bool = Field(default=False, description="Use LoRA")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    mask_prompt: bool = Field(default=False, description="Mask prompt in loss")
    log_interval: int = Field(default=10, ge=1, description="Log interval")

    @classmethod
    def from_args(cls, args: Namespace) -> SFTConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            data=Path(args.data),
            eval_data=Path(args.eval_data) if args.eval_data else None,
            output=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            mask_prompt=args.mask_prompt,
            log_interval=args.log_interval,
        )


class DPOConfig(CommandConfig):
    """Configuration for DPO training command.

    Attributes:
        model: Path or HuggingFace name of the policy model
        ref_model: Path or HuggingFace name of reference model
        data: Path to preference data
        eval_data: Path to evaluation data (optional)
        output: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        beta: DPO beta parameter
        max_length: Maximum sequence length
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
    """

    model: str = Field(..., description="Policy model path or name")
    ref_model: str | None = Field(default=None, description="Reference model path")
    data: Path = Field(..., description="Path to preference data")
    eval_data: Path | None = Field(default=None, description="Path to eval data")
    output: Path = Field(default=Path("./checkpoints/dpo"), description="Output dir")
    epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    beta: float = Field(default=0.1, gt=0, description="DPO beta")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")
    use_lora: bool = Field(default=False, description="Use LoRA")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")

    @classmethod
    def from_args(cls, args: Namespace) -> DPOConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            ref_model=getattr(args, "ref_model", None),
            data=Path(args.data),
            eval_data=Path(args.eval_data) if args.eval_data else None,
            output=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta=args.beta,
            max_length=args.max_length,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
        )

    @property
    def reference_model(self) -> str:
        """Get reference model name (defaults to policy model)."""
        return self.ref_model or self.model


class DataGenConfig(CommandConfig):
    """Configuration for data generation command.

    Attributes:
        type: Type of data to generate
        output: Output directory
        sft_samples: Number of SFT samples
        dpo_samples: Number of DPO samples
        seed: Random seed
    """

    type: DataGenType = Field(..., description="Type of data to generate")
    output: Path = Field(..., description="Output directory")
    sft_samples: int = Field(default=10000, ge=1, description="Number of SFT samples")
    dpo_samples: int = Field(default=5000, ge=1, description="Number of DPO samples")
    seed: int | None = Field(default=None, description="Random seed")

    @classmethod
    def from_args(cls, args: Namespace) -> DataGenConfig:
        """Create config from argparse Namespace."""
        return cls(
            type=DataGenType(args.type),
            output=Path(args.output),
            sft_samples=args.sft_samples,
            dpo_samples=args.dpo_samples,
            seed=args.seed,
        )


class TrainResult(CommandResult, OutputMixin):
    """Result of training command.

    Attributes:
        mode: Training mode used
        checkpoint_dir: Directory where checkpoints were saved
        epochs_completed: Number of epochs completed
    """

    mode: TrainMode = Field(..., description="Training mode")
    checkpoint_dir: Path = Field(..., description="Checkpoint directory")
    epochs_completed: int = Field(default=0, ge=0, description="Epochs completed")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header(f"{self.mode.value.upper()} Training Complete")]
        lines.append(self.format_field("Mode", self.mode.value))
        lines.append(self.format_field("Epochs", self.epochs_completed))
        lines.append(self.format_field("Checkpoints", str(self.checkpoint_dir)))
        return "\n".join(lines)


class DataGenResult(CommandResult, OutputMixin):
    """Result of data generation command.

    Attributes:
        type: Type of data generated
        output_dir: Directory where data was saved
        sft_samples: Number of SFT samples generated
        dpo_samples: Number of DPO samples generated
    """

    type: DataGenType = Field(..., description="Type of data generated")
    output_dir: Path = Field(..., description="Output directory")
    sft_samples: int = Field(default=0, ge=0, description="SFT samples generated")
    dpo_samples: int = Field(default=0, ge=0, description="DPO samples generated")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Data Generation Complete")]
        lines.append(self.format_field("Type", self.type.value))
        lines.append(self.format_field("Output", str(self.output_dir)))
        lines.append(self.format_field("SFT samples", self.sft_samples))
        lines.append(self.format_field("DPO samples", self.dpo_samples))
        return "\n".join(lines)


__all__ = [
    "DataGenConfig",
    "DataGenResult",
    "DataGenType",
    "DPOConfig",
    "SFTConfig",
    "TrainMode",
    "TrainResult",
]
