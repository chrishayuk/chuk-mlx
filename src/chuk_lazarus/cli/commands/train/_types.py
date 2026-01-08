"""Type definitions for training CLI commands.

This module contains Pydantic models and enums for training commands.
CLI commands should be thin wrappers - all business logic belongs in the framework.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin
from .._constants import DataGenType, TrainMode, TrainingDefaults


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
    epochs: int = Field(default=TrainingDefaults.SFT_EPOCHS, ge=1, description="Number of epochs")
    max_steps: int | None = Field(default=None, description="Max steps (overrides epochs)")
    batch_size: int = Field(default=TrainingDefaults.BATCH_SIZE, ge=1, description="Batch size")
    learning_rate: float = Field(
        default=TrainingDefaults.SFT_LEARNING_RATE, gt=0, description="Learning rate"
    )
    max_length: int = Field(
        default=TrainingDefaults.MAX_LENGTH, ge=1, description="Max sequence length"
    )
    use_lora: bool = Field(default=False, description="Use LoRA")
    lora_rank: int = Field(default=TrainingDefaults.LORA_RANK, ge=1, description="LoRA rank")
    mask_prompt: bool = Field(default=False, description="Mask prompt in loss")
    log_interval: int = Field(
        default=TrainingDefaults.LOG_INTERVAL, ge=1, description="Log interval"
    )

    @classmethod
    def from_args(cls, args: Namespace) -> SFTConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            data=Path(args.data),
            eval_data=Path(args.eval_data) if args.eval_data else None,
            output=Path(args.output),
            epochs=args.epochs,
            max_steps=getattr(args, "max_steps", None),
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
    epochs: int = Field(default=TrainingDefaults.DPO_EPOCHS, ge=1, description="Number of epochs")
    batch_size: int = Field(default=TrainingDefaults.BATCH_SIZE, ge=1, description="Batch size")
    learning_rate: float = Field(
        default=TrainingDefaults.DPO_LEARNING_RATE, gt=0, description="Learning rate"
    )
    beta: float = Field(default=TrainingDefaults.DPO_BETA, gt=0, description="DPO beta")
    max_length: int = Field(
        default=TrainingDefaults.MAX_LENGTH, ge=1, description="Max sequence length"
    )
    use_lora: bool = Field(default=False, description="Use LoRA")
    lora_rank: int = Field(default=TrainingDefaults.LORA_RANK, ge=1, description="LoRA rank")

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


class GRPOConfig(CommandConfig):
    """Configuration for GRPO training command.

    GRPO (Group Relative Policy Optimization) is an RL algorithm that:
    - Generates multiple responses per prompt
    - Uses group-relative advantages (no value function needed)
    - Works well with verifiable rewards (e.g., arithmetic correctness)

    Attributes:
        model: Path or HuggingFace name of the policy model
        ref_model: Path or HuggingFace name of reference model (defaults to policy)
        output: Output directory for checkpoints
        iterations: Number of training iterations
        prompts_per_iteration: Number of prompts per iteration
        group_size: Number of responses per prompt
        learning_rate: Learning rate
        kl_coef: KL penalty coefficient
        max_response_length: Maximum tokens in generated response
        temperature: Sampling temperature
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
        reward_script: Path to Python script defining reward_fn(prompt, response) -> float
    """

    model: str = Field(..., description="Policy model path or name")
    ref_model: str | None = Field(default=None, description="Reference model path")
    output: Path = Field(default=Path("./checkpoints/grpo"), description="Output dir")
    iterations: int = Field(
        default=TrainingDefaults.GRPO_ITERATIONS, ge=1, description="Training iterations"
    )
    prompts_per_iteration: int = Field(
        default=TrainingDefaults.GRPO_PROMPTS_PER_ITERATION, ge=1, description="Prompts per iteration"
    )
    group_size: int = Field(
        default=TrainingDefaults.GRPO_GROUP_SIZE, ge=2, description="Responses per prompt"
    )
    learning_rate: float = Field(
        default=TrainingDefaults.GRPO_LEARNING_RATE, gt=0, description="Learning rate"
    )
    kl_coef: float = Field(
        default=TrainingDefaults.GRPO_KL_COEF, ge=0, description="KL penalty coefficient"
    )
    max_response_length: int = Field(
        default=TrainingDefaults.GRPO_MAX_RESPONSE_LENGTH, ge=1, description="Max response tokens"
    )
    temperature: float = Field(
        default=TrainingDefaults.GRPO_TEMPERATURE, gt=0, description="Sampling temperature"
    )
    use_lora: bool = Field(default=False, description="Use LoRA")
    lora_rank: int = Field(default=TrainingDefaults.LORA_RANK, ge=1, description="LoRA rank")
    reward_script: Path | None = Field(default=None, description="Python script with reward_fn")

    @classmethod
    def from_args(cls, args: Namespace) -> GRPOConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            ref_model=getattr(args, "ref_model", None),
            output=Path(args.output),
            iterations=args.iterations,
            prompts_per_iteration=args.prompts_per_iteration,
            group_size=args.group_size,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            max_response_length=args.max_response_length,
            temperature=args.temperature,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            reward_script=Path(args.reward_script) if args.reward_script else None,
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
    "GRPOConfig",
    "SFTConfig",
    "TrainMode",
    "TrainResult",
]
