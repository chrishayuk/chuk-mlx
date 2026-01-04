"""Tests for train CLI type definitions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.cli.commands.train._types import (
    DataGenConfig,
    DataGenResult,
    DataGenType,
    DPOConfig,
    SFTConfig,
    TrainMode,
    TrainResult,
)


class TestTrainMode:
    """Tests for TrainMode enum."""

    def test_train_mode_values(self):
        """Test TrainMode enum values."""
        assert TrainMode.SFT == "sft"
        assert TrainMode.DPO == "dpo"

    def test_train_mode_is_string_enum(self):
        """Test TrainMode is a string enum."""
        assert isinstance(TrainMode.SFT, str)


class TestDataGenType:
    """Tests for DataGenType enum."""

    def test_datagen_type_values(self):
        """Test DataGenType enum values."""
        assert DataGenType.MATH == "math"
        assert DataGenType.TOOL_CALL == "tool_call"


class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_from_args_basic(self, sft_args):
        """Test creating config from args."""
        config = SFTConfig.from_args(sft_args)

        assert config.model == "test-model"
        assert config.data == Path("/path/to/train.jsonl")
        assert config.eval_data is None
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.use_lora is False

    def test_from_args_with_eval_data(self, sft_args):
        """Test config with eval data."""
        sft_args.eval_data = "/path/to/eval.jsonl"
        config = SFTConfig.from_args(sft_args)

        assert config.eval_data == Path("/path/to/eval.jsonl")

    def test_from_args_with_lora(self, sft_args):
        """Test config with LoRA."""
        sft_args.use_lora = True
        sft_args.lora_rank = 16
        config = SFTConfig.from_args(sft_args)

        assert config.use_lora is True
        assert config.lora_rank == 16

    def test_epochs_validation(self, sft_args):
        """Test epochs must be positive."""
        sft_args.epochs = 0
        with pytest.raises(ValidationError):
            SFTConfig.from_args(sft_args)

    def test_learning_rate_validation(self, sft_args):
        """Test learning rate must be positive."""
        sft_args.learning_rate = 0
        with pytest.raises(ValidationError):
            SFTConfig.from_args(sft_args)


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_from_args_basic(self, dpo_args):
        """Test creating config from args."""
        config = DPOConfig.from_args(dpo_args)

        assert config.model == "test-model"
        assert config.ref_model is None
        assert config.beta == 0.1
        assert config.use_lora is False

    def test_reference_model_defaults_to_policy(self, dpo_args):
        """Test reference model defaults to policy model."""
        config = DPOConfig.from_args(dpo_args)
        assert config.reference_model == "test-model"

    def test_reference_model_when_specified(self, dpo_args):
        """Test reference model when explicitly specified."""
        dpo_args.ref_model = "ref-model"
        config = DPOConfig.from_args(dpo_args)
        assert config.reference_model == "ref-model"

    def test_beta_validation(self, dpo_args):
        """Test beta must be positive."""
        dpo_args.beta = 0
        with pytest.raises(ValidationError):
            DPOConfig.from_args(dpo_args)


class TestDataGenConfig:
    """Tests for DataGenConfig."""

    def test_from_args_basic(self, datagen_args):
        """Test creating config from args."""
        config = DataGenConfig.from_args(datagen_args)

        assert config.type == DataGenType.MATH
        assert config.output == Path("./data/generated")
        assert config.sft_samples == 10000
        assert config.dpo_samples == 5000
        assert config.seed == 42

    def test_samples_validation(self, datagen_args):
        """Test samples must be positive."""
        datagen_args.sft_samples = 0
        with pytest.raises(ValidationError):
            DataGenConfig.from_args(datagen_args)


class TestTrainResult:
    """Tests for TrainResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = TrainResult(
            mode=TrainMode.SFT,
            checkpoint_dir=Path("./checkpoints"),
            epochs_completed=3,
        )
        assert result.mode == TrainMode.SFT
        assert result.epochs_completed == 3

    def test_to_display(self):
        """Test display formatting."""
        result = TrainResult(
            mode=TrainMode.DPO,
            checkpoint_dir=Path("./checkpoints/dpo"),
            epochs_completed=5,
        )
        display = result.to_display()
        assert "DPO Training Complete" in display
        assert "dpo" in display
        assert "5" in display


class TestDataGenResult:
    """Tests for DataGenResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = DataGenResult(
            type=DataGenType.MATH,
            output_dir=Path("./data"),
            sft_samples=1000,
            dpo_samples=500,
        )
        assert result.type == DataGenType.MATH
        assert result.sft_samples == 1000

    def test_to_display(self):
        """Test display formatting."""
        result = DataGenResult(
            type=DataGenType.MATH,
            output_dir=Path("./data"),
            sft_samples=1000,
            dpo_samples=500,
        )
        display = result.to_display()
        assert "Data Generation Complete" in display
        assert "math" in display
        assert "1000" in display
