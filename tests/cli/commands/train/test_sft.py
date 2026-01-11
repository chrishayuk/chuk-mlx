"""Tests for SFT training command."""

from argparse import Namespace
from pathlib import Path

import pytest

from chuk_lazarus.cli.commands.train._types import SFTConfig


class TestSFTConfig:
    """Tests for SFTConfig."""

    @pytest.fixture
    def basic_sft_args(self):
        """Create basic SFT args."""
        return Namespace(
            model="test-model",
            data="/path/to/train.jsonl",
            eval_data=None,
            output="/output",
            epochs=3,
            max_steps=None,
            batch_size=4,
            learning_rate=2e-5,
            max_length=512,
            use_lora=False,
            lora_rank=8,
            mask_prompt=False,
            log_interval=10,
        )

    def test_from_args(self, basic_sft_args):
        """Test creating config from args."""
        config = SFTConfig.from_args(basic_sft_args)

        assert config.model == "test-model"
        assert config.data == Path("/path/to/train.jsonl")
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.use_lora is False

    def test_from_args_with_lora(self, basic_sft_args):
        """Test creating config with LoRA enabled."""
        basic_sft_args.use_lora = True
        basic_sft_args.lora_rank = 16
        config = SFTConfig.from_args(basic_sft_args)

        assert config.use_lora is True
        assert config.lora_rank == 16

    def test_from_args_with_eval_data(self, basic_sft_args):
        """Test creating config with evaluation data."""
        basic_sft_args.eval_data = "/path/to/eval.jsonl"
        config = SFTConfig.from_args(basic_sft_args)

        assert config.eval_data == Path("/path/to/eval.jsonl")

    def test_from_args_with_mask_prompt(self, basic_sft_args):
        """Test creating config with prompt masking."""
        basic_sft_args.mask_prompt = True
        config = SFTConfig.from_args(basic_sft_args)

        assert config.mask_prompt is True

    def test_from_args_with_max_steps(self, basic_sft_args):
        """Test creating config with max_steps."""
        basic_sft_args.max_steps = 1000
        config = SFTConfig.from_args(basic_sft_args)

        assert config.max_steps == 1000

    def test_default_output_path(self, basic_sft_args):
        """Test default output path is set correctly."""
        config = SFTConfig.from_args(basic_sft_args)

        assert config.output == Path("/output")

    def test_config_is_frozen(self, basic_sft_args):
        """Test that config is immutable."""
        from pydantic import ValidationError

        config = SFTConfig.from_args(basic_sft_args)

        with pytest.raises(ValidationError):
            config.model = "other-model"
