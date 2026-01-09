"""Tests for DPO training command."""

from argparse import Namespace
from pathlib import Path

import pytest

from chuk_lazarus.cli.commands.train._types import DPOConfig


class TestDPOConfig:
    """Tests for DPOConfig."""

    @pytest.fixture
    def basic_dpo_args(self):
        """Create basic DPO args."""
        return Namespace(
            model="test-model",
            ref_model=None,
            data="/path/to/train.jsonl",
            eval_data=None,
            output="/output",
            epochs=3,
            batch_size=4,
            learning_rate=1e-6,
            beta=0.1,
            max_length=512,
            use_lora=False,
            lora_rank=8,
        )

    def test_from_args(self, basic_dpo_args):
        """Test creating config from args."""
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.model == "test-model"
        assert config.data == Path("/path/to/train.jsonl")
        assert config.epochs == 3
        assert config.beta == 0.1

    def test_from_args_with_ref_model(self, basic_dpo_args):
        """Test creating config with reference model."""
        basic_dpo_args.ref_model = "ref-model"
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.ref_model == "ref-model"
        assert config.reference_model == "ref-model"

    def test_from_args_with_lora(self, basic_dpo_args):
        """Test creating config with LoRA enabled."""
        basic_dpo_args.use_lora = True
        basic_dpo_args.lora_rank = 16
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.use_lora is True
        assert config.lora_rank == 16

    def test_reference_model_defaults_to_policy(self, basic_dpo_args):
        """Test reference_model defaults to policy model when not set."""
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.ref_model is None
        assert config.reference_model == "test-model"

    def test_from_args_with_eval_data(self, basic_dpo_args):
        """Test creating config with evaluation data."""
        basic_dpo_args.eval_data = "/path/to/eval.jsonl"
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.eval_data == Path("/path/to/eval.jsonl")

    def test_default_output_path(self, basic_dpo_args):
        """Test default output path is set correctly."""
        config = DPOConfig.from_args(basic_dpo_args)

        assert config.output == Path("/output")

    def test_config_is_frozen(self, basic_dpo_args):
        """Test that config is immutable."""
        from pydantic import ValidationError

        config = DPOConfig.from_args(basic_dpo_args)

        with pytest.raises(ValidationError):
            config.model = "other-model"
