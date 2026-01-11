"""Tests for data generation command."""

import logging
from unittest.mock import patch

import pytest

from chuk_lazarus.cli.commands.train._types import DataGenConfig, DataGenType
from chuk_lazarus.cli.commands.train.datagen import generate_data, generate_data_cmd

GENERATE_DATASET_PATCH = "chuk_lazarus.data.generators.generate_lazarus_dataset"


class TestGenerateData:
    """Tests for generate_data async command."""

    @pytest.fixture
    def basic_config(self, datagen_args):
        """Create basic datagen config."""
        return DataGenConfig.from_args(datagen_args)

    @pytest.mark.asyncio
    async def test_generate_data_math(self, basic_config, caplog):
        """Test generating math dataset."""
        with (
            patch(GENERATE_DATASET_PATCH, create=True) as mock_generate,
            caplog.at_level(logging.INFO),
        ):
            result = await generate_data(basic_config)

            # Verify generate function was called
            mock_generate.assert_called_once_with(
                output_dir="data/generated",  # Path normalizes ./data/generated
                sft_samples=10000,
                dpo_samples=5000,
                seed=42,
            )

            # Verify result
            assert result.type == DataGenType.MATH
            assert result.sft_samples == 10000
            assert result.dpo_samples == 5000

            # Check logging
            assert "Generating math dataset with 10000 SFT samples" in caplog.text
            assert "Dataset saved to" in caplog.text

    @pytest.mark.asyncio
    async def test_generate_data_custom_samples(self, datagen_args):
        """Test generating dataset with custom sample counts."""
        datagen_args.sft_samples = 5000
        datagen_args.dpo_samples = 2500
        config = DataGenConfig.from_args(datagen_args)

        with patch(GENERATE_DATASET_PATCH, create=True) as mock_generate:
            result = await generate_data(config)

            mock_generate.assert_called_once_with(
                output_dir="data/generated",  # Path normalizes
                sft_samples=5000,
                dpo_samples=2500,
                seed=42,
            )

            assert result.sft_samples == 5000
            assert result.dpo_samples == 2500

    @pytest.mark.asyncio
    async def test_generate_data_custom_seed(self, datagen_args):
        """Test generating dataset with custom random seed."""
        datagen_args.seed = 123
        config = DataGenConfig.from_args(datagen_args)

        with patch(GENERATE_DATASET_PATCH, create=True) as mock_generate:
            await generate_data(config)

            mock_generate.assert_called_once_with(
                output_dir="data/generated",  # Path normalizes
                sft_samples=10000,
                dpo_samples=5000,
                seed=123,
            )

    @pytest.mark.asyncio
    async def test_generate_data_custom_output(self, datagen_args):
        """Test generating dataset to custom output directory."""
        datagen_args.output = "/custom/path/data"
        config = DataGenConfig.from_args(datagen_args)

        with patch(GENERATE_DATASET_PATCH, create=True) as mock_generate:
            result = await generate_data(config)

            mock_generate.assert_called_once_with(
                output_dir="/custom/path/data",
                sft_samples=10000,
                dpo_samples=5000,
                seed=42,
            )

            assert str(result.output_dir) == "/custom/path/data"

    @pytest.mark.asyncio
    async def test_generate_data_unknown_type(self, datagen_args, caplog):
        """Test generating dataset with unknown type exits with error."""
        datagen_args.type = "tool_call"
        config = DataGenConfig.from_args(datagen_args)

        with (
            patch(GENERATE_DATASET_PATCH, create=True) as mock_generate,
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            await generate_data(config)

        # Verify it exits with error code 1
        assert exc_info.value.code == 1

        # Verify generate was never called
        mock_generate.assert_not_called()

        # Check error logging
        assert "Unknown data type:" in caplog.text


class TestGenerateDataCmd:
    """Tests for generate_data_cmd CLI entry point."""

    @pytest.mark.asyncio
    async def test_generate_data_cmd(self, datagen_args, capsys):
        """Test CLI entry point."""
        with patch(GENERATE_DATASET_PATCH, create=True):
            await generate_data_cmd(datagen_args)

            captured = capsys.readouterr()
            assert "Data Generation Complete" in captured.out
            assert "math" in captured.out
