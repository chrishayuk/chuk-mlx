"""Tests for batch generate command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.batching._types import GenerateConfig
from chuk_lazarus.cli.commands.data.batching.generate import data_batch_generate

LOAD_PLAN_PATCH = "chuk_lazarus.data.batching.load_batch_plan"
LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"
BATCH_WRITER_PATCH = "chuk_lazarus.data.batching.BatchWriter"
BATCH_READER_PATCH = "chuk_lazarus.data.batching.BatchReader"


class TestDataBatchGenerate:
    """Tests for data_batch_generate command."""

    @pytest.mark.asyncio
    async def test_generate_batches_jsonl(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batch files from JSONL dataset."""
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "batches"

        samples = [
            {"id": "s1", "text": "Sample 1"},
            {"id": "s2", "text": "Sample 2"},
        ]
        with open(dataset_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        config = GenerateConfig(
            plan=Path("/path/to/plan.msgpack"),
            dataset=dataset_file,
            tokenizer="gpt2",
            output=output_dir,
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 2
        mock_reader.fingerprint = "test_fp"

        mock_writer = MagicMock()
        mock_writer.write_all.return_value = ["batch_0.npz", "batch_1.npz"]

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(BATCH_WRITER_PATCH, create=True, return_value=mock_writer),
            patch(BATCH_READER_PATCH, create=True, return_value=mock_reader),
        ):
            result = await data_batch_generate(config)

            assert result.num_files == 2
            assert result.num_epochs == 2
            assert result.fingerprint == "test_fp"

    @pytest.mark.asyncio
    async def test_generate_batches_json(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batch files from JSON dataset."""
        dataset_file = tmp_path / "dataset.json"
        output_dir = tmp_path / "batches"

        samples = [
            {"sample_id": "s1", "content": "Test content"},
            {"sample_id": "s2", "input": "Test input"},
        ]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        config = GenerateConfig(
            plan=Path("/path/to/plan.msgpack"),
            dataset=dataset_file,
            tokenizer="gpt2",
            output=output_dir,
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = None

        mock_writer = MagicMock()
        mock_writer.write_all.return_value = ["batch_0.npz"]

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(BATCH_WRITER_PATCH, create=True, return_value=mock_writer),
            patch(BATCH_READER_PATCH, create=True, return_value=mock_reader),
        ):
            result = await data_batch_generate(config)

            assert result.num_files == 1
            assert result.fingerprint is None

    @pytest.mark.asyncio
    async def test_generate_batches_with_messages(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batches with chat messages format."""
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "batches"

        samples = [
            {
                "id": "s1",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        ]
        with open(dataset_file, "w") as f:
            f.write(json.dumps(samples[0]) + "\n")

        config = GenerateConfig(
            plan=Path("/path/to/plan.msgpack"),
            dataset=dataset_file,
            tokenizer="gpt2",
            output=output_dir,
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = "fp"

        mock_writer = MagicMock()
        mock_writer.write_all.return_value = ["batch_0.npz"]

        with (
            patch(LOAD_PLAN_PATCH, create=True, return_value=mock_batch_plan),
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(BATCH_WRITER_PATCH, create=True, return_value=mock_writer) as mock_writer_cls,
            patch(BATCH_READER_PATCH, create=True, return_value=mock_reader),
        ):
            await data_batch_generate(config)

            # Verify samples were passed to writer
            call_kwargs = mock_writer_cls.call_args.kwargs
            assert len(call_kwargs["samples"]) == 1
