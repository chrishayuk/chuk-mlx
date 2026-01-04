"""Tests for lengths build command."""

import json
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.data.lengths._types import LengthBuildConfig
from chuk_lazarus.cli.commands.data.lengths.build import data_lengths_build

LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"
LENGTH_CACHE_PATCH = "chuk_lazarus.data.batching.LengthCache"
FINGERPRINT_PATCH = "chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint"


class TestDataLengthsBuild:
    """Tests for data_lengths_build command."""

    @pytest.mark.asyncio
    async def test_build_with_jsonl_text_field(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building length cache from JSONL file with text field."""
        dataset_file = tmp_path / "test.jsonl"
        samples = [
            {"id": "s1", "text": "Hello world"},
            {"id": "s2", "text": "Test sample"},
        ]
        with open(dataset_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        output_file = tmp_path / "cache.db"
        config = LengthBuildConfig(
            tokenizer="test-tokenizer",
            dataset=dataset_file,
            output=output_file,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(FINGERPRINT_PATCH, create=True) as mock_fp,
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
        ):
            mock_fp.return_value = MagicMock(fingerprint="hash_123")
            mock_cache_cls.create.return_value = mock_length_cache

            result = await data_lengths_build(config)

            assert result.samples_processed == 2
            assert result.tokenizer_hash == "hash_123"
            assert mock_length_cache.add.call_count == 2

    @pytest.mark.asyncio
    async def test_build_with_content_field(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building with content field."""
        dataset_file = tmp_path / "test.json"
        samples = [{"sample_id": "s1", "content": "Hello"}]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        output_file = tmp_path / "cache.db"
        config = LengthBuildConfig(
            tokenizer="test-tokenizer",
            dataset=dataset_file,
            output=output_file,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(FINGERPRINT_PATCH, create=True, side_effect=Exception("Error")),
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
        ):
            mock_cache_cls.create.return_value = mock_length_cache

            result = await data_lengths_build(config)

            assert result.tokenizer_hash == "unknown"
            assert mock_length_cache.add.call_count == 1

    @pytest.mark.asyncio
    async def test_build_with_messages_format(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building with chat messages format."""
        dataset_file = tmp_path / "test.jsonl"
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]
        with open(dataset_file, "w") as f:
            f.write(json.dumps(samples[0]) + "\n")

        output_file = tmp_path / "cache.db"
        config = LengthBuildConfig(
            tokenizer="test-tokenizer",
            dataset=dataset_file,
            output=output_file,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(FINGERPRINT_PATCH, create=True) as mock_fp,
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
        ):
            mock_fp.return_value = MagicMock(fingerprint="hash_456")
            mock_cache_cls.create.return_value = mock_length_cache

            result = await data_lengths_build(config)

            assert result.samples_processed == 1

    @pytest.mark.asyncio
    async def test_build_auto_generates_id(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test that sample IDs are auto-generated when missing."""
        dataset_file = tmp_path / "test.json"
        samples = [{"text": "No ID here"}]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        output_file = tmp_path / "cache.db"
        config = LengthBuildConfig(
            tokenizer="test-tokenizer",
            dataset=dataset_file,
            output=output_file,
        )

        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(FINGERPRINT_PATCH, create=True) as mock_fp,
            patch(LENGTH_CACHE_PATCH, create=True) as mock_cache_cls,
        ):
            mock_fp.return_value = MagicMock(fingerprint="hash_789")
            mock_cache_cls.create.return_value = mock_length_cache

            result = await data_lengths_build(config)

            assert result.samples_processed == 1
