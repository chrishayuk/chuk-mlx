"""Tests for data command handlers in chuk-lazarus CLI."""

import json
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_length_cache():
    """Create a mock LengthCache."""
    cache = MagicMock()
    cache.__len__.return_value = 100
    cache.tokenizer_hash = "test_hash_123"
    cache.get_all.return_value = {
        "sample_0": 10,
        "sample_1": 20,
        "sample_2": 30,
        "sample_3": 15,
        "sample_4": 25,
    }

    # Make it work as async context manager
    async def mock_aenter(self):
        return self

    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        return None

    async def mock_add(sample_id, length):
        pass

    cache.__aenter__ = mock_aenter
    cache.__aexit__ = mock_aexit
    cache.add = AsyncMock()

    return cache


@pytest.fixture
def mock_batch_plan():
    """Create a mock BatchPlan."""
    plan = MagicMock()
    plan.num_epochs = 2
    plan.total_microbatches = 50
    plan.fingerprint = "test_fingerprint_abc123"

    # Mock metadata
    plan.meta = MagicMock()
    plan.meta.created_at = "2026-01-03T12:00:00"
    plan.meta.dataset_hash = "dataset_hash_123"
    plan.meta.tokenizer_hash = "tokenizer_hash_456"
    plan.meta.token_budget = 2048
    plan.meta.bucket_edges = [128, 256, 512, 1024]
    plan.meta.mode = "predictable"
    plan.meta.pad_policy = "pad_to_bucket"
    plan.meta.overflow_max = 2048
    plan.meta.seed = 42

    # Mock epoch plan
    epoch_plan = MagicMock()
    epoch_plan.num_microbatches = 25
    epoch_plan.total_samples = 500
    epoch_plan.total_tokens = 10000

    # Mock microbatch
    microbatch = MagicMock()
    microbatch.batch_size = 8
    microbatch.bucket_id = 0
    microbatch.max_len = 128
    microbatch.samples = ["sample_0", "sample_1"]

    epoch_plan.microbatches = [microbatch] * 5

    plan.get_epoch.return_value = epoch_plan
    plan.iter_epoch.return_value = iter([microbatch] * 25)
    plan.shard.return_value = plan

    return plan


class TestDataLengthsBuild:
    """Tests for data_lengths_build command."""

    def test_build_with_jsonl_text_field(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building length cache from JSONL file with text field."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

        # Create test JSONL file
        dataset_file = tmp_path / "test.jsonl"
        samples = [
            {"id": "s1", "text": "Hello world"},
            {"id": "s2", "text": "Test sample"},
        ]
        with open(dataset_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        output_file = tmp_path / "cache.db"

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        # Mock the imports inside the function
        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint") as mock_fp:
                mock_fp.return_value = MagicMock(fingerprint="hash_123")

                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("builtins.print") as mock_print:
                        data_lengths_build(args)

                    # Verify tokenizer was loaded
                    mock_load.assert_called_once_with("test-tokenizer")

                    # Verify cache was created
                    mock_create.assert_called_once()

                    # Verify add was called for each sample
                    assert mock_length_cache.add.call_count == 2

                    # Verify output was printed
                    assert mock_print.call_count > 0

    def test_build_with_json_content_field(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building length cache from JSON file with content field."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

        dataset_file = tmp_path / "test.json"
        samples = [
            {"sample_id": "s1", "content": "Hello"},
            {"sample_id": "s2", "content": "World"},
        ]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        output_file = tmp_path / "cache.db"

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch(
                "chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint",
                side_effect=Exception("Test error"),
            ):
                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("builtins.print"):
                        data_lengths_build(args)

                    # When fingerprint fails, should use "unknown"
                    assert mock_create.call_count == 1

    def test_build_with_messages_format(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building length cache with chat messages format."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

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

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint"):
                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("builtins.print"):
                        data_lengths_build(args)

                    assert mock_length_cache.add.call_count == 1

    def test_build_with_input_field(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test building length cache with input field."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

        dataset_file = tmp_path / "test.json"
        samples = [{"input": "Test input"}]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        output_file = tmp_path / "cache.db"

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint"):
                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("builtins.print"):
                        data_lengths_build(args)

                    assert mock_length_cache.add.call_count == 1

    def test_build_progress_logging(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test that progress is logged every 1000 samples."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

        dataset_file = tmp_path / "test.jsonl"
        # Create 1500 samples
        with open(dataset_file, "w") as f:
            for i in range(1500):
                f.write(json.dumps({"id": f"s{i}", "text": "test"}) + "\n")

        output_file = tmp_path / "cache.db"

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint"):
                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("chuk_lazarus.cli.commands.data.logger") as mock_logger:
                        with patch("builtins.print"):
                            data_lengths_build(args)

                        # Should log at 1000 samples
                        info_calls = [
                            call
                            for call in mock_logger.info.call_args_list
                            if "Processed" in str(call)
                        ]
                        assert len(info_calls) >= 1

    def test_build_generates_sample_id(self, tmp_path, mock_tokenizer, mock_length_cache):
        """Test that sample IDs are auto-generated when missing."""
        from chuk_lazarus.cli.commands.data import data_lengths_build

        dataset_file = tmp_path / "test.json"
        samples = [{"text": "No ID here"}]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        output_file = tmp_path / "cache.db"

        args = Namespace(
            tokenizer="test-tokenizer",
            dataset=str(dataset_file),
            output=str(output_file),
        )

        with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load:
            mock_load.return_value = mock_tokenizer

            with patch("chuk_lazarus.data.tokenizers.fingerprint.compute_fingerprint"):
                with patch("chuk_lazarus.data.batching.LengthCache.create") as mock_create:
                    mock_create.return_value = mock_length_cache

                    with patch("builtins.print"):
                        data_lengths_build(args)

                    # Should still add sample with auto-generated ID
                    assert mock_length_cache.add.call_count == 1


class TestDataLengthsStats:
    """Tests for data_lengths_stats command."""

    def test_stats_with_populated_cache(self, tmp_path, mock_length_cache):
        """Test showing statistics for a populated cache."""
        from chuk_lazarus.cli.commands.data import data_lengths_stats

        cache_file = tmp_path / "cache.db"

        args = Namespace(cache=str(cache_file))

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("builtins.print") as mock_print:
                data_lengths_stats(args)

            # Verify statistics were printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(print_calls)

            assert "Length Cache Statistics" in output
            assert "Total samples:" in output
            assert "Min length:" in output
            assert "Max length:" in output
            assert "Mean length:" in output
            assert "P10:" in output
            assert "P99:" in output

    def test_stats_with_empty_cache(self, tmp_path):
        """Test showing statistics for an empty cache."""
        from chuk_lazarus.cli.commands.data import data_lengths_stats

        cache_file = tmp_path / "cache.db"

        args = Namespace(cache=str(cache_file))

        empty_cache = MagicMock()
        empty_cache.get_all.return_value = {}
        empty_cache.tokenizer_hash = "test_hash"

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return empty_cache

            mock_load.side_effect = async_load

            with patch("builtins.print") as mock_print:
                data_lengths_stats(args)

            # Should print "Cache is empty"
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Cache is empty" in str(call) for call in print_calls)


class TestDataBatchPlanBuild:
    """Tests for data_batchplan_build command."""

    def test_build_predictable_mode(self, tmp_path, mock_length_cache, mock_batch_plan):
        """Test building batch plan in predictable mode."""
        from chuk_lazarus.cli.commands.data import data_batchplan_build

        lengths_file = tmp_path / "cache.db"
        output_file = tmp_path / "plan.msgpack"

        args = Namespace(
            lengths=str(lengths_file),
            bucket_edges="128,256,512,1024",
            token_budget=2048,
            overflow_max=2048,
            predictable=True,
            seed=42,
            epochs=2,
            output=str(output_file),
            dataset_hash="dataset_123",
        )

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls:
                mock_builder = MagicMock()
                mock_builder.build = AsyncMock(return_value=mock_batch_plan)
                mock_builder_cls.return_value = mock_builder

                with patch("chuk_lazarus.data.batching.save_batch_plan") as mock_save:
                    with patch("builtins.print") as mock_print:
                        data_batchplan_build(args)

                    # Verify save was called
                    mock_save.assert_called_once()

                    # Verify output
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    output = "\n".join(print_calls)
                    assert "Batch Plan Built" in output
                    assert "predictable" in output

    def test_build_throughput_mode(self, tmp_path, mock_length_cache, mock_batch_plan):
        """Test building batch plan in throughput mode."""
        from chuk_lazarus.cli.commands.data import data_batchplan_build

        lengths_file = tmp_path / "cache.db"
        output_file = tmp_path / "plan.msgpack"

        args = Namespace(
            lengths=str(lengths_file),
            bucket_edges="128,256,512",
            token_budget=4096,
            overflow_max=4096,
            predictable=False,
            seed=None,
            epochs=3,
            output=str(output_file),
            dataset_hash=None,
        )

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls:
                mock_builder = MagicMock()
                mock_builder.build = AsyncMock(return_value=mock_batch_plan)
                mock_builder_cls.return_value = mock_builder

                with patch("chuk_lazarus.data.batching.save_batch_plan"):
                    with patch("builtins.print") as mock_print:
                        data_batchplan_build(args)

                    print_calls = [str(call) for call in mock_print.call_args_list]
                    output = "\n".join(print_calls)
                    assert "throughput" in output


class TestDataBatchPlanInfo:
    """Tests for data_batchplan_info command."""

    def test_info_without_sharding(self, tmp_path, mock_batch_plan):
        """Test showing batch plan info without sharding."""
        from chuk_lazarus.cli.commands.data import data_batchplan_info

        plan_file = tmp_path / "plan.msgpack"

        args = Namespace(
            plan=str(plan_file),
            rank=None,
            world_size=None,
            show_batches=None,
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("builtins.print") as mock_print:
                data_batchplan_info(args)

            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(print_calls)

            assert "Batch Plan Info" in output
            assert "Fingerprint:" in output
            assert "Total batches:" in output
            assert "Epoch 0:" in output
            assert "Epoch 1:" in output

    def test_info_with_sharding(self, tmp_path, mock_batch_plan):
        """Test showing batch plan info with sharding."""
        from chuk_lazarus.cli.commands.data import data_batchplan_info

        plan_file = tmp_path / "plan.msgpack"

        args = Namespace(
            plan=str(plan_file),
            rank=1,
            world_size=4,
            show_batches=None,
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("builtins.print") as mock_print:
                data_batchplan_info(args)

            # Verify shard was called
            mock_batch_plan.shard.assert_called_once_with(1, 4)

            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(print_calls)
            assert "rank 1/4" in output

    def test_info_with_invalid_rank(self, tmp_path, mock_batch_plan):
        """Test info with invalid rank returns early."""
        from chuk_lazarus.cli.commands.data import data_batchplan_info

        plan_file = tmp_path / "plan.msgpack"

        args = Namespace(
            plan=str(plan_file),
            rank=5,
            world_size=4,
            show_batches=None,
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("builtins.print") as mock_print:
                data_batchplan_info(args)

            # Should print error and return
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Error:" in str(call) for call in print_calls)

    def test_info_with_negative_rank(self, tmp_path, mock_batch_plan):
        """Test info with negative rank returns early."""
        from chuk_lazarus.cli.commands.data import data_batchplan_info

        plan_file = tmp_path / "plan.msgpack"

        args = Namespace(
            plan=str(plan_file),
            rank=-1,
            world_size=4,
            show_batches=None,
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("builtins.print") as mock_print:
                data_batchplan_info(args)

            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Error:" in str(call) for call in print_calls)

    def test_info_show_batches(self, tmp_path, mock_batch_plan):
        """Test showing sample batches."""
        from chuk_lazarus.cli.commands.data import data_batchplan_info

        plan_file = tmp_path / "plan.msgpack"

        args = Namespace(
            plan=str(plan_file),
            rank=None,
            world_size=None,
            show_batches=3,
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("builtins.print") as mock_print:
                data_batchplan_info(args)

            print_calls = [str(call) for call in mock_print.call_args_list]
            output = "\n".join(print_calls)
            assert "Sample batches" in output
            assert "Batch 0:" in output


class TestDataBatchPlanVerify:
    """Tests for data_batchplan_verify command."""

    def test_verify_matching_fingerprints(self, tmp_path, mock_length_cache, mock_batch_plan):
        """Test verification with matching fingerprints."""
        from chuk_lazarus.cli.commands.data import data_batchplan_verify

        plan_file = tmp_path / "plan.msgpack"
        lengths_file = tmp_path / "cache.db"

        args = Namespace(plan=str(plan_file), lengths=str(lengths_file))

        # Create a rebuilt plan with same fingerprint
        rebuilt_plan = MagicMock()
        rebuilt_plan.fingerprint = mock_batch_plan.fingerprint

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_cache_load:

                async def async_load(path):
                    return mock_length_cache

                mock_cache_load.side_effect = async_load

                with patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls:
                    mock_builder = MagicMock()
                    mock_builder.build = AsyncMock(return_value=rebuilt_plan)
                    mock_builder_cls.return_value = mock_builder

                    with patch("builtins.print") as mock_print:
                        data_batchplan_verify(args)

                    print_calls = [str(call) for call in mock_print.call_args_list]
                    output = "\n".join(print_calls)
                    assert "MATCH" in output
                    assert "reproducible" in output

    def test_verify_mismatching_fingerprints(self, tmp_path, mock_length_cache, mock_batch_plan):
        """Test verification with mismatching fingerprints."""
        from chuk_lazarus.cli.commands.data import data_batchplan_verify

        plan_file = tmp_path / "plan.msgpack"
        lengths_file = tmp_path / "cache.db"

        args = Namespace(plan=str(plan_file), lengths=str(lengths_file))

        # Create a rebuilt plan with different fingerprint
        rebuilt_plan = MagicMock()
        rebuilt_plan.fingerprint = "different_fingerprint"
        rebuilt_plan.num_epochs = mock_batch_plan.num_epochs

        # Mock iter_epoch to return different batches
        rebuilt_plan.iter_epoch.return_value = iter([MagicMock()] * 20)

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_cache_load:

                async def async_load(path):
                    return mock_length_cache

                mock_cache_load.side_effect = async_load

                with patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls:
                    mock_builder = MagicMock()
                    mock_builder.build = AsyncMock(return_value=rebuilt_plan)
                    mock_builder_cls.return_value = mock_builder

                    with patch("builtins.print") as mock_print:
                        with pytest.raises(SystemExit) as exc_info:
                            data_batchplan_verify(args)

                        assert exc_info.value.code == 1

                    print_calls = [str(call) for call in mock_print.call_args_list]
                    output = "\n".join(print_calls)
                    assert "MISMATCH" in output


class TestDataBatchPlanShard:
    """Tests for data_batchplan_shard command."""

    def test_shard_creation(self, tmp_path, mock_batch_plan):
        """Test creating sharded batch plans."""
        from chuk_lazarus.cli.commands.data import data_batchplan_shard

        plan_file = tmp_path / "plan.msgpack"
        output_dir = tmp_path / "shards"

        args = Namespace(
            plan=str(plan_file),
            world_size=4,
            output=str(output_dir),
        )

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load:
            mock_load.return_value = mock_batch_plan

            with patch("chuk_lazarus.data.batching.save_batch_plan") as mock_save:
                with patch("builtins.print") as mock_print:
                    data_batchplan_shard(args)

                # Verify shard was called for each rank
                assert mock_batch_plan.shard.call_count == 4

                # Verify save was called for each rank
                assert mock_save.call_count == 4

                # Verify output directory was created
                assert output_dir.exists()

                print_calls = [str(call) for call in mock_print.call_args_list]
                output = "\n".join(print_calls)
                assert "Batch Plan Sharding" in output
                assert "Rank 0:" in output
                assert "Rank 3:" in output


class TestDataBatchingAnalyze:
    """Tests for data_batching_analyze command."""

    def test_analyze_efficiency(self, tmp_path, mock_length_cache):
        """Test analyzing batching efficiency."""
        from chuk_lazarus.cli.commands.data import data_batching_analyze

        cache_file = tmp_path / "cache.db"

        args = Namespace(
            cache=str(cache_file),
            bucket_edges="128,256,512",
            overflow_max=1024,
            output=None,
        )

        mock_report = MagicMock()
        mock_report.to_ascii.return_value = "Efficiency Report ASCII"
        mock_report.model_dump.return_value = {"efficiency": 0.85}

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_create:
                mock_create.return_value = mock_report

                with patch("builtins.print") as mock_print:
                    data_batching_analyze(args)

                # Verify report was created
                mock_create.assert_called_once()

                # Verify ASCII report was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Efficiency Report ASCII" in str(call) for call in print_calls)

    def test_analyze_with_output_file(self, tmp_path, mock_length_cache):
        """Test analyzing efficiency with JSON output."""
        from chuk_lazarus.cli.commands.data import data_batching_analyze

        cache_file = tmp_path / "cache.db"
        output_file = tmp_path / "report.json"

        args = Namespace(
            cache=str(cache_file),
            bucket_edges="128,256",
            overflow_max=512,
            output=str(output_file),
        )

        mock_report = MagicMock()
        mock_report.to_ascii.return_value = "Report"
        mock_report.model_dump.return_value = {"efficiency": 0.90}

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_create:
                mock_create.return_value = mock_report

                with patch("builtins.print"):
                    data_batching_analyze(args)

                # Verify JSON file was written
                assert output_file.exists()
                with open(output_file) as f:
                    data = json.load(f)
                    assert data["efficiency"] == 0.90


class TestDataBatchingHistogram:
    """Tests for data_batching_histogram command."""

    def test_histogram_display(self, tmp_path, mock_length_cache):
        """Test displaying length histogram."""
        from chuk_lazarus.cli.commands.data import data_batching_histogram

        cache_file = tmp_path / "cache.db"

        args = Namespace(
            cache=str(cache_file),
            bins=20,
            width=80,
        )

        mock_histogram = MagicMock()
        mock_histogram.to_ascii.return_value = "Histogram ASCII Art"
        mock_histogram.p25 = 10
        mock_histogram.p50 = 20
        mock_histogram.p75 = 30
        mock_histogram.p90 = 40
        mock_histogram.p95 = 45
        mock_histogram.p99 = 50

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_compute:
                mock_compute.return_value = mock_histogram

                with patch("builtins.print") as mock_print:
                    data_batching_histogram(args)

                # Verify histogram was computed
                mock_compute.assert_called_once()

                # Verify output
                print_calls = [str(call) for call in mock_print.call_args_list]
                output = "\n".join(print_calls)
                assert "Histogram ASCII Art" in output
                assert "Percentiles" in output
                assert "P25:" in output
                assert "P99:" in output


class TestDataBatchingSuggest:
    """Tests for data_batching_suggest command."""

    def test_suggest_minimize_waste(self, tmp_path, mock_length_cache):
        """Test suggesting bucket edges with minimize waste goal."""
        from chuk_lazarus.cli.commands.data import data_batching_suggest

        cache_file = tmp_path / "cache.db"

        args = Namespace(
            cache=str(cache_file),
            num_buckets=5,
            goal="waste",
            max_length=2048,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="minimize_waste")
        mock_suggestion.edges = [128, 256, 512, 1024, 2048]
        mock_suggestion.overflow_max = 2048
        mock_suggestion.estimated_efficiency = 0.92
        mock_suggestion.rationale = "Optimized for minimal padding waste"

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.suggest_bucket_edges") as mock_suggest:
                mock_suggest.return_value = mock_suggestion

                with patch("builtins.print") as mock_print:
                    data_batching_suggest(args)

                # Verify suggestion was created
                mock_suggest.assert_called_once()

                # Verify output
                print_calls = [str(call) for call in mock_print.call_args_list]
                output = "\n".join(print_calls)
                assert "Bucket Edge Suggestions" in output
                assert "minimize_waste" in output
                assert "128,256,512,1024,2048" in output
                assert "Use with:" in output

    def test_suggest_balance_buckets(self, tmp_path, mock_length_cache):
        """Test suggesting bucket edges with balance goal."""
        from chuk_lazarus.cli.commands.data import data_batching_suggest

        cache_file = tmp_path / "cache.db"

        args = Namespace(
            cache=str(cache_file),
            num_buckets=4,
            goal="balance",
            max_length=1024,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="balance_buckets")
        mock_suggestion.edges = [256, 512, 768, 1024]
        mock_suggestion.overflow_max = 1024
        mock_suggestion.estimated_efficiency = 0.88
        mock_suggestion.rationale = "Balanced distribution"

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.suggest_bucket_edges") as mock_suggest:
                mock_suggest.return_value = mock_suggestion

                with patch("builtins.print"):
                    data_batching_suggest(args)

                mock_suggest.assert_called_once()

    def test_suggest_minimize_memory(self, tmp_path, mock_length_cache):
        """Test suggesting bucket edges with memory goal."""
        from chuk_lazarus.cli.commands.data import data_batching_suggest

        cache_file = tmp_path / "cache.db"

        args = Namespace(
            cache=str(cache_file),
            num_buckets=3,
            goal="memory",
            max_length=512,
        )

        mock_suggestion = MagicMock()
        mock_suggestion.optimization_goal = MagicMock(value="minimize_memory")
        mock_suggestion.edges = [128, 256, 512]
        mock_suggestion.overflow_max = 512
        mock_suggestion.estimated_efficiency = 0.85
        mock_suggestion.rationale = "Memory optimized"

        with patch("chuk_lazarus.data.batching.LengthCache.load") as mock_load:

            async def async_load(path):
                return mock_length_cache

            mock_load.side_effect = async_load

            with patch("chuk_lazarus.data.batching.suggest_bucket_edges") as mock_suggest:
                mock_suggest.return_value = mock_suggestion

                with patch("builtins.print"):
                    data_batching_suggest(args)

                mock_suggest.assert_called_once()


class TestDataBatchGenerate:
    """Tests for data_batch_generate command."""

    def test_generate_batches_jsonl(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batch files from JSONL dataset."""
        from chuk_lazarus.cli.commands.data import data_batch_generate

        # Create test files
        plan_file = tmp_path / "plan.msgpack"
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "batches"

        samples = [
            {"id": "s1", "text": "Sample 1"},
            {"id": "s2", "text": "Sample 2"},
        ]
        with open(dataset_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        args = Namespace(
            plan=str(plan_file),
            dataset=str(dataset_file),
            tokenizer="test-tokenizer",
            output=str(output_dir),
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 2
        mock_reader.fingerprint = "test_fp"

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load_plan:
            mock_load_plan.return_value = mock_batch_plan

            with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load_tok:
                mock_load_tok.return_value = mock_tokenizer

                with patch("chuk_lazarus.data.batching.BatchWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.write_all.return_value = ["batch_0.npz", "batch_1.npz"]
                    mock_writer_cls.return_value = mock_writer

                    with patch("chuk_lazarus.data.batching.BatchReader") as mock_reader_cls:
                        mock_reader_cls.return_value = mock_reader

                        with patch("builtins.print") as mock_print:
                            data_batch_generate(args)

                        # Verify tokenizer was loaded
                        mock_load_tok.assert_called_once()

                        # Verify writer was created
                        mock_writer_cls.assert_called_once()

                        # Verify batches were written
                        mock_writer.write_all.assert_called_once()

                        # Verify output
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        output = "\n".join(print_calls)
                        assert "Batch Generation Complete" in output
                        assert "Files:        2" in output

    def test_generate_batches_json(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batch files from JSON dataset."""
        from chuk_lazarus.cli.commands.data import data_batch_generate

        plan_file = tmp_path / "plan.msgpack"
        dataset_file = tmp_path / "dataset.json"
        output_dir = tmp_path / "batches"

        samples = [
            {"sample_id": "s1", "content": "Test content"},
            {"sample_id": "s2", "input": "Test input"},
        ]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        args = Namespace(
            plan=str(plan_file),
            dataset=str(dataset_file),
            tokenizer="test-tokenizer",
            output=str(output_dir),
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = None

        with patch("chuk_lazarus.data.batching.load_batch_plan") as mock_load_plan:
            mock_load_plan.return_value = mock_batch_plan

            with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load_tok:
                mock_load_tok.return_value = mock_tokenizer

                with patch("chuk_lazarus.data.batching.BatchWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.write_all.return_value = ["batch_0.npz"]
                    mock_writer_cls.return_value = mock_writer

                    with patch("chuk_lazarus.data.batching.BatchReader") as mock_reader_cls:
                        mock_reader_cls.return_value = mock_reader

                        with patch("builtins.print"):
                            data_batch_generate(args)

                        mock_writer.write_all.assert_called_once()

    def test_generate_batches_with_messages(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test generating batches with chat messages format."""
        from chuk_lazarus.cli.commands.data import data_batch_generate

        plan_file = tmp_path / "plan.msgpack"
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

        args = Namespace(
            plan=str(plan_file),
            dataset=str(dataset_file),
            tokenizer="test-tokenizer",
            output=str(output_dir),
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = "fp"

        with patch("chuk_lazarus.data.batching.load_batch_plan"):
            with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load_tok:
                mock_load_tok.return_value = mock_tokenizer

                with patch("chuk_lazarus.data.batching.BatchWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.write_all.return_value = ["batch_0.npz"]
                    mock_writer_cls.return_value = mock_writer

                    with patch("chuk_lazarus.data.batching.BatchReader") as mock_reader_cls:
                        mock_reader_cls.return_value = mock_reader

                        with patch("builtins.print"):
                            data_batch_generate(args)

                        mock_writer.write_all.assert_called_once()

    def test_generate_batches_progress_logging(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test progress logging during batch generation."""
        from chuk_lazarus.cli.commands.data import data_batch_generate

        plan_file = tmp_path / "plan.msgpack"
        dataset_file = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "batches"

        # Create 1500 samples
        with open(dataset_file, "w") as f:
            for i in range(1500):
                f.write(json.dumps({"id": f"s{i}", "text": f"Sample {i}"}) + "\n")

        args = Namespace(
            plan=str(plan_file),
            dataset=str(dataset_file),
            tokenizer="test-tokenizer",
            output=str(output_dir),
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = "fp"

        with patch("chuk_lazarus.data.batching.load_batch_plan"):
            with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load_tok:
                mock_load_tok.return_value = mock_tokenizer

                with patch("chuk_lazarus.data.batching.BatchWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.write_all.return_value = ["batch_0.npz"]
                    mock_writer_cls.return_value = mock_writer

                    with patch("chuk_lazarus.data.batching.BatchReader") as mock_reader_cls:
                        mock_reader_cls.return_value = mock_reader

                        with patch("chuk_lazarus.cli.commands.data.logger") as mock_logger:
                            with patch("builtins.print"):
                                data_batch_generate(args)

                            # Should log at 1000 samples
                            info_calls = [
                                call
                                for call in mock_logger.info.call_args_list
                                if "Tokenized" in str(call)
                            ]
                            assert len(info_calls) >= 1

    def test_generate_batches_auto_generates_ids(self, tmp_path, mock_tokenizer, mock_batch_plan):
        """Test that sample IDs are auto-generated when missing."""
        from chuk_lazarus.cli.commands.data import data_batch_generate

        plan_file = tmp_path / "plan.msgpack"
        dataset_file = tmp_path / "dataset.json"
        output_dir = tmp_path / "batches"

        samples = [{"text": "No ID here"}]
        with open(dataset_file, "w") as f:
            json.dump(samples, f)

        args = Namespace(
            plan=str(plan_file),
            dataset=str(dataset_file),
            tokenizer="test-tokenizer",
            output=str(output_dir),
        )

        mock_reader = MagicMock()
        mock_reader.num_epochs = 1
        mock_reader.fingerprint = None

        with patch("chuk_lazarus.data.batching.load_batch_plan"):
            with patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer") as mock_load_tok:
                mock_load_tok.return_value = mock_tokenizer

                with patch("chuk_lazarus.data.batching.BatchWriter") as mock_writer_cls:
                    mock_writer = MagicMock()
                    mock_writer.write_all.return_value = []
                    mock_writer_cls.return_value = mock_writer

                    with patch("chuk_lazarus.data.batching.BatchReader") as mock_reader_cls:
                        mock_reader_cls.return_value = mock_reader

                        with patch("builtins.print"):
                            data_batch_generate(args)

                        # Verify writer was called with samples
                        call_args = mock_writer_cls.call_args
                        samples_dict = call_args.kwargs["samples"]
                        assert len(samples_dict) == 1
