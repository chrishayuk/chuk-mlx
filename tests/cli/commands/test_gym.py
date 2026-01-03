"""Tests for gym CLI commands."""

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGymRun:
    """Tests for gym_run command."""

    @pytest.fixture
    def gym_run_args(self):
        """Create basic arguments for gym run command."""
        return Namespace(
            tokenizer="gpt2",
            mock=True,
            num_episodes=5,
            steps_per_episode=10,
            difficulty_min=1,
            difficulty_max=5,
            success_rate=0.8,
            seed=42,
            buffer_size=1000,
            host="localhost",
            port=8023,
            transport="telnet",
            output_mode="json",
            timeout=10.0,
            retries=3,
            max_samples=None,
            output=None,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "test output"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_gym_sample(self):
        """Create a mock gym sample."""
        sample = MagicMock()
        sample.episode_id = "episode_1"
        sample.step = 1
        sample.reward = 1.0
        sample.done = False
        return sample

    @pytest.fixture
    def mock_replay_buffer(self, mock_gym_sample):
        """Create a mock replay buffer."""
        buffer = MagicMock()
        buffer.size = 10
        buffer.success_rate = 0.75
        buffer.mean_difficulty = 3.5
        buffer.mean_reward = 0.8
        buffer.add = MagicMock()
        buffer.to_dict = MagicMock(
            return_value={
                "size": 10,
                "success_rate": 0.75,
                "samples": [],
            }
        )
        return buffer

    def test_gym_run_mock_basic(
        self, gym_run_args, mock_tokenizer, mock_gym_sample, mock_replay_buffer, capsys
    ):
        """Test basic gym run with mock stream."""
        from chuk_lazarus.cli.commands.gym import gym_run

        # Mock all the dependencies - patch at their source location
        with (
            patch(
                "chuk_lazarus.utils.tokenizer_loader.load_tokenizer", return_value=mock_tokenizer
            ),
            patch(
                "chuk_lazarus.data.batching.streaming.ReplayBuffer", return_value=mock_replay_buffer
            ),
            patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
            patch("chuk_lazarus.data.batching.streaming.MockGymStream") as mock_stream_cls,
        ):
            # Create mock stream
            mock_stream = MagicMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            # Mock async iteration - return a few samples
            async def async_gen():
                for i in range(5):
                    yield mock_gym_sample

            mock_stream.__aiter__ = lambda self: async_gen()
            mock_stream_cls.return_value = mock_stream

            gym_run(gym_run_args)

            captured = capsys.readouterr()
            assert "Gym Episode Streaming" in captured.out
            assert "Summary" in captured.out

    def test_gym_run_mock_with_max_samples(
        self, gym_run_args, mock_tokenizer, mock_gym_sample, mock_replay_buffer, capsys
    ):
        """Test gym run with max samples limit."""
        from chuk_lazarus.cli.commands.gym import gym_run

        gym_run_args.max_samples = 3

        with (
            patch(
                "chuk_lazarus.utils.tokenizer_loader.load_tokenizer", return_value=mock_tokenizer
            ),
            patch(
                "chuk_lazarus.data.batching.streaming.ReplayBuffer", return_value=mock_replay_buffer
            ),
            patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
            patch("chuk_lazarus.data.batching.streaming.MockGymStream") as mock_stream_cls,
        ):
            mock_stream = MagicMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            # Generate more samples than max_samples
            async def async_gen():
                for i in range(10):
                    yield mock_gym_sample

            mock_stream.__aiter__ = lambda self: async_gen()
            mock_stream_cls.return_value = mock_stream

            gym_run(gym_run_args)

            captured = capsys.readouterr()
            assert "Summary" in captured.out

    def test_gym_run_mock_with_output(
        self, gym_run_args, mock_tokenizer, mock_gym_sample, mock_replay_buffer, capsys
    ):
        """Test gym run saving output to file."""
        from chuk_lazarus.cli.commands.gym import gym_run

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            gym_run_args.output = output_path

            with (
                patch(
                    "chuk_lazarus.utils.tokenizer_loader.load_tokenizer",
                    return_value=mock_tokenizer,
                ),
                patch(
                    "chuk_lazarus.data.batching.streaming.ReplayBuffer",
                    return_value=mock_replay_buffer,
                ),
                patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
                patch("chuk_lazarus.data.batching.streaming.MockGymStream") as mock_stream_cls,
            ):
                mock_stream = MagicMock()
                mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
                mock_stream.__aexit__ = AsyncMock(return_value=None)

                async def async_gen():
                    for i in range(5):
                        yield mock_gym_sample

                mock_stream.__aiter__ = lambda self: async_gen()
                mock_stream_cls.return_value = mock_stream

                gym_run(gym_run_args)

                # Verify file was created and contains JSON
                assert Path(output_path).exists()
                with open(output_path) as f:
                    data = json.load(f)
                    assert isinstance(data, dict)

                captured = capsys.readouterr()
                assert "Buffer saved to:" in captured.out
        finally:
            # Cleanup
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_gym_run_real_stream(
        self, gym_run_args, mock_tokenizer, mock_gym_sample, mock_replay_buffer, capsys
    ):
        """Test gym run with real (non-mock) stream configuration."""
        from chuk_lazarus.cli.commands.gym import gym_run

        gym_run_args.mock = False

        with (
            patch(
                "chuk_lazarus.utils.tokenizer_loader.load_tokenizer", return_value=mock_tokenizer
            ),
            patch(
                "chuk_lazarus.data.batching.streaming.ReplayBuffer", return_value=mock_replay_buffer
            ),
            patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
            patch("chuk_lazarus.data.batching.streaming.GymTransport"),
            patch("chuk_lazarus.data.batching.streaming.GymOutputMode"),
            patch("chuk_lazarus.data.batching.streaming.GymConfig") as mock_config,
            patch("chuk_lazarus.data.batching.streaming.GymEpisodeStream") as mock_stream_cls,
        ):
            mock_stream = MagicMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            async def async_gen():
                for i in range(3):
                    yield mock_gym_sample

            mock_stream.__aiter__ = lambda self: async_gen()
            mock_stream_cls.return_value = mock_stream

            gym_run(gym_run_args)

            # Verify the real stream was created with config
            mock_config.assert_called_once()
            mock_stream_cls.assert_called_once()

            captured = capsys.readouterr()
            assert "Summary" in captured.out

    def test_gym_run_sample_counting(
        self, gym_run_args, mock_tokenizer, mock_replay_buffer, capsys
    ):
        """Test that samples are counted correctly including print statements."""
        from chuk_lazarus.cli.commands.gym import gym_run

        with (
            patch(
                "chuk_lazarus.utils.tokenizer_loader.load_tokenizer", return_value=mock_tokenizer
            ),
            patch(
                "chuk_lazarus.data.batching.streaming.ReplayBuffer", return_value=mock_replay_buffer
            ),
            patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
            patch("chuk_lazarus.data.batching.streaming.MockGymStream") as mock_stream_cls,
        ):
            mock_stream = MagicMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=None)

            # Generate exactly 100 samples to trigger progress print
            async def async_gen():
                for i in range(100):
                    sample = MagicMock()
                    sample.episode_id = f"episode_{i // 10}"
                    yield sample

            mock_stream.__aiter__ = lambda self: async_gen()
            mock_stream_cls.return_value = mock_stream

            gym_run(gym_run_args)

            captured = capsys.readouterr()
            # Should print progress at 100 samples
            assert "Samples:" in captured.out or "Summary" in captured.out

    def test_gym_run_output_directory_creation(
        self, gym_run_args, mock_tokenizer, mock_gym_sample, mock_replay_buffer
    ):
        """Test that output directory is created if it doesn't exist."""
        from chuk_lazarus.cli.commands.gym import gym_run

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "output.json"
            gym_run_args.output = str(output_path)

            with (
                patch(
                    "chuk_lazarus.utils.tokenizer_loader.load_tokenizer",
                    return_value=mock_tokenizer,
                ),
                patch(
                    "chuk_lazarus.data.batching.streaming.ReplayBuffer",
                    return_value=mock_replay_buffer,
                ),
                patch("chuk_lazarus.data.batching.streaming.ReplayBufferConfig"),
                patch("chuk_lazarus.data.batching.streaming.MockGymStream") as mock_stream_cls,
            ):
                mock_stream = MagicMock()
                mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
                mock_stream.__aexit__ = AsyncMock(return_value=None)

                async def async_gen():
                    yield mock_gym_sample

                mock_stream.__aiter__ = lambda self: async_gen()
                mock_stream_cls.return_value = mock_stream

                gym_run(gym_run_args)

                # Verify directory was created
                assert output_path.parent.exists()
                assert output_path.exists()


class TestBenchPipeline:
    """Tests for bench_pipeline command."""

    @pytest.fixture
    def bench_args(self):
        """Create basic arguments for bench_pipeline command."""
        return Namespace(
            dataset=None,
            tokenizer="gpt2",
            max_samples=100,
            num_samples=1000,
            seed=42,
            max_length=512,
            bucket_edges="64,128,256,512",
            token_budget=4096,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        return tokenizer

    def test_bench_pipeline_synthetic_data(self, bench_args, capsys):
        """Test benchmark with synthetic data (no dataset)."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with (
            patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
            patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
            patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
            patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
            patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
            patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
            patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
        ):
            # Set up mocks
            mock_histogram = MagicMock()
            mock_histogram.min_length = 32
            mock_histogram.max_length = 512
            mock_histogram.mean_length = 200.5
            mock_histogram.median_length = 180
            mock_histogram.p90 = 400
            mock_histogram.p99 = 500
            mock_histogram.to_ascii.return_value = "Histogram"
            mock_hist.return_value = mock_histogram

            mock_analysis = MagicMock()
            mock_analysis.overall_efficiency = 0.85
            mock_analysis.to_ascii.return_value = "Bucket Analysis"
            mock_bucket.return_value = mock_analysis

            mock_plan = MagicMock()
            mock_plan.total_microbatches = 50
            mock_plan.fingerprint = "test-fingerprint"
            mock_epoch = MagicMock()
            mock_epoch.total_tokens = 200000
            mock_epoch.total_samples = 1000
            mock_microbatch = MagicMock()
            mock_microbatch.samples = ["s1", "s2"]
            mock_epoch.microbatches = [mock_microbatch] * 50
            mock_plan.get_epoch.return_value = mock_epoch

            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_plan)
            mock_builder_cls.return_value = mock_builder

            mock_bucket_spec_inst = MagicMock()
            mock_bucket_spec_inst.get_bucket_id.return_value = 0
            mock_bucket_spec_inst.get_bucket_range.return_value = (0, 128)
            mock_bucket_spec.return_value = mock_bucket_spec_inst

            mock_packed = [MagicMock(input_ids=[1, 2, 3] * 100)]
            mock_pack.return_value = mock_packed

            mock_metrics = MagicMock()
            mock_metrics.packing_ratio = 1.5
            mock_metrics.efficiency = 0.9
            mock_pack_metrics.return_value = mock_metrics

            mock_eff_report = MagicMock()
            mock_eff_report.recommendations = ["Use packing"]
            mock_eff_report.to_ascii.return_value = "Report"
            mock_report.return_value = mock_eff_report

            bench_pipeline(bench_args)

            captured = capsys.readouterr()
            assert "LAZARUS PIPELINE BENCHMARK" in captured.out
            assert "BENCHMARK SUMMARY" in captured.out
            assert "KEY INSIGHT" in captured.out

    def test_bench_pipeline_with_dataset(self, bench_args, mock_tokenizer, capsys):
        """Test benchmark with real dataset file."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test data
            for i in range(10):
                json.dump({"text": f"Sample text {i}", "id": f"sample_{i}"}, f)
                f.write("\n")
            dataset_path = f.name

        try:
            bench_args.dataset = dataset_path

            with (
                patch(
                    "chuk_lazarus.utils.tokenizer_loader.load_tokenizer",
                    return_value=mock_tokenizer,
                ),
                patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
                patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
                patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
                patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
                patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
                patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
                patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
            ):
                # Set up basic mocks
                mock_histogram = MagicMock()
                mock_histogram.min_length = 32
                mock_histogram.max_length = 100
                mock_histogram.mean_length = 50.5
                mock_histogram.median_length = 48
                mock_histogram.p90 = 80
                mock_histogram.p99 = 95
                mock_histogram.to_ascii.return_value = "Histogram"
                mock_hist.return_value = mock_histogram

                mock_analysis = MagicMock()
                mock_analysis.overall_efficiency = 0.75
                mock_analysis.to_ascii.return_value = "Bucket Analysis"
                mock_bucket.return_value = mock_analysis

                mock_plan = MagicMock()
                mock_plan.total_microbatches = 5
                mock_plan.fingerprint = "dataset-fingerprint"
                mock_epoch = MagicMock()
                mock_epoch.total_tokens = 500
                mock_epoch.total_samples = 10
                mock_microbatch = MagicMock()
                mock_microbatch.samples = ["s1", "s2"]
                mock_epoch.microbatches = [mock_microbatch] * 5
                mock_plan.get_epoch.return_value = mock_epoch

                mock_builder = MagicMock()
                mock_builder.build = AsyncMock(return_value=mock_plan)
                mock_builder_cls.return_value = mock_builder

                mock_bucket_spec_inst = MagicMock()
                mock_bucket_spec_inst.get_bucket_id.return_value = 0
                mock_bucket_spec_inst.get_bucket_range.return_value = (0, 64)
                mock_bucket_spec.return_value = mock_bucket_spec_inst

                mock_packed = [MagicMock(input_ids=[1] * 50)]
                mock_pack.return_value = mock_packed

                mock_metrics = MagicMock()
                mock_metrics.packing_ratio = 1.2
                mock_metrics.efficiency = 0.85
                mock_pack_metrics.return_value = mock_metrics

                mock_eff_report = MagicMock()
                mock_eff_report.recommendations = []
                mock_eff_report.to_ascii.return_value = "Report"
                mock_report.return_value = mock_eff_report

                bench_pipeline(bench_args)

                captured = capsys.readouterr()
                assert "Dataset:" in captured.out
                assert "Tokenizing dataset..." in captured.out
        finally:
            Path(dataset_path).unlink()

    def test_bench_pipeline_high_packing_ratio(self, bench_args, capsys):
        """Test KEY INSIGHT message when packing ratio > 1.3."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with (
            patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
            patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
            patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
            patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
            patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
            patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
            patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
        ):
            # Set up mocks with high packing ratio
            mock_histogram = MagicMock()
            mock_histogram.min_length = 32
            mock_histogram.max_length = 512
            mock_histogram.mean_length = 200.5
            mock_histogram.median_length = 180
            mock_histogram.p90 = 400
            mock_histogram.p99 = 500
            mock_histogram.to_ascii.return_value = "H"
            mock_hist.return_value = mock_histogram

            mock_analysis = MagicMock()
            mock_analysis.overall_efficiency = 0.75  # < 0.85
            mock_analysis.to_ascii.return_value = "A"
            mock_bucket.return_value = mock_analysis

            mock_plan = MagicMock()
            mock_plan.total_microbatches = 50
            mock_plan.fingerprint = "fp"
            mock_epoch = MagicMock()
            mock_epoch.total_tokens = 200000
            mock_epoch.total_samples = 1000
            mock_microbatch = MagicMock()
            mock_microbatch.samples = ["s1", "s2"]
            mock_epoch.microbatches = [mock_microbatch] * 50
            mock_plan.get_epoch.return_value = mock_epoch

            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_plan)
            mock_builder_cls.return_value = mock_builder

            mock_bucket_spec_inst = MagicMock()
            mock_bucket_spec_inst.get_bucket_id.return_value = 0
            mock_bucket_spec_inst.get_bucket_range.return_value = (0, 128)
            mock_bucket_spec.return_value = mock_bucket_spec_inst

            mock_packed = [MagicMock(input_ids=[1] * 100)]
            mock_pack.return_value = mock_packed

            # HIGH packing ratio
            mock_metrics = MagicMock()
            mock_metrics.packing_ratio = 1.5  # > 1.3
            mock_metrics.efficiency = 0.9
            mock_pack_metrics.return_value = mock_metrics

            mock_eff_report = MagicMock()
            mock_eff_report.recommendations = []
            mock_eff_report.to_ascii.return_value = "R"
            mock_report.return_value = mock_eff_report

            bench_pipeline(bench_args)

            captured = capsys.readouterr()
            assert "Packing recommended" in captured.out

    def test_bench_pipeline_high_bucket_efficiency(self, bench_args, capsys):
        """Test KEY INSIGHT message when bucket efficiency > 0.85."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with (
            patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
            patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
            patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
            patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
            patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
            patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
            patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
        ):
            mock_histogram = MagicMock()
            mock_histogram.min_length = 32
            mock_histogram.max_length = 512
            mock_histogram.mean_length = 200.5
            mock_histogram.median_length = 180
            mock_histogram.p90 = 400
            mock_histogram.p99 = 500
            mock_histogram.to_ascii.return_value = "H"
            mock_hist.return_value = mock_histogram

            # HIGH bucket efficiency
            mock_analysis = MagicMock()
            mock_analysis.overall_efficiency = 0.90  # > 0.85
            mock_analysis.to_ascii.return_value = "A"
            mock_bucket.return_value = mock_analysis

            mock_plan = MagicMock()
            mock_plan.total_microbatches = 50
            mock_plan.fingerprint = "fp"
            mock_epoch = MagicMock()
            mock_epoch.total_tokens = 200000
            mock_epoch.total_samples = 1000
            mock_microbatch = MagicMock()
            mock_microbatch.samples = ["s1", "s2"]
            mock_epoch.microbatches = [mock_microbatch] * 50
            mock_plan.get_epoch.return_value = mock_epoch

            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_plan)
            mock_builder_cls.return_value = mock_builder

            mock_bucket_spec_inst = MagicMock()
            mock_bucket_spec_inst.get_bucket_id.return_value = 0
            mock_bucket_spec_inst.get_bucket_range.return_value = (0, 128)
            mock_bucket_spec.return_value = mock_bucket_spec_inst

            mock_packed = [MagicMock(input_ids=[1] * 100)]
            mock_pack.return_value = mock_packed

            # Low packing ratio
            mock_metrics = MagicMock()
            mock_metrics.packing_ratio = 1.1  # < 1.3
            mock_metrics.efficiency = 0.9
            mock_pack_metrics.return_value = mock_metrics

            mock_eff_report = MagicMock()
            mock_eff_report.recommendations = []
            mock_eff_report.to_ascii.return_value = "R"
            mock_report.return_value = mock_eff_report

            bench_pipeline(bench_args)

            captured = capsys.readouterr()
            assert "Pad-to-bucket is sufficient" in captured.out

    def test_bench_pipeline_dataset_max_samples_break(self, bench_args, mock_tokenizer, capsys):
        """Test dataset processing stops early with max_samples."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write MORE samples than max_samples to ensure break is hit
            for i in range(200):
                json.dump({"text": f"Sample {i}"}, f)
                f.write("\n")
            dataset_path = f.name

        try:
            bench_args.dataset = dataset_path
            bench_args.max_samples = 5  # Very small limit to hit break

            with (
                patch(
                    "chuk_lazarus.utils.tokenizer_loader.load_tokenizer",
                    return_value=mock_tokenizer,
                ),
                patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
                patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
                patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
                patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
                patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
                patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
                patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
            ):
                # Minimal mocks
                mock_histogram = MagicMock()
                mock_histogram.min_length = 5
                mock_histogram.max_length = 5
                mock_histogram.mean_length = 5.0
                mock_histogram.median_length = 5
                mock_histogram.p90 = 5
                mock_histogram.p99 = 5
                mock_histogram.to_ascii.return_value = "H"
                mock_hist.return_value = mock_histogram

                mock_analysis = MagicMock()
                mock_analysis.overall_efficiency = 0.9
                mock_analysis.to_ascii.return_value = "A"
                mock_bucket.return_value = mock_analysis

                mock_plan = MagicMock()
                mock_plan.total_microbatches = 1
                mock_plan.fingerprint = "fp"
                mock_epoch = MagicMock()
                mock_epoch.total_tokens = 25
                mock_epoch.total_samples = 5
                mock_microbatch = MagicMock()
                mock_microbatch.samples = ["s1"]
                mock_epoch.microbatches = [mock_microbatch]
                mock_plan.get_epoch.return_value = mock_epoch

                mock_builder = MagicMock()
                mock_builder.build = AsyncMock(return_value=mock_plan)
                mock_builder_cls.return_value = mock_builder

                mock_bucket_spec_inst = MagicMock()
                mock_bucket_spec_inst.get_bucket_id.return_value = 0
                mock_bucket_spec_inst.get_bucket_range.return_value = (0, 64)
                mock_bucket_spec.return_value = mock_bucket_spec_inst

                mock_packed = [MagicMock(input_ids=[1] * 5)]
                mock_pack.return_value = mock_packed

                mock_metrics = MagicMock()
                mock_metrics.packing_ratio = 1.0
                mock_metrics.efficiency = 1.0
                mock_pack_metrics.return_value = mock_metrics

                mock_eff_report = MagicMock()
                mock_eff_report.recommendations = []
                mock_eff_report.to_ascii.return_value = "R"
                mock_report.return_value = mock_eff_report

                bench_pipeline(bench_args)

                # Verify only max_samples were tokenized (not all 200)
                assert mock_tokenizer.encode.call_count == 5

                captured = capsys.readouterr()
                assert "BENCHMARK SUMMARY" in captured.out
        finally:
            Path(dataset_path).unlink()

    def test_bench_pipeline_low_bucket_efficiency(self, bench_args, capsys):
        """Test KEY INSIGHT message when bucket efficiency is low."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        with (
            patch("chuk_lazarus.data.batching.compute_length_histogram") as mock_hist,
            patch("chuk_lazarus.data.batching.analyze_bucket_efficiency") as mock_bucket,
            patch("chuk_lazarus.data.batching.BatchPlanBuilder") as mock_builder_cls,
            patch("chuk_lazarus.data.batching.BucketSpec") as mock_bucket_spec,
            patch("chuk_lazarus.data.batching.pack_sequences") as mock_pack,
            patch("chuk_lazarus.data.batching.compute_packing_metrics") as mock_pack_metrics,
            patch("chuk_lazarus.data.batching.create_efficiency_report") as mock_report,
        ):
            mock_histogram = MagicMock()
            mock_histogram.min_length = 32
            mock_histogram.max_length = 512
            mock_histogram.mean_length = 200.5
            mock_histogram.median_length = 180
            mock_histogram.p90 = 400
            mock_histogram.p99 = 500
            mock_histogram.to_ascii.return_value = "H"
            mock_hist.return_value = mock_histogram

            # LOW bucket efficiency
            mock_analysis = MagicMock()
            mock_analysis.overall_efficiency = 0.65  # < 0.85
            mock_analysis.to_ascii.return_value = "A"
            mock_bucket.return_value = mock_analysis

            mock_plan = MagicMock()
            mock_plan.total_microbatches = 50
            mock_plan.fingerprint = "fp"
            mock_epoch = MagicMock()
            mock_epoch.total_tokens = 200000
            mock_epoch.total_samples = 1000
            mock_microbatch = MagicMock()
            mock_microbatch.samples = ["s1", "s2"]
            mock_epoch.microbatches = [mock_microbatch] * 50
            mock_plan.get_epoch.return_value = mock_epoch

            mock_builder = MagicMock()
            mock_builder.build = AsyncMock(return_value=mock_plan)
            mock_builder_cls.return_value = mock_builder

            mock_bucket_spec_inst = MagicMock()
            mock_bucket_spec_inst.get_bucket_id.return_value = 0
            mock_bucket_spec_inst.get_bucket_range.return_value = (0, 128)
            mock_bucket_spec.return_value = mock_bucket_spec_inst

            mock_packed = [MagicMock(input_ids=[1] * 100)]
            mock_pack.return_value = mock_packed

            # Low packing ratio
            mock_metrics = MagicMock()
            mock_metrics.packing_ratio = 1.1  # < 1.3
            mock_metrics.efficiency = 0.9
            mock_pack_metrics.return_value = mock_metrics

            mock_eff_report = MagicMock()
            mock_eff_report.recommendations = []
            mock_eff_report.to_ascii.return_value = "R"
            mock_report.return_value = mock_eff_report

            bench_pipeline(bench_args)

            captured = capsys.readouterr()
            assert "Consider adjusting bucket edges" in captured.out


class TestGymInfo:
    """Tests for gym_info command."""

    def test_gym_info_basic(self, capsys):
        """Test basic gym_info command."""
        from chuk_lazarus.cli.commands.gym import gym_info

        args = Namespace()

        with (
            patch("chuk_lazarus.data.batching.streaming.GymTransport") as mock_transport,
            patch("chuk_lazarus.data.batching.streaming.GymOutputMode") as mock_mode,
        ):
            # Mock enum iteration
            mock_transport.__iter__ = lambda self: iter(
                [
                    MagicMock(value="telnet"),
                    MagicMock(value="http"),
                    MagicMock(value="websocket"),
                ]
            )
            mock_mode.__iter__ = lambda self: iter(
                [
                    MagicMock(value="json"),
                    MagicMock(value="text"),
                ]
            )

            gym_info(args)

            captured = capsys.readouterr()
            assert "Gym Stream Configuration" in captured.out
            assert "Supported Transports:" in captured.out
            assert "Supported Output Modes:" in captured.out
            assert "Default Configuration:" in captured.out
            assert "Example Usage:" in captured.out

    def test_gym_info_displays_defaults(self, capsys):
        """Test that gym_info displays default configuration values."""
        from chuk_lazarus.cli.commands.gym import gym_info

        args = Namespace()

        with (
            patch("chuk_lazarus.data.batching.streaming.GymTransport") as mock_transport,
            patch("chuk_lazarus.data.batching.streaming.GymOutputMode") as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter([])
            mock_mode.__iter__ = lambda self: iter([])

            gym_info(args)

            captured = capsys.readouterr()
            assert "localhost" in captured.out
            assert "8023" in captured.out
            assert "telnet" in captured.out
            assert "json" in captured.out
            assert "10.0s" in captured.out
            assert "3" in captured.out

    def test_gym_info_displays_examples(self, capsys):
        """Test that gym_info displays usage examples."""
        from chuk_lazarus.cli.commands.gym import gym_info

        args = Namespace()

        with (
            patch("chuk_lazarus.data.batching.streaming.GymTransport") as mock_transport,
            patch("chuk_lazarus.data.batching.streaming.GymOutputMode") as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter([])
            mock_mode.__iter__ = lambda self: iter([])

            gym_info(args)

            captured = capsys.readouterr()
            assert "lazarus gym run" in captured.out
            assert "--mock" in captured.out
            assert "--tokenizer gpt2" in captured.out
            assert "--output buffer.json" in captured.out


class TestGymImports:
    """Tests for gym module imports."""

    def test_import_gym_functions(self):
        """Test that all gym functions can be imported."""
        from chuk_lazarus.cli.commands.gym import bench_pipeline, gym_info, gym_run

        assert callable(gym_run)
        assert callable(bench_pipeline)
        assert callable(gym_info)

    def test_module_has_logger(self):
        """Test that module has logger configured."""
        import chuk_lazarus.cli.commands.gym as gym_module

        assert hasattr(gym_module, "logger")
        assert gym_module.logger.name == "chuk_lazarus.cli.commands.gym"

    def test_module_docstring(self):
        """Test that module has docstring."""
        import chuk_lazarus.cli.commands.gym as gym_module

        assert gym_module.__doc__ is not None
        assert "Gym" in gym_module.__doc__ or "benchmark" in gym_module.__doc__
