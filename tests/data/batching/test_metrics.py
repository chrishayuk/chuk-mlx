"""Tests for batching metrics."""

from chuk_lazarus.data.batching import BatchMetrics, BatchShapeHistogram
from chuk_lazarus.data.batching.core.buckets import BucketId


class TestBatchShapeHistogram:
    """Tests for BatchShapeHistogram model."""

    def test_create_empty(self):
        """Test creating empty histogram."""
        hist = BatchShapeHistogram()
        assert hist.total_batches == 0
        assert hist.total_samples == 0

    def test_record_batch(self):
        """Test recording batches."""
        hist = BatchShapeHistogram()

        hist.record_batch(BucketId(0), 4)
        hist.record_batch(BucketId(0), 4)
        hist.record_batch(BucketId(1), 2)

        assert hist.bucket_counts[0] == 2
        assert hist.bucket_counts[1] == 1
        assert hist.bucket_samples[0] == 8
        assert hist.bucket_samples[1] == 2

    def test_total_batches(self):
        """Test total batch count."""
        hist = BatchShapeHistogram()
        hist.record_batch(BucketId(0), 4)
        hist.record_batch(BucketId(1), 2)
        hist.record_batch(BucketId(2), 3)

        assert hist.total_batches == 3

    def test_total_samples(self):
        """Test total sample count."""
        hist = BatchShapeHistogram()
        hist.record_batch(BucketId(0), 4)
        hist.record_batch(BucketId(1), 2)

        assert hist.total_samples == 6

    def test_bucket_fraction(self):
        """Test bucket fraction calculation."""
        hist = BatchShapeHistogram()
        hist.record_batch(BucketId(0), 6)
        hist.record_batch(BucketId(1), 4)

        assert abs(hist.bucket_fraction(BucketId(0)) - 0.6) < 0.001
        assert abs(hist.bucket_fraction(BucketId(1)) - 0.4) < 0.001

    def test_bucket_fraction_empty(self):
        """Test bucket fraction with empty histogram."""
        hist = BatchShapeHistogram()
        assert hist.bucket_fraction(BucketId(0)) == 0.0


class TestBatchMetrics:
    """Tests for BatchMetrics model."""

    def test_create_empty(self):
        """Test creating empty metrics."""
        metrics = BatchMetrics()
        assert metrics.total_tokens == 0
        assert metrics.padded_tokens == 0
        assert metrics.total_samples == 0

    def test_record_sample(self):
        """Test recording samples."""
        metrics = BatchMetrics()

        metrics.record_sample(
            bucket_id=BucketId(0),
            length=100,
            loss_tokens=80,
            bucket_max_length=128,
        )

        assert metrics.total_samples == 1
        assert metrics.total_tokens == 100
        assert metrics.padded_tokens == 128
        assert metrics.loss_tokens == 80

    def test_record_multiple_samples(self):
        """Test recording multiple samples."""
        metrics = BatchMetrics()

        metrics.record_sample(BucketId(0), 100, 80, 128)
        metrics.record_sample(BucketId(0), 50, 40, 128)
        metrics.record_sample(BucketId(1), 200, 150, 256)

        assert metrics.total_samples == 3
        assert metrics.total_tokens == 350
        assert metrics.padded_tokens == 128 + 128 + 256
        assert metrics.loss_tokens == 270

    def test_record_batch(self):
        """Test recording batches."""
        metrics = BatchMetrics()

        metrics.record_batch(BucketId(0), 4)
        metrics.record_batch(BucketId(1), 2)

        assert metrics.total_batches == 2

    def test_record_skip(self):
        """Test recording skipped samples."""
        metrics = BatchMetrics()

        metrics.record_skip()
        metrics.record_skip()

        assert metrics.skipped_samples == 2

    def test_record_time(self):
        """Test recording time."""
        metrics = BatchMetrics()

        metrics.record_time(1.5)
        metrics.record_time(2.5)

        assert metrics.total_time_seconds == 4.0

    def test_padding_tokens(self):
        """Test padding tokens calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 100, 128)

        assert metrics.padding_tokens == 28

    def test_padding_waste(self):
        """Test padding waste calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 100, 128)

        # 28 / 128 = 0.21875
        assert abs(metrics.padding_waste - 0.21875) < 0.001

    def test_efficiency(self):
        """Test efficiency calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 100, 128)

        # 100 / 128 = 0.78125
        assert abs(metrics.efficiency - 0.78125) < 0.001

    def test_loss_efficiency(self):
        """Test loss efficiency calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 80, 128)

        # 80 / 128 = 0.625
        assert abs(metrics.loss_efficiency - 0.625) < 0.001

    def test_tokens_per_second(self):
        """Test throughput calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 1000, 800, 1024)
        metrics.record_time(2.0)

        assert metrics.tokens_per_second == 500.0

    def test_effective_tokens_per_second(self):
        """Test effective throughput calculation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 1000, 800, 1024)
        metrics.record_time(2.0)

        assert metrics.effective_tokens_per_second == 400.0

    def test_samples_per_second(self):
        """Test sample throughput calculation."""
        metrics = BatchMetrics()
        for _ in range(10):
            metrics.record_sample(BucketId(0), 100, 100, 128)
        metrics.record_time(2.0)

        assert metrics.samples_per_second == 5.0

    def test_avg_batch_size(self):
        """Test average batch size calculation."""
        metrics = BatchMetrics()
        for _ in range(12):
            metrics.record_sample(BucketId(0), 100, 100, 128)
        metrics.record_batch(BucketId(0), 4)
        metrics.record_batch(BucketId(0), 4)
        metrics.record_batch(BucketId(0), 4)

        assert metrics.avg_batch_size == 4.0

    def test_skip_rate(self):
        """Test skip rate calculation."""
        metrics = BatchMetrics()
        for _ in range(8):
            metrics.record_sample(BucketId(0), 100, 100, 128)
        for _ in range(2):
            metrics.record_skip()

        # 2 / 10 = 0.2
        assert abs(metrics.skip_rate - 0.2) < 0.001

    def test_bucket_stats(self):
        """Test per-bucket stats."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 80, 128)
        metrics.record_sample(BucketId(0), 120, 100, 128)
        metrics.record_sample(BucketId(1), 200, 150, 256)

        assert 0 in metrics.bucket_stats
        assert 1 in metrics.bucket_stats
        assert metrics.bucket_stats[0].sample_count == 2
        assert metrics.bucket_stats[1].sample_count == 1

    def test_summary(self):
        """Test summary generation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 80, 128)
        metrics.record_batch(BucketId(0), 1)
        metrics.record_time(1.0)

        summary = metrics.summary()

        assert "total_samples" in summary
        assert "padding_waste" in summary
        assert "efficiency" in summary
        assert summary["total_samples"] == 1

    def test_bucket_summary(self):
        """Test bucket summary generation."""
        metrics = BatchMetrics()
        metrics.record_sample(BucketId(0), 100, 80, 128)
        metrics.record_sample(BucketId(1), 200, 150, 256)

        bucket_summary = metrics.bucket_summary()

        assert len(bucket_summary) == 2
        assert bucket_summary[0]["bucket_id"] == 0
        assert bucket_summary[1]["bucket_id"] == 1

    def test_zero_division_protection(self):
        """Test that empty metrics don't cause division by zero."""
        metrics = BatchMetrics()

        assert metrics.padding_waste == 0.0
        assert metrics.efficiency == 1.0
        assert metrics.loss_efficiency == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.effective_tokens_per_second == 0.0
        assert metrics.samples_per_second == 0.0
        assert metrics.avg_batch_size == 0.0
        assert metrics.skip_rate == 0.0
