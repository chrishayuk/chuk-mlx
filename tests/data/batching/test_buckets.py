"""Tests for bucket configuration."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import BucketSpec, BucketStats


class TestBucketSpec:
    """Tests for BucketSpec model."""

    def test_create_basic(self):
        """Test creating bucket spec with basic config."""
        spec = BucketSpec(
            edges=(128, 256, 512),
            overflow_max=1024,
        )
        assert spec.num_buckets == 4
        assert spec.edges == (128, 256, 512)
        assert spec.overflow_max == 1024

    def test_create_from_list(self):
        """Test creating with list (converts to tuple)."""
        spec = BucketSpec(
            edges=[128, 256, 512],
            overflow_max=1024,
        )
        assert isinstance(spec.edges, tuple)
        assert spec.edges == (128, 256, 512)

    def test_default(self):
        """Test default bucket spec."""
        spec = BucketSpec.default()
        assert spec.edges == (128, 256, 512, 1024)
        assert spec.overflow_max == 2048
        assert spec.num_buckets == 5

    def test_from_max_length(self):
        """Test creating from max length."""
        spec = BucketSpec.from_max_length(1024, num_buckets=4)
        assert len(spec.edges) == 3
        assert spec.overflow_max == 1024
        assert spec.num_buckets == 4

    def test_edges_must_be_ascending(self):
        """Test that edges must be strictly ascending."""
        with pytest.raises(ValidationError, match="strictly ascending"):
            BucketSpec(edges=(256, 128, 512), overflow_max=1024)

        with pytest.raises(ValidationError, match="strictly ascending"):
            BucketSpec(edges=(128, 128, 512), overflow_max=1024)

    def test_overflow_must_exceed_last_edge(self):
        """Test that overflow_max must exceed last edge."""
        with pytest.raises(ValidationError, match="overflow_max"):
            BucketSpec(edges=(128, 256, 512), overflow_max=512)

        with pytest.raises(ValidationError, match="overflow_max"):
            BucketSpec(edges=(128, 256, 512), overflow_max=400)

    def test_immutable(self):
        """Test that spec is immutable (frozen)."""
        spec = BucketSpec(edges=(128, 256), overflow_max=512)
        with pytest.raises(ValidationError):
            spec.overflow_max = 1024

    def test_get_bucket_id(self):
        """Test bucket ID assignment."""
        spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        # First bucket: 1-128
        assert spec.get_bucket_id(1) == 0
        assert spec.get_bucket_id(64) == 0
        assert spec.get_bucket_id(128) == 0

        # Second bucket: 129-256
        assert spec.get_bucket_id(129) == 1
        assert spec.get_bucket_id(200) == 1
        assert spec.get_bucket_id(256) == 1

        # Third bucket: 257-512
        assert spec.get_bucket_id(257) == 2
        assert spec.get_bucket_id(512) == 2

        # Overflow bucket: 513-1024
        assert spec.get_bucket_id(513) == 3
        assert spec.get_bucket_id(1024) == 3

    def test_get_bucket_max_length(self):
        """Test getting bucket max length."""
        spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        assert spec.get_bucket_max_length(0) == 128
        assert spec.get_bucket_max_length(1) == 256
        assert spec.get_bucket_max_length(2) == 512
        assert spec.get_bucket_max_length(3) == 1024

    def test_get_bucket_range(self):
        """Test getting bucket range."""
        spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        assert spec.get_bucket_range(0) == (1, 128)
        assert spec.get_bucket_range(1) == (129, 256)
        assert spec.get_bucket_range(2) == (257, 512)
        assert spec.get_bucket_range(3) == (513, 1024)

    def test_is_overflow(self):
        """Test overflow bucket detection."""
        spec = BucketSpec(edges=(128, 256), overflow_max=512)

        assert not spec.is_overflow(0)
        assert not spec.is_overflow(1)
        assert spec.is_overflow(2)

    def test_should_skip(self):
        """Test skip detection."""
        spec = BucketSpec(edges=(128, 256), overflow_max=512, min_length=10)

        # Too short
        assert spec.should_skip(5)
        assert spec.should_skip(9)
        assert not spec.should_skip(10)

        # Too long
        assert not spec.should_skip(512)
        assert spec.should_skip(513)
        assert spec.should_skip(1000)

    def test_overflow_bucket_id(self):
        """Test overflow bucket ID property."""
        spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)
        assert spec.overflow_bucket_id == 3

        spec2 = BucketSpec(edges=(128,), overflow_max=256)
        assert spec2.overflow_bucket_id == 1


class TestBucketStats:
    """Tests for BucketStats model."""

    def test_create_empty(self):
        """Test creating empty stats."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)
        assert stats.sample_count == 0
        assert stats.total_tokens == 0
        assert stats.padded_tokens == 0

    def test_add_sample(self):
        """Test adding samples."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)

        stats.add_sample(100)
        assert stats.sample_count == 1
        assert stats.total_tokens == 100
        assert stats.padded_tokens == 128

        stats.add_sample(80)
        assert stats.sample_count == 2
        assert stats.total_tokens == 180
        assert stats.padded_tokens == 256

    def test_padding_tokens(self):
        """Test padding token calculation."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)
        stats.add_sample(100)
        stats.add_sample(80)

        assert stats.padding_tokens == 256 - 180  # 76

    def test_padding_ratio(self):
        """Test padding ratio calculation."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)
        stats.add_sample(100)  # 28 padding tokens

        # 28 / 128 = 0.21875
        assert abs(stats.padding_ratio - 28 / 128) < 0.001

    def test_efficiency(self):
        """Test efficiency calculation."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)
        stats.add_sample(100)

        # 100 / 128 = 0.78125
        assert abs(stats.efficiency - 100 / 128) < 0.001

    def test_avg_length(self):
        """Test average length calculation."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)
        stats.add_sample(100)
        stats.add_sample(80)
        stats.add_sample(120)

        assert abs(stats.avg_length - 100) < 0.001  # (100+80+120)/3 = 100

    def test_empty_stats_no_division_by_zero(self):
        """Test that empty stats don't cause division by zero."""
        stats = BucketStats(bucket_id=0, bucket_max_length=128)

        assert stats.padding_ratio == 0.0
        assert stats.efficiency == 1.0
        assert stats.avg_length == 0.0
