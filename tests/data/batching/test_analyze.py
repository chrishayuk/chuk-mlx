"""Tests for the batching analysis module."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import (
    BucketSpec,
    OptimizationGoal,
)
from chuk_lazarus.data.batching.analyze import (
    BucketAnalysis,
    BucketEfficiency,
    HistogramBin,
    LengthHistogram,
    analyze_bucket_efficiency,
    compute_length_histogram,
    create_efficiency_report,
    suggest_bucket_edges,
    visualize_buckets,
)


class TestLengthHistogram:
    """Tests for compute_length_histogram."""

    def test_empty_lengths(self):
        """Test with empty input."""
        histogram = compute_length_histogram({})
        assert histogram.total_samples == 0
        assert histogram.min_length == 0
        assert histogram.max_length == 0
        assert len(histogram.bins) == 0

    def test_basic_histogram(self):
        """Test basic histogram computation."""
        lengths = {f"s{i}": i * 10 for i in range(1, 11)}  # 10, 20, ..., 100
        histogram = compute_length_histogram(lengths, num_bins=5)

        assert histogram.total_samples == 10
        assert histogram.min_length == 10
        assert histogram.max_length == 100
        assert histogram.mean_length == 55.0

    def test_percentiles(self):
        """Test percentile computation."""
        lengths = {f"s{i}": i for i in range(1, 101)}  # 1 to 100
        histogram = compute_length_histogram(lengths)

        # Percentiles are approximate - check they're in the right range
        assert 20 <= histogram.p25 <= 30
        assert 45 <= histogram.p50 <= 55
        assert 70 <= histogram.p75 <= 80
        assert 85 <= histogram.p90 <= 95
        assert 90 <= histogram.p95 <= 100
        assert 95 <= histogram.p99 <= 100

    def test_custom_bin_width(self):
        """Test with custom bin width."""
        lengths = {f"s{i}": i * 100 for i in range(1, 11)}  # 100, 200, ..., 1000
        histogram = compute_length_histogram(lengths, bin_width=200)

        # Verify bins are 200 wide
        for bin in histogram.bins:
            assert bin.max_length - bin.min_length == 200

    def test_to_ascii(self):
        """Test ASCII histogram generation."""
        lengths = {f"s{i}": i * 10 for i in range(1, 11)}
        histogram = compute_length_histogram(lengths, num_bins=5)
        ascii_output = histogram.to_ascii(width=30)

        assert "Length Distribution" in ascii_output
        assert "Total:" in ascii_output
        assert "Range:" in ascii_output


class TestHistogramBin:
    """Tests for HistogramBin model."""

    def test_create(self):
        """Test creating a histogram bin."""
        bin = HistogramBin(
            min_length=0,
            max_length=100,
            count=50,
            percentage=25.0,
        )
        assert bin.min_length == 0
        assert bin.max_length == 100
        assert bin.count == 50
        assert bin.percentage == 25.0

    def test_label(self):
        """Test bin label generation."""
        bin = HistogramBin(
            min_length=100,
            max_length=200,
            count=10,
            percentage=10.0,
        )
        assert bin.label == "100-199"


class TestBucketAnalysis:
    """Tests for analyze_bucket_efficiency."""

    def test_empty_lengths(self):
        """Test with empty input."""
        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)
        analysis = analyze_bucket_efficiency({}, bucket_spec)

        assert analysis.overall_efficiency == 0.0
        assert analysis.empty_buckets == 3

    def test_basic_analysis(self):
        """Test basic efficiency analysis."""
        # Create samples that fit in first bucket (1-128)
        lengths = {f"s{i}": 64 for i in range(10)}

        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)
        analysis = analyze_bucket_efficiency(lengths, bucket_spec)

        # All samples in bucket 0, padded to 128
        assert analysis.buckets[0].sample_count == 10
        assert analysis.buckets[0].total_tokens == 640  # 10 * 64
        assert analysis.buckets[0].padded_tokens == 1280  # 10 * 128
        assert analysis.buckets[0].efficiency == 0.5  # 64/128

        # Other buckets should be empty
        assert analysis.buckets[1].sample_count == 0
        assert analysis.buckets[2].sample_count == 0

    def test_multiple_buckets(self):
        """Test with samples in multiple buckets."""
        lengths = {
            "s1": 50,  # bucket 0
            "s2": 150,  # bucket 1
            "s3": 300,  # bucket 2 (overflow)
        }

        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=512)
        analysis = analyze_bucket_efficiency(lengths, bucket_spec)

        assert analysis.buckets[0].sample_count == 1
        assert analysis.buckets[1].sample_count == 1
        assert analysis.buckets[2].sample_count == 1
        assert analysis.empty_buckets == 0

    def test_to_ascii(self):
        """Test ASCII table generation."""
        lengths = {f"s{i}": 100 + i * 50 for i in range(10)}
        bucket_spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)
        analysis = analyze_bucket_efficiency(lengths, bucket_spec)
        ascii_output = analysis.to_ascii()

        assert "Bucket Efficiency Analysis" in ascii_output
        assert "Overall" in ascii_output


class TestBucketSuggestion:
    """Tests for suggest_bucket_edges."""

    def test_empty_lengths(self):
        """Test with empty input."""
        suggestion = suggest_bucket_edges({})

        assert suggestion.edges == ()
        assert suggestion.estimated_efficiency == 0.0

    def test_minimize_waste_goal(self):
        """Test suggestions with minimize waste goal."""
        lengths = {f"s{i}": 50 + i * 10 for i in range(100)}
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=4,
            goal=OptimizationGoal.MINIMIZE_WASTE,
        )

        assert len(suggestion.edges) > 0
        assert suggestion.optimization_goal == OptimizationGoal.MINIMIZE_WASTE
        assert suggestion.estimated_efficiency > 0

    def test_balance_buckets_goal(self):
        """Test suggestions with balance buckets goal."""
        lengths = {f"s{i}": 50 + i * 10 for i in range(100)}
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=4,
            goal=OptimizationGoal.BALANCE_BUCKETS,
        )

        assert suggestion.optimization_goal == OptimizationGoal.BALANCE_BUCKETS
        assert "balance" in suggestion.rationale.lower()

    def test_minimize_memory_goal(self):
        """Test suggestions with memory optimization goal."""
        lengths = {f"s{i}": 50 + i * 10 for i in range(100)}
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=4,
            goal=OptimizationGoal.MINIMIZE_MEMORY,
        )

        assert suggestion.optimization_goal == OptimizationGoal.MINIMIZE_MEMORY
        # Check edges are power of 2
        for edge in suggestion.edges:
            assert edge & (edge - 1) == 0 or edge == 0  # Power of 2 check

    def test_max_length(self):
        """Test with custom max length."""
        lengths = {f"s{i}": 100 for i in range(10)}
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=3,
            max_length=4096,
        )

        assert suggestion.overflow_max == 4096


class TestEfficiencyReport:
    """Tests for create_efficiency_report."""

    def test_complete_report(self):
        """Test creating a complete efficiency report."""
        lengths = {f"s{i}": 50 + i * 10 for i in range(50)}
        bucket_spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        report = create_efficiency_report(lengths, bucket_spec)

        # Verify all components are present
        assert isinstance(report.length_histogram, LengthHistogram)
        assert isinstance(report.bucket_analysis, BucketAnalysis)
        assert len(report.suggestions) == 3  # One for each goal
        assert report.total_samples == 50
        assert report.overall_efficiency > 0

    def test_recommendations_low_efficiency(self):
        """Test that recommendations are generated for low efficiency."""
        # Create data that will have low efficiency
        lengths = {f"s{i}": 10 for i in range(100)}  # Very short sequences
        bucket_spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        report = create_efficiency_report(lengths, bucket_spec)

        # Should have recommendation about low efficiency
        assert len(report.recommendations) > 0
        # Check for packing recommendation
        has_packing_rec = any("packing" in r.lower() for r in report.recommendations)
        assert has_packing_rec

    def test_to_ascii(self):
        """Test ASCII report generation."""
        lengths = {f"s{i}": 100 + i * 20 for i in range(30)}
        bucket_spec = BucketSpec(edges=(128, 256), overflow_max=1024)  # overflow_max > last edge

        report = create_efficiency_report(lengths, bucket_spec)
        ascii_output = report.to_ascii()

        assert "BATCHING EFFICIENCY REPORT" in ascii_output
        assert "Recommendations" in ascii_output


class TestVisualizeBuckets:
    """Tests for visualize_buckets function."""

    def test_visualize(self):
        """Test bucket visualization."""
        lengths = {f"s{i}": 100 + i * 50 for i in range(20)}
        bucket_spec = BucketSpec(edges=(128, 256, 512), overflow_max=1024)

        output = visualize_buckets(lengths, bucket_spec)

        assert "Bucket" in output
        assert "Efficiency" in output


class TestBucketEfficiency:
    """Tests for BucketEfficiency model."""

    def test_create(self):
        """Test creating bucket efficiency."""
        eff = BucketEfficiency(
            bucket_id=0,
            min_length=1,
            max_length=128,
            sample_count=100,
            total_tokens=6400,
            padded_tokens=12800,
            efficiency=0.5,
            waste_percentage=50.0,
        )

        assert eff.bucket_id == 0
        assert eff.waste_tokens == 6400  # 12800 - 6400

    def test_immutable(self):
        """Test that model is immutable."""
        eff = BucketEfficiency(
            bucket_id=0,
            min_length=1,
            max_length=128,
            sample_count=100,
            total_tokens=6400,
            padded_tokens=12800,
            efficiency=0.5,
            waste_percentage=50.0,
        )

        with pytest.raises(ValidationError):
            eff.bucket_id = 1
