"""Tests for MoE compression via SVD overlay representation."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.moe.moe_compression import (
    MoECompressionService,
    OverlayRepresentation,
    ProjectionOverlay,
    ReconstructionError,
    ReconstructionVerification,
    StorageEstimate,
)


class TestProjectionOverlay:
    """Tests for ProjectionOverlay Pydantic model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(2880, 2880),
            rank=2,
            num_experts=32,
            original_bytes=530841600,
            compressed_bytes=16588800,
        )
        assert overlay.name == "gate"
        assert overlay.shape == (2880, 2880)
        assert overlay.rank == 2
        assert overlay.num_experts == 32

    def test_compression_ratio_significant(self):
        """Test compression ratio for significant compression."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(2880, 2880),
            rank=2,
            num_experts=32,
            original_bytes=530841600,  # 32 * 2880 * 2880 * 2
            compressed_bytes=16958400,  # base + 32 * 2 * (2880 + 2880) * 2
        )
        # Expect ~31x compression
        assert overlay.compression_ratio > 30

    def test_compression_ratio_minimal(self):
        """Test compression ratio when minimal compression."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(1024, 2048),
            rank=756,
            num_experts=64,
            original_bytes=268435456,  # 64 * 1024 * 2048 * 2
            compressed_bytes=300000000,  # > original due to high rank
        )
        # Ratio < 1 means no compression benefit
        assert overlay.compression_ratio < 1

    def test_compression_ratio_zero_compressed(self):
        """Test compression ratio when compressed bytes is 0."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(100, 100),
            rank=1,
            num_experts=8,
            original_bytes=160000,
            compressed_bytes=0,  # Edge case
        )
        assert overlay.compression_ratio == 1.0

    def test_frozen_model(self):
        """Test that model is frozen (immutable)."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(100, 100),
            rank=1,
            num_experts=8,
            original_bytes=16000,
            compressed_bytes=4000,
        )
        with pytest.raises(ValidationError):
            overlay.name = "up"


class TestOverlayRepresentation:
    """Tests for OverlayRepresentation Pydantic model."""

    @pytest.fixture
    def sample_overlay(self):
        """Create a sample overlay representation."""
        gate = ProjectionOverlay(
            name="gate",
            shape=(2880, 2880),
            rank=2,
            num_experts=32,
            original_bytes=530841600,
            compressed_bytes=16958400,
        )
        up = ProjectionOverlay(
            name="up",
            shape=(2880, 2880),
            rank=128,
            num_experts=32,
            original_bytes=530841600,
            compressed_bytes=39813120,
        )
        down = ProjectionOverlay(
            name="down",
            shape=(2880, 2880),
            rank=64,
            num_experts=32,
            original_bytes=530841600,
            compressed_bytes=28385280,
        )
        return OverlayRepresentation(
            model_id="test-model",
            layer_idx=0,
            num_experts=32,
            gate=gate,
            up=up,
            down=down,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    def test_basic_creation(self, sample_overlay):
        """Test basic model creation."""
        assert sample_overlay.model_id == "test-model"
        assert sample_overlay.layer_idx == 0
        assert sample_overlay.num_experts == 32
        assert sample_overlay.gate_rank == 2
        assert sample_overlay.up_rank == 128
        assert sample_overlay.down_rank == 64

    def test_total_original_bytes(self, sample_overlay):
        """Test total original bytes calculation."""
        expected = 530841600 * 3  # gate + up + down
        assert sample_overlay.total_original_bytes == expected

    def test_total_compressed_bytes(self, sample_overlay):
        """Test total compressed bytes calculation."""
        expected = 16958400 + 39813120 + 28385280
        assert sample_overlay.total_compressed_bytes == expected

    def test_compression_ratio(self, sample_overlay):
        """Test overall compression ratio."""
        ratio = sample_overlay.compression_ratio
        # Should be significant compression
        assert ratio > 15

    def test_compression_ratio_zero_compressed(self):
        """Test compression ratio when compressed bytes is 0."""
        gate = ProjectionOverlay(
            name="gate",
            shape=(100, 100),
            rank=1,
            num_experts=8,
            original_bytes=160000,
            compressed_bytes=0,
        )
        up = ProjectionOverlay(
            name="up",
            shape=(100, 100),
            rank=1,
            num_experts=8,
            original_bytes=160000,
            compressed_bytes=0,
        )
        down = ProjectionOverlay(
            name="down",
            shape=(100, 100),
            rank=1,
            num_experts=8,
            original_bytes=160000,
            compressed_bytes=0,
        )
        overlay = OverlayRepresentation(
            model_id="test",
            layer_idx=0,
            num_experts=8,
            gate=gate,
            up=up,
            down=down,
            gate_rank=1,
            up_rank=1,
            down_rank=1,
        )
        assert overlay.compression_ratio == 1.0


class TestReconstructionError:
    """Tests for ReconstructionError Pydantic model."""

    def test_basic_creation(self):
        """Test basic model creation."""
        error = ReconstructionError(
            name="gate",
            mean_relative_error=0.001,
            max_relative_error=0.005,
            mean_mse=0.0001,
        )
        assert error.name == "gate"
        assert error.mean_relative_error == 0.001
        assert error.max_relative_error == 0.005
        assert error.mean_mse == 0.0001

    def test_frozen_model(self):
        """Test that model is frozen."""
        error = ReconstructionError(
            name="gate",
            mean_relative_error=0.001,
            max_relative_error=0.005,
            mean_mse=0.0001,
        )
        with pytest.raises(ValidationError):
            error.name = "up"


class TestReconstructionVerification:
    """Tests for ReconstructionVerification Pydantic model."""

    @pytest.fixture
    def passing_verification(self):
        """Create a passing verification result."""
        gate = ReconstructionError(
            name="gate",
            mean_relative_error=0.001,
            max_relative_error=0.002,
            mean_mse=0.0001,
        )
        up = ReconstructionError(
            name="up",
            mean_relative_error=0.002,
            max_relative_error=0.004,
            mean_mse=0.0002,
        )
        down = ReconstructionError(
            name="down",
            mean_relative_error=0.003,
            max_relative_error=0.006,
            mean_mse=0.0003,
        )
        return ReconstructionVerification(
            model_id="test-model",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.002,
            max_output_error=0.005,  # < 1%
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    @pytest.fixture
    def failing_verification(self):
        """Create a failing verification result."""
        gate = ReconstructionError(
            name="gate", mean_relative_error=0.05, max_relative_error=0.1, mean_mse=0.01
        )
        up = ReconstructionError(
            name="up", mean_relative_error=0.04, max_relative_error=0.08, mean_mse=0.008
        )
        down = ReconstructionError(
            name="down",
            mean_relative_error=0.06,
            max_relative_error=0.12,
            mean_mse=0.012,
        )
        return ReconstructionVerification(
            model_id="test-model",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.05,
            max_output_error=0.02,  # > 1%
            gate_rank=1,
            up_rank=8,
            down_rank=4,
        )

    def test_passed_true(self, passing_verification):
        """Test passed property when error < 1%."""
        assert passing_verification.passed is True

    def test_passed_false(self, failing_verification):
        """Test passed property when error > 1%."""
        assert failing_verification.passed is False

    def test_overall_weight_error(self, passing_verification):
        """Test overall weight error calculation."""
        expected = (0.001 + 0.002 + 0.003) / 3
        assert passing_verification.overall_weight_error == pytest.approx(expected)

    def test_basic_attributes(self, passing_verification):
        """Test basic attributes."""
        assert passing_verification.model_id == "test-model"
        assert passing_verification.layer_idx == 0
        assert passing_verification.gate_rank == 2
        assert passing_verification.up_rank == 128
        assert passing_verification.down_rank == 64


class TestStorageEstimate:
    """Tests for StorageEstimate Pydantic model."""

    @pytest.fixture
    def sample_estimate(self):
        """Create a sample storage estimate."""
        return StorageEstimate(
            model_id="test-model",
            num_layers=24,
            num_experts=32,
            original_mb=36864.0,  # ~36 GB
            compressed_mb=6912.0,  # ~7 GB
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    def test_basic_creation(self, sample_estimate):
        """Test basic model creation."""
        assert sample_estimate.model_id == "test-model"
        assert sample_estimate.num_layers == 24
        assert sample_estimate.num_experts == 32
        assert sample_estimate.original_mb == 36864.0
        assert sample_estimate.compressed_mb == 6912.0

    def test_compression_ratio(self, sample_estimate):
        """Test compression ratio calculation."""
        expected = 36864.0 / 6912.0
        assert sample_estimate.compression_ratio == pytest.approx(expected)

    def test_savings_mb(self, sample_estimate):
        """Test savings calculation."""
        expected = 36864.0 - 6912.0
        assert sample_estimate.savings_mb == pytest.approx(expected)

    def test_compression_ratio_zero_compressed(self):
        """Test compression ratio when compressed is 0."""
        estimate = StorageEstimate(
            model_id="test",
            num_layers=24,
            num_experts=32,
            original_mb=36864.0,
            compressed_mb=0.0,
            gate_rank=1,
            up_rank=1,
            down_rank=1,
        )
        assert estimate.compression_ratio == 1.0


class TestMoECompressionServiceConstants:
    """Tests for MoECompressionService constants."""

    def test_default_gate_rank(self):
        """Verify default gate rank."""
        assert MoECompressionService.DEFAULT_GATE_RANK == 2

    def test_default_up_rank(self):
        """Verify default up rank."""
        assert MoECompressionService.DEFAULT_UP_RANK == 128

    def test_default_down_rank(self):
        """Verify default down rank."""
        assert MoECompressionService.DEFAULT_DOWN_RANK == 64

    def test_variance_threshold(self):
        """Verify variance threshold is 95%."""
        assert MoECompressionService.VARIANCE_THRESHOLD == 0.95


class TestMoECompressionServiceHelpers:
    """Tests for MoECompressionService helper methods."""

    def test_compute_projection_overlay_basic(self):
        """Test _compute_projection_overlay calculation."""
        # Create mock weights array shape
        # We can't directly test with mx.array without MLX, but we can test the math

        # For a 32-expert model with 2880x2880 projections and rank 2:
        num_experts = 32
        out_dim = 2880
        in_dim = 2880
        rank = 2
        bytes_per_param = 2  # bfloat16

        # Original storage
        original_bytes = num_experts * out_dim * in_dim * bytes_per_param

        # Compressed: 1 base + num_experts * low-rank factors
        base_bytes = out_dim * in_dim * bytes_per_param
        delta_bytes = num_experts * rank * (out_dim + in_dim) * bytes_per_param
        compressed_bytes = base_bytes + delta_bytes

        # Verify math
        assert original_bytes == 32 * 2880 * 2880 * 2
        assert base_bytes == 2880 * 2880 * 2
        assert delta_bytes == 32 * 2 * (2880 + 2880) * 2

        # Compression ratio should be significant
        ratio = original_bytes / compressed_bytes
        assert ratio > 25

    def test_storage_estimate_math(self):
        """Test storage estimate calculations match expectations."""
        # GPT-OSS-like model: 24 layers, 32 experts, 2880x2880 projections
        num_layers = 24
        num_experts = 32
        dim = 2880
        bytes_per_param = 2

        # Original: 3 projections per expert
        original_per_layer = num_experts * 3 * dim * dim * bytes_per_param
        original_total = original_per_layer * num_layers
        original_mb = original_total / (1024 * 1024)

        # Compressed with ranks 2, 128, 64
        gate_rank, up_rank, down_rank = 2, 128, 64

        base_per_layer = 3 * dim * dim * bytes_per_param
        deltas_per_layer = (
            num_experts
            * (gate_rank * (dim + dim) + up_rank * (dim + dim) + down_rank * (dim + dim))
            * bytes_per_param
        )

        compressed_per_layer = base_per_layer + deltas_per_layer
        compressed_total = compressed_per_layer * num_layers
        compressed_mb = compressed_total / (1024 * 1024)

        # Verify significant compression
        ratio = original_mb / compressed_mb
        assert ratio > 4  # Should be ~5-6x compression


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_projection_overlay_minimum_values(self):
        """Test ProjectionOverlay with minimum valid values."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(1, 1),
            rank=1,
            num_experts=1,
            original_bytes=2,
            compressed_bytes=1,
        )
        assert overlay.compression_ratio == 2.0

    def test_storage_estimate_minimum_values(self):
        """Test StorageEstimate with minimum valid values."""
        estimate = StorageEstimate(
            model_id="tiny-model",
            num_layers=1,
            num_experts=1,
            original_mb=0.001,
            compressed_mb=0.001,
            gate_rank=1,
            up_rank=1,
            down_rank=1,
        )
        assert estimate.compression_ratio == 1.0
        assert estimate.savings_mb == 0.0

    def test_reconstruction_error_perfect(self):
        """Test ReconstructionError with zero error."""
        error = ReconstructionError(
            name="gate",
            mean_relative_error=0.0,
            max_relative_error=0.0,
            mean_mse=0.0,
        )
        assert error.mean_relative_error == 0.0

    def test_verification_at_threshold(self):
        """Test verification at exactly 1% threshold."""
        gate = ReconstructionError(
            name="gate",
            mean_relative_error=0.01,
            max_relative_error=0.01,
            mean_mse=0.001,
        )
        up = ReconstructionError(
            name="up", mean_relative_error=0.01, max_relative_error=0.01, mean_mse=0.001
        )
        down = ReconstructionError(
            name="down",
            mean_relative_error=0.01,
            max_relative_error=0.01,
            mean_mse=0.001,
        )

        # At exactly 1%, should fail (uses < not <=)
        verification = ReconstructionVerification(
            model_id="test",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.01,
            max_output_error=0.01,  # exactly 1%
            gate_rank=1,
            up_rank=1,
            down_rank=1,
        )
        assert verification.passed is False

        # Just under 1%, should pass
        verification_pass = ReconstructionVerification(
            model_id="test",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.009,
            max_output_error=0.0099,  # just under 1%
            gate_rank=1,
            up_rank=1,
            down_rank=1,
        )
        assert verification_pass.passed is True


class TestJSONSerialization:
    """Tests for JSON serialization of models."""

    def test_projection_overlay_json(self):
        """Test ProjectionOverlay JSON serialization."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(2880, 2880),
            rank=2,
            num_experts=32,
            original_bytes=530841600,
            compressed_bytes=16958400,
        )
        json_str = overlay.model_dump_json()
        assert "gate" in json_str
        assert "2880" in json_str

    def test_storage_estimate_json(self):
        """Test StorageEstimate JSON serialization."""
        estimate = StorageEstimate(
            model_id="test/model",
            num_layers=24,
            num_experts=32,
            original_mb=36864.0,
            compressed_mb=6912.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        json_str = estimate.model_dump_json()
        assert "test/model" in json_str
        assert "36864" in json_str

    def test_reconstruction_verification_json(self):
        """Test ReconstructionVerification JSON serialization."""
        gate = ReconstructionError(
            name="gate",
            mean_relative_error=0.001,
            max_relative_error=0.002,
            mean_mse=0.0001,
        )
        up = ReconstructionError(
            name="up",
            mean_relative_error=0.002,
            max_relative_error=0.004,
            mean_mse=0.0002,
        )
        down = ReconstructionError(
            name="down",
            mean_relative_error=0.003,
            max_relative_error=0.006,
            mean_mse=0.0003,
        )
        verification = ReconstructionVerification(
            model_id="test-model",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.002,
            max_output_error=0.005,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        json_str = verification.model_dump_json()
        assert "test-model" in json_str
        assert "gate" in json_str
