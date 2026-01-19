"""Tests for MoE type and compression formatters."""

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.formatters import (
    format_moe_type_comparison,
    format_moe_type_result,
    format_overlay_result,
    format_storage_estimate,
    format_verification_result,
)
from chuk_lazarus.introspection.moe import MoEType
from chuk_lazarus.introspection.moe.moe_compression import (
    OverlayRepresentation,
    ProjectionOverlay,
    ReconstructionError,
    ReconstructionVerification,
    StorageEstimate,
)
from chuk_lazarus.introspection.moe.moe_type import (
    MoETypeAnalysis,
    ProjectionRankAnalysis,
)


class TestFormatMoETypeResult:
    """Tests for format_moe_type_result formatter."""

    @pytest.fixture
    def pseudo_moe_analysis(self):
        """Create a pseudo-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="openai/gpt-oss-20b",
            layer_idx=0,
            num_experts=32,
            moe_type=MoEType.UPCYCLED,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(2880, 2880), max_rank=2880, effective_rank_95=1
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(2880, 2880), max_rank=2880, effective_rank_95=337
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2880, 2880), max_rank=2880, effective_rank_95=206
            ),
            mean_cosine_similarity=0.418,
            std_cosine_similarity=0.163,
        )

    @pytest.fixture
    def native_moe_analysis(self):
        """Create a native-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="allenai/OLMoE-1B-7B-0924",
            layer_idx=0,
            num_experts=64,
            moe_type=MoEType.PRETRAINED_MOE,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(1024, 2048), max_rank=1024, effective_rank_95=755
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(1024, 2048), max_rank=1024, effective_rank_95=772
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2048, 1024), max_rank=1024, effective_rank_95=785
            ),
            mean_cosine_similarity=0.0,
            std_cosine_similarity=0.001,
        )

    @pytest.fixture
    def unknown_moe_analysis(self):
        """Create an unknown-MoE analysis result."""
        return MoETypeAnalysis(
            model_id="unknown/model",
            layer_idx=0,
            num_experts=8,
            moe_type=MoEType.UNKNOWN,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(512, 512), max_rank=512, effective_rank_95=128
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(512, 1024), max_rank=512, effective_rank_95=256
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(1024, 512), max_rank=512, effective_rank_95=256
            ),
            mean_cosine_similarity=0.15,
            std_cosine_similarity=0.05,
        )

    def test_pseudo_moe_output(self, pseudo_moe_analysis):
        """Test formatter output for pseudo-MoE."""
        output = format_moe_type_result(pseudo_moe_analysis)

        assert "MOE TYPE ANALYSIS" in output
        assert "openai/gpt-oss-20b" in output
        assert "UPCYCLED" in output
        assert "Gate Rank:" in output
        assert "0.0%" in output  # Gate rank ratio
        assert "0.418" in output  # Cosine similarity
        assert "Yes" in output  # Compressible

    def test_native_moe_output(self, native_moe_analysis):
        """Test formatter output for native-MoE."""
        output = format_moe_type_result(native_moe_analysis)

        assert "PRETRAINED" in output
        assert "allenai/OLMoE-1B-7B-0924" in output
        assert "755" in output  # Gate rank
        assert "0.000" in output  # Cosine similarity
        assert "No" in output  # Not compressible

    def test_unknown_moe_output(self, unknown_moe_analysis):
        """Test formatter output for unknown-MoE."""
        output = format_moe_type_result(unknown_moe_analysis)

        assert "UNKNOWN" in output
        assert "unknown/model" in output

    def test_header_format(self, pseudo_moe_analysis):
        """Test that header is properly formatted."""
        output = format_moe_type_result(pseudo_moe_analysis)

        assert "=" * 70 in output
        assert output.count("=" * 70) >= 2  # Header and footer

    def test_evidence_section(self, pseudo_moe_analysis):
        """Test evidence section formatting."""
        output = format_moe_type_result(pseudo_moe_analysis)

        assert "Evidence:" in output
        assert "Gate Rank:" in output
        assert "Up Rank:" in output
        assert "Down Rank:" in output
        assert "Cosine Similarity:" in output

    def test_compression_section(self, pseudo_moe_analysis):
        """Test compression section formatting."""
        output = format_moe_type_result(pseudo_moe_analysis)

        assert "Compression:" in output
        assert "Compressible:" in output
        assert "Estimated Ratio:" in output


class TestFormatMoETypeComparison:
    """Tests for format_moe_type_comparison formatter."""

    @pytest.fixture
    def pseudo_analysis(self):
        """Create pseudo-MoE analysis."""
        return MoETypeAnalysis(
            model_id="openai/gpt-oss-20b",
            layer_idx=0,
            num_experts=32,
            moe_type=MoEType.UPCYCLED,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(2880, 2880), max_rank=2880, effective_rank_95=1
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(2880, 2880), max_rank=2880, effective_rank_95=337
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2880, 2880), max_rank=2880, effective_rank_95=206
            ),
            mean_cosine_similarity=0.418,
            std_cosine_similarity=0.163,
        )

    @pytest.fixture
    def native_analysis(self):
        """Create native-MoE analysis."""
        return MoETypeAnalysis(
            model_id="allenai/OLMoE-1B-7B-0924",
            layer_idx=0,
            num_experts=64,
            moe_type=MoEType.PRETRAINED_MOE,
            gate=ProjectionRankAnalysis(
                name="gate", shape=(1024, 2048), max_rank=1024, effective_rank_95=755
            ),
            up=ProjectionRankAnalysis(
                name="up", shape=(1024, 2048), max_rank=1024, effective_rank_95=772
            ),
            down=ProjectionRankAnalysis(
                name="down", shape=(2048, 1024), max_rank=1024, effective_rank_95=785
            ),
            mean_cosine_similarity=0.0,
            std_cosine_similarity=0.001,
        )

    def test_comparison_table_structure(self, pseudo_analysis, native_analysis):
        """Test comparison table structure."""
        output = format_moe_type_comparison(pseudo_analysis, native_analysis)

        assert "MOE TYPE COMPARISON" in output
        assert "Metric" in output
        assert "+-" in output  # Table borders

    def test_comparison_includes_both_models(self, pseudo_analysis, native_analysis):
        """Test both models are included."""
        output = format_moe_type_comparison(pseudo_analysis, native_analysis)

        # Model names are truncated to 14 chars
        assert "gpt-oss-20b" in output
        assert "OLMoE-1B-7B-09" in output

    def test_comparison_type_row(self, pseudo_analysis, native_analysis):
        """Test Type row in comparison."""
        output = format_moe_type_comparison(pseudo_analysis, native_analysis)

        assert "UPCYCLED" in output
        assert "PRETRAINED" in output

    def test_comparison_gate_rank_row(self, pseudo_analysis, native_analysis):
        """Test Gate Rank row in comparison."""
        output = format_moe_type_comparison(pseudo_analysis, native_analysis)

        assert "Gate Rank" in output
        assert "1/" in output  # Pseudo gate rank
        assert "755/" in output  # Native gate rank

    def test_comparison_compressible_row(self, pseudo_analysis, native_analysis):
        """Test Compressible row in comparison."""
        output = format_moe_type_comparison(pseudo_analysis, native_analysis)

        assert "Compressible" in output
        assert "Yes" in output
        assert "No" in output


class TestFormatOverlayResult:
    """Tests for format_overlay_result formatter."""

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

    def test_overlay_header(self, sample_overlay):
        """Test overlay result header."""
        output = format_overlay_result(sample_overlay)

        assert "OVERLAY REPRESENTATION" in output

    def test_overlay_model_info(self, sample_overlay):
        """Test model info in overlay result."""
        output = format_overlay_result(sample_overlay)

        assert "Model:" in output
        assert "test-model" in output
        assert "Layer:" in output
        assert "Experts:" in output
        assert "32" in output

    def test_overlay_projection_analysis(self, sample_overlay):
        """Test projection analysis in overlay result."""
        output = format_overlay_result(sample_overlay)

        assert "Projection Analysis:" in output
        assert "Gate:" in output
        assert "Up:" in output
        assert "Down:" in output
        assert "rank=" in output
        assert "shape=" in output
        assert "compression:" in output

    def test_overlay_storage_section(self, sample_overlay):
        """Test storage section in overlay result."""
        output = format_overlay_result(sample_overlay)

        assert "Storage:" in output
        assert "Original:" in output
        assert "Compressed:" in output
        assert "Ratio:" in output
        assert "MB" in output


class TestFormatVerificationResult:
    """Tests for format_verification_result formatter."""

    @pytest.fixture
    def passing_verification(self):
        """Create a passing verification result."""
        gate = ReconstructionError(
            name="gate", mean_relative_error=0.001, max_relative_error=0.002, mean_mse=0.0001
        )
        up = ReconstructionError(
            name="up", mean_relative_error=0.002, max_relative_error=0.004, mean_mse=0.0002
        )
        down = ReconstructionError(
            name="down", mean_relative_error=0.003, max_relative_error=0.006, mean_mse=0.0003
        )
        return ReconstructionVerification(
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
            name="down", mean_relative_error=0.06, max_relative_error=0.12, mean_mse=0.012
        )
        return ReconstructionVerification(
            model_id="test-model",
            layer_idx=0,
            gate=gate,
            up=up,
            down=down,
            mean_output_error=0.05,
            max_output_error=0.02,
            gate_rank=1,
            up_rank=8,
            down_rank=4,
        )

    def test_verification_header(self, passing_verification):
        """Test verification result header."""
        output = format_verification_result(passing_verification)

        assert "RECONSTRUCTION VERIFICATION" in output

    def test_verification_passed_status(self, passing_verification):
        """Test passed status formatting."""
        output = format_verification_result(passing_verification)

        assert "PASSED" in output
        assert "✓" in output
        assert "suitable for production" in output

    def test_verification_failed_status(self, failing_verification):
        """Test failed status formatting."""
        output = format_verification_result(failing_verification)

        assert "FAILED" in output
        assert "✗" in output
        assert "increase ranks" in output

    def test_verification_ranks(self, passing_verification):
        """Test ranks display."""
        output = format_verification_result(passing_verification)

        assert "Ranks:" in output
        assert "gate=2" in output
        assert "up=128" in output
        assert "down=64" in output

    def test_verification_weight_errors(self, passing_verification):
        """Test weight error section."""
        output = format_verification_result(passing_verification)

        assert "Weight Reconstruction Errors:" in output
        assert "Gate:" in output
        assert "Up:" in output
        assert "Down:" in output

    def test_verification_output_errors(self, passing_verification):
        """Test output error section."""
        output = format_verification_result(passing_verification)

        assert "Output Reconstruction Errors:" in output
        assert "Mean:" in output
        assert "Max:" in output


class TestFormatStorageEstimate:
    """Tests for format_storage_estimate formatter."""

    @pytest.fixture
    def sample_estimate(self):
        """Create a sample storage estimate."""
        return StorageEstimate(
            model_id="test-model",
            num_layers=24,
            num_experts=32,
            original_mb=36864.0,
            compressed_mb=6912.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    def test_storage_header(self, sample_estimate):
        """Test storage estimate header."""
        output = format_storage_estimate(sample_estimate)

        assert "STORAGE ESTIMATE" in output

    def test_storage_model_info(self, sample_estimate):
        """Test model info in storage estimate."""
        output = format_storage_estimate(sample_estimate)

        assert "Model:" in output
        assert "test-model" in output
        assert "Layers:" in output
        assert "24 MoE layers" in output
        assert "Experts:" in output
        assert "32 per layer" in output

    def test_storage_ranks(self, sample_estimate):
        """Test ranks display."""
        output = format_storage_estimate(sample_estimate)

        assert "Ranks:" in output
        assert "gate=2" in output
        assert "up=128" in output
        assert "down=64" in output

    def test_storage_values(self, sample_estimate):
        """Test storage values in output."""
        output = format_storage_estimate(sample_estimate)

        assert "Full Model Storage:" in output
        assert "Original:" in output
        assert "Compressed:" in output
        assert "Savings:" in output
        assert "36864" in output
        assert "6912" in output

    def test_storage_breakdown(self, sample_estimate):
        """Test breakdown section."""
        output = format_storage_estimate(sample_estimate)

        assert "Breakdown:" in output
        assert "Base experts (shared):" in output
        assert "Low-rank deltas:" in output
