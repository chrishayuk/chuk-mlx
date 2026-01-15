"""Tests for MoE overlay CLI handlers."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_compute import (
    _async_moe_overlay_compute,
    handle_moe_overlay_compute,
)
from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_estimate import (
    _async_moe_overlay_estimate,
    handle_moe_overlay_estimate,
)
from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_verify import (
    _async_moe_overlay_verify,
    handle_moe_overlay_verify,
)
from chuk_lazarus.introspection.moe.moe_compression import (
    MoECompressionService,
    OverlayRepresentation,
    ProjectionOverlay,
    ReconstructionError,
    ReconstructionVerification,
    StorageEstimate,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_overlay_result():
    """Create a mock overlay representation result."""
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
        model_id="test/model",
        layer_idx=0,
        num_experts=32,
        gate=gate,
        up=up,
        down=down,
        gate_rank=2,
        up_rank=128,
        down_rank=64,
    )


@pytest.fixture
def mock_verification_result():
    """Create a mock reconstruction verification result."""
    gate = ReconstructionError(
        name="gate",
        mean_relative_error=0.001,
        max_relative_error=0.002,
        mean_mse=0.0001,
    )
    up = ReconstructionError(
        name="up", mean_relative_error=0.002, max_relative_error=0.004, mean_mse=0.0002
    )
    down = ReconstructionError(
        name="down",
        mean_relative_error=0.003,
        max_relative_error=0.006,
        mean_mse=0.0003,
    )
    return ReconstructionVerification(
        model_id="test/model",
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
def mock_storage_estimate():
    """Create a mock storage estimate result."""
    return StorageEstimate(
        model_id="test/model",
        num_layers=24,
        num_experts=32,
        original_mb=36864.0,
        compressed_mb=6912.0,
        gate_rank=2,
        up_rank=128,
        down_rank=64,
    )


# =============================================================================
# Tests for moe-overlay-compute handler
# =============================================================================


class TestHandleMoEOverlayCompute:
    """Tests for handle_moe_overlay_compute function."""

    def test_handle_moe_overlay_compute_calls_asyncio_run(self):
        """Test that handle_moe_overlay_compute calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_compute.asyncio"
        ) as mock_asyncio:
            handle_moe_overlay_compute(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncMoEOverlayCompute:
    """Tests for _async_moe_overlay_compute function."""

    @pytest.mark.asyncio
    async def test_successful_compute(self, capsys, mock_overlay_result):
        """Test successful overlay compute execution."""
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=None,
        )

        with patch.object(
            MoECompressionService, "compute_overlay", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.return_value = mock_overlay_result

            await _async_moe_overlay_compute(args)

            captured = capsys.readouterr()
            assert "Computing overlay representation" in captured.out
            assert "auto-selecting" in captured.out
            assert "OVERLAY REPRESENTATION" in captured.out

    @pytest.mark.asyncio
    async def test_compute_with_custom_ranks(self, capsys, mock_overlay_result):
        """Test compute with custom ranks."""
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=4,
            up_rank=64,
            down_rank=32,
            output=None,
        )

        with patch.object(
            MoECompressionService, "compute_overlay", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.return_value = mock_overlay_result

            await _async_moe_overlay_compute(args)

            captured = capsys.readouterr()
            assert "Ranks: gate=4" in captured.out

            # Verify service called with correct args
            mock_compute.assert_called_once_with(
                "test/model",
                layer=None,
                gate_rank=4,
                up_rank=64,
                down_rank=32,
            )

    @pytest.mark.asyncio
    async def test_compute_with_output_file(self, capsys, mock_overlay_result, tmp_path):
        """Test compute with output file."""
        output_file = tmp_path / "overlay.json"
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=str(output_file),
        )

        with patch.object(
            MoECompressionService, "compute_overlay", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.return_value = mock_overlay_result

            await _async_moe_overlay_compute(args)

            captured = capsys.readouterr()
            assert "Saved to:" in captured.out
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_compute_missing_attributes(self, capsys, mock_overlay_result):
        """Test compute handles missing optional attributes."""
        args = Namespace(model="test/model")

        with patch.object(
            MoECompressionService, "compute_overlay", new_callable=AsyncMock
        ) as mock_compute:
            mock_compute.return_value = mock_overlay_result

            await _async_moe_overlay_compute(args)

            captured = capsys.readouterr()
            assert "Computing overlay representation" in captured.out


# =============================================================================
# Tests for moe-overlay-verify handler
# =============================================================================


class TestHandleMoEOverlayVerify:
    """Tests for handle_moe_overlay_verify function."""

    def test_handle_moe_overlay_verify_calls_asyncio_run(self):
        """Test that handle_moe_overlay_verify calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_verify.asyncio"
        ) as mock_asyncio:
            handle_moe_overlay_verify(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncMoEOverlayVerify:
    """Tests for _async_moe_overlay_verify function."""

    @pytest.mark.asyncio
    async def test_successful_verify(self, capsys, mock_verification_result):
        """Test successful overlay verify execution."""
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=None,
        )

        with patch.object(
            MoECompressionService, "verify_reconstruction", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.return_value = mock_verification_result

            await _async_moe_overlay_verify(args)

            captured = capsys.readouterr()
            assert "Verifying overlay reconstruction" in captured.out
            assert "RECONSTRUCTION VERIFICATION" in captured.out

    @pytest.mark.asyncio
    async def test_verify_with_custom_ranks(self, capsys, mock_verification_result):
        """Test verify with custom ranks."""
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=4,
            up_rank=64,
            down_rank=32,
            output=None,
        )

        with patch.object(
            MoECompressionService, "verify_reconstruction", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.return_value = mock_verification_result

            await _async_moe_overlay_verify(args)

            captured = capsys.readouterr()
            assert "Ranks: gate=4" in captured.out

    @pytest.mark.asyncio
    async def test_verify_with_output_file(self, capsys, mock_verification_result, tmp_path):
        """Test verify with output file."""
        output_file = tmp_path / "verify.json"
        args = Namespace(
            model="test/model",
            layer=None,
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=str(output_file),
        )

        with patch.object(
            MoECompressionService, "verify_reconstruction", new_callable=AsyncMock
        ) as mock_verify:
            mock_verify.return_value = mock_verification_result

            await _async_moe_overlay_verify(args)

            captured = capsys.readouterr()
            assert "Saved to:" in captured.out
            assert output_file.exists()


# =============================================================================
# Tests for moe-overlay-estimate handler
# =============================================================================


class TestHandleMoEOverlayEstimate:
    """Tests for handle_moe_overlay_estimate function."""

    def test_handle_moe_overlay_estimate_calls_asyncio_run(self):
        """Test that handle_moe_overlay_estimate calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.moe_overlay_estimate.asyncio"
        ) as mock_asyncio:
            handle_moe_overlay_estimate(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncMoEOverlayEstimate:
    """Tests for _async_moe_overlay_estimate function."""

    @pytest.mark.asyncio
    async def test_successful_estimate(self, capsys, mock_storage_estimate):
        """Test successful overlay estimate execution."""
        args = Namespace(
            model="test/model",
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=None,
        )

        with patch.object(
            MoECompressionService, "estimate_savings", new_callable=AsyncMock
        ) as mock_estimate:
            mock_estimate.return_value = mock_storage_estimate

            await _async_moe_overlay_estimate(args)

            captured = capsys.readouterr()
            assert "Estimating storage savings" in captured.out
            assert "STORAGE ESTIMATE" in captured.out

    @pytest.mark.asyncio
    async def test_estimate_with_custom_ranks(self, capsys, mock_storage_estimate):
        """Test estimate with custom ranks."""
        args = Namespace(
            model="test/model",
            gate_rank=4,
            up_rank=64,
            down_rank=32,
            output=None,
        )

        with patch.object(
            MoECompressionService, "estimate_savings", new_callable=AsyncMock
        ) as mock_estimate:
            mock_estimate.return_value = mock_storage_estimate

            await _async_moe_overlay_estimate(args)

            captured = capsys.readouterr()
            assert "Ranks: gate=4" in captured.out

            # Verify service called with correct args
            mock_estimate.assert_called_once_with(
                "test/model",
                gate_rank=4,
                up_rank=64,
                down_rank=32,
            )

    @pytest.mark.asyncio
    async def test_estimate_with_output_file(self, capsys, mock_storage_estimate, tmp_path):
        """Test estimate with output file."""
        output_file = tmp_path / "estimate.json"
        args = Namespace(
            model="test/model",
            gate_rank=None,
            up_rank=None,
            down_rank=None,
            output=str(output_file),
        )

        with patch.object(
            MoECompressionService, "estimate_savings", new_callable=AsyncMock
        ) as mock_estimate:
            mock_estimate.return_value = mock_storage_estimate

            await _async_moe_overlay_estimate(args)

            captured = capsys.readouterr()
            assert "Saved to:" in captured.out
            assert output_file.exists()

    @pytest.mark.asyncio
    async def test_estimate_missing_attributes(self, capsys, mock_storage_estimate):
        """Test estimate handles missing optional attributes."""
        args = Namespace(model="test/model")

        with patch.object(
            MoECompressionService, "estimate_savings", new_callable=AsyncMock
        ) as mock_estimate:
            mock_estimate.return_value = mock_storage_estimate

            await _async_moe_overlay_estimate(args)

            captured = capsys.readouterr()
            assert "Estimating storage savings" in captured.out
