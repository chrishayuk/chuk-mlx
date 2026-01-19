"""Tests for moe_compression.py to improve coverage."""

import mlx.core as mx
import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.moe.moe_compression import (
    CompressionConfig,
    CompressionResult,
    MoECompressionService,
    OverlayRepresentation,
    ProjectionOverlay,
    ReconstructionError,
    ReconstructionVerification,
    StorageEstimate,
)


class TestProjectionOverlay:
    """Tests for ProjectionOverlay class."""

    def test_compression_ratio_normal(self):
        """Test compression_ratio property with normal values (line 54-56)."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(100, 100),
            rank=10,
            num_experts=8,
            original_bytes=10000,
            compressed_bytes=2000,
        )
        assert overlay.compression_ratio == 5.0

    def test_compression_ratio_zero_compressed(self):
        """Test compression_ratio when compressed_bytes is 0 (line 56)."""
        overlay = ProjectionOverlay(
            name="up",
            shape=(100, 100),
            rank=10,
            num_experts=8,
            original_bytes=10000,
            compressed_bytes=0,
        )
        assert overlay.compression_ratio == 1.0


class TestOverlayRepresentation:
    """Tests for OverlayRepresentation class."""

    def _create_overlay(
        self,
        gate_bytes: tuple[int, int] = (1000, 200),
        up_bytes: tuple[int, int] = (2000, 400),
        down_bytes: tuple[int, int] = (1500, 300),
    ) -> OverlayRepresentation:
        """Helper to create OverlayRepresentation."""
        return OverlayRepresentation(
            model_id="test",
            layer_idx=0,
            num_experts=8,
            gate=ProjectionOverlay(
                name="gate",
                shape=(100, 100),
                rank=2,
                num_experts=8,
                original_bytes=gate_bytes[0],
                compressed_bytes=gate_bytes[1],
            ),
            up=ProjectionOverlay(
                name="up",
                shape=(100, 400),
                rank=128,
                num_experts=8,
                original_bytes=up_bytes[0],
                compressed_bytes=up_bytes[1],
            ),
            down=ProjectionOverlay(
                name="down",
                shape=(400, 100),
                rank=64,
                num_experts=8,
                original_bytes=down_bytes[0],
                compressed_bytes=down_bytes[1],
            ),
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    def test_total_original_bytes(self):
        """Test total_original_bytes property (lines 79-81)."""
        overlay = self._create_overlay()
        assert overlay.total_original_bytes == 1000 + 2000 + 1500

    def test_total_compressed_bytes(self):
        """Test total_compressed_bytes property (lines 83-86)."""
        overlay = self._create_overlay()
        assert overlay.total_compressed_bytes == 200 + 400 + 300

    def test_compression_ratio(self):
        """Test compression_ratio property (lines 88-95)."""
        overlay = self._create_overlay()
        expected = (1000 + 2000 + 1500) / (200 + 400 + 300)
        assert abs(overlay.compression_ratio - expected) < 0.001

    def test_compression_ratio_zero_compressed(self):
        """Test compression_ratio when compressed_bytes is 0."""
        overlay = self._create_overlay(
            gate_bytes=(1000, 0),
            up_bytes=(2000, 0),
            down_bytes=(1500, 0),
        )
        assert overlay.compression_ratio == 1.0


class TestReconstructionError:
    """Tests for ReconstructionError class."""

    def test_creation(self):
        """Test ReconstructionError creation."""
        error = ReconstructionError(
            name="gate",
            mean_relative_error=0.01,
            max_relative_error=0.05,
            mean_mse=0.001,
        )
        assert error.name == "gate"
        assert error.mean_relative_error == 0.01
        assert error.max_relative_error == 0.05
        assert error.mean_mse == 0.001


class TestReconstructionVerification:
    """Tests for ReconstructionVerification class."""

    def _create_verification(
        self,
        max_output_error: float = 0.005,
        gate_error: float = 0.01,
        up_error: float = 0.02,
        down_error: float = 0.015,
    ) -> ReconstructionVerification:
        """Helper to create ReconstructionVerification."""
        return ReconstructionVerification(
            model_id="test",
            layer_idx=0,
            gate=ReconstructionError(
                name="gate",
                mean_relative_error=gate_error,
                max_relative_error=gate_error * 2,
                mean_mse=0.001,
            ),
            up=ReconstructionError(
                name="up",
                mean_relative_error=up_error,
                max_relative_error=up_error * 2,
                mean_mse=0.001,
            ),
            down=ReconstructionError(
                name="down",
                mean_relative_error=down_error,
                max_relative_error=down_error * 2,
                mean_mse=0.001,
            ),
            mean_output_error=max_output_error * 0.5,
            max_output_error=max_output_error,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )

    def test_passed_true(self):
        """Test passed property when error < 1% (line 132-134)."""
        verification = self._create_verification(max_output_error=0.005)
        assert verification.passed is True

    def test_passed_false(self):
        """Test passed property when error >= 1%."""
        verification = self._create_verification(max_output_error=0.02)
        assert verification.passed is False

    def test_passed_boundary(self):
        """Test passed property at 1% boundary."""
        # Exactly at 1% should fail
        verification = self._create_verification(max_output_error=0.01)
        assert verification.passed is False
        # Just under 1% should pass
        verification = self._create_verification(max_output_error=0.0099)
        assert verification.passed is True

    def test_overall_weight_error(self):
        """Test overall_weight_error property (lines 137-143)."""
        verification = self._create_verification(gate_error=0.01, up_error=0.02, down_error=0.03)
        expected = (0.01 + 0.02 + 0.03) / 3
        assert abs(verification.overall_weight_error - expected) < 0.0001


class TestStorageEstimate:
    """Tests for StorageEstimate class."""

    def test_compression_ratio(self):
        """Test compression_ratio property (line 164-167)."""
        estimate = StorageEstimate(
            model_id="test",
            num_layers=24,
            num_experts=32,
            original_mb=1000.0,
            compressed_mb=200.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        assert estimate.compression_ratio == 5.0

    def test_compression_ratio_zero_compressed(self):
        """Test compression_ratio when compressed_mb is 0."""
        estimate = StorageEstimate(
            model_id="test",
            num_layers=24,
            num_experts=32,
            original_mb=1000.0,
            compressed_mb=0.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        assert estimate.compression_ratio == 1.0

    def test_savings_mb(self):
        """Test savings_mb property (lines 169-172)."""
        estimate = StorageEstimate(
            model_id="test",
            num_layers=24,
            num_experts=32,
            original_mb=1000.0,
            compressed_mb=200.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        assert estimate.savings_mb == 800.0


class TestCompressionConfig:
    """Tests for CompressionConfig class."""

    def test_compression_ratio(self):
        """Test compression_ratio property (lines 202-205)."""
        config = CompressionConfig(
            model_id="test",
            num_layers=24,
            num_experts=32,
            moe_layer_indices=list(range(24)),
            gate_shape=(100, 100),
            up_shape=(100, 400),
            down_shape=(400, 100),
            gate_rank=2,
            up_rank=128,
            down_rank=64,
            has_biases=False,
            original_bytes=10000000,
            compressed_bytes=2000000,
        )
        assert config.compression_ratio == 5.0

    def test_compression_ratio_zero_compressed(self):
        """Test compression_ratio when compressed_bytes is 0."""
        config = CompressionConfig(
            model_id="test",
            num_layers=24,
            num_experts=32,
            moe_layer_indices=list(range(24)),
            gate_shape=(100, 100),
            up_shape=(100, 400),
            down_shape=(400, 100),
            gate_rank=2,
            up_rank=128,
            down_rank=64,
            has_biases=False,
            original_bytes=10000000,
            compressed_bytes=0,
        )
        assert config.compression_ratio == 1.0


class TestCompressionResult:
    """Tests for CompressionResult class."""

    def _create_result(
        self, original_bytes: int = 10000000, compressed_bytes: int = 2000000
    ) -> CompressionResult:
        """Helper to create CompressionResult."""
        config = CompressionConfig(
            model_id="test",
            num_layers=24,
            num_experts=32,
            moe_layer_indices=list(range(24)),
            gate_shape=(100, 100),
            up_shape=(100, 400),
            down_shape=(400, 100),
            gate_rank=2,
            up_rank=128,
            down_rank=64,
            has_biases=False,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )
        return CompressionResult(
            output_path="/tmp/compressed",
            config=config,
            mean_reconstruction_error=0.005,
            max_reconstruction_error=0.008,
        )

    def test_compression_ratio(self):
        """Test compression_ratio property (lines 218-221)."""
        result = self._create_result()
        assert result.compression_ratio == 5.0

    def test_original_mb(self):
        """Test original_mb property (lines 223-226)."""
        result = self._create_result(original_bytes=10 * 1024 * 1024)
        assert result.original_mb == 10.0

    def test_compressed_mb(self):
        """Test compressed_mb property (lines 228-231)."""
        result = self._create_result(compressed_bytes=2 * 1024 * 1024)
        assert result.compressed_mb == 2.0


class TestMoECompressionServiceConstants:
    """Tests for MoECompressionService constants."""

    def test_default_gate_rank(self):
        """Test DEFAULT_GATE_RANK constant."""
        assert MoECompressionService.DEFAULT_GATE_RANK == 2

    def test_default_up_rank(self):
        """Test DEFAULT_UP_RANK constant."""
        assert MoECompressionService.DEFAULT_UP_RANK == 128

    def test_default_down_rank(self):
        """Test DEFAULT_DOWN_RANK constant."""
        assert MoECompressionService.DEFAULT_DOWN_RANK == 64

    def test_variance_threshold(self):
        """Test VARIANCE_THRESHOLD constant."""
        assert MoECompressionService.VARIANCE_THRESHOLD == 0.95


class TestPydanticModelsFrozen:
    """Tests for frozen Pydantic models."""

    def test_projection_overlay_frozen(self):
        """Test ProjectionOverlay is frozen."""
        overlay = ProjectionOverlay(
            name="gate",
            shape=(100, 100),
            rank=10,
            num_experts=8,
            original_bytes=1000,
            compressed_bytes=200,
        )
        with pytest.raises(ValidationError):
            overlay.rank = 20

    def test_reconstruction_error_frozen(self):
        """Test ReconstructionError is frozen."""
        error = ReconstructionError(
            name="gate",
            mean_relative_error=0.01,
            max_relative_error=0.05,
            mean_mse=0.001,
        )
        with pytest.raises(ValidationError):
            error.name = "up"

    def test_storage_estimate_frozen(self):
        """Test StorageEstimate is frozen."""
        estimate = StorageEstimate(
            model_id="test",
            num_layers=24,
            num_experts=32,
            original_mb=1000.0,
            compressed_mb=200.0,
            gate_rank=2,
            up_rank=128,
            down_rank=64,
        )
        with pytest.raises(ValidationError):
            estimate.original_mb = 2000.0


class TestOverlayExperts:
    """Tests for OverlayExperts class."""

    def _create_config(self) -> CompressionConfig:
        """Create a test CompressionConfig."""
        return CompressionConfig(
            model_id="test-model",
            num_layers=2,
            num_experts=4,
            moe_layer_indices=[0, 1],
            gate_shape=(10, 10),
            up_shape=(10, 40),
            down_shape=(40, 10),
            gate_rank=2,
            up_rank=8,
            down_rank=4,
            has_biases=False,
            original_bytes=10000,
            compressed_bytes=2000,
        )

    def _create_overlay_experts(self, with_biases: bool = False):
        """Create OverlayExperts with mock weights."""
        import mlx.core as mx

        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        config = self._create_config()

        # Create base weights for layer 0, gate projection
        base_weights = {
            "layer_0_gate_base": mx.zeros((10, 10)),
        }

        # Create delta weights (U and V matrices) for expert 0
        delta_weights = {
            "layer_0_gate_expert_0_U": mx.ones((10, 2)),  # (out_dim, rank)
            "layer_0_gate_expert_0_V": mx.ones((2, 10)),  # (rank, in_dim)
        }

        biases = None
        if with_biases:
            biases = {
                "gate_bias": mx.zeros((4, 10)),  # (num_experts, out_dim)
            }

        return OverlayExperts(config, base_weights, delta_weights, biases)

    def test_init(self):
        """Test OverlayExperts initialization (lines 1104-1115)."""
        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        config = self._create_config()
        base = {"layer_0_gate_base": mx.zeros((10, 10))}
        deltas = {"layer_0_gate_expert_0_U": mx.ones((10, 2))}

        experts = OverlayExperts(config, base, deltas)

        assert experts.config is config
        assert experts._base is base
        assert experts._deltas is deltas
        assert experts._biases == {}

    def test_init_with_biases(self):
        """Test OverlayExperts initialization with biases."""
        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        config = self._create_config()
        base = {"layer_0_gate_base": mx.zeros((10, 10))}
        deltas = {"layer_0_gate_expert_0_U": mx.ones((10, 2))}
        biases = {"gate_bias": mx.zeros((4, 10))}

        experts = OverlayExperts(config, base, deltas, biases)

        assert experts._biases is biases

    def test_num_layers_property(self):
        """Test num_layers property (lines 1241-1244)."""
        experts = self._create_overlay_experts()
        assert experts.num_layers == 2

    def test_num_experts_property(self):
        """Test num_experts property (lines 1246-1249)."""
        experts = self._create_overlay_experts()
        assert experts.num_experts == 4

    def test_moe_layer_indices_property(self):
        """Test moe_layer_indices property (lines 1251-1254)."""
        experts = self._create_overlay_experts()
        assert experts.moe_layer_indices == [0, 1]

    def test_memory_usage_mb(self):
        """Test memory_usage_mb method (lines 1256-1263)."""
        experts = self._create_overlay_experts()
        usage = experts.memory_usage_mb()
        # Should be a positive number
        assert usage > 0
        assert isinstance(usage, float)

    def test_get_expert_weight(self):
        """Test get_expert_weight method (lines 1157-1192)."""
        experts = self._create_overlay_experts()
        weight = experts.get_expert_weight(layer=0, projection="gate", expert=0)
        # base (zeros) + U @ V = rank * ones
        assert weight.shape == (10, 10)

    def test_get_expert_weight_missing_base(self):
        """Test get_expert_weight raises KeyError for missing base (lines 1179-1180)."""
        experts = self._create_overlay_experts()
        with pytest.raises(KeyError, match="Base weight not found"):
            experts.get_expert_weight(layer=99, projection="gate", expert=0)

    def test_get_expert_weight_missing_delta(self):
        """Test get_expert_weight raises KeyError for missing delta (lines 1181-1182)."""
        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        config = self._create_config()
        # Base exists but no deltas
        base = {"layer_0_gate_base": mx.zeros((10, 10))}
        deltas = {}

        experts = OverlayExperts(config, base, deltas)

        with pytest.raises(KeyError, match="Delta U not found"):
            experts.get_expert_weight(layer=0, projection="gate", expert=0)

    def test_apply_expert(self):
        """Test apply_expert method (lines 1194-1239)."""
        experts = self._create_overlay_experts()
        x = mx.ones((1, 10))
        result = experts.apply_expert(layer=0, projection="gate", expert=0, x=x)
        assert result.shape == (1, 10)

    def test_apply_expert_with_biases(self):
        """Test apply_expert with biases (lines 1234-1237)."""
        experts = self._create_overlay_experts(with_biases=True)
        x = mx.ones((1, 10))
        result = experts.apply_expert(layer=0, projection="gate", expert=0, x=x)
        assert result.shape == (1, 10)


class TestOverlayExpertsLoad:
    """Tests for OverlayExperts.load method."""

    def test_load_config_not_found(self, tmp_path):
        """Test load raises FileNotFoundError for missing config (lines 1126-1127)."""
        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        with pytest.raises(FileNotFoundError, match="Config not found"):
            OverlayExperts.load(tmp_path)

    def test_load_base_weights_not_found(self, tmp_path):
        """Test load raises FileNotFoundError for missing base weights (lines 1135-1136)."""
        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        # Create config file
        config = CompressionConfig(
            model_id="test",
            num_layers=2,
            num_experts=4,
            moe_layer_indices=[0, 1],
            gate_shape=(10, 10),
            up_shape=(10, 40),
            down_shape=(40, 10),
            gate_rank=2,
            up_rank=8,
            down_rank=4,
            has_biases=False,
            original_bytes=10000,
            compressed_bytes=2000,
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(config.model_dump_json())

        with pytest.raises(FileNotFoundError, match="Base weights not found"):
            OverlayExperts.load(tmp_path)

    def test_load_deltas_not_found(self, tmp_path):
        """Test load raises FileNotFoundError for missing deltas (lines 1137-1138)."""
        import numpy as np
        from safetensors.numpy import save_file as save_safetensors

        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        # Create config file
        config = CompressionConfig(
            model_id="test",
            num_layers=2,
            num_experts=4,
            moe_layer_indices=[0, 1],
            gate_shape=(10, 10),
            up_shape=(10, 40),
            down_shape=(40, 10),
            gate_rank=2,
            up_rank=8,
            down_rank=4,
            has_biases=False,
            original_bytes=10000,
            compressed_bytes=2000,
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(config.model_dump_json())

        # Create base weights file (safetensors needs numpy arrays)
        base_weights = {"layer_0_gate_base": np.zeros((10, 10), dtype=np.float16)}
        save_safetensors(base_weights, str(tmp_path / "base_weights.safetensors"))

        with pytest.raises(FileNotFoundError, match="Deltas not found"):
            OverlayExperts.load(tmp_path)

    def test_load_success(self, tmp_path):
        """Test successful load (lines 1117-1155)."""
        import numpy as np
        from safetensors.numpy import save_file as save_safetensors

        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        # Create config file
        config = CompressionConfig(
            model_id="test",
            num_layers=2,
            num_experts=4,
            moe_layer_indices=[0, 1],
            gate_shape=(10, 10),
            up_shape=(10, 40),
            down_shape=(40, 10),
            gate_rank=2,
            up_rank=8,
            down_rank=4,
            has_biases=False,
            original_bytes=10000,
            compressed_bytes=2000,
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(config.model_dump_json())

        # Create base weights file
        base_weights = {"layer_0_gate_base": np.zeros((10, 10), dtype=np.float16)}
        save_safetensors(base_weights, str(tmp_path / "base_weights.safetensors"))

        # Create deltas file
        deltas = {
            "layer_0_gate_expert_0_U": np.ones((10, 2), dtype=np.float16),
            "layer_0_gate_expert_0_V": np.ones((2, 10), dtype=np.float16),
        }
        save_safetensors(deltas, str(tmp_path / "deltas.safetensors"))

        experts = OverlayExperts.load(tmp_path)

        assert experts.num_layers == 2
        assert experts.num_experts == 4

    def test_load_with_biases(self, tmp_path):
        """Test load with biases file (lines 1143-1148)."""
        import numpy as np
        from safetensors.numpy import save_file as save_safetensors

        from chuk_lazarus.introspection.moe.moe_compression import OverlayExperts

        # Create config file
        config = CompressionConfig(
            model_id="test",
            num_layers=2,
            num_experts=4,
            moe_layer_indices=[0, 1],
            gate_shape=(10, 10),
            up_shape=(10, 40),
            down_shape=(40, 10),
            gate_rank=2,
            up_rank=8,
            down_rank=4,
            has_biases=True,
            original_bytes=10000,
            compressed_bytes=2000,
        )
        config_path = tmp_path / "config.json"
        config_path.write_text(config.model_dump_json())

        # Create files
        base_weights = {"layer_0_gate_base": np.zeros((10, 10), dtype=np.float16)}
        save_safetensors(base_weights, str(tmp_path / "base_weights.safetensors"))

        deltas = {
            "layer_0_gate_expert_0_U": np.ones((10, 2), dtype=np.float16),
            "layer_0_gate_expert_0_V": np.ones((2, 10), dtype=np.float16),
        }
        save_safetensors(deltas, str(tmp_path / "deltas.safetensors"))

        biases = {"gate_bias": np.zeros((4, 10), dtype=np.float16)}
        save_safetensors(biases, str(tmp_path / "biases.safetensors"))

        experts = OverlayExperts.load(tmp_path)

        assert "gate_bias" in experts._biases
