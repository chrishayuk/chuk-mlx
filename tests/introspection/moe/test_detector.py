"""Tests for MoE architecture detection."""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.introspection.moe.detector import (
    detect_moe_architecture,
    get_moe_layer_info,
    get_moe_layers,
    is_moe_model,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture

# =============================================================================
# Mock Models for Testing
# =============================================================================


class MockRouter(nn.Module):
    """Mock router for testing."""

    def __init__(self, num_experts: int = 8, num_experts_per_tok: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight = mx.zeros((num_experts, 64))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return mx.zeros((x.shape[0], 2)), mx.zeros((x.shape[0], 2), dtype=mx.int32)


class MockMoE(nn.Module):
    """Mock MoE layer for testing."""

    def __init__(self, num_experts: int = 8, num_experts_per_tok: int = 2):
        super().__init__()
        self.router = MockRouter(num_experts, num_experts_per_tok)
        self.experts = [nn.Linear(64, 64) for _ in range(num_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockMoEWithSharedExpert(nn.Module):
    """Mock MoE with shared expert (Llama4 style)."""

    def __init__(self):
        super().__init__()
        self.router = MockRouter()
        self.shared_expert = nn.Linear(64, 64)
        self.experts = [nn.Linear(64, 64) for _ in range(8)]

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockBatchedExperts:
    """Mock batched experts (GPT-OSS style)."""

    def __init__(self):
        self.gate_up_proj_blocks = mx.zeros((8, 64, 128))


class MockGPTOSSMoE(nn.Module):
    """Mock GPT-OSS style MoE."""

    def __init__(self):
        super().__init__()
        self.router = MockRouter(32, 4)
        self.experts = MockBatchedExperts()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockTransformerLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, moe: nn.Module | None = None):
        super().__init__()
        self.mlp = moe if moe else MockMoE()

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class MockHybridLayer(nn.Module):
    """Mock hybrid layer with Mamba."""

    def __init__(self):
        super().__init__()
        self.mlp = MockMoE()
        self.mamba = nn.Linear(64, 64)  # Fake Mamba block

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockDenseLayer(nn.Module):
    """Mock dense layer (non-MoE)."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(64, 128)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class MockModel(nn.Module):
    """Mock transformer model."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.model = type("InnerModel", (), {"layers": layers})()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockModelDirect(nn.Module):
    """Mock model with direct layers attribute."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockTransformerModel(nn.Module):
    """Mock model with transformer attribute."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.transformer = type("Transformer", (), {"layers": layers})()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockDecoderModel(nn.Module):
    """Mock model with decoder attribute."""

    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.decoder = type("Decoder", (), {"layers": layers})()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class MockEmptyModel(nn.Module):
    """Mock model with no layers."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


# =============================================================================
# Tests
# =============================================================================


class TestDetectMoEArchitecture:
    """Tests for detect_moe_architecture function."""

    def test_generic_moe(self):
        """Test detection of generic MoE with router only (no experts list)."""

        # A model with just a router but no experts list returns GENERIC
        class RouterOnlyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.router = MockRouter()
                # No experts list, so not MIXTRAL

            def __call__(self, x):
                return x

        class RouterOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = RouterOnlyMoE()

            def __call__(self, x):
                return x

        layers = [RouterOnlyLayer()]
        model = MockModel(layers)
        assert detect_moe_architecture(model) == MoEArchitecture.GENERIC

    def test_mixtral_style(self):
        """Test detection of Mixtral-style MoE."""
        moe = MockMoE(8, 2)
        layers = [MockTransformerLayer(moe)]
        model = MockModel(layers)
        # Generic MoE with list experts returns MIXTRAL
        assert detect_moe_architecture(model) == MoEArchitecture.MIXTRAL

    def test_llama4_style(self):
        """Test detection of Llama4-style MoE with shared expert."""
        moe = MockMoEWithSharedExpert()
        layers = [MockTransformerLayer(moe)]
        model = MockModel(layers)
        assert detect_moe_architecture(model) == MoEArchitecture.LLAMA4

    def test_gpt_oss_style(self):
        """Test detection of GPT-OSS-style batched MoE."""
        moe = MockGPTOSSMoE()
        layers = [MockTransformerLayer(moe)]
        model = MockModel(layers)
        assert detect_moe_architecture(model) == MoEArchitecture.GPT_OSS

    def test_granite_hybrid(self):
        """Test detection of Granite hybrid with Mamba."""
        layers = [MockHybridLayer()]
        model = MockModel(layers)
        assert detect_moe_architecture(model) == MoEArchitecture.GRANITE_HYBRID

    def test_no_layers(self):
        """Test empty model returns GENERIC."""
        model = MockEmptyModel()
        assert detect_moe_architecture(model) == MoEArchitecture.GENERIC

    def test_dense_only(self):
        """Test dense model returns GENERIC."""
        layers = [MockDenseLayer()]
        model = MockModel(layers)
        assert detect_moe_architecture(model) == MoEArchitecture.GENERIC


class TestGetMoELayerInfo:
    """Tests for get_moe_layer_info function."""

    def test_valid_moe_layer(self):
        """Test getting info from valid MoE layer."""
        layers = [MockTransformerLayer(MockMoE(8, 2))]
        model = MockModel(layers)
        info = get_moe_layer_info(model, 0)

        assert info is not None
        assert info.layer_idx == 0
        assert info.num_experts == 8
        assert info.num_experts_per_tok == 2
        assert info.has_shared_expert is False

    def test_layer_out_of_range(self):
        """Test out of range layer returns None."""
        layers = [MockTransformerLayer()]
        model = MockModel(layers)
        info = get_moe_layer_info(model, 10)
        assert info is None

    def test_dense_layer(self):
        """Test dense layer returns None."""
        layers = [MockDenseLayer()]
        model = MockModel(layers)
        info = get_moe_layer_info(model, 0)
        assert info is None

    def test_shared_expert_detection(self):
        """Test shared expert is detected."""
        moe = MockMoEWithSharedExpert()
        layers = [MockTransformerLayer(moe)]
        model = MockModel(layers)
        info = get_moe_layer_info(model, 0)

        assert info is not None
        assert info.has_shared_expert is True

    def test_no_mlp(self):
        """Test layer without mlp returns None."""
        layer = nn.Module()
        layers = [layer]
        model = MockModelDirect(layers)
        info = get_moe_layer_info(model, 0)
        assert info is None


class TestGetMoELayers:
    """Tests for get_moe_layers function."""

    def test_all_moe_layers(self):
        """Test finding all MoE layers."""
        layers = [MockTransformerLayer(), MockTransformerLayer()]
        model = MockModel(layers)
        moe_layers = get_moe_layers(model)
        assert moe_layers == [0, 1]

    def test_mixed_layers(self):
        """Test finding MoE layers in mixed model."""
        layers = [MockDenseLayer(), MockTransformerLayer(), MockDenseLayer()]
        model = MockModel(layers)
        moe_layers = get_moe_layers(model)
        assert moe_layers == [1]

    def test_no_moe_layers(self):
        """Test model with no MoE layers."""
        layers = [MockDenseLayer(), MockDenseLayer()]
        model = MockModel(layers)
        moe_layers = get_moe_layers(model)
        assert moe_layers == []

    def test_empty_model(self):
        """Test empty model."""
        model = MockEmptyModel()
        moe_layers = get_moe_layers(model)
        assert moe_layers == []


class TestIsMoEModel:
    """Tests for is_moe_model function."""

    def test_moe_model(self):
        """Test MoE model returns True."""
        layers = [MockTransformerLayer()]
        model = MockModel(layers)
        assert is_moe_model(model) is True

    def test_dense_model(self):
        """Test dense model returns False."""
        layers = [MockDenseLayer()]
        model = MockModel(layers)
        assert is_moe_model(model) is False

    def test_empty_model(self):
        """Test empty model returns False."""
        model = MockEmptyModel()
        assert is_moe_model(model) is False


class TestModelLayerExtraction:
    """Tests for layer extraction from different model structures."""

    def test_model_attribute(self):
        """Test extraction via model attribute."""
        layers = [MockTransformerLayer()]
        model = MockModel(layers)
        assert len(get_moe_layers(model)) == 1

    def test_transformer_attribute(self):
        """Test extraction via transformer attribute."""
        layers = [MockTransformerLayer()]
        model = MockTransformerModel(layers)
        assert len(get_moe_layers(model)) == 1

    def test_decoder_attribute(self):
        """Test extraction via decoder attribute."""
        layers = [MockTransformerLayer()]
        model = MockDecoderModel(layers)
        assert len(get_moe_layers(model)) == 1

    def test_direct_layers(self):
        """Test extraction via direct layers attribute."""
        layers = [MockTransformerLayer()]
        model = MockModelDirect(layers)
        assert len(get_moe_layers(model)) == 1
