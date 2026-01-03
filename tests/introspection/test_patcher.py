"""Tests for introspection patcher module."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from chuk_lazarus.introspection.enums import PatchEffect
from chuk_lazarus.introspection.patcher import (
    ActivationPatcher,
    CommutativityAnalyzer,
    LayerPatch,
)


class MockConfig:
    """Mock configuration."""

    def __init__(self, hidden_size: int = 64, model_id: str = "test-model"):
        self.hidden_size = hidden_size
        self.model_id = model_id


class MockEmbedding(nn.Module):
    """Mock embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.weight[input_ids]


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((hidden_size, hidden_size))

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        # Simple linear transformation
        if x.ndim == 3:
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
            out_flat = x_flat @ self.weight
            return out_flat.reshape(batch, seq, dim)
        return x @ self.weight


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()

        class InnerModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers):
                super().__init__()
                self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
                self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
                self.norm = nn.RMSNorm(hidden_size)

        self.model = InnerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        return self.lm_head(h)


class MockTokenizer:
    """Mock tokenizer."""

    def __init__(self):
        self._vocab = {str(i): i for i in range(100)}
        self._reverse = {i: str(i) for i in range(100)}

    def encode(self, text: str) -> list[int]:
        # Simple: use character codes
        return [ord(c) % 100 for c in text]

    def decode(self, ids: list[int]) -> str:
        # Return first token as string
        if isinstance(ids, list) and len(ids) > 0:
            return str(ids[0])
        return str(ids)


class TestLayerPatch:
    """Tests for LayerPatch dataclass."""

    def test_init_defaults(self):
        activation = np.random.randn(64)
        patch = LayerPatch(layer=5, activation=activation)
        assert patch.layer == 5
        assert patch.blend == 1.0
        assert patch.position == -1

    def test_init_custom(self):
        activation = np.random.randn(64)
        patch = LayerPatch(
            layer=10,
            activation=activation,
            blend=0.5,
            position=3,
        )
        assert patch.layer == 10
        assert patch.blend == 0.5
        assert patch.position == 3

    def test_with_mx_array(self):
        activation = mx.random.normal((64,))
        patch = LayerPatch(layer=2, activation=activation)
        assert isinstance(patch.activation, mx.array)


class TestActivationPatcher:
    """Tests for ActivationPatcher class."""

    def test_init(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        patcher = ActivationPatcher(model, tokenizer, config)
        assert patcher.model is model
        assert patcher.tokenizer is tokenizer
        assert patcher.config is config
        assert hasattr(patcher, "_accessor")

    def test_init_without_config(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        assert patcher.config is None

    @pytest.mark.asyncio
    async def test_capture_activation(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig()

        patcher = ActivationPatcher(model, tokenizer, config)
        activation = await patcher.capture_activation("test", layer=2)

        assert isinstance(activation, np.ndarray)
        assert activation.ndim == 1
        assert activation.shape[0] == 64  # hidden_size

    @pytest.mark.asyncio
    async def test_capture_activation_position(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        # Capture at specific position
        activation = await patcher.capture_activation("test", layer=1, position=0)

        assert isinstance(activation, np.ndarray)
        assert activation.shape[0] == 64

    @pytest.mark.asyncio
    async def test_patch_and_predict(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        patcher = ActivationPatcher(model, tokenizer, config)

        # Get source activation
        source_activation = await patcher.capture_activation("source", layer=2)

        # Patch into target
        top_token, top_prob = await patcher.patch_and_predict(
            target_prompt="target",
            source_activation=source_activation,
            layer=2,
            blend=1.0,
        )

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)
        assert 0 <= top_prob <= 1

    @pytest.mark.asyncio
    async def test_patch_and_predict_blend(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        source_activation = await patcher.capture_activation("source", layer=1)

        # Test different blend factors
        token_full, prob_full = await patcher.patch_and_predict(
            "target", source_activation, layer=1, blend=1.0
        )

        token_half, prob_half = await patcher.patch_and_predict(
            "target", source_activation, layer=1, blend=0.5
        )

        # Results should be valid
        assert isinstance(token_full, str)
        assert isinstance(token_half, str)

    @pytest.mark.asyncio
    async def test_patch_and_predict_with_numpy(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        # Use numpy array directly
        source_activation = np.random.randn(64).astype(np.float32)

        top_token, top_prob = await patcher.patch_and_predict("target", source_activation, layer=1)

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)

    @pytest.mark.asyncio
    async def test_sweep_layers(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig()

        patcher = ActivationPatcher(model, tokenizer, config)

        result = await patcher.sweep_layers(
            target_prompt="target",
            source_prompt="source",
            layers=[0, 2],
            blend=1.0,
        )

        assert result.model_id == "test-model"
        assert result.source_prompt == "source"
        assert result.target_prompt == "target"
        assert result.blend == 1.0
        assert len(result.layer_results) == 2

        # Check layer results
        for layer_result in result.layer_results:
            assert layer_result.layer in [0, 2]
            assert isinstance(layer_result.top_token, str)
            assert isinstance(layer_result.top_prob, float)
            assert isinstance(layer_result.effect, PatchEffect)

    @pytest.mark.asyncio
    async def test_sweep_layers_default(self):
        model = MockModel(num_layers=20)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)

        result = await patcher.sweep_layers(
            target_prompt="target",
            source_prompt="source",
        )

        # Should test every ~10th layer by default
        assert len(result.layer_results) > 0
        assert len(result.layer_results) <= 20

    @pytest.mark.asyncio
    async def test_sweep_layers_with_answers(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)

        result = await patcher.sweep_layers(
            target_prompt="2+2=",
            source_prompt="3+3=",
            layers=[1, 2],
            source_answer="6",
            target_answer="4",
        )

        assert result.source_answer == "6"
        assert result.target_answer == "4"

    @pytest.mark.asyncio
    async def test_effect_detection(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)

        result = await patcher.sweep_layers(
            target_prompt="target",
            source_prompt="source",
            layers=[1],
            source_answer="expected",
        )

        # Effect should be determined
        layer_result = result.layer_results[0]
        assert layer_result.effect in [
            PatchEffect.NO_CHANGE,
            PatchEffect.TRANSFERRED,
            PatchEffect.STILL_TARGET,
            PatchEffect.CHANGED,
        ]


class TestCommutativityAnalyzer:
    """Tests for CommutativityAnalyzer class."""

    def test_init(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = CommutativityAnalyzer(model, tokenizer, config)
        assert analyzer.model is model
        assert analyzer.tokenizer is tokenizer
        assert analyzer.config is config

    @pytest.mark.asyncio
    async def test_get_activation(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = CommutativityAnalyzer(model, tokenizer)
        activation = await analyzer.get_activation("test", layer=2)

        assert isinstance(activation, np.ndarray)
        assert activation.ndim == 1

    @pytest.mark.asyncio
    async def test_analyze_default_layer(self):
        model = MockModel(num_layers=10)
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = CommutativityAnalyzer(model, tokenizer, config)

        # Test with small set of pairs
        pairs = [
            ("2*3=", "3*2="),
            ("4*5=", "5*4="),
        ]

        result = await analyzer.analyze(pairs=pairs)

        assert result.model_id == "test-model"
        assert result.layer == 6  # 60% of 10 layers
        assert result.num_pairs == 2
        assert len(result.pairs) == 2

        # Check statistics (allow small floating point tolerance)
        assert -0.01 <= result.mean_similarity <= 1.01
        assert result.min_similarity <= result.max_similarity

    @pytest.mark.asyncio
    async def test_analyze_specific_layer(self):
        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()

        analyzer = CommutativityAnalyzer(model, tokenizer)

        pairs = [("2*3=", "3*2=")]
        result = await analyzer.analyze(layer=5, pairs=pairs)

        assert result.layer == 5
        assert result.num_pairs == 1

    @pytest.mark.asyncio
    async def test_analyze_default_pairs(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = CommutativityAnalyzer(model, tokenizer)

        # Use default pairs (all single-digit multiplication)
        result = await analyzer.analyze(layer=2)

        # Should have many pairs (2-9 range)
        assert result.num_pairs > 20
        assert len(result.pairs) == result.num_pairs

    @pytest.mark.asyncio
    async def test_similarity_values(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = CommutativityAnalyzer(model, tokenizer)

        pairs = [
            ("2*3=", "3*2="),
            ("5*6=", "6*5="),
        ]

        result = await analyzer.analyze(layer=1, pairs=pairs)

        # Check individual pair results
        for pair in result.pairs:
            assert pair.prompt_a in ["2*3=", "5*6="]
            assert pair.prompt_b in ["3*2=", "6*5="]
            # Allow small floating point tolerance
            assert -1.0001 <= pair.similarity <= 1.0001

    @pytest.mark.asyncio
    async def test_statistics_calculation(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = CommutativityAnalyzer(model, tokenizer)

        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        result = await analyzer.analyze(layer=0, pairs=pairs)

        # Check that statistics are calculated
        assert isinstance(result.mean_similarity, float)
        assert isinstance(result.std_similarity, float)
        assert isinstance(result.min_similarity, float)
        assert isinstance(result.max_similarity, float)

        # Min should be <= mean <= max
        assert result.min_similarity <= result.mean_similarity <= result.max_similarity


class TestPatchedLayerWrapper:
    """Test the internal patched layer wrapper."""

    @pytest.mark.asyncio
    async def test_wrapper_preserves_attributes(self):
        model = MockModel(num_layers=2)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)

        original_layer = patcher._accessor.get_layer(0)
        source_activation = mx.random.normal((64,))

        wrapped = patcher._create_patched_layer(
            original_layer, source_activation, blend=1.0, position=-1
        )

        # Wrapper should preserve original layer functionality
        assert hasattr(wrapped, "_wrapped")
        assert wrapped._wrapped is original_layer

    @pytest.mark.asyncio
    async def test_wrapper_patches_activation(self):
        model = MockModel(num_layers=2)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        original_layer = patcher._accessor.get_layer(0)

        # Create known source activation
        source_activation = mx.ones((64,))

        wrapped = patcher._create_patched_layer(
            original_layer, source_activation, blend=1.0, position=-1
        )

        # Run through wrapper
        input_h = mx.random.normal((1, 3, 64))
        output_h = wrapped(input_h)

        # Should produce output (exact value depends on layer implementation)
        assert output_h.shape == input_h.shape

    @pytest.mark.asyncio
    async def test_wrapper_blend_factor(self):
        model = MockModel(num_layers=2)
        tokenizer = MockTokenizer()

        patcher = ActivationPatcher(model, tokenizer)
        original_layer = patcher._accessor.get_layer(0)
        source_activation = mx.random.normal((64,))

        # Test different blend factors
        wrapped_full = patcher._create_patched_layer(
            original_layer, source_activation, blend=1.0, position=-1
        )
        wrapped_half = patcher._create_patched_layer(
            original_layer, source_activation, blend=0.5, position=-1
        )

        input_h = mx.random.normal((1, 3, 64))
        output_full = wrapped_full(input_h)
        output_half = wrapped_half(input_h)

        # Both should produce valid outputs
        assert output_full.shape == input_h.shape
        assert output_half.shape == input_h.shape
