"""Tests for async model analyzer."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.analyzer import (
    AnalysisConfig,
    AnalysisResult,
    LayerPredictionResult,
    LayerStrategy,
    ModelAnalyzer,
    ModelInfo,
    TokenEvolutionResult,
    TokenPrediction,
    _is_quantized_model,
)
from chuk_lazarus.models_v2.families.registry import (
    ModelFamilyType,
    detect_model_family,
)


# Reuse simple model from other tests
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SimpleMLP(hidden_size)

    def __call__(self, x: mx.array, mask=None, cache=None) -> tuple[mx.array, None]:
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x, None


class SimpleTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [SimpleTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)
        self.hidden_size = hidden_size

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class SimpleForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = SimpleTransformerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_word_embeddings = False

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        # Simple encoding: return ASCII values mod vocab_size
        if not text:
            return []
        return [ord(c) % self.vocab_size for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i % 26) + 65) for i in ids)

    def get_vocab(self):
        return {chr(i + 65): i for i in range(26)}


class TestLayerStrategy:
    """Tests for LayerStrategy enum."""

    def test_enum_values(self):
        assert LayerStrategy.ALL.value == "all"
        assert LayerStrategy.EVENLY_SPACED.value == "evenly_spaced"
        assert LayerStrategy.FIRST_LAST.value == "first_last"
        assert LayerStrategy.CUSTOM.value == "custom"


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_default_config(self):
        config = AnalysisConfig()
        assert config.layer_strategy == LayerStrategy.EVENLY_SPACED
        assert config.layer_step == 4
        assert config.top_k == 5
        assert config.track_tokens == []

    def test_custom_config(self):
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.ALL,
            layer_step=2,
            top_k=10,
            track_tokens=["hello", "world"],
        )
        assert config.layer_strategy == LayerStrategy.ALL
        assert config.layer_step == 2
        assert config.top_k == 10
        assert config.track_tokens == ["hello", "world"]

    def test_custom_layers(self):
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.CUSTOM,
            custom_layers=[0, 4, 8, 12],
        )
        assert config.custom_layers == [0, 4, 8, 12]


class TestTokenPrediction:
    """Tests for TokenPrediction model."""

    def test_creation(self):
        pred = TokenPrediction(
            token="hello",
            token_id=42,
            probability=0.85,
            rank=1,
        )
        assert pred.token == "hello"
        assert pred.token_id == 42
        assert pred.probability == 0.85
        assert pred.rank == 1

    def test_frozen(self):
        from pydantic import ValidationError

        pred = TokenPrediction(token="test", token_id=1, probability=0.5, rank=1)
        # Pydantic frozen models raise validation error on modification
        with pytest.raises(ValidationError):
            pred.token = "modified"


class TestLayerPredictionResult:
    """Tests for LayerPredictionResult model."""

    def test_creation(self):
        predictions = [
            TokenPrediction(token="a", token_id=1, probability=0.5, rank=1),
            TokenPrediction(token="b", token_id=2, probability=0.3, rank=2),
        ]
        result = LayerPredictionResult(layer_idx=4, predictions=predictions)
        assert result.layer_idx == 4
        assert len(result.predictions) == 2

    def test_top_token_property(self):
        predictions = [
            TokenPrediction(token="first", token_id=1, probability=0.7, rank=1),
        ]
        result = LayerPredictionResult(layer_idx=0, predictions=predictions)
        assert result.top_token == "first"
        assert result.top_probability == 0.7

    def test_empty_predictions(self):
        result = LayerPredictionResult(layer_idx=0, predictions=[])
        assert result.top_token == ""
        assert result.top_probability == 0.0


class TestTokenEvolutionResult:
    """Tests for TokenEvolutionResult model."""

    def test_creation(self):
        result = TokenEvolutionResult(
            token="test",
            token_id=42,
            layer_probabilities={0: 0.1, 4: 0.3, 8: 0.8},
            layer_ranks={0: 50, 4: 10, 8: 1},
            emergence_layer=8,
        )
        assert result.token == "test"
        assert result.emergence_layer == 8
        assert result.layer_probabilities[4] == 0.3


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_creation(self):
        final_pred = [TokenPrediction(token="x", token_id=1, probability=0.9, rank=1)]
        result = AnalysisResult(
            prompt="test",
            tokens=["t", "e", "s", "t"],
            num_layers=12,
            captured_layers=[0, 4, 8, 11],
            final_prediction=final_pred,
            layer_predictions=[],
        )
        assert result.prompt == "test"
        assert result.predicted_token == "x"
        assert result.predicted_probability == 0.9

    def test_empty_final_prediction(self):
        result = AnalysisResult(
            prompt="test",
            tokens=["t"],
            num_layers=4,
            captured_layers=[0, 3],
            final_prediction=[],
            layer_predictions=[],
        )
        assert result.predicted_token == ""
        assert result.predicted_probability == 0.0


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_creation(self):
        info = ModelInfo(
            model_id="test-model",
            num_layers=12,
            hidden_size=768,
            vocab_size=32000,
            has_tied_embeddings=True,
        )
        assert info.model_id == "test-model"
        assert info.num_layers == 12
        assert info.has_tied_embeddings


class TestModelAnalyzer:
    """Tests for ModelAnalyzer class."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def analyzer(self, model, tokenizer):
        return ModelAnalyzer.from_model(model, tokenizer, model_id="test-model")

    def test_from_model(self, model, tokenizer):
        analyzer = ModelAnalyzer.from_model(model, tokenizer, model_id="custom")
        assert analyzer._model is model
        assert analyzer._tokenizer is tokenizer
        assert analyzer._model_id == "custom"

    def test_model_info(self, analyzer):
        info = analyzer.model_info
        assert isinstance(info, ModelInfo)
        assert info.model_id == "test-model"
        assert info.num_layers == 4
        assert info.hidden_size == 64
        assert info.vocab_size == 100

    def test_analyze_sync(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL, top_k=3)
        result = analyzer._analyze_sync("test", config)

        assert isinstance(result, AnalysisResult)
        assert result.prompt == "test"
        assert len(result.tokens) > 0
        assert result.num_layers == 4
        assert len(result.layer_predictions) > 0

    def test_get_layers_to_capture_all(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL)
        layers = analyzer._get_layers_to_capture(4, config)
        assert layers == [0, 1, 2, 3]

    def test_get_layers_to_capture_first_last(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST)
        layers = analyzer._get_layers_to_capture(12, config)
        assert layers == [0, 11]

    def test_get_layers_to_capture_evenly_spaced(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.EVENLY_SPACED, layer_step=4)
        layers = analyzer._get_layers_to_capture(16, config)
        assert 0 in layers
        assert 15 in layers  # Last layer always included
        assert 4 in layers
        assert 8 in layers

    def test_get_layers_to_capture_custom(self, analyzer):
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.CUSTOM,
            custom_layers=[0, 5, 10],
        )
        layers = analyzer._get_layers_to_capture(12, config)
        assert layers == [0, 5, 10]

    def test_get_layers_to_capture_custom_empty(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.CUSTOM, custom_layers=None)
        layers = analyzer._get_layers_to_capture(12, config)
        assert layers == [0, 11]  # Falls back to first/last

    def test_get_top_predictions(self, analyzer):
        logits = mx.random.normal(shape=(100,))
        predictions = analyzer._get_top_predictions(logits, top_k=5)

        assert len(predictions) == 5
        for i, pred in enumerate(predictions):
            assert pred.rank == i + 1
            assert 0 <= pred.probability <= 1

    def test_embedding_scale(self, model, tokenizer):
        # Test with explicit embedding scale
        analyzer = ModelAnalyzer.from_model(
            model, tokenizer, model_id="scaled", embedding_scale=33.94
        )
        assert analyzer._embedding_scale == 33.94


class TestModelAnalyzerSync:
    """Synchronous tests for ModelAnalyzer using _analyze_sync."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    def test_analyze_sync(self, model, tokenizer):
        analyzer = ModelAnalyzer.from_model(model, tokenizer)
        config = AnalysisConfig()
        result = analyzer._analyze_sync("hello world", config)

        assert isinstance(result, AnalysisResult)
        assert result.prompt == "hello world"
        assert len(result.final_prediction) > 0

    def test_analyze_sync_with_config(self, model, tokenizer):
        analyzer = ModelAnalyzer.from_model(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL, top_k=3)
        result = analyzer._analyze_sync("test", config)

        assert isinstance(result, AnalysisResult)
        assert len(result.layer_predictions) == 4  # All layers

    def test_analyze_sync_with_tracking(self, model, tokenizer):
        analyzer = ModelAnalyzer.from_model(model, tokenizer)
        config = AnalysisConfig(track_tokens=["A", "B"])
        result = analyzer._analyze_sync("test", config)

        # Token evolutions may or may not be populated depending on tokenizer
        assert isinstance(result.token_evolutions, list)


class TestAnalyzerHelpers:
    """Test helper functions in analyzer module."""

    def test_is_quantized_model(self):
        # Quantized in config
        assert _is_quantized_model({"quantization_config": {}}, "model")

        # Quantized in model ID
        assert _is_quantized_model({}, "mlx-community/model-4bit")
        assert _is_quantized_model({}, "mlx-community/model-8bit")

        # Not quantized
        assert not _is_quantized_model({}, "regular-model")

    def test_detect_model_family(self):
        """Test model family detection via registry."""
        # Gemma variants
        assert detect_model_family({"model_type": "gemma"}) == ModelFamilyType.GEMMA
        assert detect_model_family({"model_type": "gemma3"}) == ModelFamilyType.GEMMA
        assert detect_model_family({"model_type": "gemma3_text"}) == ModelFamilyType.GEMMA

        # Llama variants
        assert detect_model_family({"model_type": "llama"}) == ModelFamilyType.LLAMA
        assert detect_model_family({"model_type": "mistral"}) == ModelFamilyType.LLAMA

        # Other families
        assert detect_model_family({"model_type": "qwen3"}) == ModelFamilyType.QWEN3
        assert detect_model_family({"model_type": "granite"}) == ModelFamilyType.GRANITE


class TestAnalyzerInternals:
    """Test internal methods of ModelAnalyzer."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def analyzer(self, model, tokenizer):
        return ModelAnalyzer.from_model(model, tokenizer)

    def test_get_num_layers(self, analyzer):
        assert analyzer._get_num_layers() == 4

    def test_get_hidden_size(self, analyzer):
        assert analyzer._get_hidden_size() == 64

    def test_get_vocab_size(self, analyzer):
        assert analyzer._get_vocab_size() == 100

    def test_analyze_sync(self, analyzer):
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL)
        result = analyzer._analyze_sync("hello", config)
        assert result is not None
        assert len(result.layer_predictions) > 0

    def test_get_top_predictions(self, analyzer):
        logits = mx.random.normal(shape=(100,))
        predictions = analyzer._get_top_predictions(logits, top_k=5)
        assert len(predictions) == 5
        for pred in predictions:
            assert pred.rank >= 1
            assert pred.probability >= 0


class TestAsyncMethods:
    """Test async methods of ModelAnalyzer."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def analyzer(self, model, tokenizer):
        return ModelAnalyzer.from_model(model, tokenizer)

    @pytest.mark.asyncio
    async def test_analyze_async(self, analyzer):
        """Test async analyze method."""
        result = await analyzer.analyze("hello world")

        assert isinstance(result, AnalysisResult)
        assert result.prompt == "hello world"
        assert len(result.final_prediction) > 0

    @pytest.mark.asyncio
    async def test_analyze_with_config(self, analyzer):
        """Test analyze with explicit config."""
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL, top_k=3)
        result = await analyzer.analyze("test", config)

        assert isinstance(result, AnalysisResult)
        assert len(result.layer_predictions) == 4

    @pytest.mark.asyncio
    async def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        prompts = ["hello", "world", "test"]
        results = await analyzer.analyze_batch(prompts)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, AnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_batch_with_config(self, analyzer):
        """Test batch analysis with shared config."""
        prompts = ["hello", "world"]
        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST)
        results = await analyzer.analyze_batch(prompts, config)

        assert len(results) == 2
        for result in results:
            assert len(result.captured_layers) == 2  # first and last


class TestFromPretrainedMocked:
    """Test from_pretrained with mocked model loading."""

    @pytest.mark.asyncio
    async def test_from_pretrained_context_manager(self):
        """Test from_pretrained as async context manager with mocking."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)
        mock_config = ModelConfig(vocab_size=100, hidden_size=64, num_hidden_layers=4)

        with patch(
            "chuk_lazarus.introspection.analyzer._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained("test-model") as analyzer:
                assert analyzer._model is mock_model
                assert analyzer._tokenizer is mock_tokenizer
                assert analyzer._model_id == "test-model"
                assert analyzer._config is mock_config

    @pytest.mark.asyncio
    async def test_from_pretrained_with_embedding_scale(self):
        """Test from_pretrained with embedding scale override."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)
        mock_config = ModelConfig(vocab_size=100, hidden_size=64, num_hidden_layers=4)

        with patch(
            "chuk_lazarus.introspection.analyzer._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained(
                "test-model", embedding_scale=33.94
            ) as analyzer:
                # Explicit override takes precedence
                assert analyzer._embedding_scale == 33.94

    @pytest.mark.asyncio
    async def test_from_pretrained_analyze_flow(self):
        """Test full flow through from_pretrained and analyze."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)
        mock_config = ModelConfig(vocab_size=100, hidden_size=64, num_hidden_layers=4)

        with patch(
            "chuk_lazarus.introspection.analyzer._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained("test-model") as analyzer:
                result = await analyzer.analyze("hello")
                assert isinstance(result, AnalysisResult)
                assert result.prompt == "hello"


class TestAnalyzePromptConvenience:
    """Test the analyze_prompt convenience function."""

    @pytest.mark.asyncio
    async def test_analyze_prompt(self):
        """Test analyze_prompt convenience function with mocking."""
        from chuk_lazarus.introspection.analyzer import analyze_prompt
        from chuk_lazarus.models_v2.core.config import ModelConfig

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)
        mock_config = ModelConfig(vocab_size=100, hidden_size=64, num_hidden_layers=4)

        with patch(
            "chuk_lazarus.introspection.analyzer._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            result = await analyze_prompt("test-model", "hello world")

            assert isinstance(result, AnalysisResult)
            assert result.prompt == "hello world"

    @pytest.mark.asyncio
    async def test_analyze_prompt_with_config(self):
        """Test analyze_prompt with config."""
        from chuk_lazarus.introspection.analyzer import analyze_prompt
        from chuk_lazarus.models_v2.core.config import ModelConfig

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)
        mock_config = ModelConfig(vocab_size=100, hidden_size=64, num_hidden_layers=4)

        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL, top_k=3)

        with patch(
            "chuk_lazarus.introspection.analyzer._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            result = await analyze_prompt("test-model", "test", config)

            assert isinstance(result, AnalysisResult)
            assert len(result.layer_predictions) == 4  # All layers


class TestLoadModelSync:
    """Test _load_model_sync with mocking."""

    @pytest.mark.skip(reason="mlx_lm fallback removed - now uses native loader only")
    def test_load_model_sync_mlx_lm_fallback(self):
        """Test _load_model_sync falls back to mlx_lm for unknown model types."""
        import json
        import sys
        import tempfile
        from pathlib import Path

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)

        # Create a mock mlx_lm module
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config.json with UNKNOWN model type to force mlx_lm fallback
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "model_type": "unknown_custom_model",  # Unknown type forces fallback
                        "hidden_size": 64,
                        "vocab_size": 100,
                        "num_hidden_layers": 4,
                    },
                    f,
                )

            # Mock HFLoader.download to return our temp path
            mock_result = MagicMock()
            mock_result.model_path = Path(tmpdir)

            # Add mock mlx_lm to sys.modules
            sys.modules["mlx_lm"] = mock_mlx_lm

            try:
                with patch(
                    "chuk_lazarus.inference.loader.HFLoader.download",
                    return_value=mock_result,
                ):
                    from chuk_lazarus.introspection.analyzer import _load_model_sync

                    model, tokenizer, config = _load_model_sync("test/model")

                    assert model is mock_model
                    assert tokenizer is mock_tokenizer
            finally:
                # Clean up the mock module
                if "mlx_lm" in sys.modules:
                    del sys.modules["mlx_lm"]

    @pytest.mark.skip(reason="mlx_lm fallback removed - now uses native loader only")
    def test_load_model_sync_quantized_uses_mlx_lm(self):
        """Test that quantized models use mlx_lm."""
        import json
        import sys
        import tempfile
        from pathlib import Path

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump({"model_type": "gemma", "quantization_config": {}}, f)

            mock_result = MagicMock()
            mock_result.model_path = Path(tmpdir)

            sys.modules["mlx_lm"] = mock_mlx_lm

            try:
                with patch(
                    "chuk_lazarus.inference.loader.HFLoader.download",
                    return_value=mock_result,
                ):
                    from chuk_lazarus.introspection.analyzer import _load_model_sync

                    model, tokenizer, config = _load_model_sync("model-4bit")

                    assert model is mock_model
            finally:
                if "mlx_lm" in sys.modules:
                    del sys.modules["mlx_lm"]

    @pytest.mark.skip(reason="mlx_lm fallback removed - now uses native loader only")
    def test_load_model_sync_gemma_adds_embedding_scale_via_mlx_lm(self):
        """Test that Gemma models loaded via mlx_lm get embedding scale attached.

        When a Gemma model is quantized (or for other reasons uses mlx_lm fallback),
        we attach _embedding_scale_for_hooks so logit lens works correctly.
        """
        import json
        import sys
        import tempfile
        from pathlib import Path

        mock_model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        mock_tokenizer = MockTokenizer(vocab_size=100)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                # Mark as quantized to force mlx_lm fallback
                json.dump(
                    {
                        "model_type": "gemma",
                        "hidden_size": 1024,
                        "quantization_config": {"bits": 4},  # Forces mlx_lm fallback
                    },
                    f,
                )

            mock_result = MagicMock()
            mock_result.model_path = Path(tmpdir)

            sys.modules["mlx_lm"] = mock_mlx_lm

            try:
                with patch(
                    "chuk_lazarus.inference.loader.HFLoader.download",
                    return_value=mock_result,
                ):
                    from chuk_lazarus.introspection.analyzer import _load_model_sync

                    model, tokenizer, config = _load_model_sync("gemma-model-4bit")

                    # Check embedding scale was attached (mlx_lm fallback path)
                    assert hasattr(model, "_embedding_scale_for_hooks")
                    assert model._embedding_scale_for_hooks == 32.0  # sqrt(1024)
            finally:
                if "mlx_lm" in sys.modules:
                    del sys.modules["mlx_lm"]


class TestModelFamilyType:
    """Test ModelFamilyType enum from registry."""

    def test_model_family_type_values(self):
        """Test ModelFamilyType enum values."""
        assert ModelFamilyType.GEMMA.value == "gemma"
        assert ModelFamilyType.LLAMA.value == "llama"
        assert ModelFamilyType.QWEN3.value == "qwen3"
        assert ModelFamilyType.GRANITE.value == "granite"


class TestLoadWithFamilyRegistry:
    """Test _load_with_family_registry function."""

    @pytest.mark.skip(
        reason="_load_with_family_registry function removed - logic integrated into _load_model_sync"
    )
    def test_load_with_family_registry_no_family_info(self):
        """Test loader raises when no family info found."""
        pass

    def test_family_registry_detects_all_families(self):
        """Test that all model families can be detected."""
        # Test detection of various model types
        test_cases = [
            ({"model_type": "llama"}, ModelFamilyType.LLAMA),
            ({"model_type": "gemma"}, ModelFamilyType.GEMMA),
            ({"model_type": "gemma3_text"}, ModelFamilyType.GEMMA),
            ({"model_type": "qwen3"}, ModelFamilyType.QWEN3),
            ({"model_type": "granite"}, ModelFamilyType.GRANITE),
            ({"model_type": "jamba"}, ModelFamilyType.JAMBA),
            ({"model_type": "mamba"}, ModelFamilyType.MAMBA),
            ({"model_type": "starcoder2"}, ModelFamilyType.STARCODER2),
        ]

        for config, expected_family in test_cases:
            result = detect_model_family(config)
            assert result == expected_family, f"Failed for {config}"


class TestModelAnalyzerFallbacks:
    """Test fallback behavior in ModelAnalyzer internal methods."""

    def test_get_num_layers_fallback(self):
        """Test fallback when model structure is different."""

        class WeirdModel(nn.Module):
            """Model without typical layer structure."""

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def __call__(self, x):
                return self.fc(x)

        model = WeirdModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        # Should fallback to 32
        assert analyzer._get_num_layers() == 32

    def test_get_hidden_size_fallback(self):
        """Test fallback when hidden_size is not accessible."""

        class WeirdModel(nn.Module):
            """Model without hidden_size attribute."""

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def __call__(self, x):
                return self.fc(x)

        model = WeirdModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        # Should fallback to 4096
        assert analyzer._get_hidden_size() == 4096

    def test_get_vocab_size_len_fallback(self):
        """Test vocab_size fallback to len(tokenizer)."""

        class TokenizerNoVocabSize:
            """Tokenizer without vocab_size but with __len__."""

            def __len__(self):
                return 50000

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                return "test"

        model = SimpleForCausalLM()
        tokenizer = TokenizerNoVocabSize()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        # Should use len(tokenizer)
        assert analyzer._get_vocab_size() == 50000

    def test_get_num_layers_direct_layers(self):
        """Test get_num_layers with model.layers directly."""

        class ModelWithDirectLayers(nn.Module):
            """Model with layers directly on model (not model.model)."""

            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(64, 64) for _ in range(8)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = ModelWithDirectLayers()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        assert analyzer._get_num_layers() == 8

    def test_get_hidden_size_from_args(self):
        """Test get_hidden_size from model.args."""

        class Args:
            hidden_size = 768

        class ModelWithArgs(nn.Module):
            def __init__(self):
                super().__init__()
                self.args = Args()
                self.fc = nn.Linear(768, 768)

            def __call__(self, x):
                return self.fc(x)

        model = ModelWithArgs()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        assert analyzer._get_hidden_size() == 768


class TestAnalyzeWithTokenTracking:
    """Test token tracking during analysis."""

    @pytest.fixture
    def analyzer(self):
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        return ModelAnalyzer.from_model(model, tokenizer)

    def test_analyze_with_track_tokens(self, analyzer):
        """Test analyzing with token tracking."""
        config = AnalysisConfig(track_tokens=["A", "B", "C"])
        result = analyzer._analyze_sync("test input", config)

        assert isinstance(result, AnalysisResult)
        # Token evolutions should be a list (may be empty if tokens not found)
        assert isinstance(result.token_evolutions, list)

    def test_analyze_with_invalid_track_token(self, analyzer):
        """Test that invalid tokens are silently skipped."""
        # Use a token that won't be in vocabulary
        config = AnalysisConfig(track_tokens=["NONEXISTENT_TOKEN_XYZ"])
        result = analyzer._analyze_sync("test", config)

        # Should not raise, just skip the token
        assert isinstance(result, AnalysisResult)


class TestModelInfoTiedEmbeddings:
    """Test model_info with tied embeddings."""

    def test_model_info_tied_embeddings_true(self):
        """Test model_info when tie_word_embeddings is True."""

        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleTransformerModel()
                self.lm_head = nn.Linear(64, 100, bias=False)
                self.tie_word_embeddings = True

            def __call__(self, x):
                return self.lm_head(self.model(x))

        model = TiedModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer, model_id="tied-model")

        info = analyzer.model_info
        assert info.has_tied_embeddings is True

    def test_model_info_tied_embeddings_false(self):
        """Test model_info when tie_word_embeddings is False or absent."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        info = analyzer.model_info
        assert info.has_tied_embeddings is False
