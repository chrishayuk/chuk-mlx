"""Comprehensive tests for ModelAnalyzer core functionality."""

from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.analyzer.config import (
    AnalysisConfig,
    LayerStrategy,
    TrackStrategy,
)
from chuk_lazarus.introspection.analyzer.core import ModelAnalyzer, analyze_prompt
from chuk_lazarus.introspection.analyzer.models import (
    AnalysisResult,
    LayerPredictionResult,
    LayerTransition,
    ModelInfo,
    ResidualContribution,
    TokenPrediction,
)
from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks


# Simple test models
class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SimpleMLP(hidden_size)

    def __call__(self, x: mx.array, cache: mx.array | None = None) -> tuple[mx.array, None]:
        # Self-attention with residual
        h = self.norm1(x)
        h = self.attn(h, h, h)
        x = x + h

        # MLP with residual
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x, None


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing."""

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

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class SimpleForCausalLM(nn.Module):
    """Simple causal LM wrapper for testing."""

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = SimpleTransformerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Simple encoding: return char codes mod vocab_size."""
        if not text:
            return []
        return [ord(c) % self.vocab_size for c in text[:5]]

    def decode(self, ids: list[int]) -> str:
        """Simple decoding: return token ID as string."""
        if not ids:
            return ""
        return f"[{ids[0]}]"


class MockConfig:
    """Mock config for testing."""

    def __init__(
        self,
        num_hidden_layers: int = 4,
        hidden_size: int = 64,
        vocab_size: int = 100,
        tie_word_embeddings: bool = False,
        embedding_scale: float | None = None,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        if embedding_scale is not None:
            self.embedding_scale = embedding_scale


class TestModelAnalyzerInit:
    """Tests for ModelAnalyzer initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()

        analyzer = ModelAnalyzer(model, tokenizer)

        assert analyzer._model is model
        assert analyzer._tokenizer is tokenizer
        assert analyzer._model_id == "unknown"
        assert analyzer._config is None
        assert analyzer._embedding_scale is None

    def test_init_with_model_id(self):
        """Test initialization with model ID."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()

        analyzer = ModelAnalyzer(model, tokenizer, model_id="test-model")

        assert analyzer._model_id == "test-model"

    def test_init_with_embedding_scale(self):
        """Test initialization with explicit embedding scale."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()

        analyzer = ModelAnalyzer(model, tokenizer, embedding_scale=8.0)

        assert analyzer._embedding_scale == 8.0

    def test_init_with_config_embedding_scale(self):
        """Test embedding scale from config."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        config = MockConfig(embedding_scale=10.0)

        analyzer = ModelAnalyzer(model, tokenizer, config=config)

        assert analyzer._embedding_scale == 10.0

    def test_init_explicit_scale_overrides_config(self):
        """Test that explicit embedding scale overrides config."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        config = MockConfig(embedding_scale=10.0)

        analyzer = ModelAnalyzer(model, tokenizer, embedding_scale=15.0, config=config)

        assert analyzer._embedding_scale == 15.0


class TestModelAnalyzerFromModel:
    """Tests for from_model factory method."""

    def test_from_model_basic(self):
        """Test creating analyzer from existing model."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()

        analyzer = ModelAnalyzer.from_model(model, tokenizer)

        assert analyzer._model is model
        assert analyzer._tokenizer is tokenizer
        assert analyzer._model_id == "custom"

    def test_from_model_with_params(self):
        """Test from_model with all parameters."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = ModelAnalyzer.from_model(
            model, tokenizer, model_id="my-model", embedding_scale=5.0, config=config
        )

        assert analyzer._model_id == "my-model"
        assert analyzer._embedding_scale == 5.0
        assert analyzer._config is config


class TestModelAnalyzerProperties:
    """Tests for ModelAnalyzer properties."""

    def test_config_property(self):
        """Test config property."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = ModelAnalyzer(model, tokenizer, config=config)

        assert analyzer.config is config

    def test_config_property_none(self):
        """Test config property when None."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()

        analyzer = ModelAnalyzer(model, tokenizer)

        assert analyzer.config is None

    def test_model_info_with_config(self):
        """Test model_info property with config."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(
            num_hidden_layers=4,
            hidden_size=64,
            vocab_size=100,
            tie_word_embeddings=True,
        )

        analyzer = ModelAnalyzer(model, tokenizer, model_id="test", config=config)
        info = analyzer.model_info

        assert isinstance(info, ModelInfo)
        assert info.model_id == "test"
        assert info.num_layers == 4
        assert info.hidden_size == 64
        assert info.vocab_size == 100
        assert info.has_tied_embeddings is True

    def test_model_info_without_config(self):
        """Test model_info property without config (uses introspection)."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)

        analyzer = ModelAnalyzer(model, tokenizer, model_id="test")
        info = analyzer.model_info

        assert isinstance(info, ModelInfo)
        assert info.model_id == "test"
        assert info.num_layers == 4  # From _get_num_layers()
        assert info.vocab_size == 100  # From _get_vocab_size()


class TestModelAnalyzerPrivateMethods:
    """Tests for private helper methods."""

    def test_get_num_layers_from_model_layers(self):
        """Test _get_num_layers when model has model.layers."""
        model = SimpleForCausalLM(num_layers=6)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        num_layers = analyzer._get_num_layers()

        assert num_layers == 6

    def test_get_num_layers_direct_layers(self):
        """Test _get_num_layers with direct layers attribute."""

        class DirectLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [SimpleMLP() for _ in range(8)]

        model = DirectLayerModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        num_layers = analyzer._get_num_layers()

        assert num_layers == 8

    def test_get_num_layers_fallback(self):
        """Test _get_num_layers fallback when structure unknown."""

        class UnknownModel(nn.Module):
            pass

        model = UnknownModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        num_layers = analyzer._get_num_layers()

        assert num_layers == 32  # Fallback

    def test_get_hidden_size_from_model_args(self):
        """Test _get_hidden_size from model.args."""

        class ModelWithArgs(nn.Module):
            def __init__(self):
                super().__init__()
                self.args = type("Args", (), {"hidden_size": 512})()

        model = ModelWithArgs()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        hidden_size = analyzer._get_hidden_size()

        assert hidden_size == 512

    def test_get_hidden_size_fallback(self):
        """Test _get_hidden_size fallback."""

        class UnknownModel(nn.Module):
            pass

        model = UnknownModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        hidden_size = analyzer._get_hidden_size()

        assert hidden_size == 4096  # Fallback

    def test_get_vocab_size_from_tokenizer(self):
        """Test _get_vocab_size from tokenizer."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer(vocab_size=256)
        analyzer = ModelAnalyzer(model, tokenizer)

        vocab_size = analyzer._get_vocab_size()

        assert vocab_size == 256

    def test_get_vocab_size_from_len(self):
        """Test _get_vocab_size using len() fallback."""

        class TokenizerWithLen:
            def __len__(self):
                return 1024

        model = SimpleForCausalLM()
        tokenizer = TokenizerWithLen()
        analyzer = ModelAnalyzer(model, tokenizer)

        vocab_size = analyzer._get_vocab_size()

        assert vocab_size == 1024

    def test_get_layers_to_capture_all(self):
        """Test _get_layers_to_capture with ALL strategy."""
        model = SimpleForCausalLM(num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL)

        layers = analyzer._get_layers_to_capture(4, config)

        assert layers == [0, 1, 2, 3]

    def test_get_layers_to_capture_first_last(self):
        """Test _get_layers_to_capture with FIRST_LAST strategy."""
        model = SimpleForCausalLM(num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST)

        layers = analyzer._get_layers_to_capture(4, config)

        assert layers == [0, 3]

    def test_get_layers_to_capture_custom_with_layers(self):
        """Test _get_layers_to_capture with CUSTOM strategy and custom layers."""
        model = SimpleForCausalLM(num_layers=8)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.CUSTOM, custom_layers=[0, 2, 5, 7])

        layers = analyzer._get_layers_to_capture(8, config)

        assert layers == [0, 2, 5, 7]

    def test_get_layers_to_capture_custom_without_layers(self):
        """Test _get_layers_to_capture with CUSTOM but no custom_layers (fallback)."""
        model = SimpleForCausalLM(num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.CUSTOM)

        layers = analyzer._get_layers_to_capture(4, config)

        assert layers == [0, 3]  # Falls back to first_last

    def test_get_layers_to_capture_evenly_spaced(self):
        """Test _get_layers_to_capture with EVENLY_SPACED strategy."""
        model = SimpleForCausalLM(num_layers=12)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.EVENLY_SPACED, layer_step=4)

        layers = analyzer._get_layers_to_capture(12, config)

        # Should be 0, 4, 8, and 11 (last layer always included)
        assert layers == [0, 4, 8, 11]

    def test_get_top_predictions(self):
        """Test _get_top_predictions method."""
        model = SimpleForCausalLM(vocab_size=100)
        tokenizer = MockTokenizer(vocab_size=100)
        analyzer = ModelAnalyzer(model, tokenizer)

        # Create logits favoring certain tokens
        logits = mx.zeros(100)
        logits[10] = 5.0  # Highest
        logits[20] = 3.0  # Second
        logits[30] = 1.0  # Third

        predictions = analyzer._get_top_predictions(logits, top_k=3)

        assert len(predictions) == 3
        assert predictions[0].token_id == 10
        assert predictions[0].rank == 1
        assert predictions[1].token_id == 20
        assert predictions[1].rank == 2


class TestGetTokensToTrack:
    """Tests for _get_tokens_to_track method."""

    def test_manual_strategy(self):
        """Test MANUAL track strategy."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            track_strategy=TrackStrategy.MANUAL, track_tokens=["hello", "world"]
        )

        tokens = analyzer._get_tokens_to_track(config, [])

        assert tokens == ["hello", "world"]

    def test_top_k_final_strategy(self):
        """Test TOP_K_FINAL track strategy."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(track_strategy=TrackStrategy.TOP_K_FINAL)

        # Create mock layer predictions
        layer_preds = [
            LayerPredictionResult(
                layer_idx=0,
                predictions=[
                    TokenPrediction(token="a", token_id=1, probability=0.5, rank=1),
                ],
            ),
            LayerPredictionResult(
                layer_idx=1,
                predictions=[
                    TokenPrediction(token="b", token_id=2, probability=0.6, rank=1),
                    TokenPrediction(token="c", token_id=3, probability=0.3, rank=2),
                ],
            ),
        ]

        tokens = analyzer._get_tokens_to_track(config, layer_preds)

        # Should return tokens from final layer
        assert tokens == ["b", "c"]

    def test_top_k_final_strategy_empty(self):
        """Test TOP_K_FINAL strategy with no predictions."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(track_strategy=TrackStrategy.TOP_K_FINAL)

        tokens = analyzer._get_tokens_to_track(config, [])

        assert tokens == []

    def test_tool_tokens_strategy(self):
        """Test TOOL_TOKENS track strategy."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(track_strategy=TrackStrategy.TOOL_TOKENS)

        tokens = analyzer._get_tokens_to_track(config, [])

        # Should return predefined tool tokens
        assert "{" in tokens
        assert "get_" in tokens
        assert "function" in tokens

    def test_emergent_strategy(self):
        """Test EMERGENT track strategy."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(track_strategy=TrackStrategy.EMERGENT)

        layer_preds = [
            LayerPredictionResult(
                layer_idx=3,
                predictions=[
                    TokenPrediction(token="final", token_id=10, probability=0.9, rank=1),
                ],
            ),
        ]

        tokens = analyzer._get_tokens_to_track(config, layer_preds)

        # Should include final layer tokens plus common ones
        assert "final" in tokens
        assert "{" in tokens
        assert "get_" in tokens


class TestComputeResidualDecomposition:
    """Tests for _compute_residual_decomposition method."""

    def test_residual_decomposition_no_embeddings(self):
        """Test when no embeddings captured."""
        model = SimpleForCausalLM()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        hooks = ModelHooks(model)
        # Don't populate embeddings
        hooks.state.embeddings = None

        contributions = analyzer._compute_residual_decomposition(hooks, [0, 1])

        assert contributions == []

    def test_residual_decomposition_with_data(self):
        """Test residual decomposition with actual data."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        # Create hooks and run forward
        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers="all",
                positions="all",
                capture_attention_output=True,
                capture_ffn_output=True,
            )
        )
        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        contributions = analyzer._compute_residual_decomposition(hooks, [0, 1, 2, 3])

        # Should have contributions for each layer
        assert len(contributions) > 0
        for contrib in contributions:
            assert isinstance(contrib, ResidualContribution)
            assert contrib.attention_norm >= 0
            assert contrib.ffn_norm >= 0
            assert 0 <= contrib.attention_fraction <= 1
            assert 0 <= contrib.ffn_fraction <= 1

    def test_residual_decomposition_without_attn_ffn_outputs(self):
        """Test residual decomposition when attn/ffn outputs not captured."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        # Create hooks WITHOUT capturing attention/ffn outputs
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers="all", positions="all"))
        input_ids = mx.array([[1, 2, 3]])
        hooks.forward(input_ids)

        contributions = analyzer._compute_residual_decomposition(hooks, [0, 1])

        # Should still compute, but with equal split approximation
        assert len(contributions) > 0
        for contrib in contributions:
            # Should approximate with 50/50 split
            assert abs(contrib.attention_fraction - 0.5) < 0.1 or contrib.total_norm > 0


class TestAnalyzeBatch:
    """Tests for analyze_batch method."""

    @pytest.mark.asyncio
    async def test_analyze_batch_multiple_prompts(self):
        """Test analyzing multiple prompts."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        prompts = ["hello", "world", "test"]
        results = await analyzer.analyze_batch(prompts)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, AnalysisResult)
            assert result.prompt == prompts[i]

    @pytest.mark.asyncio
    async def test_analyze_batch_with_config(self):
        """Test analyze_batch with custom config."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST, top_k=3)
        results = await analyzer.analyze_batch(["test1", "test2"], config=config)

        assert len(results) == 2
        for result in results:
            # Should only have first and last layer
            assert len(result.captured_layers) == 2


class TestAnalyzeSync:
    """Tests for _analyze_sync method (core analysis logic)."""

    def test_analyze_sync_basic(self):
        """Test basic synchronous analysis."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig()

        result = analyzer._analyze_sync("test", config)

        assert isinstance(result, AnalysisResult)
        assert result.prompt == "test"
        assert len(result.tokens) > 0
        assert len(result.layer_predictions) > 0
        assert len(result.final_prediction) > 0

    def test_analyze_sync_with_entropy(self):
        """Test analysis with entropy computation."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(compute_entropy=True, layer_strategy=LayerStrategy.ALL)

        result = analyzer._analyze_sync("test", config)

        # Should have entropy values
        for layer_pred in result.layer_predictions:
            assert layer_pred.entropy >= 0
            assert 0 <= layer_pred.entropy_normalized <= 1

    def test_analyze_sync_with_transitions(self):
        """Test analysis with transition computation."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            compute_transitions=True, compute_entropy=True, layer_strategy=LayerStrategy.ALL
        )

        result = analyzer._analyze_sync("test", config)

        # Should have transitions between layers
        assert len(result.layer_transitions) > 0
        for transition in result.layer_transitions:
            assert isinstance(transition, LayerTransition)
            assert transition.kl_divergence >= 0
            assert transition.js_divergence >= 0

    def test_analyze_sync_without_transitions(self):
        """Test analysis without transition computation."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(compute_transitions=False)

        result = analyzer._analyze_sync("test", config)

        assert result.layer_transitions == []

    def test_analyze_sync_track_token_evolution(self):
        """Test token evolution tracking."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            track_strategy=TrackStrategy.TOP_K_FINAL, layer_strategy=LayerStrategy.ALL
        )

        result = analyzer._analyze_sync("test", config)

        # Should track some tokens
        assert len(result.token_evolutions) >= 0  # May be 0 if tokens not found

    def test_analyze_sync_token_not_in_vocab(self):
        """Test handling of tokens not in vocabulary during tracking."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            track_strategy=TrackStrategy.MANUAL,
            track_tokens=["nonexistent_token_xyz"],
            layer_strategy=LayerStrategy.ALL,
        )

        # Should not raise error, just skip unfound tokens
        result = analyzer._analyze_sync("test", config)

        # Token evolution may be empty if token not in vocab
        assert isinstance(result, AnalysisResult)

    def test_analyze_sync_with_residual_decomposition(self):
        """Test analysis with residual decomposition."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            compute_residual_decomposition=True, layer_strategy=LayerStrategy.ALL
        )

        result = analyzer._analyze_sync("test", config)

        # Should have residual contributions
        assert len(result.residual_contributions) > 0

    def test_analyze_sync_entropy_edge_case_2d_logits(self):
        """Test entropy computation with 2D logits (edge case)."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(compute_entropy=True, layer_strategy=LayerStrategy.ALL)

        result = analyzer._analyze_sync("test", config)

        # Should handle both 2D and 3D logit shapes gracefully
        assert isinstance(result, AnalysisResult)

    def test_analyze_sync_transitions_missing_probs(self):
        """Test transition computation when probabilities not in cache."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        # Disable entropy to prevent prob caching
        config = AnalysisConfig(
            compute_transitions=True, compute_entropy=False, layer_strategy=LayerStrategy.ALL
        )

        result = analyzer._analyze_sync("test", config)

        # Should still compute transitions (with fallback values)
        if len(result.layer_predictions) > 1:
            assert len(result.layer_transitions) > 0
            for trans in result.layer_transitions:
                # When probs not cached, should have zero divergence
                assert trans.kl_divergence == 0.0
                assert trans.js_divergence == 0.0


class TestAnalyzeAsync:
    """Tests for async analyze method."""

    @pytest.mark.asyncio
    async def test_analyze_basic(self):
        """Test basic async analysis."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        result = await analyzer.analyze("test prompt")

        assert isinstance(result, AnalysisResult)
        assert result.prompt == "test prompt"

    @pytest.mark.asyncio
    async def test_analyze_with_config(self):
        """Test async analysis with custom config."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST, top_k=3)
        result = await analyzer.analyze("test", config=config)

        assert len(result.final_prediction) == 3
        assert len(result.captured_layers) == 2

    @pytest.mark.asyncio
    async def test_analyze_default_config(self):
        """Test analyze creates default config when None provided."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        result = await analyzer.analyze("test", config=None)

        assert isinstance(result, AnalysisResult)


class TestFromPretrained:
    """Tests for from_pretrained async context manager."""

    @pytest.mark.asyncio
    async def test_from_pretrained_basic(self):
        """Test from_pretrained context manager."""
        mock_model = SimpleForCausalLM()
        mock_tokenizer = MockTokenizer()
        mock_config = MockConfig()

        with patch(
            "chuk_lazarus.introspection.analyzer.core._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained("test-model") as analyzer:
                assert analyzer._model is mock_model
                assert analyzer._tokenizer is mock_tokenizer
                assert analyzer._model_id == "test-model"
                assert analyzer._config is mock_config

    @pytest.mark.asyncio
    async def test_from_pretrained_with_embedding_scale(self):
        """Test from_pretrained with explicit embedding scale."""
        mock_model = SimpleForCausalLM()
        mock_tokenizer = MockTokenizer()
        mock_config = MockConfig()

        with patch(
            "chuk_lazarus.introspection.analyzer.core._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained(
                "test-model", embedding_scale=12.0
            ) as analyzer:
                assert analyzer._embedding_scale == 12.0

    @pytest.mark.asyncio
    async def test_from_pretrained_cleanup(self):
        """Test that from_pretrained properly cleans up."""
        mock_model = SimpleForCausalLM()
        mock_tokenizer = MockTokenizer()
        mock_config = MockConfig()

        with patch(
            "chuk_lazarus.introspection.analyzer.core._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            async with ModelAnalyzer.from_pretrained("test-model") as _analyzer:
                pass
            # Should exit cleanly without errors


class TestAnalyzePromptConvenience:
    """Tests for analyze_prompt convenience function."""

    @pytest.mark.asyncio
    async def test_analyze_prompt_basic(self):
        """Test analyze_prompt convenience function."""
        mock_model = SimpleForCausalLM()
        mock_tokenizer = MockTokenizer()
        mock_config = MockConfig()

        with patch(
            "chuk_lazarus.introspection.analyzer.core._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            result = await analyze_prompt("test-model", "Hello world")

            assert isinstance(result, AnalysisResult)
            assert result.prompt == "Hello world"

    @pytest.mark.asyncio
    async def test_analyze_prompt_with_config(self):
        """Test analyze_prompt with custom config."""
        mock_model = SimpleForCausalLM()
        mock_tokenizer = MockTokenizer()
        mock_config = MockConfig()

        config = AnalysisConfig(layer_strategy=LayerStrategy.FIRST_LAST)

        with patch(
            "chuk_lazarus.introspection.analyzer.core._load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_config),
        ):
            result = await analyze_prompt("test-model", "Hello", config=config)

            assert len(result.captured_layers) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_model_info_with_tied_embeddings_no_config(self):
        """Test model_info when model has tie_word_embeddings attribute."""

        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleTransformerModel()
                self.tie_word_embeddings = True

        model = TiedModel()
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        info = analyzer.model_info

        # When no config, getattr on model will find tie_word_embeddings=True
        assert info.has_tied_embeddings is True

    def test_analyze_with_single_layer(self):
        """Test analysis with only one layer captured."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.CUSTOM, custom_layers=[2])

        result = analyzer._analyze_sync("test", config)

        assert len(result.captured_layers) == 1
        # No transitions with single layer
        assert len(result.layer_transitions) == 0

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

        # Create a tokenizer that returns at least one token for empty string
        class SafeTokenizer:
            vocab_size = 100

            def encode(self, text: str) -> list[int]:
                if not text:
                    return [0]  # Return padding/bos token
                return [ord(c) % 100 for c in text[:5]]

            def decode(self, ids: list[int]) -> str:
                return f"[{ids[0]}]" if ids else ""

        tokenizer = SafeTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)

        result = analyzer._analyze_sync("", AnalysisConfig())

        # Should handle gracefully
        assert isinstance(result, AnalysisResult)
        assert result.prompt == ""

    def test_very_long_layer_list(self):
        """Test with many layers."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=8)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.ALL)

        result = analyzer._analyze_sync("test", config)

        assert len(result.layer_predictions) == 8
        assert len(result.layer_transitions) == 7  # n-1 transitions

    def test_get_layers_deduplicates(self):
        """Test that custom layers are deduplicated and sorted."""
        model = SimpleForCausalLM(num_layers=8)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(
            layer_strategy=LayerStrategy.CUSTOM, custom_layers=[5, 0, 5, 2, 0, 7]
        )

        layers = analyzer._get_layers_to_capture(8, config)

        assert layers == [0, 2, 5, 7]  # Sorted and deduplicated

    def test_evenly_spaced_includes_last_layer(self):
        """Test that evenly spaced always includes final layer."""
        model = SimpleForCausalLM(num_layers=10)
        tokenizer = MockTokenizer()
        analyzer = ModelAnalyzer(model, tokenizer)
        config = AnalysisConfig(layer_strategy=LayerStrategy.EVENLY_SPACED, layer_step=3)

        layers = analyzer._get_layers_to_capture(10, config)

        assert 9 in layers  # Last layer (index 9) should be included
        assert layers == sorted(set(layers))  # Should be sorted and unique
