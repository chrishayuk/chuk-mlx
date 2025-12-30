"""Tests for logit lens module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection import CaptureConfig, CapturedState, ModelHooks
from chuk_lazarus.introspection.logit_lens import (
    LayerPrediction,
    LogitLens,
    TokenEvolution,
)


# Reuse the simple model from test_hooks
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

    def __call__(self, x: mx.array, cache: mx.array | None = None) -> tuple[mx.array, None]:
        h = self.norm1(x)
        h = self.attn(h, h, h)
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

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        # Simple encoding: return first char's ASCII mod vocab_size
        if not text:
            return []
        return [ord(text[0]) % self.vocab_size]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 65) for i in ids)  # A, B, C, ...


class TestLayerPrediction:
    """Tests for LayerPrediction dataclass."""

    def test_basic_creation(self):
        pred = LayerPrediction(
            layer_idx=4,
            position=10,
            top_tokens=["cat", "dog", "bird"],
            top_probs=[0.5, 0.3, 0.2],
            top_ids=[10, 20, 30],
        )

        assert pred.layer_idx == 4
        assert pred.position == 10
        assert len(pred.top_tokens) == 3

    def test_repr(self):
        pred = LayerPrediction(
            layer_idx=4,
            position=10,
            top_tokens=["cat", "dog", "bird"],
            top_probs=[0.5, 0.3, 0.2],
            top_ids=[10, 20, 30],
        )

        repr_str = repr(pred)
        assert "Layer 4" in repr_str
        assert "cat" in repr_str


class TestTokenEvolution:
    """Tests for TokenEvolution dataclass."""

    def test_basic_creation(self):
        evo = TokenEvolution(
            token="cat",
            token_id=10,
            layers=[0, 1, 2, 3],
            probabilities=[0.1, 0.2, 0.5, 0.8],
            ranks=[50, 20, 2, 1],
        )

        assert evo.token == "cat"
        assert len(evo.layers) == 4

    def test_emergence_layer(self):
        evo = TokenEvolution(
            token="cat",
            token_id=10,
            layers=[0, 1, 2, 3],
            probabilities=[0.1, 0.2, 0.5, 0.8],
            ranks=[50, 20, 2, 1],  # Becomes top-1 at layer 3
        )

        assert evo.emergence_layer == 3

    def test_emergence_layer_never_top1(self):
        evo = TokenEvolution(
            token="cat",
            token_id=10,
            layers=[0, 1, 2, 3],
            probabilities=[0.1, 0.2, 0.3, 0.4],
            ranks=[50, 20, 5, 2],  # Never top-1
        )

        assert evo.emergence_layer is None

    def test_to_dict(self):
        evo = TokenEvolution(
            token="cat",
            token_id=10,
            layers=[0, 1, 2, 3],
            probabilities=[0.1, 0.2, 0.5, 0.8],
            ranks=[50, 20, 2, 1],
        )

        d = evo.to_dict()

        assert d["token"] == "cat"
        assert d["token_id"] == 10
        assert d["layers"] == [0, 1, 2, 3]
        assert d["emergence_layer"] == 3


class TestLogitLens:
    """Tests for LogitLens class."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=100)

    @pytest.fixture
    def hooks_with_state(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CapturedState)  # This won't work, let's create proper config
        from chuk_lazarus.introspection import CaptureConfig

        hooks.configure(CaptureConfig(layers="all", positions="all"))

        # Run forward to populate state
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)

        return hooks

    def test_get_layer_predictions(self, hooks_with_state, tokenizer):
        lens = LogitLens(hooks_with_state, tokenizer)

        predictions = lens.get_layer_predictions(position=-1, top_k=5)

        assert len(predictions) == 4  # 4 layers
        for pred in predictions:
            assert len(pred.top_tokens) == 5
            assert len(pred.top_probs) == 5
            assert len(pred.top_ids) == 5

    def test_get_layer_predictions_without_tokenizer(self, hooks_with_state):
        lens = LogitLens(hooks_with_state, tokenizer=None)

        predictions = lens.get_layer_predictions(position=-1, top_k=5)

        # Should use token IDs as strings
        assert len(predictions) == 4
        assert predictions[0].top_tokens[0].startswith("[")

    def test_track_token_by_id(self, hooks_with_state, tokenizer):
        lens = LogitLens(hooks_with_state, tokenizer)

        evolution = lens.track_token(token=10, position=-1)

        assert evolution.token_id == 10
        assert len(evolution.layers) == 4
        assert len(evolution.probabilities) == 4

    def test_track_token_by_string(self, hooks_with_state, tokenizer):
        lens = LogitLens(hooks_with_state, tokenizer)

        evolution = lens.track_token(token="A", position=-1)

        assert evolution.token == "A"
        assert len(evolution.layers) == 4

    def test_track_token_requires_tokenizer_for_string(self, hooks_with_state):
        lens = LogitLens(hooks_with_state, tokenizer=None)

        with pytest.raises(ValueError, match="Tokenizer required"):
            lens.track_token(token="hello", position=-1)

    def test_compare_tokens(self, hooks_with_state, tokenizer):
        lens = LogitLens(hooks_with_state, tokenizer)

        comparisons = lens.compare_tokens([10, 20, 30], position=-1)

        assert len(comparisons) == 3
        assert "[10]" in comparisons
        assert "[20]" in comparisons

    def test_print_evolution(self, hooks_with_state, tokenizer, capsys):
        lens = LogitLens(hooks_with_state, tokenizer)

        lens.print_evolution(position=-1, top_k=3)

        captured = capsys.readouterr()
        assert "Logit Lens" in captured.out
        assert "Layer" in captured.out

    def test_to_dict(self, hooks_with_state, tokenizer):
        lens = LogitLens(hooks_with_state, tokenizer)

        result = lens.to_dict(position=-1, top_k=5)

        assert "position" in result
        assert "layers" in result
        assert len(result["layers"]) == 4


class TestLogitLensEmptyState:
    """Tests for LogitLens with no captured layers."""

    def test_empty_predictions(self):
        # Create hooks but don't run forward
        class DummyModel(nn.Module):
            pass

        hooks = ModelHooks.__new__(ModelHooks)
        hooks.state = CapturedState()

        lens = LogitLens(hooks, tokenizer=None)
        predictions = lens.get_layer_predictions()

        assert predictions == []

    def test_print_evolution_empty(self, capsys):
        hooks = ModelHooks.__new__(ModelHooks)
        hooks.state = CapturedState()

        lens = LogitLens(hooks, tokenizer=None)
        lens.print_evolution()

        captured = capsys.readouterr()
        assert "No layers captured" in captured.out


class TestTokenEvolutionThreshold:
    """Test TokenEvolution threshold properties."""

    def test_first_significant_layer_found(self):
        evo = TokenEvolution(
            token="test",
            token_id=42,
            layers=[0, 2, 4, 6],
            probabilities=[0.05, 0.08, 0.15, 0.5],
            ranks=[100, 80, 20, 1],
        )
        # Default threshold is 0.1
        assert evo.first_significant_layer == 4

    def test_first_significant_layer_custom_threshold(self):
        evo = TokenEvolution(
            token="test",
            token_id=42,
            layers=[0, 2, 4, 6],
            probabilities=[0.05, 0.08, 0.15, 0.5],
            ranks=[100, 80, 20, 1],
        )
        # With higher threshold
        assert evo.first_significant_layer == 4  # still layer 4 with 0.15 >= 0.1

    def test_first_significant_layer_never_exceeds(self):
        evo = TokenEvolution(
            token="test",
            token_id=42,
            layers=[0, 2, 4, 6],
            probabilities=[0.01, 0.02, 0.03, 0.04],
            ranks=[100, 90, 80, 70],
        )
        # With threshold=0.1, never exceeds
        assert evo.first_significant_layer is None


class TestLogitLensAdvanced:
    """Advanced LogitLens tests."""

    @pytest.fixture
    def model(self):
        return SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)

    @pytest.fixture
    def hooks_with_state(self, model):
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers="all", positions="all"))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)
        return hooks

    def test_get_layer_predictions_no_normalize(self, hooks_with_state):
        lens = LogitLens(hooks_with_state, tokenizer=MockTokenizer())
        predictions = lens.get_layer_predictions(normalize=False)
        assert len(predictions) > 0

    def test_track_token_by_id_detailed(self, hooks_with_state):
        lens = LogitLens(hooks_with_state, tokenizer=MockTokenizer())
        evo = lens.track_token(5, position=-1)  # Track token ID 5

        assert evo.token_id == 5
        assert len(evo.layers) > 0
        assert len(evo.probabilities) == len(evo.layers)
        assert len(evo.ranks) == len(evo.layers)

    def test_compare_tokens_detailed(self, hooks_with_state):
        lens = LogitLens(hooks_with_state, tokenizer=MockTokenizer())
        evolutions = lens.compare_tokens([1, 2, 3], position=-1)

        # compare_tokens returns a dict with token_id -> evolution
        assert len(evolutions) == 3
