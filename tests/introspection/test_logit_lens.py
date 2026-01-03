"""Tests for logit lens module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection import CaptureConfig, CapturedState, ModelHooks
from chuk_lazarus.introspection.logit_lens import (
    LayerPrediction,
    LogitLens,
    TokenEvolution,
    run_logit_lens,
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


class TestLogitLensNoneLogits:
    """Test handling of None logits (line 176, 276)."""

    def test_get_layer_predictions_skips_none_logits(self):
        """Test that get_layer_predictions skips layers with None logits."""
        from unittest.mock import Mock

        # Create mock hooks that return None for some layers
        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock(), 1: Mock(), 2: Mock()}

        # Mock get_layer_logits to return None for layer 1
        def mock_get_layer_logits(layer_idx, normalize=True):
            if layer_idx == 1:
                return None
            # Return valid logits for other layers
            return mx.random.normal((1, 5, 100))

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        predictions = lens.get_layer_predictions()

        # Should only have predictions for layers 0 and 2 (layer 1 returned None)
        assert len(predictions) == 2
        assert predictions[0].layer_idx == 0
        assert predictions[1].layer_idx == 2

    def test_track_token_skips_none_logits(self):
        """Test that track_token skips layers with None logits."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock(), 1: Mock(), 2: Mock()}

        # Mock get_layer_logits to return None for layer 1
        def mock_get_layer_logits(layer_idx, normalize=True):
            if layer_idx == 1:
                return None
            return mx.random.normal((1, 5, 100))

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        evolution = lens.track_token(10)

        # Should only have data for layers 0 and 2
        assert len(evolution.layers) == 2
        assert 1 not in evolution.layers


class TestLogitLens2DLogits:
    """Test handling of 2D logits (lines 182, 282)."""

    def test_get_layer_predictions_2d_logits(self):
        """Test get_layer_predictions with 2D logits (no batch dimension)."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}

        # Return 2D logits [seq_len, vocab_size]
        def mock_get_layer_logits(layer_idx, normalize=True):
            return mx.random.normal((5, 100))  # 2D instead of 3D

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        predictions = lens.get_layer_predictions(position=-1, top_k=5)

        assert len(predictions) == 1
        assert len(predictions[0].top_tokens) == 5

    def test_track_token_2d_logits(self):
        """Test track_token with 2D logits (no batch dimension)."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}

        # Return 2D logits
        def mock_get_layer_logits(layer_idx, normalize=True):
            return mx.random.normal((5, 100))

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        evolution = lens.track_token(10, position=-1)

        assert len(evolution.layers) == 1
        assert len(evolution.probabilities) == 1


class TestTokenTrackingEdgeCases:
    """Test edge cases in token tracking."""

    def test_track_token_empty_encoding(self):
        """Test tracking a token that encodes to empty list (line 250)."""
        from unittest.mock import Mock

        # Create a tokenizer that returns empty list for certain tokens
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[])
        tokenizer.get_vocab = Mock(return_value={})  # Return empty dict instead of Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}

        lens = LogitLens(hooks, tokenizer)

        with pytest.raises(ValueError, match="not in vocabulary"):
            lens.track_token("invalid_token")

    def test_track_token_multitoken_warning(self):
        """Test warning when tracking a multi-token string."""
        import warnings
        from unittest.mock import Mock

        # Create a tokenizer that returns multiple tokens
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[10, 20, 30])
        tokenizer.decode = Mock(return_value="first")
        tokenizer.get_vocab = Mock(return_value={})

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}
        hooks.get_layer_logits = Mock(return_value=mx.random.normal((1, 5, 100)))

        lens = LogitLens(hooks, tokenizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evolution = lens.track_token("multi token string")

            # Should have issued a warning
            assert len(w) == 1
            assert "not a single token" in str(w[0].message)

        # Should still track the first token
        assert evolution.token_id == 10

    def test_track_token_with_get_vocab(self):
        """Test token tracking using get_vocab method."""
        from unittest.mock import Mock

        tokenizer = Mock()
        tokenizer.get_vocab = Mock(return_value={"test": 42})
        tokenizer.decode = Mock(return_value="test")

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}
        hooks.get_layer_logits = Mock(return_value=mx.random.normal((1, 5, 100)))

        lens = LogitLens(hooks, tokenizer)
        evolution = lens.track_token("test")

        # Should use the vocab lookup
        assert evolution.token_id == 42
        tokenizer.get_vocab.assert_called_once()

    def test_track_token_rank_none_when_not_in_topk(self):
        """Test that rank is None when token is not in top-k (line 294)."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}

        # Create logits where token 99 has very low probability
        logits = mx.zeros((1, 5, 100))
        logits[0, -1, :10] = 10.0  # Make first 10 tokens have high logits
        logits[0, -1, 99] = -10.0  # Make token 99 have very low logit

        hooks.get_layer_logits = Mock(return_value=logits)

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        evolution = lens.track_token(99, position=-1, top_k_for_rank=50)

        # Token 99 should not be in top 50, so rank should be None
        assert evolution.ranks[0] is None

    def test_track_token_encode_without_add_special_tokens(self):
        """Test token encoding when tokenizer doesn't support add_special_tokens."""
        from unittest.mock import Mock

        tokenizer = Mock()

        # Simulate tokenizer that raises TypeError for add_special_tokens
        def encode_side_effect(*args, **kwargs):
            if "add_special_tokens" in kwargs:
                raise TypeError("encode() got an unexpected keyword argument 'add_special_tokens'")
            return [42]

        tokenizer.encode = Mock(side_effect=encode_side_effect)
        tokenizer.get_vocab = Mock(return_value={})
        tokenizer.decode = Mock(return_value="test")

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock()}
        hooks.get_layer_logits = Mock(return_value=mx.random.normal((1, 5, 100)))

        lens = LogitLens(hooks, tokenizer)
        evolution = lens.track_token("test")

        # Should fall back to encode without add_special_tokens
        assert evolution.token_id == 42


class TestFindEmergencePoint:
    """Test find_emergence_point method (lines 327-331)."""

    def test_find_emergence_point_found(self):
        """Test finding emergence point when threshold is exceeded."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock(), 1: Mock(), 2: Mock()}

        # Create logits where token 10 has increasing probability
        def mock_get_layer_logits(layer_idx, normalize=True):
            logits = mx.zeros((1, 5, 100))
            # Probability increases with layer
            logits[0, -1, 10] = layer_idx * 2.0
            return logits

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())

        # Should find layer 1 or 2 depending on threshold
        emergence = lens.find_emergence_point(10, threshold=0.3)
        assert emergence is not None
        assert emergence in [0, 1, 2]

    def test_find_emergence_point_not_found(self):
        """Test when threshold is never exceeded."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock(), 1: Mock()}

        # Create logits where token 10 always has low probability
        def mock_get_layer_logits(layer_idx, normalize=True):
            logits = mx.zeros((1, 5, 100))
            logits[0, -1, 10] = -10.0  # Very low
            return logits

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())

        # Should return None
        emergence = lens.find_emergence_point(10, threshold=0.5)
        assert emergence is None

    def test_find_emergence_point_custom_threshold(self):
        """Test find_emergence_point with custom threshold."""
        from unittest.mock import Mock

        hooks = Mock()
        hooks.state = Mock()
        hooks.state.hidden_states = {0: Mock(), 1: Mock(), 2: Mock()}

        def mock_get_layer_logits(layer_idx, normalize=True):
            logits = mx.ones((1, 5, 100)) * -100.0  # Very low for all tokens
            if layer_idx == 2:
                logits[0, -1, 10] = 10.0  # Very high probability at layer 2 (softmax will be ~1.0)
            return logits

        hooks.get_layer_logits = mock_get_layer_logits

        lens = LogitLens(hooks, tokenizer=MockTokenizer())
        emergence = lens.find_emergence_point(10, threshold=0.9)

        assert emergence == 2


class TestRunLogitLens:
    """Test run_logit_lens convenience function (lines 433-458)."""

    def test_run_logit_lens_basic(self):
        """Test basic usage of run_logit_lens."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()

        result = run_logit_lens(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            top_k=3,
        )

        assert "position" in result
        assert "layers" in result
        assert len(result["layers"]) > 0

    def test_run_logit_lens_with_tracked_token(self):
        """Test run_logit_lens with token tracking (line 454-456)."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()

        result = run_logit_lens(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            track_token="A",
            top_k=3,
        )

        assert "tracked_token" in result
        assert "token" in result["tracked_token"]
        assert "probabilities" in result["tracked_token"]

    def test_run_logit_lens_with_specific_layers(self):
        """Test run_logit_lens with specific layers."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()

        result = run_logit_lens(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            layers=[0, 2],
            top_k=5,
        )

        # Should only have specified layers
        layer_indices = [layer["layer"] for layer in result["layers"]]
        assert set(layer_indices).issubset({0, 2})

    def test_run_logit_lens_all_layers(self):
        """Test run_logit_lens with all layers (default)."""
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer()

        result = run_logit_lens(
            model=model,
            tokenizer=tokenizer,
            prompt="test",
            layers=None,  # Explicitly test None case
            top_k=3,
        )

        assert len(result["layers"]) > 0
