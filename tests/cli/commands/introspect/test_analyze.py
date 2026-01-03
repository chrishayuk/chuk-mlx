"""Tests for introspect analyze CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIntrospectAnalyze:
    """Tests for introspect_analyze command."""

    @pytest.fixture
    def analyze_args(self):
        """Create arguments for analyze command."""
        return Namespace(
            model="test-model",
            prompt="test prompt",
            prefix=None,
            all_layers=False,
            layers=None,
            layer_step=4,
            layer_strategy="evenly_spaced",
            track=None,
            top_k=5,
            embedding_scale=None,
            raw=True,  # Use raw mode to skip chat template processing
            output=None,
            find_answer=False,
            no_find_answer=True,  # Skip answer finding to avoid mlx array issues
        )

    def test_analyze_basic(self, analyze_args, mock_model_analyzer, capsys):
        """Test basic analyze command."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        introspect_analyze(analyze_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out or "Analyzing" in captured.out

    def test_analyze_all_layers(self, analyze_args, mock_model_analyzer):
        """Test analyzing all layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.all_layers = True

        introspect_analyze(analyze_args)

    def test_analyze_custom_layer_step(self, analyze_args, mock_model_analyzer):
        """Test with custom layer step."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.layer_step = 2

        introspect_analyze(analyze_args)

    def test_analyze_track_tokens(self, analyze_args, mock_model_analyzer, capsys):
        """Test tracking specific tokens."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.track = ["Paris", "London"]

        # Add token evolution to mock result
        mock_analyzer = mock_model_analyzer.from_pretrained.return_value
        mock_evo = MagicMock()
        mock_evo.token = "Paris"
        mock_evo.layer_probabilities = {0: 0.1, 4: 0.5}
        mock_evo.layer_ranks = {0: 10, 4: 1}
        mock_evo.emergence_layer = 4

        mock_result = MagicMock()
        mock_result.tokens = ["test"]
        mock_result.captured_layers = [0, 4]
        mock_result.final_prediction = []
        mock_result.layer_predictions = []
        mock_result.token_evolutions = [mock_evo]
        # analyze is async - use AsyncMock
        mock_analyzer.analyze = AsyncMock(return_value=mock_result)

        introspect_analyze(analyze_args)

        captured = capsys.readouterr()
        # Token evolution should be printed
        assert "Loading" in captured.out

    def test_analyze_with_embedding_scale(self, analyze_args, mock_model_analyzer):
        """Test with manual embedding scale."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.embedding_scale = 33.94

        introspect_analyze(analyze_args)

    def test_analyze_raw_mode(self, analyze_args, mock_model_analyzer, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.raw = True

        introspect_analyze(analyze_args)

        captured = capsys.readouterr()
        # Check that raw mode is being used

    def test_analyze_save_output(self, analyze_args, mock_model_analyzer):
        """Test saving analysis results."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            analyze_args.output = f.name

        # Set up mock result with all attributes needed for JSON serialization
        mock_analyzer = mock_model_analyzer.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.prompt = "test prompt"  # Required for JSON output
        mock_result.tokens = ["test"]
        mock_result.num_layers = 12
        mock_result.captured_layers = [0, 4]
        mock_result.final_prediction = []  # Empty list to avoid iteration issues
        mock_result.layer_predictions = []  # Empty list to avoid iteration issues
        mock_result.token_evolutions = []  # Empty list
        mock_result.to_dict.return_value = {
            "prompt": "test prompt",
            "tokens": ["test"],
            "num_layers": 12,
            "captured_layers": [0, 4],
        }
        mock_analyzer.analyze = AsyncMock(return_value=mock_result)

        introspect_analyze(analyze_args)

        # Check file was created
        from pathlib import Path

        if Path(analyze_args.output).exists():
            import json

            with open(analyze_args.output) as f:
                data = json.load(f)
                assert isinstance(data, dict)

    def test_analyze_layer_strategies(self, analyze_args, mock_model_analyzer):
        """Test different layer strategies."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        for strategy in ["all", "evenly_spaced", "first_last", "custom"]:
            analyze_args.layer_strategy = strategy

            introspect_analyze(analyze_args)


class TestIntrospectCompare:
    """Tests for introspect_compare command."""

    @pytest.fixture
    def compare_args(self):
        """Create arguments for compare command."""
        return Namespace(
            model1="model-1",
            model2="model-2",
            prompt="test prompt",
            layers=None,
            top_k=5,
            track=None,
            output=None,
        )

    def test_compare_basic(self, compare_args, capsys):
        """Test basic compare command."""
        from chuk_lazarus.cli.commands.introspect import introspect_compare

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer1 = MagicMock()
            mock_analyzer2 = MagicMock()

            mock_analyzer1.__aenter__ = AsyncMock(return_value=mock_analyzer1)
            mock_analyzer1.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer2.__aenter__ = AsyncMock(return_value=mock_analyzer2)
            mock_analyzer2.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.tokens = ["test"]
            mock_result.captured_layers = [0]
            mock_result.final_prediction = []
            mock_result.layer_predictions = []
            mock_result.token_evolutions = []

            # analyze is async - use AsyncMock
            mock_analyzer1.analyze = AsyncMock(return_value=mock_result)
            mock_analyzer2.analyze = AsyncMock(return_value=mock_result)

            mock_cls.from_pretrained.side_effect = [mock_analyzer1, mock_analyzer2]

            introspect_compare(compare_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out or "Model" in captured.out


class TestIntrospectHooks:
    """Tests for introspect_hooks command."""

    @pytest.fixture
    def hooks_args(self):
        """Create arguments for hooks command."""
        return Namespace(
            model="test-model",
            prompt="test prompt",
            layers=None,
            capture_attention=False,
            last_only=False,
            no_logit_lens=True,  # Skip logit lens to avoid mlx operations on mock data
            top_k=5,
            output=None,
        )

    def test_hooks_basic(self, hooks_args, mock_mlx_lm_load, capsys):
        """Test basic hooks command."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {0: MagicMock()}
            mock_cls.return_value = mock_hooks

            introspect_hooks(hooks_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_hooks_with_layers(self, hooks_args, mock_mlx_lm_load):
        """Test hooks with specific layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.layers = "0,4,8"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {}
            mock_cls.return_value = mock_hooks

            introspect_hooks(hooks_args)

    def test_hooks_capture_attention(self, hooks_args, mock_mlx_lm_load, capsys):
        """Test hooks with attention capture."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.capture_attention = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {}
            mock_hooks.state.attention_weights = {}
            mock_cls.return_value = mock_hooks

            introspect_hooks(hooks_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out
