"""Tests for attention_pattern handler."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern import (
    _print_attention_weights,
    _print_header,
    _print_insight,
    handle_attention_pattern,
)


class TestHandleAttentionPattern:
    """Tests for handle_attention_pattern function."""

    def test_handle_attention_pattern_calls_asyncio_run(self):
        """Test that handle_attention_pattern calls asyncio.run."""
        args = Namespace(model="test/model", prompt="test prompt")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_layer(self):
        """Test handle_attention_pattern with layer parameter."""
        args = Namespace(model="test/model", prompt="test prompt", layer=5)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_position(self):
        """Test handle_attention_pattern with position parameter."""
        args = Namespace(model="test/model", prompt="test", position=2)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_head(self):
        """Test handle_attention_pattern with head parameter."""
        args = Namespace(model="test/model", prompt="test", head=0)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_attention_pattern_with_top_k(self):
        """Test handle_attention_pattern with top_k parameter."""
        args = Namespace(model="test/model", prompt="test", top_k=10)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()


class TestPrintHeader:
    """Tests for _print_header function."""

    def test_print_header_basic(self, capsys):
        """Test _print_header prints expected sections."""
        from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
            AttentionPatternConfig,
        )

        config = AttentionPatternConfig(model="test/model", prompt="Hello world")
        _print_header(config)

        captured = capsys.readouterr()
        assert "ATTENTION PATTERN ANALYSIS" in captured.out
        assert "WHAT THIS SHOWS" in captured.out
        assert "test/model" in captured.out
        assert "Hello world" in captured.out

    def test_print_header_with_position(self, capsys):
        """Test _print_header with position."""
        from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
            AttentionPatternConfig,
        )

        config = AttentionPatternConfig(model="test/model", prompt="Test", position=2)
        _print_header(config)

        captured = capsys.readouterr()
        assert "EXPERIMENT" in captured.out


class TestPrintAttentionWeights:
    """Tests for _print_attention_weights function."""

    def test_print_attention_weights_basic(self, capsys):
        """Test _print_attention_weights prints attention data."""
        result = MagicMock()
        result.query_position = 2
        result.query_token = "test"
        result.attention_weights = [(0, 0.5), (1, 0.3), (2, 0.2)]
        result.self_attention = 0.2

        tokens = ["Hello", "world", "test"]
        _print_attention_weights(result, tokens)

        captured = capsys.readouterr()
        assert "ATTENTION WEIGHTS" in captured.out
        assert 'Position 2: "test"' in captured.out

    def test_print_attention_weights_self_attention_marker(self, capsys):
        """Test _print_attention_weights marks self-attention."""
        result = MagicMock()
        result.query_position = 1
        result.query_token = "world"
        result.attention_weights = [(0, 0.5), (1, 0.5)]  # pos 1 in top-k
        result.self_attention = 0.5

        tokens = ["Hello", "world"]
        _print_attention_weights(result, tokens)

        captured = capsys.readouterr()
        assert "(self)" in captured.out

    def test_print_attention_weights_self_not_in_topk(self, capsys):
        """Test _print_attention_weights shows self-attention when not in top-k."""
        result = MagicMock()
        result.query_position = 2
        result.query_token = "test"
        result.attention_weights = [(0, 0.8), (1, 0.2)]  # pos 2 not included
        result.self_attention = 0.05

        tokens = ["Hello", "world", "test"]
        _print_attention_weights(result, tokens)

        captured = capsys.readouterr()
        assert "Self-attention" in captured.out


class TestPrintInsight:
    """Tests for _print_insight function."""

    def test_print_insight(self, capsys):
        """Test _print_insight prints key insight section."""
        _print_insight()

        captured = capsys.readouterr()
        assert "KEY INSIGHT" in captured.out
        assert "attention" in captured.out.lower()
        assert "router" in captured.out.lower()


class TestAttentionPatternConfig:
    """Tests for AttentionPatternConfig."""

    def test_from_args_basic(self):
        """Test basic config creation from args."""
        from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
            AttentionPatternConfig,
        )

        args = MagicMock()
        args.model = "test/model"
        args.prompt = "Hello world"
        args.layer = None
        args.position = None
        args.head = None
        args.top_k = 5

        config = AttentionPatternConfig.from_args(args)

        assert config.model == "test/model"
        assert config.prompt == "Hello world"
        assert config.layer is None
        assert config.top_k == 5

    def test_from_args_with_all_options(self):
        """Test config creation with all options."""
        from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
            AttentionPatternConfig,
        )

        args = MagicMock()
        args.model = "test/model"
        args.prompt = "Test prompt"
        args.layer = 12
        args.position = 3
        args.head = 0
        args.top_k = 10

        config = AttentionPatternConfig.from_args(args)

        assert config.layer == 12
        assert config.position == 3
        assert config.head == 0
        assert config.top_k == 10
