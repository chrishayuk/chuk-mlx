"""Tests for moe_expert handlers."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
    handle_ablate,
    _async_ablate,
)
from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern import (
    handle_attention_pattern,
    _async_attention_pattern,
    _print_header,
    _print_attention_weights,
    _print_insight,
)
from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing import (
    handle_attention_routing,
)


class TestHandleAblate:
    """Tests for handle_ablate function."""

    def test_calls_asyncio_run(self):
        """Test that handle_ablate calls asyncio.run."""
        args = Namespace(model="test", expert=5, prompt="test", benchmark=False)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.asyncio"
        ) as mock_asyncio:
            handle_ablate(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncAblate:
    """Tests for _async_ablate function."""

    @pytest.mark.asyncio
    async def test_ablate_with_single_expert(self, capsys):
        """Test ablation with single expert."""
        args = Namespace(
            model="test-model",
            expert=5,
            prompt="2+2=",
            max_tokens=100,
            benchmark=False,
        )

        mock_router = MagicMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router._generate_normal_sync = MagicMock(return_value="4")
        mock_router.generate_with_ablation = AsyncMock(return_value=("5", {}))

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter.from_pretrained",
            new_callable=AsyncMock,
            return_value=mock_router,
        ):
            await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_with_multiple_experts(self, capsys):
        """Test ablation with multiple experts."""
        args = Namespace(
            model="test-model",
            experts="5,6,7",
            prompt="2+2=",
            max_tokens=100,
            benchmark=False,
        )

        mock_router = MagicMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router._generate_normal_sync = MagicMock(return_value="4")
        mock_router.generate_with_ablation = AsyncMock(return_value=("5", {}))

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter.from_pretrained",
            new_callable=AsyncMock,
            return_value=mock_router,
        ):
            await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_invalid_experts_format(self, capsys):
        """Test ablation with invalid experts format."""
        args = Namespace(
            model="test-model",
            experts="invalid",
            prompt="2+2=",
        )

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Invalid experts format" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_missing_expert(self, capsys):
        """Test ablation with missing expert argument."""
        args = Namespace(
            model="test-model",
            prompt="2+2=",
        )

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "required" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_missing_prompt(self, capsys):
        """Test ablation with missing prompt."""
        args = Namespace(
            model="test-model",
            expert=5,
        )

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "required" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_with_benchmark(self, capsys):
        """Test ablation with benchmark flag."""
        args = Namespace(
            model="test-model",
            expert=5,
            prompt="2+2=",
            max_tokens=10,
            benchmark=True,
        )

        mock_router = MagicMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router._generate_normal_sync = MagicMock(return_value="4")  # sync method
        mock_router.generate_with_ablation = AsyncMock(return_value=("5", {}))

        # Mock the benchmark functions
        mock_benchmarks = MagicMock()
        mock_problem = MagicMock()
        mock_problem.prompt = "3+3="
        mock_problem.answer = "6"
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter.from_pretrained",
                new_callable=AsyncMock,
                return_value=mock_router,
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks",
                return_value=mock_benchmarks,
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.AblationBenchmarkService.create_problem_result",
                return_value=MagicMock(status="correct"),
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.AblationBenchmarkService.format_summary",
                return_value="Summary",
            ),
        ):
            await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out


class TestHandleAttentionPattern:
    """Tests for handle_attention_pattern function."""

    def test_calls_asyncio_run(self):
        """Test that handle_attention_pattern calls asyncio.run."""
        args = Namespace(model="test", prompt="test", position=None, layer=None, head=None, top_k=5)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.asyncio"
        ) as mock_asyncio:
            handle_attention_pattern(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncAttentionPattern:
    """Tests for _async_attention_pattern function."""

    @pytest.mark.asyncio
    async def test_attention_pattern_basic(self, capsys):
        """Test basic attention pattern analysis."""
        args = Namespace(
            model="test-model",
            prompt="King is to queen",
            position=None,
            layer=None,
            head=None,
            top_k=5,
        )

        mock_info = MagicMock()
        mock_info.moe_layers = [0, 2, 4, 6]
        mock_info.total_layers = 8

        mock_router = AsyncMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router.info = mock_info
        mock_router.tokenizer = MagicMock()
        mock_router.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_router.tokenizer.decode.side_effect = lambda x: f"tok{x[0]}"

        mock_attn_result = MagicMock()
        mock_attn_result.query_position = 4
        mock_attn_result.query_token = "test"
        mock_attn_result.attention_weights = [(0, 0.5), (1, 0.3)]
        mock_attn_result.self_attention = 0.1

        mock_weights = MagicMock()
        mock_weights.positions = {4: MagicMock(expert_indices=[5, 6], weights=[0.7, 0.3])}

        mock_router.capture_router_weights = AsyncMock(return_value=[mock_weights])

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.ExpertRouter.from_pretrained",
                return_value=mock_router,
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.MoEAnalysisService.capture_attention_weights",
                new_callable=AsyncMock,
                return_value=mock_attn_result,
            ),
        ):
            await _async_attention_pattern(args)

        captured = capsys.readouterr()
        assert "ATTENTION PATTERN ANALYSIS" in captured.out
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_attention_pattern_with_layer(self, capsys):
        """Test attention pattern with specified layer."""
        args = Namespace(
            model="test-model",
            prompt="test",
            position=2,
            layer=5,
            head=3,
            top_k=3,
        )

        mock_info = MagicMock()
        mock_info.moe_layers = [0, 2, 4, 6]
        mock_info.total_layers = 8

        mock_router = AsyncMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router.info = mock_info
        mock_router.tokenizer = MagicMock()
        mock_router.tokenizer.encode.return_value = [1, 2, 3]
        mock_router.tokenizer.decode.side_effect = lambda x: f"tok{x[0]}"

        mock_attn_result = MagicMock()
        mock_attn_result.query_position = 2
        mock_attn_result.query_token = "test"
        mock_attn_result.attention_weights = [(0, 0.5)]
        mock_attn_result.self_attention = 0.3

        mock_router.capture_router_weights = AsyncMock(return_value=[])

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.ExpertRouter.from_pretrained",
                return_value=mock_router,
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.MoEAnalysisService.capture_attention_weights",
                new_callable=AsyncMock,
                return_value=mock_attn_result,
            ),
        ):
            await _async_attention_pattern(args)

        captured = capsys.readouterr()
        assert "Using head 3" in captured.out

    @pytest.mark.asyncio
    async def test_attention_pattern_negative_position(self, capsys):
        """Test attention pattern with negative position."""
        args = Namespace(
            model="test-model",
            prompt="one two three",
            position=-1,
            layer=None,
            head=None,
            top_k=5,
        )

        mock_info = MagicMock()
        mock_info.moe_layers = [0, 2]
        mock_info.total_layers = 4

        mock_router = AsyncMock()
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)
        mock_router.info = mock_info
        mock_router.tokenizer = MagicMock()
        mock_router.tokenizer.encode.return_value = [1, 2, 3]
        mock_router.tokenizer.decode.side_effect = lambda x: f"tok{x[0]}"

        mock_attn_result = MagicMock()
        mock_attn_result.query_position = 2
        mock_attn_result.query_token = "three"
        mock_attn_result.attention_weights = [(2, 0.8)]  # Self is in top-k
        mock_attn_result.self_attention = 0.8

        mock_router.capture_router_weights = AsyncMock(return_value=[])

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.ExpertRouter.from_pretrained",
                return_value=mock_router,
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_pattern.MoEAnalysisService.capture_attention_weights",
                new_callable=AsyncMock,
                return_value=mock_attn_result,
            ),
        ):
            await _async_attention_pattern(args)

        captured = capsys.readouterr()
        # Should show self attention marker
        assert "(self)" in captured.out


class TestPrintFunctions:
    """Tests for print helper functions."""

    def test_print_header(self, capsys):
        """Test _print_header function."""
        from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
            AttentionPatternConfig,
        )

        config = AttentionPatternConfig(
            model="test-model",
            prompt="test prompt",
        )
        _print_header(config)

        captured = capsys.readouterr()
        assert "ATTENTION PATTERN ANALYSIS" in captured.out
        assert "test-model" in captured.out

    def test_print_attention_weights(self, capsys):
        """Test _print_attention_weights function."""
        result = MagicMock()
        result.query_position = 3
        result.query_token = "test"
        result.attention_weights = [(0, 0.5), (1, 0.3), (2, 0.15)]
        result.self_attention = 0.05

        tokens = ["a", "b", "c", "test"]
        _print_attention_weights(result, tokens)

        captured = capsys.readouterr()
        assert "ATTENTION WEIGHTS" in captured.out
        assert "Position 3" in captured.out
        assert "0.500" in captured.out or "0.5" in captured.out

    def test_print_attention_weights_self_not_in_top_k(self, capsys):
        """Test _print_attention_weights when self not in top-k."""
        result = MagicMock()
        result.query_position = 3
        result.query_token = "test"
        result.attention_weights = [(0, 0.5), (1, 0.3)]  # Position 3 not included
        result.self_attention = 0.05

        tokens = ["a", "b", "c", "test"]
        _print_attention_weights(result, tokens)

        captured = capsys.readouterr()
        assert "Self-attention" in captured.out

    def test_print_insight(self, capsys):
        """Test _print_insight function."""
        _print_insight()

        captured = capsys.readouterr()
        assert "KEY INSIGHT" in captured.out
        assert "attention" in captured.out.lower()


class TestHandleAttentionRouting:
    """Tests for handle_attention_routing function."""

    def test_calls_asyncio_run(self):
        """Test that handle_attention_routing calls asyncio.run."""
        args = Namespace(model="test", context="analogy")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.attention_routing.asyncio"
        ) as mock_asyncio:
            handle_attention_routing(args)
            mock_asyncio.run.assert_called_once()
