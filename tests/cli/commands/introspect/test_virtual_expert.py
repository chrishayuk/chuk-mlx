"""Tests for virtual_expert CLI commands."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.virtual_expert import (
    introspect_virtual_expert,
)


class TestIntrospectVirtualExpert:
    """Tests for introspect_virtual_expert function."""

    @pytest.fixture
    def basic_args(self):
        """Create basic args for virtual expert command."""
        return Namespace(
            model="test/model",
            action="solve",
            prompt="2+2=",
            layer=None,
            expert=None,
        )

    @pytest.mark.asyncio
    async def test_solve_action(self, basic_args, capsys):
        """Test solve action calls VirtualExpertService.solve."""
        mock_result = MagicMock()
        mock_result.to_display.return_value = "Result: 4"

        with patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService:
            MockService.solve = AsyncMock(return_value=mock_result)

            await introspect_virtual_expert(basic_args)

            MockService.solve.assert_called_once()
            captured = capsys.readouterr()
            assert "Result: 4" in captured.out

    @pytest.mark.asyncio
    async def test_analyze_action(self, basic_args, capsys):
        """Test analyze action calls VirtualExpertService.analyze."""
        basic_args.action = "analyze"
        basic_args.prompt = None
        basic_args.test_file = None

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Analysis Results"

        with (
            patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService,
            patch("chuk_lazarus.datasets.load_expert_test_categories") as mock_load,
        ):
            MockService.analyze = AsyncMock(return_value=mock_result)
            mock_load.return_value = {"test": ["prompt1"]}

            await introspect_virtual_expert(basic_args)

            MockService.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_benchmark_action(self, basic_args, capsys):
        """Test benchmark action calls VirtualExpertService.benchmark."""
        basic_args.action = "benchmark"
        basic_args.prompt = None
        basic_args.benchmark_file = None

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Benchmark Results"

        with (
            patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService,
            patch("chuk_lazarus.datasets.load_expert_benchmark") as mock_load,
        ):
            MockService.benchmark = AsyncMock(return_value=mock_result)
            mock_load.return_value = [{"prompt": "2+2=", "answer": 4}]

            await introspect_virtual_expert(basic_args)

            MockService.benchmark.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_action(self, basic_args, capsys):
        """Test compare action calls VirtualExpertService.compare."""
        basic_args.action = "compare"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Comparison Results"

        with patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService:
            MockService.compare = AsyncMock(return_value=mock_result)

            await introspect_virtual_expert(basic_args)

            MockService.compare.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_action(self, basic_args, capsys):
        """Test interactive action calls VirtualExpertService.interactive."""
        basic_args.action = "interactive"
        basic_args.prompt = None

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Interactive session ended"

        with patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService:
            MockService.interactive = AsyncMock(return_value=mock_result)

            await introspect_virtual_expert(basic_args)

            MockService.interactive.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_action(self, basic_args, capsys):
        """Test unknown action prints error."""
        basic_args.action = "unknown"

        await introspect_virtual_expert(basic_args)

        captured = capsys.readouterr()
        assert "Unknown action: unknown" in captured.out

    @pytest.mark.asyncio
    async def test_default_action_is_solve(self, capsys):
        """Test that default action is solve."""
        args = Namespace(
            model="test/model",
            prompt="2+2=",
            layer=None,
            expert=None,
            # No action specified
        )

        mock_result = MagicMock()
        mock_result.to_display.return_value = "Result: 4"

        with patch("chuk_lazarus.introspection.virtual_expert.VirtualExpertService") as MockService:
            MockService.solve = AsyncMock(return_value=mock_result)

            await introspect_virtual_expert(args)

            MockService.solve.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_requires_prompt(self, basic_args):
        """Test solve action raises error without prompt."""
        basic_args.action = "solve"
        basic_args.prompt = None

        with pytest.raises(ValueError, match="--prompt required"):
            await introspect_virtual_expert(basic_args)

    @pytest.mark.asyncio
    async def test_compare_requires_prompt(self, basic_args):
        """Test compare action raises error without prompt."""
        basic_args.action = "compare"
        basic_args.prompt = None

        with pytest.raises(ValueError, match="--prompt required"):
            await introspect_virtual_expert(basic_args)


class TestVirtualExpertAction:
    """Tests for VirtualExpertAction enum."""

    def test_action_values(self):
        """Test all action values are defined."""
        from chuk_lazarus.introspection.virtual_expert import VirtualExpertAction

        assert VirtualExpertAction.ANALYZE.value == "analyze"
        assert VirtualExpertAction.SOLVE.value == "solve"
        assert VirtualExpertAction.BENCHMARK.value == "benchmark"
        assert VirtualExpertAction.COMPARE.value == "compare"
        assert VirtualExpertAction.INTERACTIVE.value == "interactive"


class TestVirtualExpertConfig:
    """Tests for VirtualExpertConfig model."""

    def test_config_creation(self):
        """Test creating config with required fields."""
        from chuk_lazarus.introspection.virtual_expert import VirtualExpertConfig

        config = VirtualExpertConfig(
            model="test/model",
            prompt="2+2=",
        )
        assert config.model == "test/model"
        assert config.prompt == "2+2="

    def test_config_optional_fields(self):
        """Test config with optional fields."""
        from chuk_lazarus.introspection.virtual_expert import VirtualExpertConfig

        config = VirtualExpertConfig(
            model="test/model",
            layer=5,
            expert=3,
            prompt="test",
        )
        assert config.layer == 5
        assert config.expert == 3
