"""Tests for ablate handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
    _async_ablate,
    _check_answer,
    handle_ablate,
)


class TestHandleAblate:
    """Tests for handle_ablate function."""

    def test_handle_ablate_calls_asyncio_run(self):
        """Test that handle_ablate calls asyncio.run."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="Test",
            max_tokens=100,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.asyncio"
        ) as mock_asyncio:
            handle_ablate(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncAblate:
    """Tests for _async_ablate function."""

    @pytest.mark.asyncio
    async def test_missing_expert_prints_error(self, capsys):
        """Test that missing expert prints error."""
        args = Namespace(model="test/model", prompt="Test")

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Error: --expert/-e or --experts is required" in captured.out

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model", expert=6)

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_invalid_experts_format_prints_error(self, capsys):
        """Test that invalid experts format prints error."""
        args = Namespace(
            model="test/model",
            experts="not,valid",
            prompt="Test",
        )

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Error: Invalid experts format" in captured.out

    @pytest.mark.asyncio
    async def test_successful_ablate_single_expert(self, capsys):
        """Test successful ablation with single expert."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="127 * 89 = ",
            max_tokens=100,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="11303")
        mock_router.generate_with_ablation = AsyncMock(return_value=("different", {}))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_ablate(args)

            captured = capsys.readouterr()
            assert "ABLATION" in captured.out
            assert "Expert(s) 6" in captured.out

    @pytest.mark.asyncio
    async def test_successful_ablate_multiple_experts(self, capsys):
        """Test successful ablation with multiple experts."""
        args = Namespace(
            model="test/model",
            experts="6,7,20",
            prompt="Test",
            max_tokens=100,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="normal")
        mock_router.generate_with_ablation = AsyncMock(return_value=("ablated", {}))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_ablate(args)

            captured = capsys.readouterr()
            assert "6, 7, 20" in captured.out


class TestCheckAnswer:
    """Tests for _check_answer helper."""

    def test_correct_answer(self):
        """Test matching answer."""
        assert _check_answer("The answer is 42", 42) is True

    def test_wrong_answer(self):
        """Test non-matching answer."""
        assert _check_answer("The answer is 42", 100) is False

    def test_no_number_in_output(self):
        """Test output with no number."""
        assert _check_answer("No number here", 42) is False

    def test_negative_number(self):
        """Test negative number matching."""
        assert _check_answer("-123", -123) is True

    def test_first_number_is_used(self):
        """Test that first number is extracted."""
        assert _check_answer("3 plus 4 equals 7", 3) is True


class TestAblationBenchmark:
    """Tests for _run_ablation_benchmark function."""

    @pytest.mark.asyncio
    async def test_benchmark_run(self, capsys):
        """Test running ablation benchmark."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
            _run_ablation_benchmark,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="42")
        mock_router.generate_with_ablation = AsyncMock(return_value=("42", {}))

        # Mock the benchmarks
        mock_problem = MagicMock()
        mock_problem.prompt = "6 * 7 = "
        mock_problem.answer = 42

        mock_benchmarks = MagicMock()
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks"
        ) as mock_get_bench:
            mock_get_bench.return_value = mock_benchmarks
            await _run_ablation_benchmark(mock_router, [6], "test-model", 100)

        captured = capsys.readouterr()
        assert "ABLATION BENCHMARK" in captured.out
        assert "Expert(s) 6" in captured.out

    @pytest.mark.asyncio
    async def test_benchmark_shows_broken_when_ablation_breaks(self, capsys):
        """Test that BROKEN is shown when ablation causes failure."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
            _run_ablation_benchmark,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="42")
        mock_router.generate_with_ablation = AsyncMock(return_value=("wrong", {}))

        mock_problem = MagicMock()
        mock_problem.prompt = "6 * 7 = "
        mock_problem.answer = 42

        mock_benchmarks = MagicMock()
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks"
        ) as mock_get_bench:
            mock_get_bench.return_value = mock_benchmarks
            await _run_ablation_benchmark(mock_router, [6], "test-model", 100)

        captured = capsys.readouterr()
        assert "BROKEN" in captured.out
        assert "caused" in captured.out and "additional failures" in captured.out

    @pytest.mark.asyncio
    async def test_benchmark_shows_fixed_when_ablation_improves(self, capsys):
        """Test that FIXED is shown when ablation improves result."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
            _run_ablation_benchmark,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="wrong")
        mock_router.generate_with_ablation = AsyncMock(return_value=("42", {}))

        mock_problem = MagicMock()
        mock_problem.prompt = "6 * 7 = "
        mock_problem.answer = 42

        mock_benchmarks = MagicMock()
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks"
        ) as mock_get_bench:
            mock_get_bench.return_value = mock_benchmarks
            await _run_ablation_benchmark(mock_router, [6], "test-model", 100)

        captured = capsys.readouterr()
        assert "FIXED" in captured.out
        assert "improved" in captured.out

    @pytest.mark.asyncio
    async def test_benchmark_shows_no_change(self, capsys):
        """Test that no change message is shown when accuracy is same."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
            _run_ablation_benchmark,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="wrong")
        mock_router.generate_with_ablation = AsyncMock(return_value=("also wrong", {}))

        mock_problem = MagicMock()
        mock_problem.prompt = "6 * 7 = "
        mock_problem.answer = 42

        mock_benchmarks = MagicMock()
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks"
        ) as mock_get_bench:
            mock_get_bench.return_value = mock_benchmarks
            await _run_ablation_benchmark(mock_router, [6], "test-model", 100)

        captured = capsys.readouterr()
        assert "No change in accuracy" in captured.out

    @pytest.mark.asyncio
    async def test_ablate_with_benchmark_flag(self, capsys):
        """Test ablation with benchmark flag enabled."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="Test",
            max_tokens=100,
            benchmark=True,
        )

        mock_router = AsyncMock()
        mock_router._generate_normal_sync = MagicMock(return_value="42")
        mock_router.generate_with_ablation = AsyncMock(return_value=("42", {}))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        mock_problem = MagicMock()
        mock_problem.prompt = "6 * 7 = "
        mock_problem.answer = 42

        mock_benchmarks = MagicMock()
        mock_benchmarks.get_all_problems.return_value = [mock_problem]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.get_arithmetic_benchmarks"
            ) as mock_get_bench,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_bench.return_value = mock_benchmarks

            await _async_ablate(args)

        captured = capsys.readouterr()
        assert "ABLATION BENCHMARK" in captured.out
