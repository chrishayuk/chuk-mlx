"""Tests for ablate handler."""

from argparse import Namespace
from unittest.mock import patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate import (
    _async_ablate,
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
        """Test that missing prompt prints error when benchmark not specified."""
        args = Namespace(model="test/model", expert=6, benchmark=False)

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required for ablate action" in captured.out

    @pytest.mark.asyncio
    async def test_invalid_experts_format_prints_error(self, capsys):
        """Test that invalid experts format prints error."""
        args = Namespace(
            model="test/model",
            experts="a,b,c",  # Invalid format
            prompt="Test",
            benchmark=False,
        )

        await _async_ablate(args)

        captured = capsys.readouterr()
        assert "Error: Invalid experts format" in captured.out

    @pytest.mark.asyncio
    async def test_multiple_experts_parsing(self, capsys):
        """Test that multiple experts are parsed correctly."""
        args = Namespace(
            model="test/model",
            experts="1,2,3",
            prompt="Test",
            benchmark=False,
            max_tokens=100,
            layer=None,
        )

        # Mock ExpertRouter to avoid actual model loading
        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.ablate.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained.side_effect = Exception("Test bypass")

            # The handler will raise the exception from from_pretrained
            with pytest.raises(Exception, match="Test bypass"):
                await _async_ablate(args)


class TestAblationBenchmarkService:
    """Tests for AblationBenchmarkService helper functions."""

    def test_check_answer_correct(self):
        """Test check_answer returns True for correct answer."""
        from chuk_lazarus.introspection.moe.ablation_service import (
            AblationBenchmarkService,
        )

        assert AblationBenchmarkService.check_answer("The answer is 42", 42) is True
        assert AblationBenchmarkService.check_answer("42", 42) is True

    def test_check_answer_incorrect(self):
        """Test check_answer returns False for incorrect answer."""
        from chuk_lazarus.introspection.moe.ablation_service import (
            AblationBenchmarkService,
        )

        assert AblationBenchmarkService.check_answer("The answer is 41", 42) is False
        assert AblationBenchmarkService.check_answer("no number here", 42) is False

    def test_check_answer_negative(self):
        """Test check_answer handles negative numbers."""
        from chuk_lazarus.introspection.moe.ablation_service import (
            AblationBenchmarkService,
        )

        assert AblationBenchmarkService.check_answer("-5", -5) is True
        assert AblationBenchmarkService.check_answer("The result is -10", -10) is True


class TestBenchmarkProblemResult:
    """Tests for BenchmarkProblemResult model."""

    def test_status_broken(self):
        """Test status is BROKEN when normal correct but ablated incorrect."""
        from chuk_lazarus.introspection.moe.ablation_service import BenchmarkProblemResult

        result = BenchmarkProblemResult(
            prompt="2+2=",
            expected_answer=4,
            normal_output="4",
            ablated_output="5",
            normal_correct=True,
            ablated_correct=False,
        )
        assert result.status == "BROKEN"

    def test_status_fixed(self):
        """Test status is FIXED when normal incorrect but ablated correct."""
        from chuk_lazarus.introspection.moe.ablation_service import BenchmarkProblemResult

        result = BenchmarkProblemResult(
            prompt="2+2=",
            expected_answer=4,
            normal_output="5",
            ablated_output="4",
            normal_correct=False,
            ablated_correct=True,
        )
        assert result.status == "FIXED"

    def test_status_empty(self):
        """Test status is empty when both same."""
        from chuk_lazarus.introspection.moe.ablation_service import BenchmarkProblemResult

        result = BenchmarkProblemResult(
            prompt="2+2=",
            expected_answer=4,
            normal_output="4",
            ablated_output="4",
            normal_correct=True,
            ablated_correct=True,
        )
        assert result.status == ""


class TestAblationBenchmarkResult:
    """Tests for AblationBenchmarkResult model."""

    def test_accuracy_calculations(self):
        """Test accuracy computed fields."""
        from chuk_lazarus.introspection.moe.ablation_service import (
            AblationBenchmarkResult,
            BenchmarkProblemResult,
        )

        problems = [
            BenchmarkProblemResult(
                prompt="1+1=",
                expected_answer=2,
                normal_output="2",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2+2=",
                expected_answer=4,
                normal_output="4",
                ablated_output="4",
                normal_correct=True,
                ablated_correct=True,
            ),
        ]

        result = AblationBenchmarkResult(expert_indices=[6], problems=problems)

        assert result.normal_correct_count == 2
        assert result.ablated_correct_count == 1
        assert result.normal_accuracy == 1.0
        assert result.ablated_accuracy == 0.5
        assert result.accuracy_diff == -1
        assert result.broken_count == 1
        assert result.fixed_count == 0
