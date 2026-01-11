"""Tests for introspect ablation CLI commands."""

from argparse import Namespace

import pytest


class TestIntrospectAblate:
    """Tests for introspect_ablate command."""

    @pytest.fixture
    def ablate_args(self):
        """Create arguments for ablate command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

    def test_ablate_no_prompt_error(self, capsys):
        """Test error when no prompt provided."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts=None,
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        with pytest.raises(ValueError, match="--prompt"):
            introspect_ablate(args)

    def test_ablate_prompt_without_criterion_error(self, capsys):
        """Test error when prompt without criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        with pytest.raises(ValueError, match="--criterion"):
            introspect_ablate(args)


class TestAblationConfig:
    """Tests for AblationConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.model == "test-model"
        assert config.prompt == "2+2="
        assert config.criterion == "4"
        assert config.component == "mlp"

    def test_from_args_with_prompts(self):
        """Test creating config with multi-prompt format."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=:4|3+3=:6",
            criterion=None,
            layers="20,21,22",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.prompts == "2+2=:4|3+3=:6"

    def test_from_args_multi_mode(self):
        """Test creating config with multi mode."""
        from chuk_lazarus.cli.commands.introspect._types import AblationConfig

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21,22",
            component="mlp",
            multi=True,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        config = AblationConfig.from_args(args)

        assert config.multi is True


class TestParsePrompts:
    """Tests for prompt parsing utilities."""

    def test_parse_prompts(self):
        """Test parsing prompts from pipe-separated string."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("2+2=|3+3=|4+4=")
        assert len(prompts) == 3
        assert prompts[0] == "2+2="

    def test_parse_layers_range(self):
        """Test parsing layer range."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("20-23")
        assert layers == [20, 21, 22, 23]

    def test_parse_layers_comma_separated(self):
        """Test parsing comma-separated layers."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("20,21,22")
        assert layers == [20, 21, 22]

    def test_parse_layers_none(self):
        """Test parsing None returns None."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers(None)
        assert layers is None


class TestAblationResult:
    """Tests for AblationResult type."""

    def test_result_creation(self):
        """Test creating ablation result."""
        from chuk_lazarus.cli.commands.introspect._types import AblationResult

        result = AblationResult(
            prompt="2+2=",
            expected="4",
            ablation="L20 MLP",
            output="4",
            correct=True,
        )

        assert result.prompt == "2+2="
        assert result.correct is True

    def test_result_to_display(self):
        """Test result display format."""
        from chuk_lazarus.cli.commands.introspect._types import AblationResult

        result = AblationResult(
            prompt="2+2=",
            expected="4",
            ablation="L20 MLP",
            output="4",
            correct=True,
        )

        display = result.to_display()
        assert "PASS" in display
        assert "L20 MLP" in display


class TestPrintMultiPromptResults:
    """Tests for _print_multi_prompt_results helper function."""

    def test_print_results_basic(self, capsys):
        """Test basic multi-prompt results printing."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        # Create mock results
        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "4"
        mock_single_result.passes_criterion = True

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=False)

        captured = capsys.readouterr()
        assert "MULTI-PROMPT ABLATION TEST" in captured.out
        assert "L20 MLP" in captured.out

    def test_print_results_verbose(self, capsys):
        """Test verbose multi-prompt results printing."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "4"
        mock_single_result.passes_criterion = True

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=True)

        captured = capsys.readouterr()
        assert "FULL OUTPUTS" in captured.out
        assert "PASS" in captured.out

    def test_print_results_failing(self, capsys):
        """Test multi-prompt results with failures."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_prompt_results,
        )

        mock_single_result = MagicMock()
        mock_single_result.prompt = "2+2="
        mock_single_result.output = "5"
        mock_single_result.passes_criterion = False

        mock_ablation_result = MagicMock()
        mock_ablation_result.ablation_name = "L20 MLP"
        mock_ablation_result.results = [mock_single_result]

        prompt_pairs = [("2+2=", "4")]

        _print_multi_prompt_results([mock_ablation_result], prompt_pairs, verbose=True)

        captured = capsys.readouterr()
        assert "FAIL" in captured.out


class TestPrintMultiAblationResults:
    """Tests for _print_multi_ablation_results helper function."""

    def test_causal_result(self, capsys):
        """Test printing causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "wrong"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "CAUSAL" in captured.out
        assert "breaks the criterion" in captured.out

    def test_inverse_causal_result(self, capsys):
        """Test printing inverse causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = False
        baseline.output = "wrong"

        ablated = MagicMock()
        ablated.passes_criterion = True
        ablated.output = "4"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "INVERSE CAUSAL" in captured.out
        assert "enables the criterion" in captured.out

    def test_not_causal_result(self, capsys):
        """Test printing not causal ablation result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = True
        ablated.output = "4"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "NOT CAUSAL" in captured.out
        assert "doesn't affect" in captured.out

    def test_baseline_fails_result(self, capsys):
        """Test printing baseline fails result."""
        from unittest.mock import MagicMock

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _print_multi_ablation_results,
        )

        baseline = MagicMock()
        baseline.passes_criterion = False
        baseline.output = "wrong"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "also wrong"

        _print_multi_ablation_results("2+2=", "4", [20, 21], baseline, ablated)

        captured = capsys.readouterr()
        assert "BASELINE FAILS" in captured.out


class TestAsyncIntrospectAblate:
    """Tests for _async_introspect_ablate function."""

    @pytest.mark.asyncio
    async def test_multi_prompt_mode(self, capsys):
        """Test ablation with multiple prompts."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=:4|3+3=:6",
            criterion=None,
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()
        mock_result.ablation_name = "Baseline"
        mock_single = MagicMock()
        mock_single.prompt = "2+2="
        mock_single.output = "4"
        mock_single.passes_criterion = True
        mock_result.results = [mock_single]

        # Need to patch at the introspection module level, not CLI level
        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", "4"), ("3+3=", "6")],
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_prompt_ablation",
                new_callable=AsyncMock,
                return_value=[mock_result],
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_multi_layer_ablation_mode(self, capsys):
        """Test multi-layer ablation mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=True,  # Multi-layer mode
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        baseline = MagicMock()
        baseline.passes_criterion = True
        baseline.output = "4"

        ablated = MagicMock()
        ablated.passes_criterion = False
        ablated.output = "wrong"

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_ablation",
                new_callable=AsyncMock,
                return_value=(baseline, ablated),
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Ablating layers together" in captured.out
        assert "CAUSAL" in captured.out

    @pytest.mark.asyncio
    async def test_sweep_mode(self, capsys):
        """Test ablation sweep mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,  # Sweep mode
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Sweeping layers individually" in captured.out

    @pytest.mark.asyncio
    async def test_sweep_mode_with_output(self, tmp_path, capsys):
        """Test ablation sweep mode with output file."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        output_file = str(tmp_path / "results.json")

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=output_file,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        capsys.readouterr()  # Clear output
        mock_study.save_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_attention_component(self, capsys):
        """Test ablation with attention component."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="attention",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Component: attention" in captured.out

    @pytest.mark.asyncio
    async def test_both_component(self, capsys):
        """Test ablation with both components."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="both",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Component: both" in captured.out

    @pytest.mark.asyncio
    async def test_raw_mode(self, capsys):
        """Test ablation with raw mode."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers="20,21",
            component="mlp",
            multi=False,
            raw=True,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Mode: RAW" in captured.out

    @pytest.mark.asyncio
    async def test_no_layers_uses_all(self, capsys):
        """Test ablation without explicit layers uses all layers."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt="2+2=",
            prompts=None,
            criterion="4",
            layers=None,  # No layers specified
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24
        mock_result = MagicMock()

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_ablation_sweep",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        # Should use all 24 layers
        assert "Sweeping layers individually" in captured.out

    @pytest.mark.asyncio
    async def test_prompts_without_expected_uses_criterion(self, capsys):
        """Test prompts without expected values use criterion."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=|3+3=",  # No expected values
            criterion="4",  # Use criterion for all
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        mock_result = MagicMock()
        mock_result.ablation_name = "Baseline"
        mock_single = MagicMock()
        mock_single.prompt = "2+2="
        mock_single.output = "4"
        mock_single.passes_criterion = True
        mock_result.results = [mock_single]

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", ""), ("3+3=", "")],  # No expected values
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.run_multi_prompt_ablation",
                new_callable=AsyncMock,
                return_value=[mock_result],
            ),
        ):
            await _async_introspect_ablate(args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    @pytest.mark.asyncio
    async def test_prompts_without_expected_or_criterion_error(self):
        """Test prompts without expected values or criterion raises error."""
        from unittest.mock import MagicMock, patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            _async_introspect_ablate,
        )

        args = Namespace(
            model="test-model",
            prompt=None,
            prompts="2+2=|3+3=",  # No expected values
            criterion=None,  # No criterion either
            layers="20,21",
            component="mlp",
            multi=False,
            raw=False,
            max_tokens=50,
            verbose=False,
            output=None,
        )

        mock_study = MagicMock()
        mock_study.adapter.num_layers = 24

        with (
            patch(
                "chuk_lazarus.introspection.ablation.AblationStudy.from_pretrained",
                return_value=mock_study,
            ),
            patch(
                "chuk_lazarus.introspection.ablation.AblationService.parse_prompt_pairs",
                return_value=[("2+2=", ""), ("3+3=", "")],
            ),
            pytest.raises(ValueError, match="has no expected value"),
        ):
            await _async_introspect_ablate(args)


class TestIntrospectWeightDiff:
    """Tests for introspect_weight_diff command."""

    def test_calls_asyncio_run(self):
        """Test that introspect_weight_diff calls asyncio.run."""
        from unittest.mock import patch

        from chuk_lazarus.cli.commands.introspect.ablation import introspect_weight_diff

        args = Namespace(base="test/base", finetuned="test/ft", output=None)

        with patch("chuk_lazarus.cli.commands.introspect.ablation.asyncio") as mock_asyncio:
            introspect_weight_diff(args)
            mock_asyncio.run.assert_called_once()


class TestIntrospectActivationDiff:
    """Tests for introspect_activation_diff command."""

    def test_calls_asyncio_run(self):
        """Test that introspect_activation_diff calls asyncio.run."""
        from unittest.mock import patch

        from chuk_lazarus.cli.commands.introspect.ablation import (
            introspect_activation_diff,
        )

        args = Namespace(base="test/base", finetuned="test/ft", prompts="test", output=None)

        with patch("chuk_lazarus.cli.commands.introspect.ablation.asyncio") as mock_asyncio:
            introspect_activation_diff(args)
            mock_asyncio.run.assert_called_once()
