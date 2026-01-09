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
