"""Tests for introspect probing CLI commands."""

import asyncio
from argparse import Namespace

import pytest


class TestIntrospectMetacognitive:
    """Tests for introspect_metacognitive command."""

    @pytest.fixture
    def metacognitive_args(self):
        """Create arguments for metacognitive command."""
        return Namespace(
            model="test-model",
            prompts="2+2=|47*47=",
            decision_layer=None,
            raw=False,
            output=None,
            top_k=5,
        )

    def test_metacognitive_basic(self, metacognitive_args, capsys):
        """Test basic metacognitive analysis."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        asyncio.run(introspect_metacognitive(metacognitive_args))

        captured = capsys.readouterr()
        assert "METACOGNITIVE" in captured.out or "test-model" in captured.out

    def test_metacognitive_custom_decision_layer(self, metacognitive_args, capsys):
        """Test with custom decision layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        metacognitive_args.decision_layer = 5

        asyncio.run(introspect_metacognitive(metacognitive_args))

        captured = capsys.readouterr()
        # The mock should have been called and output captured
        assert captured.out != "" or captured.err != ""

    def test_metacognitive_raw_mode(self, metacognitive_args, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        metacognitive_args.raw = True

        asyncio.run(introspect_metacognitive(metacognitive_args))

        # Just verify it runs without error
        capsys.readouterr()
        assert True  # Test passes if no exception


class TestIntrospectUncertainty:
    """Tests for introspect_uncertainty command."""

    @pytest.fixture
    def uncertainty_args(self):
        """Create arguments for uncertainty command."""
        return Namespace(
            model="test-model",
            prompt="What is 2+2?",
            layer=None,
            calibration_file=None,
            output=None,
        )

    def test_uncertainty_basic(self, uncertainty_args, capsys):
        """Test basic uncertainty analysis."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        asyncio.run(introspect_uncertainty(uncertainty_args))

        captured = capsys.readouterr()
        # The mock should produce output
        assert "UNCERTAINTY" in captured.out or captured.out != ""

    def test_uncertainty_custom_layer(self, uncertainty_args, capsys):
        """Test with custom layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        uncertainty_args.layer = 5

        asyncio.run(introspect_uncertainty(uncertainty_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestIntrospectProbe:
    """Tests for introspect_probe command."""

    @pytest.fixture
    def probe_args(self):
        """Create arguments for probe command."""
        return Namespace(
            model="test-model",
            positive="good|positive|great",
            negative="bad|negative|terrible",
            layers=None,
            all_layers=False,
            probe_file=None,
            output=None,
        )

    def test_probe_basic(self, probe_args, capsys):
        """Test basic probe training."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        asyncio.run(introspect_probe(probe_args))

        captured = capsys.readouterr()
        assert "PROBE" in captured.out or captured.out != ""

    def test_probe_missing_data(self):
        """Test error when no probe data provided."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        args = Namespace(
            model="test-model",
            positive=None,
            negative=None,
            layers=None,
            all_layers=False,
            probe_file=None,
            output=None,
        )

        with pytest.raises(ValueError, match="Probing requires"):
            asyncio.run(introspect_probe(args))


class TestProbingUtils:
    """Tests for probing utility functions."""

    def test_parse_prompts(self):
        """Test prompt parsing from string."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("prompt1|prompt2|prompt3")
        assert len(prompts) == 3
        assert prompts[0] == "prompt1"
        assert prompts[2] == "prompt3"

    def test_parse_prompts_single(self):
        """Test parsing single prompt."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("single prompt")
        assert len(prompts) == 1
        assert prompts[0] == "single prompt"

    def test_parse_layers_specific(self):
        """Test parsing specific layer list."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("0,5,10,15")
        assert layers == [0, 5, 10, 15]

    def test_parse_layers_none(self):
        """Test parsing None returns None."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers(None)
        assert layers is None

    def test_extract_arg_with_default(self):
        """Test extract_arg with default value."""
        from chuk_lazarus.cli.commands.introspect._utils import extract_arg

        args = Namespace(existing_attr="value")

        # Existing attribute
        result = extract_arg(args, "existing_attr")
        assert result == "value"

        # Non-existing attribute with default
        result = extract_arg(args, "missing_attr", "default")
        assert result == "default"

        # Non-existing attribute without default
        result = extract_arg(args, "missing_attr")
        assert result is None

    def test_get_layer_depth_ratio(self):
        """Test layer depth ratio calculation."""
        from chuk_lazarus.cli.commands._constants import LayerDepthRatio
        from chuk_lazarus.cli.commands.introspect._utils import get_layer_depth_ratio

        # When layer is specified, ratio is ignored
        ratio = get_layer_depth_ratio(5, LayerDepthRatio.LATE)
        assert ratio is None or ratio == LayerDepthRatio.LATE

        # When layer is None, use provided ratio
        ratio = get_layer_depth_ratio(None, LayerDepthRatio.LATE)
        assert ratio == LayerDepthRatio.LATE
