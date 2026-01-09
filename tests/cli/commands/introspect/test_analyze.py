"""Tests for introspect analyze CLI commands."""

import asyncio
from argparse import Namespace

import pytest


class TestIntrospectAnalyze:
    """Tests for introspect_analyze command."""

    @pytest.fixture
    def analyze_args(self):
        """Create arguments for analyze command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            prefix=None,
            adapter=None,
            embedding_scale=None,
            raw=False,
            layers=None,
            all_layers=False,
            layer_strategy="evenly_spaced",
            layer_step=4,
            top_k=10,
            track=None,
            steer=None,
            steer_neuron=None,
            steer_layer=None,
            strength=None,
            inject_layer=None,
            inject_token=None,
            inject_blend=1.0,
            compute_override="none",
            compute_layer=None,
            find_answer=None,
            no_find_answer=False,
            gen_tokens=30,
            expected=None,
            output=None,
        )

    def test_analyze_requires_prompt(self):
        """Test that analyze requires --prompt or --prefix."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        args = Namespace(
            model="test-model",
            prompt=None,
            prefix=None,
            adapter=None,
            embedding_scale=None,
            raw=False,
            layers=None,
            all_layers=False,
            layer_strategy="evenly_spaced",
            layer_step=4,
            top_k=10,
            track=None,
            steer=None,
            steer_neuron=None,
            steer_layer=None,
            strength=None,
            inject_layer=None,
            inject_token=None,
            inject_blend=1.0,
            compute_override="none",
            compute_layer=None,
            find_answer=None,
            no_find_answer=False,
            gen_tokens=30,
            expected=None,
            output=None,
        )

        with pytest.raises(ValueError, match="--prompt"):
            asyncio.run(introspect_analyze(args))

    def test_analyze_basic(self, analyze_args, capsys):
        """Test basic analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert "LOGIT" in captured.out or "ANALYSIS" in captured.out

    def test_analyze_with_prefix(self, analyze_args, capsys):
        """Test analysis with prefix mode."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.prompt = None
        analyze_args.prefix = "The answer is"

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_custom_layers(self, analyze_args, capsys):
        """Test analysis with custom layer selection."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.layers = "0,4,8,12"

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_all_layers(self, analyze_args, capsys):
        """Test analysis with all layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.all_layers = True

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_raw_mode(self, analyze_args, capsys):
        """Test analysis in raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        analyze_args.raw = True

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_analyze_with_output(self, analyze_args, tmp_path, capsys):
        """Test analysis with output file."""
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        output_file = tmp_path / "analysis.json"
        analyze_args.output = str(output_file)

        asyncio.run(introspect_analyze(analyze_args))

        captured = capsys.readouterr()
        assert "saved to" in captured.out


class TestIntrospectCompare:
    """Tests for introspect_compare command."""

    @pytest.fixture
    def compare_args(self):
        """Create arguments for compare command."""
        return Namespace(
            model1="model-a",
            model2="model-b",
            prompt="2+2=",
            top_k=10,
            track=None,
        )

    def test_compare_basic(self, compare_args, capsys):
        """Test basic model comparison."""
        from chuk_lazarus.cli.commands.introspect import introspect_compare

        asyncio.run(introspect_compare(compare_args))

        captured = capsys.readouterr()
        assert "COMPARISON" in captured.out or "Model" in captured.out


class TestIntrospectHooks:
    """Tests for introspect_hooks command."""

    @pytest.fixture
    def hooks_args(self):
        """Create arguments for hooks command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            layers=None,
            capture_attention=False,
            last_only=False,
            no_logit_lens=False,
        )

    def test_hooks_basic(self, hooks_args, capsys):
        """Test basic hooks demonstration."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert "HOOKS" in captured.out or "Captured" in captured.out

    def test_hooks_with_layers(self, hooks_args, capsys):
        """Test hooks with custom layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.layers = "0,4,8,12"

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_hooks_with_attention(self, hooks_args, capsys):
        """Test hooks with attention capture."""
        from chuk_lazarus.cli.commands.introspect import introspect_hooks

        hooks_args.capture_attention = True

        asyncio.run(introspect_hooks(hooks_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestAnalysisConfig:
    """Tests for analysis configuration types."""

    def test_delimiters(self):
        """Test delimiter constants."""
        from chuk_lazarus.cli.commands._constants import Delimiters

        assert Delimiters.LAYER_SEPARATOR == ","
        assert Delimiters.PROMPT_SEPARATOR == "|"
