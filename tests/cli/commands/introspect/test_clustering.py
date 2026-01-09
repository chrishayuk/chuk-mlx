"""Tests for introspect clustering CLI commands."""

import asyncio
from argparse import Namespace

import pytest

from .conftest import requires_sklearn


@requires_sklearn
class TestIntrospectActivationCluster:
    """Tests for introspect_activation_cluster command."""

    @pytest.fixture
    def cluster_args(self):
        """Create arguments for activation cluster command."""
        return Namespace(
            model="test-model",
            class_a="2+2=|5+5=|10+10=",
            class_b="47*47=|67*83=|97*89=",
            label_a="easy",
            label_b="hard",
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
            output=None,
        )

    @pytest.fixture
    def multi_class_args(self):
        """Create arguments for multi-class clustering."""
        return Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=["2+2=|3+3=", "47*47=", "100-50="],
            labels=["addition", "multiplication", "subtraction"],
            layer=6,
            save_plot=None,
            output=None,
        )

    def test_cluster_requires_prompts(self):
        """Test that cluster requires prompt input."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
        )

        with pytest.raises(ValueError, match="Must provide"):
            asyncio.run(introspect_activation_cluster(args))

    def test_cluster_requires_min_prompts(self):
        """Test that cluster requires at least 2 prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a="2+2=",
            class_b=None,
            label_a="single",
            label_b=None,
            prompt_groups=None,
            labels=None,
            layer=None,
            save_plot=None,
        )

        with pytest.raises(ValueError, match="at least 2 prompts"):
            asyncio.run(introspect_activation_cluster(args))

    def test_cluster_basic(self, cluster_args, capsys):
        """Test basic two-class clustering."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        asyncio.run(introspect_activation_cluster(cluster_args))

        captured = capsys.readouterr()
        assert "CLUSTERING" in captured.out or "test-model" in captured.out

    def test_cluster_with_layer(self, cluster_args, capsys):
        """Test clustering with specific layer."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        cluster_args.layer = 6

        asyncio.run(introspect_activation_cluster(cluster_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_cluster_multi_class(self, multi_class_args, capsys):
        """Test multi-class clustering."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        asyncio.run(introspect_activation_cluster(multi_class_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_cluster_label_count_mismatch(self):
        """Test error when prompt groups and labels don't match."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_cluster

        args = Namespace(
            model="test-model",
            class_a=None,
            class_b=None,
            label_a=None,
            label_b=None,
            prompt_groups=["2+2=|3+3=", "47*47="],
            labels=["only_one_label"],  # Mismatch
            layer=None,
            save_plot=None,
        )

        with pytest.raises(ValueError, match="must match"):
            asyncio.run(introspect_activation_cluster(args))


class TestClusteringConfig:
    """Tests for clustering configuration."""

    def test_parse_prompts(self):
        """Test prompt parsing."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("2+2=|3+3=|4+4=")
        assert len(prompts) == 3
        assert prompts[0] == "2+2="

    def test_parse_layers(self):
        """Test layer parsing."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_layers

        layers = parse_layers("4,6,8")
        assert layers == [4, 6, 8]

    def test_display_defaults(self):
        """Test display default constants."""
        from chuk_lazarus.cli.commands._constants import DisplayDefaults

        assert DisplayDefaults.ASCII_GRID_WIDTH > 0
        assert DisplayDefaults.ASCII_GRID_HEIGHT > 0
