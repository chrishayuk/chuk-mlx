"""Tests for introspect embedding CLI commands."""

import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .conftest import requires_sklearn


@requires_sklearn
class TestIntrospectEmbedding:
    """Tests for introspect_embedding command."""

    @pytest.fixture
    def embedding_args(self):
        """Create arguments for embedding command."""
        return Namespace(
            model="test-model",
            layers=None,
            operation=None,
            output=None,
        )

    def test_embedding_basic(self, embedding_args, mock_ablation_study, capsys):
        """Test basic embedding analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            # Mock embedding access
            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out
                        assert "TASK TYPE DETECTION" in captured.out

    def test_embedding_specific_layers(self, embedding_args, mock_ablation_study, capsys):
        """Test embedding analysis at specific layers."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        embedding_args.layers = "0,4,8"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_embedding_specific_operation(self, embedding_args, mock_ablation_study, capsys):
        """Test embedding analysis with specific operation."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        embedding_args.operation = "mult"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        captured = capsys.readouterr()
                        assert "Loading model" in captured.out

    def test_embedding_save_output(self, embedding_args, mock_ablation_study):
        """Test saving embedding analysis results."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_embedding

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            embedding_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            mock_study = mock_ablation_study.from_pretrained.return_value
            mock_model = MagicMock()
            mock_model.model.embed_tokens.return_value = mx.zeros((1, 5, 768))
            mock_study.adapter.model = mock_model

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.LinearRegression") as mock_lin:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_lin.return_value = mock_reg

                    with patch("sklearn.model_selection.cross_val_score") as mock_cv:
                        mock_cv.return_value = np.array([0.9, 0.95, 0.92])

                        introspect_embedding(embedding_args)

                        if Path(embedding_args.output).exists():
                            import json

                            with open(embedding_args.output) as f:
                                data = json.load(f)
                                assert "results" in data


@requires_sklearn
class TestIntrospectEarlyLayers:
    """Tests for introspect_early_layers command."""

    @pytest.fixture
    def early_layers_args(self):
        """Create arguments for early layers command."""
        return Namespace(
            model="test-model",
            layers=None,
            operations=None,
            digits=None,
            analyze_positions=False,
            output=None,
        )

    def test_early_layers_basic(self, early_layers_args, mock_ablation_study, capsys):
        """Test basic early layers analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "REPRESENTATION SIMILARITY" in captured.out

    def test_early_layers_specific_layers(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis at specific layers."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.layers = "0,2,4"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "0, 2, 4" in captured.out

    def test_early_layers_specific_operations(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis with specific operations."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.operations = "*,+"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "*, +" in captured.out or "Operations" in captured.out

    def test_early_layers_custom_digits(self, early_layers_args, mock_ablation_study, capsys):
        """Test early layers analysis with custom digit range."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.digits = "2-5"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "2-5" in captured.out or "Digit" in captured.out

    def test_early_layers_with_position_analysis(
        self, early_layers_args, mock_ablation_study, capsys
    ):
        """Test early layers analysis with position analysis."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        early_layers_args.analyze_positions = True

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 5, 768)),
                1: mx.zeros((1, 5, 768)),
                2: mx.zeros((1, 5, 768)),
                4: mx.zeros((1, 5, 768)),
                8: mx.zeros((1, 5, 768)),
            }
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    captured = capsys.readouterr()
                    assert "Loading model" in captured.out
                    assert "POSITION-WISE ANALYSIS" in captured.out

    def test_early_layers_save_output(self, early_layers_args, mock_ablation_study):
        """Test saving early layers analysis results."""
        import mlx.core as mx

        from chuk_lazarus.cli.commands.introspect import introspect_early_layers

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            early_layers_args.output = f.name

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()
            mock_hooks.state.hidden_states = {
                0: mx.zeros((1, 1, 768)),
                1: mx.zeros((1, 1, 768)),
                2: mx.zeros((1, 1, 768)),
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            with patch("sklearn.linear_model.LogisticRegression") as mock_lr:
                mock_probe = MagicMock()
                mock_probe.fit.return_value = mock_probe
                mock_probe.score.return_value = 0.95
                mock_lr.return_value = mock_probe

                with patch("sklearn.linear_model.Ridge") as mock_ridge:
                    mock_reg = MagicMock()
                    mock_reg.fit.return_value = mock_reg
                    # Return prediction with same size as input
                    mock_reg.predict.side_effect = lambda X: np.ones(len(X)) * 5
                    mock_reg.score.return_value = 0.85
                    mock_ridge.return_value = mock_reg

                    introspect_early_layers(early_layers_args)

                    if Path(early_layers_args.output).exists():
                        import json

                        with open(early_layers_args.output) as f:
                            data = json.load(f)
                            assert "probe_results" in data
