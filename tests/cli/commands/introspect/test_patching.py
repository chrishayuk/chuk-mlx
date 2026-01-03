"""Tests for introspect patching CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIntrospectCommutativity:
    """Tests for introspect_commutativity command."""

    @pytest.fixture
    def commutativity_args(self):
        """Create arguments for commutativity command."""
        return Namespace(
            model="test-model",
            layer=None,
            pairs=None,
            output=None,
        )

    def test_commutativity_basic(self, commutativity_args, mock_ablation_study, capsys):
        """Test basic commutativity analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_commutativity

        with patch("chuk_lazarus.introspection.CommutativityAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()

            # Mock async analyze
            mock_result = MagicMock()
            mock_result.layer = 12
            mock_result.num_pairs = 5
            mock_result.mean_similarity = 0.998
            mock_result.std_similarity = 0.001
            mock_result.min_similarity = 0.995
            mock_result.max_similarity = 0.999
            mock_result.level = MagicMock()
            mock_result.level.value = "very_high"
            mock_result.interpretation = "Lookup table detected"
            mock_result.pairs = []
            mock_result.model_dump.return_value = {}

            async def mock_analyze(**kwargs):
                return mock_result

            mock_analyzer.analyze = mock_analyze
            mock_cls.return_value = mock_analyzer

            introspect_commutativity(commutativity_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_commutativity_explicit_pairs(self, commutativity_args, mock_ablation_study):
        """Test commutativity with explicit pairs."""
        from chuk_lazarus.cli.commands.introspect import introspect_commutativity

        commutativity_args.pairs = "2*3,3*2|7*8,8*7"

        with patch("chuk_lazarus.introspection.CommutativityAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()

            mock_result = MagicMock()
            mock_result.layer = 12
            mock_result.num_pairs = 2
            mock_result.mean_similarity = 0.998
            mock_result.std_similarity = 0.001
            mock_result.min_similarity = 0.995
            mock_result.max_similarity = 0.999
            mock_result.level = MagicMock(value="very_high")
            mock_result.interpretation = "Test"
            mock_result.pairs = []
            mock_result.model_dump.return_value = {}

            async def mock_analyze(**kwargs):
                return mock_result

            mock_analyzer.analyze = mock_analyze
            mock_cls.return_value = mock_analyzer

            introspect_commutativity(commutativity_args)

    def test_commutativity_specific_layer(self, commutativity_args, mock_ablation_study):
        """Test commutativity at specific layer."""
        from chuk_lazarus.cli.commands.introspect import introspect_commutativity

        commutativity_args.layer = 15

        with patch("chuk_lazarus.introspection.CommutativityAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()

            mock_result = MagicMock()
            mock_result.layer = 15
            mock_result.num_pairs = 5
            mock_result.mean_similarity = 0.998
            mock_result.std_similarity = 0.001
            mock_result.min_similarity = 0.995
            mock_result.max_similarity = 0.999
            mock_result.level = MagicMock(value="very_high")
            mock_result.interpretation = "Test"
            mock_result.pairs = []
            mock_result.model_dump.return_value = {}

            async def mock_analyze(**kwargs):
                assert kwargs.get("layer") == 15
                return mock_result

            mock_analyzer.analyze = mock_analyze
            mock_cls.return_value = mock_analyzer

            introspect_commutativity(commutativity_args)

    def test_commutativity_save_output(self, commutativity_args, mock_ablation_study):
        """Test saving commutativity results."""
        from chuk_lazarus.cli.commands.introspect import introspect_commutativity

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            commutativity_args.output = f.name

        with patch("chuk_lazarus.introspection.CommutativityAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()

            mock_result = MagicMock()
            mock_result.layer = 12
            mock_result.num_pairs = 5
            mock_result.mean_similarity = 0.998
            mock_result.std_similarity = 0.001
            mock_result.min_similarity = 0.995
            mock_result.max_similarity = 0.999
            mock_result.level = MagicMock(value="very_high")
            mock_result.interpretation = "Test"
            mock_result.pairs = []
            mock_result.model_dump.return_value = {"test": "data"}

            async def mock_analyze(**kwargs):
                return mock_result

            mock_analyzer.analyze = mock_analyze
            mock_cls.return_value = mock_analyzer

            introspect_commutativity(commutativity_args)

            # Check file was created
            from pathlib import Path

            assert Path(commutativity_args.output).exists()


class TestIntrospectPatch:
    """Tests for introspect_patch command."""

    @pytest.fixture
    def patch_args(self):
        """Create arguments for patch command."""
        return Namespace(
            model="test-model",
            source="7*8=",
            target="7+8=",
            layer=None,
            layers=None,
            position="last",
            blend=1.0,
            max_tokens=10,
            output=None,
        )

    def test_patch_basic(self, patch_args, mock_ablation_study, capsys):
        """Test basic activation patching."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        with patch("chuk_lazarus.introspection.ActivationPatcher") as mock_cls:
            mock_patcher = MagicMock()

            # Create layer result mock
            mock_layer_result = MagicMock()
            mock_layer_result.layer = 12
            mock_layer_result.top_token = "56"
            mock_layer_result.top_prob = 0.95
            mock_layer_result.effect = MagicMock()
            mock_layer_result.effect.value = "full_transfer"

            # Create sweep result mock
            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {"baseline": "15"}

            # sweep_layers is an async method
            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_patch_specific_layer(self, patch_args, mock_ablation_study):
        """Test patching at specific layer."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        patch_args.layer = 15

        with patch("chuk_lazarus.introspection.ActivationPatcher") as mock_cls:
            mock_patcher = MagicMock()

            # Create layer result mock
            mock_layer_result = MagicMock()
            mock_layer_result.layer = 15
            mock_layer_result.top_token = "56"
            mock_layer_result.top_prob = 0.95
            mock_layer_result.effect = MagicMock()
            mock_layer_result.effect.value = "full_transfer"

            # Create sweep result mock
            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {"baseline": "15"}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_cls.return_value = mock_patcher

            introspect_patch(patch_args)

    def test_patch_layer_sweep(self, patch_args, mock_ablation_study, capsys):
        """Test patching across multiple layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        patch_args.layers = "10-15"

        with patch("chuk_lazarus.introspection.ActivationPatcher") as mock_cls:
            mock_patcher = MagicMock()

            # Create layer result mock
            mock_layer_result = MagicMock()
            mock_layer_result.layer = 10
            mock_layer_result.top_token = "15"
            mock_layer_result.top_prob = 0.8
            mock_layer_result.effect = MagicMock()
            mock_layer_result.effect.value = "no_effect"

            # Create sweep result mock
            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {"baseline": "15"}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            captured = capsys.readouterr()
            assert "Layer" in captured.out or "Loading" in captured.out

    def test_patch_save_output(self, patch_args, mock_ablation_study):
        """Test saving patch results."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            patch_args.output = f.name

        with patch("chuk_lazarus.introspection.ActivationPatcher") as mock_cls:
            mock_patcher = MagicMock()

            # Create layer result mock
            mock_layer_result = MagicMock()
            mock_layer_result.layer = 12
            mock_layer_result.top_token = "56"
            mock_layer_result.top_prob = 0.95
            mock_layer_result.effect = MagicMock()
            mock_layer_result.effect.value = "full_transfer"

            # Create sweep result mock
            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {"baseline": "15", "layer_results": []}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            # Check file was created
            from pathlib import Path

            # Check file was created with valid JSON
            assert Path(patch_args.output).exists()
            import json

            with open(patch_args.output) as f:
                data = json.load(f)
                assert isinstance(data, dict)
