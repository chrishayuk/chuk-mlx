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

    def test_commutativity_with_pairs_output(self, commutativity_args, mock_ablation_study, capsys):
        """Test commutativity with actual pair results in output."""
        from chuk_lazarus.cli.commands.introspect import introspect_commutativity

        with patch("chuk_lazarus.introspection.CommutativityAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()

            # Create mock pair with actual values
            mock_pair = MagicMock()
            mock_pair.prompt_a = "2*3"
            mock_pair.prompt_b = "3*2"
            mock_pair.similarity = 0.998

            mock_result = MagicMock()
            mock_result.layer = 12
            mock_result.num_pairs = 1
            mock_result.mean_similarity = 0.998
            mock_result.std_similarity = 0.001
            mock_result.min_similarity = 0.998
            mock_result.max_similarity = 0.998
            mock_result.level = MagicMock(value="very_high")
            mock_result.interpretation = "Test"
            mock_result.pairs = [mock_pair]  # Include pairs to cover line 61
            mock_result.model_dump.return_value = {}

            async def mock_analyze(**kwargs):
                return mock_result

            mock_analyzer.analyze = mock_analyze
            mock_cls.return_value = mock_analyzer

            introspect_commutativity(commutativity_args)

            captured = capsys.readouterr()
            # Verify pair was printed
            assert "2*3" in captured.out
            assert "3*2" in captured.out


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

    def test_patch_with_source_answer(self, patch_args, mock_ablation_study, capsys):
        """Test patching when source has expected answer."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        with (
            patch("chuk_lazarus.introspection.ActivationPatcher") as mock_patcher_cls,
            patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract,
        ):
            # Mock extract_expected_answer to return values
            mock_extract.side_effect = lambda p: "56" if "7*8" in p else "15"

            mock_patcher = MagicMock()

            mock_layer_result = MagicMock()
            mock_layer_result.layer = 12
            mock_layer_result.top_token = "56"
            mock_layer_result.top_prob = 0.95
            mock_layer_result.effect = MagicMock(value="no_effect")

            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_patcher_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            captured = capsys.readouterr()
            # Verify answers were printed (lines 122, 124)
            assert "Source answer: 56" in captured.out
            assert "Target answer: 15" in captured.out

    def test_patch_with_layer_arg(self, patch_args, mock_ablation_study):
        """Test patching with single layer argument (line 129)."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        patch_args.layer = 8
        patch_args.layers = None

        with (
            patch("chuk_lazarus.introspection.ActivationPatcher") as mock_patcher_cls,
            patch("chuk_lazarus.introspection.parse_layers_arg") as mock_parse,
        ):
            # parse_layers_arg returns None to trigger use of args.layer
            mock_parse.return_value = None

            mock_patcher = MagicMock()

            mock_layer_result = MagicMock()
            mock_layer_result.layer = 8
            mock_layer_result.top_token = "56"
            mock_layer_result.top_prob = 0.95
            mock_layer_result.effect = MagicMock(value="no_effect")

            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_patcher_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            # Verify sweep_layers was called with the single layer
            call_args = mock_patcher.sweep_layers.call_args
            assert 8 in call_args.kwargs["layers"]

    def test_patch_with_default_layer_sweep(self, patch_args, mock_ablation_study, capsys):
        """Test patching with default layer sweep (lines 132-133)."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        # No layers or layer specified
        patch_args.layer = None
        patch_args.layers = None

        with (
            patch("chuk_lazarus.introspection.ActivationPatcher") as mock_patcher_cls,
            patch("chuk_lazarus.introspection.parse_layers_arg") as mock_parse,
        ):
            # parse_layers_arg returns None to trigger default sweep
            mock_parse.return_value = None

            mock_patcher = MagicMock()

            mock_layer_result = MagicMock()
            mock_layer_result.layer = 0
            mock_layer_result.top_token = "15"
            mock_layer_result.top_prob = 0.8
            mock_layer_result.effect = MagicMock(value="no_effect")

            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result]
            mock_sweep_result.model_dump.return_value = {}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_patcher_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            # Verify sweep_layers was called with default layer sweep
            call_args = mock_patcher.sweep_layers.call_args
            assert "layers" in call_args.kwargs
            # Default should create evenly spaced layers
            layers = call_args.kwargs["layers"]
            assert isinstance(layers, list)

    def test_patch_with_transferred_effect(self, patch_args, mock_ablation_study, capsys):
        """Test patching that shows transfer (line 172)."""
        from chuk_lazarus.cli.commands.introspect import introspect_patch

        with patch("chuk_lazarus.introspection.ActivationPatcher") as mock_patcher_cls:
            mock_patcher = MagicMock()

            # Create layer result with "transferred" effect
            mock_layer_result1 = MagicMock()
            mock_layer_result1.layer = 10
            mock_layer_result1.top_token = "56"
            mock_layer_result1.top_prob = 0.95
            mock_layer_result1.effect = MagicMock(value="transferred")

            mock_layer_result2 = MagicMock()
            mock_layer_result2.layer = 11
            mock_layer_result2.top_token = "56"
            mock_layer_result2.top_prob = 0.98
            mock_layer_result2.effect = MagicMock(value="transferred")

            mock_sweep_result = MagicMock()
            mock_sweep_result.baseline_token = "15"
            mock_sweep_result.baseline_prob = 0.9
            mock_sweep_result.layer_results = [mock_layer_result1, mock_layer_result2]
            mock_sweep_result.model_dump.return_value = {}

            mock_patcher.sweep_layers = AsyncMock(return_value=mock_sweep_result)
            mock_patcher_cls.return_value = mock_patcher

            introspect_patch(patch_args)

            captured = capsys.readouterr()
            # Verify transfer message was printed (line 172)
            assert "Source answer transferred at layers" in captured.out
            assert "10" in captured.out
            assert "11" in captured.out
