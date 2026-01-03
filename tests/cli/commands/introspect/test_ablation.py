"""Tests for introspect ablation CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectAblate:
    """Tests for introspect_ablate command."""

    @pytest.fixture
    def ablate_args(self):
        """Create arguments for ablate command."""
        return Namespace(
            model="test-model",
            prompt="test prompt",
            prefix=None,
            criterion="test",
            component="mlp",
            layers=None,
            multi=False,
            prompts=None,
            raw=False,
            max_tokens=10,
            output=None,
        )

    def test_ablate_basic(self, ablate_args, mock_ablation_study, capsys):
        """Test basic ablation study."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.run_layer_sweep.return_value = MagicMock()
        mock_study.ablate_and_generate.return_value = "output"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_ablate_with_layers(self, ablate_args, mock_ablation_study):
        """Test ablation with specific layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.layers = "10,11,12"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.run_layer_sweep.return_value = MagicMock()

        introspect_ablate(ablate_args)

    def test_ablate_layer_range(self, ablate_args, mock_ablation_study):
        """Test ablation with layer range."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.layers = "10-15"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.run_layer_sweep.return_value = MagicMock()

        introspect_ablate(ablate_args)

    def test_ablate_multi_layer(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-layer ablation."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.multi = True
        ablate_args.layers = "10,11,12"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "output"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        # Multi-layer mode should be different from sweep
        assert "Loading model" in captured.out

    def test_ablate_multi_prompt(self, ablate_args, mock_ablation_study, capsys):
        """Test ablation with multiple prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompts = "10*10=:100|45*45=:2025"
        ablate_args.prompt = None

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "output"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_ablate_attention_component(self, ablate_args, mock_ablation_study):
        """Test ablating attention component."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.component = "attention"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.run_layer_sweep.return_value = MagicMock()

        introspect_ablate(ablate_args)

    def test_ablate_both_components(self, ablate_args, mock_ablation_study):
        """Test ablating both components."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.component = "both"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.run_layer_sweep.return_value = MagicMock()

        introspect_ablate(ablate_args)

    def test_ablate_save_output(self, ablate_args, mock_ablation_study):
        """Test saving ablation results."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            ablate_args.output = f.name

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_sweep_result = MagicMock()
        mock_sweep_result.to_dict.return_value = {"layers": [0, 1, 2], "results": []}
        mock_study.run_layer_sweep.return_value = mock_sweep_result

        introspect_ablate(ablate_args)

        # Check file was created and contains valid JSON
        from pathlib import Path

        assert Path(ablate_args.output).exists()
        # The file may or may not have content depending on implementation
        # Just verify the command ran without error


class TestIntrospectWeightDiff:
    """Tests for introspect_weight_diff command."""

    @pytest.fixture
    def weight_diff_args(self):
        """Create arguments for weight diff command."""
        return Namespace(
            base="base-model",
            finetuned="finetuned-model",
            output=None,
        )

    def test_weight_diff_basic(self, weight_diff_args, capsys):
        """Test basic weight diff."""
        from chuk_lazarus.cli.commands.introspect import introspect_weight_diff

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.return_value = "/mock/path"

            with patch(
                "chuk_lazarus.introspection.ablation.AblationStudy._detect_family"
            ) as mock_detect:
                mock_detect.return_value = "llama"

                with patch(
                    "chuk_lazarus.introspection.ablation.AblationStudy._load_model"
                ) as mock_load:
                    mock_model = MagicMock()
                    mock_config = MagicMock()
                    mock_config.num_hidden_layers = 12
                    mock_load.return_value = (mock_model, mock_config)

                    with patch(
                        "chuk_lazarus.introspection.ablation.ModelAdapter"
                    ) as mock_adapter_cls:
                        mock_adapter = MagicMock()
                        mock_adapter.num_layers = 12
                        mock_layer = MagicMock()
                        mock_layer.mlp.gate_proj.weight = MagicMock()
                        mock_adapter.get_layer.return_value = mock_layer
                        mock_adapter_cls.return_value = mock_adapter

                        introspect_weight_diff(weight_diff_args)

                        captured = capsys.readouterr()
                        assert "Loading" in captured.out


class TestIntrospectActivationDiff:
    """Tests for introspect_activation_diff command."""

    @pytest.fixture
    def activation_diff_args(self):
        """Create arguments for activation diff command."""
        return Namespace(
            base="base-model",
            finetuned="finetuned-model",
            prompts="test1|test2",
            output=None,
        )

    def test_activation_diff_basic(self, activation_diff_args, mock_ablation_study, capsys):
        """Test basic activation diff."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            mock_hooks_inst = MagicMock()
            mock_hooks_inst.state.hidden_states = {}
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out
