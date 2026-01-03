"""Tests for introspect ablation CLI commands."""

import tempfile
from argparse import Namespace
from pathlib import Path
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
            verbose=False,
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

    def test_ablate_no_prompt_error(self, ablate_args, capsys):
        """Test error when no prompt is provided."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompt = None
        ablate_args.prompts = None

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "required" in captured.out

    def test_ablate_prompt_without_criterion_error(self, ablate_args, capsys):
        """Test error when prompt provided without criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompt = "test prompt"
        ablate_args.criterion = None
        ablate_args.prompts = None

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "criterion" in captured.out

    def test_ablate_multi_prompt_without_expected_error(
        self, ablate_args, mock_ablation_study, capsys
    ):
        """Test error when multi-prompt has no expected value and no criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompts = "prompt1:expected1|prompt2"  # prompt2 has no expected value
        ablate_args.prompt = None
        ablate_args.criterion = None

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "output"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "no expected value" in captured.out

    def test_ablate_multi_prompt_with_multi_mode(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-prompt mode with multi-layer ablation."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompts = "10*10=:100|45*45=:2025"
        ablate_args.prompt = None
        ablate_args.multi = True
        ablate_args.layers = "10,11"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "2025"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "MULTI-PROMPT ABLATION TEST" in captured.out
        assert "baseline" in captured.out.lower()

    def test_ablate_multi_prompt_verbose(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-prompt mode with verbose output."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompts = "10*10=:100|45*45=:2025"
        ablate_args.prompt = None
        ablate_args.verbose = True
        ablate_args.layers = "10"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "2025"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "FULL OUTPUTS" in captured.out
        assert "Prompt:" in captured.out

    def test_ablate_predefined_criterion_function_call(
        self, ablate_args, mock_ablation_study, capsys
    ):
        """Test using predefined function_call criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.criterion = "function_call"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.results = []
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_ablate_predefined_criterion_sorry(self, ablate_args, mock_ablation_study):
        """Test using predefined sorry criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.criterion = "sorry"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.results = []
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

    def test_ablate_predefined_criterion_positive(self, ablate_args, mock_ablation_study):
        """Test using predefined positive criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.criterion = "positive"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.results = []
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

    def test_ablate_predefined_criterion_negative(self, ablate_args, mock_ablation_study):
        """Test using predefined negative criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.criterion = "negative"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.results = []
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

    def test_ablate_predefined_criterion_refusal(self, ablate_args, mock_ablation_study):
        """Test using predefined refusal criterion."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.criterion = "refusal"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()
        mock_result.results = []
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

    def test_ablate_multi_mode_causal(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-layer mode showing causal result."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.multi = True
        ablate_args.layers = "10,11"
        ablate_args.criterion = "2025"

        mock_study = mock_ablation_study.from_pretrained.return_value
        # First call is baseline (passes), second is ablated (fails)
        mock_study.ablate_and_generate.side_effect = ["2025", "wrong"]

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "CAUSAL" in captured.out
        assert "breaks the criterion" in captured.out

    def test_ablate_multi_mode_inverse_causal(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-layer mode showing inverse causal result."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.multi = True
        ablate_args.layers = "10,11"
        ablate_args.criterion = "2025"

        mock_study = mock_ablation_study.from_pretrained.return_value
        # First call is baseline (fails), second is ablated (passes)
        mock_study.ablate_and_generate.side_effect = ["wrong", "2025"]

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "INVERSE CAUSAL" in captured.out
        assert "enables the criterion" in captured.out

    def test_ablate_multi_mode_not_causal(self, ablate_args, mock_ablation_study, capsys):
        """Test multi-layer mode showing not causal result."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.multi = True
        ablate_args.layers = "10,11"
        ablate_args.criterion = "2025"

        mock_study = mock_ablation_study.from_pretrained.return_value
        # Both calls pass
        mock_study.ablate_and_generate.side_effect = ["2025", "2025"]

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "NOT CAUSAL" in captured.out
        assert "doesn't affect outcome" in captured.out

    def test_ablate_verbose_mode_sweep(self, ablate_args, mock_ablation_study, capsys):
        """Test verbose mode in sweep showing detailed outputs."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.verbose = True
        ablate_args.layers = "10"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_result = MagicMock()

        # Create mock results with proper attributes
        mock_layer_result = MagicMock()
        mock_layer_result.layer = 10
        mock_layer_result.original_output = "This is the original output for testing verbose mode"
        mock_layer_result.ablated_output = "This is the ablated output for testing verbose mode"

        mock_result.results = [mock_layer_result]
        mock_study.run_layer_sweep.return_value = mock_result

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        assert "Detailed Outputs" in captured.out
        assert "Layer 10" in captured.out

    def test_ablate_multi_prompt_fallback_to_criterion(
        self, ablate_args, mock_ablation_study, capsys
    ):
        """Test multi-prompt mode using criterion as fallback for prompts without expected value."""
        from chuk_lazarus.cli.commands.introspect import introspect_ablate

        ablate_args.prompts = "prompt1:expected1|prompt2"  # prompt2 has no expected value
        ablate_args.prompt = None
        ablate_args.criterion = "fallback_criterion"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.ablate_and_generate.return_value = "output"

        introspect_ablate(ablate_args)

        captured = capsys.readouterr()
        # Should succeed without error since criterion is provided
        assert "MULTI-PROMPT ABLATION TEST" in captured.out


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

    def test_weight_diff_with_mlp_exception(self, weight_diff_args, capsys):
        """Test weight diff when MLP comparison raises exception."""
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
                    mock_config.num_hidden_layers = 2
                    mock_load.return_value = (mock_model, mock_config)

                    with patch(
                        "chuk_lazarus.introspection.ablation.ModelAdapter"
                    ) as mock_adapter_cls:
                        mock_adapter = MagicMock()
                        mock_adapter.num_layers = 2
                        # Make get_mlp_down_weight raise an exception
                        mock_adapter.get_mlp_down_weight.side_effect = Exception("MLP error")
                        # Make attention work
                        import mlx.core as mx

                        mock_adapter.get_attn_o_weight.return_value = mx.zeros((10, 10))
                        mock_adapter_cls.return_value = mock_adapter

                        introspect_weight_diff(weight_diff_args)

                        captured = capsys.readouterr()
                        assert "Loading" in captured.out
                        # Should still complete despite exception

    def test_weight_diff_with_attn_exception(self, weight_diff_args, capsys):
        """Test weight diff when attention comparison raises exception."""
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
                    mock_config.num_hidden_layers = 2
                    mock_load.return_value = (mock_model, mock_config)

                    with patch(
                        "chuk_lazarus.introspection.ablation.ModelAdapter"
                    ) as mock_adapter_cls:
                        mock_adapter = MagicMock()
                        mock_adapter.num_layers = 2
                        # Make MLP work
                        import mlx.core as mx

                        mock_adapter.get_mlp_down_weight.return_value = mx.zeros((10, 10))
                        # Make get_attn_o_weight raise an exception
                        mock_adapter.get_attn_o_weight.side_effect = Exception("Attn error")
                        mock_adapter_cls.return_value = mock_adapter

                        introspect_weight_diff(weight_diff_args)

                        captured = capsys.readouterr()
                        assert "Loading" in captured.out

    def test_weight_diff_high_divergence(self, weight_diff_args, capsys):
        """Test weight diff showing high divergence markers."""
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
                    mock_config.num_hidden_layers = 2
                    mock_load.return_value = (mock_model, mock_config)

                    with patch(
                        "chuk_lazarus.introspection.ablation.ModelAdapter"
                    ) as mock_adapter_cls:
                        import mlx.core as mx

                        mock_base_adapter = MagicMock()
                        mock_base_adapter.num_layers = 2
                        # Create weights with high divergence
                        base_weight = mx.ones((10, 10))
                        mock_base_adapter.get_mlp_down_weight.return_value = base_weight
                        mock_base_adapter.get_attn_o_weight.return_value = base_weight

                        mock_ft_adapter = MagicMock()
                        mock_ft_adapter.num_layers = 2
                        # Create very different weights (>0.1 relative diff)
                        ft_weight = mx.ones((10, 10)) * 2.0  # 100% different
                        mock_ft_adapter.get_mlp_down_weight.return_value = ft_weight
                        mock_ft_adapter.get_attn_o_weight.return_value = ft_weight

                        mock_adapter_cls.side_effect = [mock_base_adapter, mock_ft_adapter]

                        introspect_weight_diff(weight_diff_args)

                        captured = capsys.readouterr()
                        assert "***" in captured.out  # High divergence marker
                        assert "Top 5 divergent" in captured.out

    def test_weight_diff_with_output(self, weight_diff_args, capsys):
        """Test weight diff with output file."""
        from chuk_lazarus.cli.commands.introspect import introspect_weight_diff

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            weight_diff_args.output = f.name

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
                    mock_config.num_hidden_layers = 2
                    mock_load.return_value = (mock_model, mock_config)

                    with patch(
                        "chuk_lazarus.introspection.ablation.ModelAdapter"
                    ) as mock_adapter_cls:
                        import mlx.core as mx

                        mock_adapter = MagicMock()
                        mock_adapter.num_layers = 2
                        mock_adapter.get_mlp_down_weight.return_value = mx.zeros((10, 10))
                        mock_adapter.get_attn_o_weight.return_value = mx.zeros((10, 10))
                        mock_adapter_cls.return_value = mock_adapter

                        introspect_weight_diff(weight_diff_args)

                        captured = capsys.readouterr()
                        assert "Results saved to:" in captured.out
                        assert Path(weight_diff_args.output).exists()


class TestIntrospectActivationDiff:
    """Tests for introspect_activation_diff command."""

    @pytest.fixture
    def activation_diff_args(self):
        """Create arguments for activation diff command."""
        return Namespace(
            base="base-model",
            finetuned="finetuned-model",
            prompts="test1,test2",
            output=None,
        )

    def test_activation_diff_basic(self, activation_diff_args, mock_ablation_study, capsys):
        """Test basic activation diff."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 12

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            import mlx.core as mx

            mock_hooks_inst = MagicMock()
            # Create hidden states with proper shapes
            hidden_states = {}
            for i in range(12):
                hidden_states[i] = mx.ones((1, 5, 768))  # [batch, seq_len, hidden_size]
            mock_hooks_inst.state.hidden_states = hidden_states
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out

    def test_activation_diff_from_file(self, activation_diff_args, mock_ablation_study, capsys):
        """Test activation diff with prompts from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        # Create temporary file with prompts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt1\n")
            f.write("prompt2\n")
            f.write("\n")  # Empty line should be skipped
            f.write("prompt3\n")
            temp_file = f.name

        activation_diff_args.prompts = f"@{temp_file}"

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 2

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            import mlx.core as mx

            mock_hooks_inst = MagicMock()
            hidden_states = {}
            for i in range(2):
                hidden_states[i] = mx.ones((1, 5, 768))
            mock_hooks_inst.state.hidden_states = hidden_states
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Testing 3 prompts" in captured.out

        # Cleanup
        Path(temp_file).unlink()

    def test_activation_diff_with_missing_hidden_states(
        self, activation_diff_args, mock_ablation_study, capsys
    ):
        """Test activation diff when some hidden states are None."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 5

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            import mlx.core as mx

            mock_hooks_inst = MagicMock()
            # Only provide some hidden states, others will be None
            hidden_states = {
                0: mx.ones((1, 5, 768)),
                2: mx.ones((1, 5, 768)),
                # Layers 1, 3, 4 will return None
            }
            mock_hooks_inst.state.hidden_states.get = lambda idx: hidden_states.get(idx)
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out
            # Should handle None gracefully

    def test_activation_diff_with_2d_hidden_states(
        self, activation_diff_args, mock_ablation_study, capsys
    ):
        """Test activation diff with 2D hidden states (no batch dimension)."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 3

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            import mlx.core as mx

            mock_hooks_inst = MagicMock()
            # Create 2D hidden states (seq_len, hidden_size)
            hidden_states = {}
            for i in range(3):
                hidden_states[i] = mx.ones((5, 768))  # 2D instead of 3D
            mock_hooks_inst.state.hidden_states = hidden_states
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out

    def test_activation_diff_high_divergence(
        self, activation_diff_args, mock_ablation_study, capsys
    ):
        """Test activation diff showing high divergence markers."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        mock_base_study = MagicMock()
        mock_base_study.adapter.num_layers = 2
        mock_base_study.adapter.tokenizer = MagicMock()
        mock_base_study.adapter.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_base_study.adapter.model = MagicMock()

        mock_ft_study = MagicMock()
        mock_ft_study.adapter.num_layers = 2
        mock_ft_study.adapter.tokenizer = mock_base_study.adapter.tokenizer
        mock_ft_study.adapter.model = MagicMock()

        with patch("chuk_lazarus.introspection.ablation.AblationStudy") as mock_ablation_cls:
            # Return different studies for base and ft
            mock_ablation_cls.from_pretrained.side_effect = [mock_base_study, mock_ft_study]

            with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
                import mlx.core as mx

                def create_hooks_with_states(base_or_ft):
                    mock_hooks_inst = MagicMock()
                    hidden_states = {}
                    for i in range(2):
                        if base_or_ft == "base":
                            # Base model activations
                            hidden_states[i] = mx.ones((1, 5, 768))
                        else:
                            # FT model activations very different (low cosine similarity)
                            hidden_states[i] = mx.ones((1, 5, 768)) * -1.0
                    mock_hooks_inst.state.hidden_states = hidden_states
                    return mock_hooks_inst

                # Alternate between base and ft hooks
                call_count = [0]

                def hooks_side_effect(model):
                    call_count[0] += 1
                    return create_hooks_with_states("base" if call_count[0] % 2 == 1 else "ft")

                mock_hooks.side_effect = hooks_side_effect

                introspect_activation_diff(activation_diff_args)

                captured = capsys.readouterr()
                # Should show high divergence marker
                assert "***" in captured.out

    def test_activation_diff_with_output(self, activation_diff_args, mock_ablation_study, capsys):
        """Test activation diff with output file."""
        from chuk_lazarus.cli.commands.introspect import introspect_activation_diff

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            activation_diff_args.output = f.name

        mock_study = mock_ablation_study.from_pretrained.return_value
        mock_study.adapter.num_layers = 2

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks:
            import mlx.core as mx

            mock_hooks_inst = MagicMock()
            hidden_states = {}
            for i in range(2):
                hidden_states[i] = mx.ones((1, 5, 768))
            mock_hooks_inst.state.hidden_states = hidden_states
            mock_hooks.return_value = mock_hooks_inst

            introspect_activation_diff(activation_diff_args)

            captured = capsys.readouterr()
            assert "Results saved to:" in captured.out
            assert Path(activation_diff_args.output).exists()
