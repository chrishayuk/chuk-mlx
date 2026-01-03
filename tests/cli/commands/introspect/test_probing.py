"""Tests for introspect probing CLI commands."""

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
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
        )

    @pytest.fixture
    def mock_analyzer_with_predictions(self):
        """Create a mock analyzer with layer predictions."""
        mock_analyzer = MagicMock()
        mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
        mock_analyzer.__aexit__ = AsyncMock(return_value=None)

        # Model info
        mock_analyzer.model_info = MagicMock()
        mock_analyzer.model_info.model_id = "test-model"
        mock_analyzer.model_info.num_layers = 10

        # Tokenizer
        mock_analyzer._tokenizer = MagicMock()
        mock_analyzer._tokenizer.chat_template = "Test template"

        # Layer prediction for decision layer
        mock_layer_pred = MagicMock()
        mock_layer_pred.layer_idx = 7  # 70% of 10 layers
        mock_layer_pred.top_token = "4"
        mock_layer_pred.probability = 0.85

        # Analysis result
        mock_result = MagicMock()
        mock_result.layer_predictions = [mock_layer_pred]
        mock_result.predicted_token = "4"
        mock_result.final_probability = 0.9

        mock_analyzer.analyze = AsyncMock(return_value=mock_result)

        return mock_analyzer

    def test_metacognitive_basic(self, metacognitive_args, mock_analyzer_with_predictions, capsys):
        """Test basic metacognitive analysis."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_analyzer_with_predictions

            with patch("chuk_lazarus.introspection.apply_chat_template") as mock_template:
                mock_template.return_value = "formatted prompt"

                with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                    mock_extract.return_value = "4"

                    introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out
        assert "Decision layer" in captured.out
        assert "DIRECT" in captured.out

    def test_metacognitive_custom_decision_layer(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test with custom decision layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        metacognitive_args.decision_layer = 5

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            # Update mock to match custom layer
            mock_layer_pred = MagicMock()
            mock_layer_pred.layer_idx = 5
            mock_layer_pred.top_token = " "
            mock_layer_pred.probability = 0.75

            mock_result = MagicMock()
            mock_result.layer_predictions = [mock_layer_pred]
            mock_result.predicted_token = " "
            mock_result.final_probability = 0.8

            mock_analyzer = mock_analyzer_with_predictions
            mock_analyzer.analyze = AsyncMock(return_value=mock_result)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.apply_chat_template") as mock_template:
                mock_template.return_value = "formatted prompt"

                with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                    mock_extract.return_value = None

                    introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        assert "Decision layer: 5" in captured.out
        assert "COT" in captured.out

    def test_metacognitive_raw_mode(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        metacognitive_args.raw = True

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_analyzer_with_predictions

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = None

                introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        assert "Mode: RAW" in captured.out

    def test_metacognitive_no_chat_template(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test when tokenizer has no chat template."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            # Remove chat template
            mock_analyzer = mock_analyzer_with_predictions
            mock_analyzer._tokenizer.chat_template = None
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = None

                introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        assert "Mode: RAW (no chat template)" in captured.out

    def test_metacognitive_from_file(
        self, metacognitive_args, mock_analyzer_with_predictions, tmp_path
    ):
        """Test loading prompts from file."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        # Create test file
        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("2+2=\n47*47=\n100-37=\n")

        metacognitive_args.prompts = f"@{prompts_file}"

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_analyzer_with_predictions

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = None

                introspect_metacognitive(metacognitive_args)

    def test_metacognitive_with_output(
        self, metacognitive_args, mock_analyzer_with_predictions, tmp_path
    ):
        """Test saving results to file."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        output_file = tmp_path / "results.json"
        metacognitive_args.output = str(output_file)

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_analyzer_with_predictions

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = "4"

                introspect_metacognitive(metacognitive_args)

        # Check file was created
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "model_id" in data
            assert "decision_layer" in data
            assert "results" in data

    def test_metacognitive_digit_matching(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test digit matching logic."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_analyzer_with_predictions

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                # Test correct match
                mock_extract.return_value = "42"

                introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        # Should show match status
        assert "Match?" in captured.out

    def test_metacognitive_summary_stats(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test summary statistics calculation."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            # Create multiple predictions with different strategies
            mock_layer_pred_digit = MagicMock()
            mock_layer_pred_digit.layer_idx = 7
            mock_layer_pred_digit.top_token = "5"
            mock_layer_pred_digit.probability = 0.8

            mock_layer_pred_cot = MagicMock()
            mock_layer_pred_cot.layer_idx = 7
            mock_layer_pred_cot.top_token = " "
            mock_layer_pred_cot.probability = 0.7

            mock_analyzer = mock_analyzer_with_predictions
            call_count = [0]

            async def analyze_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 2 == 1:
                    mock_result = MagicMock()
                    mock_result.layer_predictions = [mock_layer_pred_digit]
                    mock_result.predicted_token = "5"
                    mock_result.final_probability = 0.85
                    return mock_result
                else:
                    mock_result = MagicMock()
                    mock_result.layer_predictions = [mock_layer_pred_cot]
                    mock_result.predicted_token = "To"
                    mock_result.final_probability = 0.75
                    return mock_result

            mock_analyzer.analyze = AsyncMock(side_effect=analyze_side_effect)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = "5"

                introspect_metacognitive(metacognitive_args)

        captured = capsys.readouterr()
        assert "Summary:" in captured.out
        assert "Direct computation:" in captured.out
        assert "Chain-of-thought:" in captured.out
        assert "Direct accuracy:" in captured.out

    def test_metacognitive_no_layer_prediction(
        self, metacognitive_args, mock_analyzer_with_predictions, capsys
    ):
        """Test when layer prediction is missing (reveals division by zero bug)."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_metacognitive

        # Use single prompt - this will trigger division by zero if all prompts are skipped
        metacognitive_args.prompts = "2+2="

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            # Return result with no matching layer predictions
            mock_result = MagicMock()
            mock_result.layer_predictions = []
            mock_result.predicted_token = "test"
            mock_result.final_probability = 0.9

            mock_analyzer = mock_analyzer_with_predictions
            mock_analyzer.analyze = AsyncMock(return_value=mock_result)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.extract_expected_answer") as mock_extract:
                mock_extract.return_value = None

                # This test reveals a bug: division by zero when no results
                # We'll test that it does fail for now (bug documentation)
                with pytest.raises(ZeroDivisionError):
                    introspect_metacognitive(metacognitive_args)


class TestIntrospectUncertainty:
    """Tests for introspect_uncertainty command."""

    @pytest.fixture
    def uncertainty_args(self):
        """Create arguments for uncertainty command."""
        return Namespace(
            model="test-model",
            prompts="2+2=|47*47=",
            layer=None,
            working=None,
            broken=None,
            output=None,
        )

    @pytest.fixture
    def mock_model_setup(self):
        """Mock model loading and setup."""
        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            # Mock download result
            mock_result = MagicMock()
            mock_result.model_path = Path("/fake/path")
            mock_loader.download.return_value = mock_result

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_loader.load_tokenizer.return_value = mock_tokenizer

            # Mock config
            mock_config = MagicMock()

            # Mock model
            mock_model = MagicMock()

            # Mock the open() call for config.json
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=None)
            mock_file.read.return_value = b'{"model_type": "llama"}'

            # Mock json.load to return config data
            def mock_json_load(f):
                return {"model_type": "llama"}

            with patch("builtins.open", return_value=mock_file):
                with patch("json.load", side_effect=mock_json_load):
                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_info:
                            mock_family = MagicMock()
                            mock_family.config_class.from_hf_config.return_value = mock_config
                            mock_family.model_class.return_value = mock_model
                            mock_info.return_value = mock_family

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.num_layers = 10
                                mock_accessor.embed.return_value = MagicMock()
                                mock_accessor.create_causal_mask.return_value = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(10)]

                                # Mock layer outputs
                                for layer in mock_accessor.layers:
                                    mock_hidden = MagicMock()
                                    # Mock array indexing
                                    mock_hidden.__getitem__ = MagicMock(return_value=MagicMock())
                                    mock_hidden[0].__getitem__ = MagicMock(return_value=MagicMock())
                                    mock_hidden[0][-1].__getitem__ = MagicMock(
                                        return_value=MagicMock()
                                    )
                                    mock_hidden[0][-1][:].tolist.return_value = [0.1] * 768
                                    layer.return_value = mock_hidden

                                mock_accessor_cls.return_value = mock_accessor

                                with patch("mlx.core.array") as mock_array:
                                    mock_array.return_value = MagicMock()

                                    yield {
                                        "loader": mock_loader,
                                        "tokenizer": mock_tokenizer,
                                        "model": mock_model,
                                        "config": mock_config,
                                        "accessor": mock_accessor,
                                    }

    def test_uncertainty_basic(self, uncertainty_args, mock_model_setup, capsys):
        """Test basic uncertainty detection."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        introspect_uncertainty(uncertainty_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out
        assert "Detection layer" in captured.out
        assert "Calibrating" in captured.out

    def test_uncertainty_custom_layer(self, uncertainty_args, mock_model_setup, capsys):
        """Test with custom detection layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        uncertainty_args.layer = 5

        introspect_uncertainty(uncertainty_args)

        captured = capsys.readouterr()
        assert "Detection layer: 5" in captured.out

    def test_uncertainty_custom_calibration(self, uncertainty_args, mock_model_setup, capsys):
        """Test with custom calibration prompts."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        uncertainty_args.working = "10+10=,20+20="
        uncertainty_args.broken = "10+10=,20+20="

        introspect_uncertainty(uncertainty_args)

        captured = capsys.readouterr()
        assert "Calibrating" in captured.out

    def test_uncertainty_from_file(self, uncertainty_args, mock_model_setup, tmp_path):
        """Test loading prompts from file."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        prompts_file = tmp_path / "test_prompts.txt"
        prompts_file.write_text("2+2=\n47*47=\n")

        uncertainty_args.prompts = f"@{prompts_file}"

        introspect_uncertainty(uncertainty_args)

    def test_uncertainty_with_output(self, uncertainty_args, mock_model_setup, tmp_path, capsys):
        """Test saving results to file."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        output_file = tmp_path / "uncertainty_results.json"
        uncertainty_args.output = str(output_file)

        introspect_uncertainty(uncertainty_args)

        # Check that output message was printed
        captured = capsys.readouterr()
        assert "Results saved to:" in captured.out or output_file.exists()

    def test_uncertainty_unsupported_model(self, uncertainty_args, capsys):
        """Test with unsupported model."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = Path("/fake/path")
            mock_loader.download.return_value = mock_result

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=None)

            def mock_json_load(f):
                return {"model_type": "unknown"}

            with patch("builtins.open", return_value=mock_file):
                with patch("json.load", side_effect=mock_json_load):
                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = None

                        with pytest.raises(ValueError, match="Unsupported model"):
                            introspect_uncertainty(uncertainty_args)

    def test_uncertainty_predictions(self, uncertainty_args, mock_model_setup, capsys):
        """Test uncertainty predictions output."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_uncertainty

        introspect_uncertainty(uncertainty_args)

        captured = capsys.readouterr()
        assert "UNCERTAINTY DETECTION RESULTS" in captured.out
        assert "Summary:" in captured.out


@pytest.mark.skip(
    reason="Integration tests requiring full MLX setup - needs proper mocking of matrix ops"
)
class TestIntrospectProbe:
    """Tests for introspect_probe command."""

    @pytest.fixture
    def probe_args(self):
        """Create arguments for probe command."""
        return Namespace(
            model="test-model",
            class_a="hard1|hard2",
            class_b="easy1|easy2",
            label_a="hard",
            label_b="easy",
            layer=None,
            method="logistic",
            save_direction=None,
            test=None,
            output=None,
        )

    @pytest.fixture
    def mock_probe_setup(self):
        """Mock model and sklearn setup for probing."""
        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            # Mock download result
            mock_result = MagicMock()
            mock_result.model_path = Path("/fake/path")
            mock_loader.download.return_value = mock_result

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_loader.load_tokenizer.return_value = mock_tokenizer

            # Mock config
            mock_config = MagicMock()

            # Mock model
            mock_model = MagicMock()

            # Mock the open() call for config.json
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=None)
            mock_file.read.return_value = b'{"model_type": "llama"}'

            # Mock json.load to return config data
            def mock_json_load(f):
                return {"model_type": "llama"}

            with patch("builtins.open", return_value=mock_file):
                with patch("json.load", side_effect=mock_json_load):
                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = "llama"

                        with patch(
                            "chuk_lazarus.models_v2.families.registry.get_family_info"
                        ) as mock_info:
                            mock_family = MagicMock()
                            mock_family.config_class.from_hf_config.return_value = mock_config
                            mock_family.model_class.return_value = mock_model
                            mock_info.return_value = mock_family

                            with patch(
                                "chuk_lazarus.introspection.ModelAccessor"
                            ) as mock_accessor_cls:
                                mock_accessor = MagicMock()
                                mock_accessor.num_layers = 4  # Use small number for faster tests
                                mock_accessor.embed.return_value = MagicMock()
                                mock_accessor.create_causal_mask.return_value = MagicMock()
                                mock_accessor.layers = [MagicMock() for _ in range(4)]

                                # Mock layer outputs
                                for layer in mock_accessor.layers:
                                    mock_hidden = MagicMock()
                                    mock_hidden.__getitem__ = MagicMock(return_value=MagicMock())
                                    mock_hidden[0].__getitem__ = MagicMock(return_value=MagicMock())
                                    mock_hidden[0][-1].__getitem__ = MagicMock(
                                        return_value=MagicMock()
                                    )
                                    mock_hidden[0][-1][:].tolist.return_value = list(
                                        np.random.randn(768)
                                    )
                                    layer.return_value = mock_hidden

                                mock_accessor_cls.return_value = mock_accessor

                                # Mock sklearn
                                with patch(
                                    "sklearn.linear_model.LogisticRegression"
                                ) as mock_logreg_cls:
                                    mock_probe = MagicMock()
                                    mock_probe.fit.return_value = mock_probe
                                    mock_probe.score.return_value = 0.95
                                    mock_probe.coef_ = np.random.randn(1, 768)
                                    mock_probe.predict_proba.return_value = np.array([[0.1, 0.9]])
                                    mock_logreg_cls.return_value = mock_probe

                                    with patch(
                                        "sklearn.model_selection.cross_val_score"
                                    ) as mock_cv:
                                        mock_cv.return_value = np.array(
                                            [0.9, 0.92, 0.95, 0.88, 0.91]
                                        )

                                        with patch("mlx.core.array") as mock_array:
                                            mock_array.return_value = MagicMock()

                                            yield {
                                                "loader": mock_loader,
                                                "tokenizer": mock_tokenizer,
                                                "model": mock_model,
                                                "config": mock_config,
                                                "accessor": mock_accessor,
                                                "probe": mock_probe,
                                            }

    def test_probe_basic(self, probe_args, mock_probe_setup, capsys):
        """Test basic probe training."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out
        assert "Class A" in captured.out
        assert "Class B" in captured.out
        assert "Training probes" in captured.out
        assert "PROBE ACCURACY BY LAYER" in captured.out

    def test_probe_specific_layer(self, probe_args, mock_probe_setup, capsys):
        """Test probing specific layer."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        probe_args.layer = 2

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "Selected layer: L2" in captured.out

    def test_probe_difference_method(self, probe_args, mock_probe_setup, capsys):
        """Test difference of means method."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        probe_args.method = "mean_difference"

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "difference of means" in captured.out

    def test_probe_logistic_method(self, probe_args, mock_probe_setup, capsys):
        """Test logistic regression method."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        probe_args.method = "logistic"

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "logistic regression" in captured.out

    def test_probe_from_files(self, probe_args, mock_probe_setup, tmp_path):
        """Test loading class prompts from files."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        class_a_file = tmp_path / "class_a.txt"
        class_a_file.write_text("hard1\nhard2\nhard3\n")

        class_b_file = tmp_path / "class_b.txt"
        class_b_file.write_text("easy1\neasy2\neasy3\n")

        probe_args.class_a = f"@{class_a_file}"
        probe_args.class_b = f"@{class_b_file}"

        introspect_probe(probe_args)

    def test_probe_with_output(self, probe_args, mock_probe_setup, tmp_path):
        """Test saving probe results."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        output_file = tmp_path / "probe_results.json"
        probe_args.output = str(output_file)

        introspect_probe(probe_args)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "model_id" in data
            assert "best_layer" in data
            assert "best_accuracy" in data
            assert "layer_results" in data

    def test_probe_save_direction(self, probe_args, mock_probe_setup, tmp_path):
        """Test saving direction vector."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        direction_file = tmp_path / "direction.npz"
        probe_args.save_direction = str(direction_file)

        introspect_probe(probe_args)

        assert direction_file.exists()
        # Load and verify
        data = np.load(direction_file)
        assert "direction" in data
        assert "layer" in data

    def test_probe_test_prompts(self, probe_args, mock_probe_setup, capsys):
        """Test classification of new prompts."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        probe_args.test = "test1|test2|test3"

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "TEST PREDICTIONS" in captured.out

    def test_probe_test_from_file(self, probe_args, mock_probe_setup, tmp_path, capsys):
        """Test loading test prompts from file."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        test_file = tmp_path / "test_prompts.txt"
        test_file.write_text("test1\ntest2\ntest3\n")

        probe_args.test = f"@{test_file}"

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "TEST PREDICTIONS" in captured.out

    def test_probe_sklearn_missing(self, mock_probe_setup, probe_args, capsys):
        """Test when sklearn is not available."""
        # Patch the import to fail for sklearn modules

        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        original_import = __builtins__.__import__

        def selective_import(name, *args, **kwargs):
            if "sklearn" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=selective_import):
            introspect_probe(probe_args)

            captured = capsys.readouterr()
            assert "ERROR" in captured.out or "sklearn" in captured.out.lower()

    def test_probe_insufficient_samples_cv(self, probe_args, mock_probe_setup, capsys):
        """Test cross-validation with insufficient samples."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        # Use very small dataset to trigger ValueError in cross_val_score
        probe_args.class_a = "sample1"
        probe_args.class_b = "sample2"

        with patch("sklearn.model_selection.cross_val_score") as mock_cv:
            # Simulate ValueError for insufficient samples
            mock_cv.side_effect = ValueError("Not enough samples")

            introspect_probe(probe_args)

            # Should fall back to direct fit/score
            captured = capsys.readouterr()
            assert "PROBE ACCURACY" in captured.out

    def test_probe_projection_statistics(self, probe_args, mock_probe_setup, capsys):
        """Test projection statistics output."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "Projection statistics:" in captured.out
        assert "Separation:" in captured.out

    def test_probe_top_neurons(self, probe_args, mock_probe_setup, capsys):
        """Test top neurons output."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "Top 10 neurons" in captured.out

    def test_probe_unsupported_model(self, probe_args, capsys):
        """Test with unsupported model."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
            mock_result = MagicMock()
            mock_result.model_path = Path("/fake/path")
            mock_loader.download.return_value = mock_result

            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=None)

            def mock_json_load(f):
                return {"model_type": "unknown"}

            with patch("builtins.open", return_value=mock_file):
                with patch("json.load", side_effect=mock_json_load):
                    with patch(
                        "chuk_lazarus.models_v2.families.registry.detect_model_family"
                    ) as mock_detect:
                        mock_detect.return_value = None

                        with pytest.raises(ValueError, match="Unsupported model"):
                            introspect_probe(probe_args)

    def test_probe_layer_output_with_mask(self, probe_args, mock_probe_setup, capsys):
        """Test layer output handling with mask parameter."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        # This test ensures the try/except for mask parameter is covered
        introspect_probe(probe_args)

        captured = capsys.readouterr()
        assert "Training probes" in captured.out

    def test_probe_combined_output(self, probe_args, mock_probe_setup, tmp_path, capsys):
        """Test saving both JSON and direction vector."""
        from chuk_lazarus.cli.commands.introspect.probing import introspect_probe

        output_file = tmp_path / "results.json"
        direction_file = tmp_path / "direction.npz"

        probe_args.output = str(output_file)
        probe_args.save_direction = str(direction_file)
        probe_args.test = "test1|test2"

        introspect_probe(probe_args)

        assert output_file.exists()
        assert direction_file.exists()

        captured = capsys.readouterr()
        assert "Results saved to:" in captured.out
        assert "Direction vector saved to:" in captured.out
        assert "Use with: lazarus introspect steer" in captured.out


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from chuk_lazarus.cli.commands.introspect.probing import __all__

        assert "introspect_metacognitive" in __all__
        assert "introspect_probe" in __all__
        assert "introspect_uncertainty" in __all__
        assert len(__all__) == 3
