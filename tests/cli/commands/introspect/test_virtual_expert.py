"""Tests for virtual_expert CLI commands."""

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


@pytest.fixture
def mock_model():
    """Create a mock model with MoE structure."""
    model = MagicMock()

    # Create layers with MoE structure
    layers = []
    for i in range(12):
        layer = MagicMock()
        layer.mlp = MagicMock()
        if i % 2 == 0:  # Make even layers MoE layers
            layer.mlp.router = MagicMock()
        layers.append(layer)

    model.model.layers = layers
    return model


@pytest.fixture
def mock_dense_model():
    """Create a mock model without MoE structure (dense)."""

    # Create a real class to avoid MagicMock's hasattr behavior
    class DenseMLP:
        pass  # No router attribute

    class DenseLayer:
        def __init__(self):
            self.mlp = DenseMLP()

    class DenseLayerList:
        def __init__(self):
            self._layers = [DenseLayer() for _ in range(12)]

        def __iter__(self):
            return iter(self._layers)

        def __len__(self):
            return len(self._layers)

        def __getitem__(self, idx):
            return self._layers[idx]

    class DenseModelBackbone:
        def __init__(self):
            self.layers = DenseLayerList()

    model = MagicMock()
    model.model = DenseModelBackbone()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    return tokenizer


@pytest.fixture
def mock_load_model(mock_model, mock_tokenizer):
    """Mock the _load_model function."""
    with patch("chuk_lazarus.cli.commands.introspect.virtual_expert._load_model") as mock_load:
        mock_load.return_value = (mock_model, mock_tokenizer)
        yield mock_load


@pytest.fixture
def mock_load_dense_model(mock_dense_model, mock_tokenizer):
    """Mock the _load_model function to return dense model."""
    with patch("chuk_lazarus.cli.commands.introspect.virtual_expert._load_model") as mock_load:
        mock_load.return_value = (mock_dense_model, mock_tokenizer)
        yield mock_load


@pytest.fixture
def mock_virtual_moe_wrapper():
    """Mock VirtualMoEWrapper."""
    with patch("chuk_lazarus.inference.VirtualMoEWrapper") as mock_wrapper_class:
        mock_wrapper = MagicMock()
        mock_wrapper.calibrate = MagicMock()
        mock_wrapper.routing_threshold = 0.5

        # Mock solve method
        mock_result = MagicMock()
        mock_result.answer = "11303"
        mock_result.is_correct = True
        mock_result.plugin_name = "math"
        mock_result.used_virtual_expert = True
        mock_result.routing_score = 0.95
        mock_wrapper.solve.return_value = mock_result

        # Mock benchmark method
        mock_analysis = MagicMock()
        mock_analysis.model_name = "test-model"
        mock_analysis.total_problems = 10
        mock_analysis.correct_without_virtual = 5
        mock_analysis.correct_with_virtual = 9
        mock_analysis.accuracy_without = 0.5
        mock_analysis.accuracy_with = 0.9
        mock_analysis.improvement = 0.4
        mock_analysis.times_virtual_used = 7
        mock_analysis.avg_routing_score = 0.85
        mock_analysis.plugins_used = {"math": 7}

        # Mock individual results
        mock_problem_result = MagicMock()
        mock_problem_result.prompt = "2 + 2 = "
        mock_problem_result.answer = "4"
        mock_problem_result.correct_answer = 4
        mock_problem_result.is_correct = True
        mock_problem_result.used_virtual_expert = False
        mock_analysis.results = [mock_problem_result]
        mock_analysis.model_dump.return_value = {"total_problems": 10}

        mock_wrapper.benchmark.return_value = mock_analysis

        # Mock compare method
        mock_wrapper.compare = MagicMock()

        # Mock _generate_direct method
        mock_wrapper._generate_direct.return_value = "42"

        mock_wrapper_class.return_value = mock_wrapper
        yield mock_wrapper_class


@pytest.fixture
def mock_virtual_dense_wrapper():
    """Mock VirtualDenseWrapper."""
    # Need to mock the entire class to avoid calling __init__
    mock_wrapper = MagicMock()
    mock_wrapper.calibrate = MagicMock()
    mock_wrapper.routing_threshold = 0.5

    # Mock solve method
    mock_result = MagicMock()
    mock_result.answer = "11303"
    mock_result.is_correct = True
    mock_result.plugin_name = "math"
    mock_result.used_virtual_expert = True
    mock_result.routing_score = 0.95
    mock_wrapper.solve.return_value = mock_result

    # Mock benchmark method
    mock_analysis = MagicMock()
    mock_analysis.model_name = "test-model"
    mock_analysis.total_problems = 10
    mock_analysis.correct_without_virtual = 5
    mock_analysis.correct_with_virtual = 9
    mock_analysis.accuracy_without = 0.5
    mock_analysis.accuracy_with = 0.9
    mock_analysis.improvement = 0.4
    mock_analysis.times_virtual_used = 7
    mock_analysis.avg_routing_score = 0.85
    mock_analysis.plugins_used = {"math": 7}

    # Mock individual results
    mock_problem_result = MagicMock()
    mock_problem_result.prompt = "2 + 2 = "
    mock_problem_result.answer = "4"
    mock_problem_result.correct_answer = 4
    mock_problem_result.is_correct = True
    mock_problem_result.used_virtual_expert = False
    mock_analysis.results = [mock_problem_result]
    mock_analysis.model_dump.return_value = {"total_problems": 10}

    mock_wrapper.benchmark.return_value = mock_analysis

    # Mock compare method
    mock_wrapper.compare = MagicMock()

    # Mock _generate_direct method
    mock_wrapper._generate_direct.return_value = "42"

    with patch(
        "chuk_lazarus.inference.VirtualDenseWrapper", return_value=mock_wrapper
    ) as mock_wrapper_class:
        yield mock_wrapper_class


class TestIntrospectVirtualExpert:
    """Tests for introspect_virtual_expert command dispatcher."""

    def test_analyze_action(self, mock_load_model, capsys):
        """Test analyze action is dispatched correctly."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="analyze", model="test-model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._analyze_experts"
        ) as mock_analyze:
            introspect_virtual_expert(args)
            mock_analyze.assert_called_once_with(args)

    def test_solve_action(self, mock_load_model):
        """Test solve action is dispatched correctly."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="solve", model="test-model", prompt="2+2")

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._solve_with_expert"
        ) as mock_solve:
            introspect_virtual_expert(args)
            mock_solve.assert_called_once_with(args)

    def test_benchmark_action(self, mock_load_model):
        """Test benchmark action is dispatched correctly."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="benchmark", model="test-model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._run_benchmark"
        ) as mock_benchmark:
            introspect_virtual_expert(args)
            mock_benchmark.assert_called_once_with(args)

    def test_compare_action(self, mock_load_model):
        """Test compare action is dispatched correctly."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="compare", model="test-model", prompt="2+2")

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._compare_approaches"
        ) as mock_compare:
            introspect_virtual_expert(args)
            mock_compare.assert_called_once_with(args)

    def test_interactive_action(self, mock_load_model):
        """Test interactive action is dispatched correctly."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="interactive", model="test-model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._interactive_mode"
        ) as mock_interactive:
            introspect_virtual_expert(args)
            mock_interactive.assert_called_once_with(args)

    def test_unknown_action(self, capsys):
        """Test unknown action prints error."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(action="invalid_action", model="test-model")

        introspect_virtual_expert(args)

        captured = capsys.readouterr()
        assert "Unknown action: invalid_action" in captured.out

    def test_default_action_is_solve(self, mock_load_model):
        """Test default action when not specified."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            introspect_virtual_expert,
        )

        args = Namespace(model="test-model", prompt="2+2")
        # No action attribute

        with patch(
            "chuk_lazarus.cli.commands.introspect.virtual_expert._solve_with_expert"
        ) as mock_solve:
            introspect_virtual_expert(args)
            mock_solve.assert_called_once_with(args)


class TestLoadModel:
    """Tests for _load_model helper function."""

    def test_load_model_success(self):
        """Test successful model loading."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _load_model

        mock_result = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_result.model_path = mock_path

        mock_config = {
            "model_type": "llama",
            "hidden_size": 768,
            "num_hidden_layers": 12,
        }

        mock_family_info = MagicMock()
        mock_config_obj = MagicMock()
        mock_model_obj = MagicMock()
        mock_model_obj.model.layers = [MagicMock() for _ in range(12)]
        mock_family_info.config_class.from_hf_config.return_value = mock_config_obj
        mock_family_info.model_class.return_value = mock_model_obj

        mock_tokenizer = MagicMock()

        with (
            patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader,
            patch("chuk_lazarus.models_v2.families.registry.detect_model_family") as mock_detect,
            patch("chuk_lazarus.models_v2.families.registry.get_family_info") as mock_get_family,
            patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
        ):
            mock_loader.download.return_value = mock_result
            mock_loader.apply_weights_to_model.return_value = None
            mock_loader.load_tokenizer.return_value = mock_tokenizer
            mock_detect.return_value = "llama"
            mock_get_family.return_value = mock_family_info

            model, tokenizer = _load_model("test-model")

            assert model == mock_model_obj
            assert tokenizer == mock_tokenizer
            mock_loader.download.assert_called_once_with("test-model")

    def test_load_model_unsupported_family(self, capsys):
        """Test loading unsupported model family."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _load_model

        mock_result = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_result.model_path = mock_path

        mock_config = {"model_type": "unknown"}

        with (
            patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader,
            patch("chuk_lazarus.models_v2.families.registry.detect_model_family") as mock_detect,
            patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
        ):
            mock_loader.download.return_value = mock_result
            mock_detect.return_value = None

            with pytest.raises(ValueError, match="Unsupported model"):
                _load_model("unsupported-model")


class TestIsMoeModel:
    """Tests for _is_moe_model helper function."""

    def test_is_moe_model_true(self, mock_model):
        """Test detecting MoE model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _is_moe_model

        assert _is_moe_model(mock_model) is True

    def test_is_moe_model_false(self, mock_dense_model):
        """Test detecting non-MoE model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _is_moe_model

        assert _is_moe_model(mock_dense_model) is False

    def test_is_moe_model_with_mixed_layers(self):
        """Test model with some MoE and some non-MoE layers."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _is_moe_model

        model = MagicMock()
        layers = []
        for i in range(12):
            layer = MagicMock()
            layer.mlp = MagicMock()
            if i == 5:  # Only one MoE layer
                layer.mlp.router = MagicMock()
            layers.append(layer)
        model.model.layers = layers

        assert _is_moe_model(model) is True


class TestAnalyzeExperts:
    """Tests for _analyze_experts function."""

    def test_analyze_experts_moe_model(self, mock_load_model, capsys):
        """Test analyzing experts on MoE model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _analyze_experts

        args = Namespace(model="test-model")

        mock_moe_layer_info = MagicMock()
        mock_moe_layer_info.num_experts = 32

        mock_hooks = MagicMock()
        mock_hooks.state.selected_experts = {
            0: MagicMock()  # layer 0
        }
        # Mock the selected experts array
        mock_experts_array = MagicMock()
        mock_experts_array.tolist.return_value = [1, 5, 10]
        mock_hooks.state.selected_experts[0].__getitem__.return_value = mock_experts_array

        with (
            patch("mlx.core.array") as mock_array,
            patch("chuk_lazarus.introspection.moe.get_moe_layer_info") as mock_get_info,
            patch("chuk_lazarus.introspection.moe.MoEHooks") as mock_hooks_class,
            patch("chuk_lazarus.introspection.moe.MoECaptureConfig"),
        ):
            mock_array.return_value = MagicMock()
            mock_get_info.return_value = mock_moe_layer_info
            mock_hooks_class.return_value = mock_hooks

            _analyze_experts(args)

            captured = capsys.readouterr()
            assert "EXPERT CATEGORY ANALYSIS" in captured.out
            assert "Model: test-model" in captured.out

    def test_analyze_experts_non_moe_model(self, mock_load_dense_model, capsys):
        """Test analyzing experts on non-MoE model prints error."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _analyze_experts

        args = Namespace(model="test-model")

        _analyze_experts(args)

        captured = capsys.readouterr()
        assert "Model is not MoE" in captured.out


class TestSolveWithExpert:
    """Tests for _solve_with_expert function."""

    def test_solve_with_moe_wrapper(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test solving with MoE model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _solve_with_expert(args)

        captured = capsys.readouterr()
        assert "Calibrating virtual expert" in captured.out
        assert "Answer: 11303" in captured.out
        assert "Correct: True" in captured.out
        assert "Plugin: math" in captured.out
        assert "Used virtual expert: True" in captured.out
        assert "Routing score: 0.95" in captured.out

    def test_solve_with_dense_wrapper(
        self, mock_load_dense_model, mock_virtual_dense_wrapper, capsys
    ):
        """Test solving with dense model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _solve_with_expert(args)

        captured = capsys.readouterr()
        assert "Calibrating virtual expert" in captured.out
        assert "Answer: 11303" in captured.out

    def test_solve_adds_equals_sign(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test that equals sign is added if missing."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _solve_with_expert(args)

        # Check that solve was called with "= " appended
        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.solve.assert_called_once()
        call_arg = wrapper.solve.call_args[0][0]
        assert call_arg.endswith("= ")

    def test_solve_keeps_existing_equals(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test that existing equals sign is preserved."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        args = Namespace(model="test-model", prompt="127 * 89 = ")

        _solve_with_expert(args)

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.solve.assert_called_once()
        call_arg = wrapper.solve.call_args[0][0]
        assert call_arg == "127 * 89 = "

    def test_solve_without_routing_score(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test solving when routing_score is None."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        # Modify mock to return None routing score
        wrapper = mock_virtual_moe_wrapper.return_value
        mock_result = wrapper.solve.return_value
        mock_result.routing_score = None

        args = Namespace(model="test-model", prompt="2+2")

        _solve_with_expert(args)

        captured = capsys.readouterr()
        # Should not contain routing score line
        assert "Routing score" not in captured.out

    def test_solve_without_plugin_name(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test solving when plugin_name is None."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        # Modify mock to return None plugin name
        wrapper = mock_virtual_moe_wrapper.return_value
        mock_result = wrapper.solve.return_value
        mock_result.plugin_name = None

        args = Namespace(model="test-model", prompt="2+2")

        _solve_with_expert(args)

        captured = capsys.readouterr()
        # Should not contain plugin line
        assert "Plugin:" not in captured.out


class TestRunBenchmark:
    """Tests for _run_benchmark function."""

    def test_benchmark_with_default_problems(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test benchmark with default problem set."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        args = Namespace(model="test-model", problems=None, output=None)

        _run_benchmark(args)

        captured = capsys.readouterr()
        assert "BENCHMARK RESULTS" in captured.out
        assert "Model: test-model" in captured.out
        assert "Total problems: 10" in captured.out
        assert "Improvement:" in captured.out

    def test_benchmark_with_custom_problems_string(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test benchmark with custom problems from string."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        args = Namespace(model="test-model", problems="2+2=|3+3=|4+4=", output=None)

        _run_benchmark(args)

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.benchmark.assert_called_once()
        problems = wrapper.benchmark.call_args[0][0]
        assert problems == ["2+2=", "3+3=", "4+4="]

    def test_benchmark_with_problems_from_file(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test benchmark with problems loaded from file."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2 + 2 = \n")
            f.write("5 * 5 = \n")
            f.write("10 - 3 = \n")
            f.flush()
            file_path = f.name

        try:
            args = Namespace(model="test-model", problems=f"@{file_path}", output=None)

            _run_benchmark(args)

            wrapper = mock_virtual_moe_wrapper.return_value
            wrapper.benchmark.assert_called_once()
            problems = wrapper.benchmark.call_args[0][0]
            # File reading strips trailing whitespace, so check without trailing space
            assert "2 + 2 =" in problems or "2 + 2 = " in problems
            assert "5 * 5 =" in problems or "5 * 5 = " in problems
            assert "10 - 3 =" in problems or "10 - 3 = " in problems
        finally:
            Path(file_path).unlink()

    def test_benchmark_with_output_file(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test benchmark saves results to file."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            args = Namespace(model="test-model", problems=None, output=output_path)

            _run_benchmark(args)

            # Check file was created
            assert Path(output_path).exists()
            with open(output_path) as f:
                data = json.load(f)
                assert data == {"total_problems": 10}
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()

    def test_benchmark_displays_per_problem_breakdown(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test benchmark displays per-problem breakdown."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        args = Namespace(model="test-model", problems=None, output=None)

        _run_benchmark(args)

        captured = capsys.readouterr()
        assert "PER-PROBLEM BREAKDOWN" in captured.out
        assert "2 + 2 = " in captured.out
        assert "OK" in captured.out or "X" in captured.out

    def test_benchmark_with_dense_model(
        self, mock_load_dense_model, mock_virtual_dense_wrapper, capsys
    ):
        """Test benchmark with dense model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        args = Namespace(model="test-model", problems=None, output=None)

        _run_benchmark(args)

        captured = capsys.readouterr()
        assert "BENCHMARK RESULTS" in captured.out

    def test_benchmark_displays_plugins_used(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test benchmark displays plugins used."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        args = Namespace(model="test-model", problems=None, output=None)

        _run_benchmark(args)

        captured = capsys.readouterr()
        assert "Plugins used:" in captured.out


class TestCompareApproaches:
    """Tests for _compare_approaches function."""

    def test_compare_with_moe_model(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test comparing approaches with MoE model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _compare_approaches,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _compare_approaches(args)

        captured = capsys.readouterr()
        assert "Calibrating virtual expert" in captured.out

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.compare.assert_called_once()

    def test_compare_with_dense_model(self, mock_load_dense_model, mock_virtual_dense_wrapper):
        """Test comparing approaches with dense model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _compare_approaches,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _compare_approaches(args)

        wrapper = mock_virtual_dense_wrapper.return_value
        wrapper.compare.assert_called_once()

    def test_compare_adds_equals_sign(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test that equals sign is added if missing."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _compare_approaches,
        )

        args = Namespace(model="test-model", prompt="127 * 89")

        _compare_approaches(args)

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.compare.assert_called_once()
        call_arg = wrapper.compare.call_args[0][0]
        assert call_arg.endswith("= ")


class TestInteractiveMode:
    """Tests for _interactive_mode function."""

    def test_interactive_quit_command(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test quitting interactive mode with !quit."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "VIRTUAL EXPERT - INTERACTIVE MODE" in captured.out
        assert "Goodbye!" in captured.out

    def test_interactive_eof(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test exiting interactive mode with EOF."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=EOFError):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_keyboard_interrupt(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test exiting interactive mode with Ctrl-C."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_solve_expression(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test solving expression in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Answer: 11303" in captured.out
        assert "Correct: True" in captured.out

    def test_interactive_model_command(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test !model command in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!model 2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Model only: 42" in captured.out

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper._generate_direct.assert_called_once()

    def test_interactive_compare_command(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test !compare command in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!compare 2 + 2", "!quit"]):
            _interactive_mode(args)

        wrapper = mock_virtual_moe_wrapper.return_value
        wrapper.compare.assert_called_once()

    def test_interactive_threshold_command(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test !threshold command in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!threshold 0.7", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Set threshold to 0.7" in captured.out

        wrapper = mock_virtual_moe_wrapper.return_value
        assert wrapper.routing_threshold == 0.7

    def test_interactive_threshold_invalid(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test !threshold command with invalid value."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!threshold abc", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Invalid threshold" in captured.out

    def test_interactive_unknown_command(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test unknown command in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!unknown", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_interactive_empty_input(self, mock_load_model, mock_virtual_moe_wrapper, capsys):
        """Test empty input in interactive mode."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["", "!quit"]):
            _interactive_mode(args)

        # Should just continue without error
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_interactive_with_dense_model(
        self, mock_load_dense_model, mock_virtual_dense_wrapper, capsys
    ):
        """Test interactive mode with dense model."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "VIRTUAL EXPERT - INTERACTIVE MODE (Dense)" in captured.out

    def test_interactive_with_moe_model_shows_type(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test interactive mode shows MoE type."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "VIRTUAL EXPERT - INTERACTIVE MODE (MoE)" in captured.out

    def test_interactive_model_command_without_expression(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test !model command without expression."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!model", "!quit"]):
            _interactive_mode(args)

        # Should show unknown command message since no expression provided
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_interactive_compare_command_without_expression(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test !compare command without expression."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!compare", "!quit"]):
            _interactive_mode(args)

        # Should show unknown command message since no expression provided
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_interactive_threshold_command_without_value(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test !threshold command without value."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["!threshold", "!quit"]):
            _interactive_mode(args)

        # Should show unknown command message since no value provided
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_interactive_displays_routing_score(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test interactive mode displays routing score."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Routing score: 0.95" in captured.out

    def test_interactive_displays_plugin_when_used(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test interactive mode displays plugin when virtual expert is used."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        with patch("builtins.input", side_effect=["2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        assert "Plugin: math" in captured.out

    def test_interactive_no_plugin_when_not_used(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test interactive mode doesn't display plugin when not used."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        # Modify mock to not use virtual expert
        wrapper = mock_virtual_moe_wrapper.return_value
        mock_result = wrapper.solve.return_value
        mock_result.used_virtual_expert = False
        mock_result.plugin_name = None

        with patch("builtins.input", side_effect=["2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        # Plugin line should not appear
        lines = captured.out.split("\n")
        plugin_lines = [line for line in lines if line.strip().startswith("Plugin:")]
        assert len(plugin_lines) == 0

    def test_interactive_no_routing_score_when_none(
        self, mock_load_model, mock_virtual_moe_wrapper, capsys
    ):
        """Test interactive mode doesn't display routing score when None."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _interactive_mode,
        )

        args = Namespace(model="test-model")

        # Modify mock to return None routing score
        wrapper = mock_virtual_moe_wrapper.return_value
        mock_result = wrapper.solve.return_value
        mock_result.routing_score = None

        with patch("builtins.input", side_effect=["2 + 2", "!quit"]):
            _interactive_mode(args)

        captured = capsys.readouterr()
        # Routing score line should not appear when None
        lines = captured.out.split("\n")
        routing_lines = [line for line in lines if "Routing score:" in line]
        assert len(routing_lines) == 0


class TestAnalyzeExpertsEdgeCases:
    """Additional tests for _analyze_experts edge cases."""

    def test_analyze_experts_with_expert_activations(self, mock_load_model, capsys):
        """Test analyzing experts with actual expert activations."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _analyze_experts

        args = Namespace(model="test-model")

        mock_moe_layer_info = MagicMock()
        mock_moe_layer_info.num_experts = 32

        # Create a custom dict-like class to track expert calls
        call_count = [0]

        class ExpertDict(dict):
            def __getitem__(self, key):
                # Return different expert indices based on call count
                # to simulate different categories activating different experts
                idx = call_count[0]
                call_count[0] += 1

                mock_experts_array = MagicMock()
                # First half use expert 1, second half use expert 5
                # This tests the logic for counting and sorting experts
                if idx < 8:
                    mock_experts_array.tolist.return_value = [1, 2, 3]
                else:
                    mock_experts_array.tolist.return_value = [5, 6, 7]

                return MagicMock(__getitem__=lambda s, k: mock_experts_array)

        mock_hooks = MagicMock()
        mock_hooks.state.selected_experts = ExpertDict({0: None})

        with (
            patch("mlx.core.array") as mock_array,
            patch("chuk_lazarus.introspection.moe.get_moe_layer_info") as mock_get_info,
            patch("chuk_lazarus.introspection.moe.MoEHooks") as mock_hooks_class,
            patch("chuk_lazarus.introspection.moe.MoECaptureConfig"),
        ):
            mock_array.return_value = MagicMock()
            mock_get_info.return_value = mock_moe_layer_info
            mock_hooks_class.return_value = mock_hooks

            _analyze_experts(args)

            captured = capsys.readouterr()
            # Should show the analysis table
            assert "Expert" in captured.out
            assert "MATH" in captured.out


class TestSolveEdgeCases:
    """Additional edge case tests for solve functionality."""

    def test_solve_with_equals_only(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test solving when prompt ends with just '='."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import (
            _solve_with_expert,
        )

        args = Namespace(model="test-model", prompt="127 * 89=")

        _solve_with_expert(args)

        wrapper = mock_virtual_moe_wrapper.return_value
        # Should not add another equals sign
        call_arg = wrapper.solve.call_args[0][0]
        assert call_arg.count("=") == 1


class TestBenchmarkEdgeCases:
    """Additional edge case tests for benchmark functionality."""

    def test_benchmark_empty_file(self, mock_load_model, mock_virtual_moe_wrapper):
        """Test benchmark with empty file."""
        from chuk_lazarus.cli.commands.introspect.virtual_expert import _run_benchmark

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write only blank lines
            f.write("\n\n\n")
            f.flush()
            file_path = f.name

        try:
            args = Namespace(model="test-model", problems=f"@{file_path}", output=None)

            _run_benchmark(args)

            wrapper = mock_virtual_moe_wrapper.return_value
            # Should be called with empty list or handle gracefully
            wrapper.benchmark.assert_called_once()
        finally:
            Path(file_path).unlink()


class TestModuleExports:
    """Test module exports."""

    def test_module_has_all(self):
        """Test that __all__ is defined."""
        import chuk_lazarus.cli.commands.introspect.virtual_expert as module

        assert hasattr(module, "__all__")
        assert "introspect_virtual_expert" in module.__all__
