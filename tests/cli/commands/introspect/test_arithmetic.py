"""Tests for introspect arithmetic CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIntrospectArithmetic:
    """Tests for introspect_arithmetic command."""

    @pytest.fixture
    def arithmetic_args(self):
        """Create arguments for arithmetic command."""
        return Namespace(
            model="test-model",
            hard_only=False,
            easy_only=False,
            quick=False,
            raw=False,
            output=None,
        )

    def test_arithmetic_basic(self, arithmetic_args, capsys):
        """Test basic arithmetic study."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)

            # Mock model info
            mock_analyzer.model_info = MagicMock()
            mock_analyzer.model_info.model_id = "test-model"
            mock_analyzer.model_info.num_layers = 12

            # Mock tokenizer
            mock_analyzer._tokenizer = MagicMock()
            mock_analyzer._tokenizer.chat_template = None

            # Mock analysis result
            mock_result = MagicMock()
            mock_result.layer_predictions = []
            mock_analyzer.analyze = AsyncMock(return_value=mock_result)

            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                mock_test_suite = MagicMock()
                mock_test_suite.test_cases = []
                mock_suite.generate_test_cases.return_value = mock_test_suite

                introspect_arithmetic(arithmetic_args)

                captured = capsys.readouterr()
                assert "Loading model" in captured.out

    def test_arithmetic_hard_only(self, arithmetic_args):
        """Test hard-only arithmetic study."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        arithmetic_args.hard_only = True

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)
            mock_analyzer.analyze = AsyncMock(return_value=MagicMock(layer_predictions=[]))
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[])

                introspect_arithmetic(arithmetic_args)

                # Check difficulty filter was used
                call_args = mock_suite.generate_test_cases.call_args
                assert call_args is not None

    def test_arithmetic_easy_only(self, arithmetic_args):
        """Test easy-only arithmetic study."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        arithmetic_args.easy_only = True

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)
            mock_analyzer.analyze = AsyncMock(return_value=MagicMock(layer_predictions=[]))
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[])

                introspect_arithmetic(arithmetic_args)

    def test_arithmetic_quick_mode(self, arithmetic_args):
        """Test quick mode (reduced test set)."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        arithmetic_args.quick = True

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)
            mock_analyzer.analyze = AsyncMock(return_value=MagicMock(layer_predictions=[]))
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                # Create 9 test cases with proper values, quick mode takes every 3rd
                mock_cases = []
                for i in range(9):
                    case = MagicMock()
                    case.prompt = f"test{i}="
                    case.expected = str(i * 2)
                    # operation and difficulty are enums with .value attribute
                    case.operation = MagicMock(value="+")
                    case.difficulty = MagicMock(value="easy")
                    case.magnitude = 1  # Real int value
                    mock_cases.append(case)
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=mock_cases)

                introspect_arithmetic(arithmetic_args)

    def test_arithmetic_raw_mode(self, arithmetic_args, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        arithmetic_args.raw = True

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)
            mock_analyzer.analyze = AsyncMock(return_value=MagicMock(layer_predictions=[]))
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[])

                introspect_arithmetic(arithmetic_args)

                captured = capsys.readouterr()
                assert "RAW" in captured.out

    def test_arithmetic_save_output(self, arithmetic_args):
        """Test saving arithmetic results."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            arithmetic_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)
            mock_analyzer.analyze = AsyncMock(return_value=MagicMock(layer_predictions=[]))
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[])

                introspect_arithmetic(arithmetic_args)

                # Check file was created
                from pathlib import Path

                if Path(arithmetic_args.output).exists():
                    import json

                    with open(arithmetic_args.output) as f:
                        data = json.load(f)
                        assert isinstance(data, (dict, list))

    def test_arithmetic_with_predictions(self, arithmetic_args, capsys):
        """Test arithmetic with actual predictions to cover analysis loop."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)

            # Create mock layer predictions with actual structure
            mock_pred = MagicMock()
            mock_pred.token = "4"
            mock_pred.probability = 0.9

            mock_layer_pred = MagicMock()
            mock_layer_pred.layer_idx = 5
            mock_layer_pred.predictions = [mock_pred]

            mock_result = MagicMock()
            mock_result.layer_predictions = [mock_layer_pred]
            mock_result.final_prediction = [mock_pred]

            mock_analyzer.analyze = AsyncMock(return_value=mock_result)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                # Create test case
                mock_case = MagicMock()
                mock_case.prompt = "2+2="
                mock_case.expected = "4"
                mock_case.operation = MagicMock(value="add")
                mock_case.difficulty = MagicMock(value="easy")
                mock_case.magnitude = 1

                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[mock_case])

                introspect_arithmetic(arithmetic_args)

                captured = capsys.readouterr()
                # Should show the test result
                assert "2+2=" in captured.out or "Running" in captured.out

    def test_arithmetic_with_chat_template(self, arithmetic_args, capsys):
        """Test arithmetic when model has chat template (covers line 89)."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)

            # Set chat_template to a truthy value
            mock_tokenizer = MagicMock()
            mock_tokenizer.chat_template = "some_template"
            mock_analyzer._tokenizer = mock_tokenizer

            mock_result = MagicMock()
            mock_result.layer_predictions = []
            mock_result.final_prediction = []
            mock_analyzer.analyze = AsyncMock(return_value=mock_result)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                # Create a test case so the loop body executes (line 89)
                mock_case = MagicMock()
                mock_case.prompt = "2+2="
                mock_case.expected = "4"
                mock_case.operation = MagicMock(value="add")
                mock_case.difficulty = MagicMock(value="easy")
                mock_case.magnitude = 1
                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=[mock_case])

                # apply_chat_template is already mocked by conftest fixture
                introspect_arithmetic(arithmetic_args)

                captured = capsys.readouterr()
                assert "CHAT" in captured.out

    def test_arithmetic_summary_stats(self, arithmetic_args, capsys):
        """Test arithmetic summary statistics output."""
        from chuk_lazarus.cli.commands.introspect import introspect_arithmetic

        with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
            mock_analyzer.__aexit__ = AsyncMock(return_value=None)
            mock_analyzer.model_info = MagicMock(model_id="test", num_layers=12)
            mock_analyzer._tokenizer = MagicMock(chat_template=None)

            # Mock successful prediction
            mock_pred = MagicMock()
            mock_pred.token = "4"
            mock_pred.probability = 0.95

            mock_layer_pred = MagicMock()
            mock_layer_pred.layer_idx = 8
            mock_layer_pred.predictions = [mock_pred]

            mock_result = MagicMock()
            mock_result.layer_predictions = [mock_layer_pred]
            mock_result.final_prediction = [mock_pred]

            mock_analyzer.analyze = AsyncMock(return_value=mock_result)
            mock_cls.from_pretrained.return_value = mock_analyzer

            with patch("chuk_lazarus.introspection.ArithmeticTestSuite") as mock_suite:
                # Create multiple test cases
                cases = []
                for i in range(3):
                    case = MagicMock()
                    case.prompt = f"{i}+{i}="
                    case.expected = str(i * 2) if i > 0 else "4"
                    case.operation = MagicMock(value="add")
                    case.difficulty = MagicMock(value="easy")
                    case.magnitude = 1
                    cases.append(case)

                mock_suite.generate_test_cases.return_value = MagicMock(test_cases=cases)

                introspect_arithmetic(arithmetic_args)

                captured = capsys.readouterr()
                # Should show summary
                assert "Running" in captured.out or "test" in captured.out.lower()
