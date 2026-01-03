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
