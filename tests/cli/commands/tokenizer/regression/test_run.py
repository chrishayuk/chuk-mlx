"""Tests for regression_run command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.tokenizer._types import RegressionRunConfig
from chuk_lazarus.cli.commands.tokenizer.regression.run import regression_run


class TestRegressionRunConfig:
    """Tests for RegressionRunConfig."""

    def test_from_args(self):
        """Test config creation."""
        args = MagicMock()
        args.tokenizer = "gpt2"
        args.tests = "/path/to/tests.yaml"

        config = RegressionRunConfig.from_args(args)

        assert config.tokenizer == "gpt2"
        assert config.tests == Path("/path/to/tests.yaml")


class TestRegressionRun:
    """Tests for regression_run function."""

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.regression.load_tests_from_yaml")
    @patch("chuk_lazarus.data.tokenizers.regression.run_token_tests")
    def test_regression_run_all_pass(
        self, mock_run_tests, mock_load_tests, mock_load_tokenizer, capsys
    ):
        """Test regression run with all tests passing."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_suite = MagicMock()
        mock_suite.name = "Test Suite"
        mock_suite.tests = [MagicMock()]
        mock_load_tests.return_value = mock_suite

        mock_result = MagicMock()
        mock_result.total_tests = 5
        mock_result.passed = 5
        mock_result.failed = 0
        mock_result.results = []
        mock_run_tests.return_value = mock_result

        config = RegressionRunConfig(tokenizer="gpt2", tests=Path("/path/to/tests.yaml"))
        result = regression_run(config)

        captured = capsys.readouterr()
        assert "Regression Test Results" in captured.out
        assert "Suite: Test Suite" in captured.out
        assert "Tests: 5" in captured.out
        assert "Passed: 5" in captured.out
        assert "All tests passed!" in captured.out
        assert result.passed == 5
        assert result.failed == 0

    @patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer")
    @patch("chuk_lazarus.data.tokenizers.regression.load_tests_from_yaml")
    @patch("chuk_lazarus.data.tokenizers.regression.run_token_tests")
    def test_regression_run_with_failures(
        self, mock_run_tests, mock_load_tests, mock_load_tokenizer, capsys
    ):
        """Test regression run with failing tests."""
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        mock_suite = MagicMock()
        mock_suite.name = "Test Suite"
        mock_suite.tests = [MagicMock()]
        mock_load_tests.return_value = mock_suite

        mock_failed_test = MagicMock()
        mock_failed_test.passed = False
        mock_failed_test.test_name = "test_encode_hello"
        mock_failed_test.message = "Expected 5 tokens, got 6"

        mock_result = MagicMock()
        mock_result.total_tests = 5
        mock_result.passed = 4
        mock_result.failed = 1
        mock_result.results = [mock_failed_test]
        mock_run_tests.return_value = mock_result

        config = RegressionRunConfig(tokenizer="gpt2", tests=Path("/path/to/tests.yaml"))

        try:
            regression_run(config)
        except SystemExit as e:
            # Expected to exit with code 1 on failures
            assert e.code == 1

        captured = capsys.readouterr()
        assert "Failed tests:" in captured.out
        assert "test_encode_hello" in captured.out
