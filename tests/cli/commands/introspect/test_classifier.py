"""Tests for introspect classifier CLI commands."""

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIntrospectClassifier:
    """Tests for introspect_classifier command."""

    @pytest.fixture
    def classifier_args(self):
        """Create basic classifier arguments."""
        return Namespace(
            model="test-model",
            classes=["add:1+1|2+2", "mult:2*2|3*3"],
            category=None,
            categories_file=None,
            layers=None,
            all_layers=False,
            output=None,
        )

    @pytest.mark.asyncio
    async def test_classifier_with_classes_arg(self, classifier_args, capsys):
        """Test classifier with --classes argument format."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        mock_result = MagicMock()
        mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"

        with patch(
            "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await introspect_classifier(classifier_args)

        captured = capsys.readouterr()
        assert "CLASSIFIER RESULTS" in captured.out

    @pytest.mark.asyncio
    async def test_classifier_with_category_arg(self, capsys):
        """Test classifier with --category argument format."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        args = Namespace(
            model="test-model",
            classes=None,
            category=["add|1+1|2+2", "mult|2*2|3*3"],
            categories_file=None,
            layers=None,
            all_layers=False,
            output=None,
        )

        mock_result = MagicMock()
        mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"

        with patch(
            "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await introspect_classifier(args)

        captured = capsys.readouterr()
        assert "CLASSIFIER RESULTS" in captured.out

    @pytest.mark.asyncio
    async def test_classifier_with_categories_file(self, capsys):
        """Test classifier with --categories-file argument."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        # Create temp categories file
        categories = {
            "add": ["1+1=", "2+2="],
            "mult": ["2*2=", "3*3="],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(categories, f)
            categories_file = f.name

        try:
            args = Namespace(
                model="test-model",
                classes=None,
                category=None,
                categories_file=categories_file,
                layers=None,
                all_layers=False,
                output=None,
            )

            mock_result = MagicMock()
            mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"

            with patch(
                "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
                new_callable=AsyncMock,
                return_value=mock_result,
            ):
                await introspect_classifier(args)

            captured = capsys.readouterr()
            assert "CLASSIFIER RESULTS" in captured.out
        finally:
            Path(categories_file).unlink()

    @pytest.mark.asyncio
    async def test_classifier_with_layers(self, classifier_args, capsys):
        """Test classifier with --layers argument."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        classifier_args.layers = "4,8,12"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"

        with patch(
            "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_train:
            await introspect_classifier(classifier_args)

            # Verify layers were parsed
            call_args = mock_train.call_args[0]
            config = call_args[0]
            assert config.layers == [4, 8, 12]

    @pytest.mark.asyncio
    async def test_classifier_with_all_layers(self, classifier_args, capsys):
        """Test classifier with --all-layers flag."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        classifier_args.all_layers = True

        mock_result = MagicMock()
        mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"

        with patch(
            "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_train:
            await introspect_classifier(classifier_args)

            call_args = mock_train.call_args[0]
            config = call_args[0]
            assert config.all_layers is True

    @pytest.mark.asyncio
    async def test_classifier_with_output(self, classifier_args, tmp_path, capsys):
        """Test classifier with --output argument."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        output_file = str(tmp_path / "results.json")
        classifier_args.output = output_file

        mock_result = MagicMock()
        mock_result.to_display.return_value = "CLASSIFIER RESULTS\nAccuracy: 0.95"
        mock_result.save = MagicMock()

        with patch(
            "chuk_lazarus.introspection.classifier.ClassifierService.train_and_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await introspect_classifier(classifier_args)

        captured = capsys.readouterr()
        assert "Results saved to:" in captured.out
        mock_result.save.assert_called_once_with(output_file)

    @pytest.mark.asyncio
    async def test_classifier_invalid_class_format_error(self, capsys):
        """Test classifier raises error for invalid --classes format."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        args = Namespace(
            model="test-model",
            classes=["invalid_no_colon"],  # Missing colon
            category=None,
            categories_file=None,
            layers=None,
            all_layers=False,
            output=None,
        )

        with pytest.raises(ValueError, match="Invalid class format"):
            await introspect_classifier(args)

    @pytest.mark.asyncio
    async def test_classifier_invalid_category_format_error(self, capsys):
        """Test classifier raises error for invalid --category format."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        args = Namespace(
            model="test-model",
            classes=None,
            category=["only_label"],  # Missing prompts
            categories_file=None,
            layers=None,
            all_layers=False,
            output=None,
        )

        with pytest.raises(ValueError, match="Invalid category format"):
            await introspect_classifier(args)

    @pytest.mark.asyncio
    async def test_classifier_too_few_categories_error(self, capsys):
        """Test classifier raises error for fewer than 2 categories."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_classifier

        args = Namespace(
            model="test-model",
            classes=["only_one:prompt1|prompt2"],  # Only 1 category
            category=None,
            categories_file=None,
            layers=None,
            all_layers=False,
            output=None,
        )

        with pytest.raises(ValueError, match="at least 2 categories"):
            await introspect_classifier(args)


class TestIntrospectLogitLens:
    """Tests for introspect_logit_lens command."""

    @pytest.fixture
    def logit_lens_args(self):
        """Create basic logit lens arguments."""
        return Namespace(
            model="test-model",
            prompts="2+2=",
            prompt=None,
            layers=None,
            layer_step=4,
            top_k=5,
            track=None,
        )

    @pytest.mark.asyncio
    async def test_logit_lens_basic(self, logit_lens_args, capsys):
        """Test basic logit lens analysis."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_logit_lens

        mock_result = MagicMock()
        mock_result.to_display.return_value = "LOGIT LENS RESULTS\nLayer 0: token=4"

        with patch(
            "chuk_lazarus.introspection.logit_lens.LogitLensService.analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            await introspect_logit_lens(logit_lens_args)

        captured = capsys.readouterr()
        assert "LOGIT LENS RESULTS" in captured.out

    @pytest.mark.asyncio
    async def test_logit_lens_with_layers(self, logit_lens_args, capsys):
        """Test logit lens with --layers argument."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_logit_lens

        logit_lens_args.layers = "0,4,8,12"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "LOGIT LENS RESULTS\nLayer 0: token=4"

        with patch(
            "chuk_lazarus.introspection.logit_lens.LogitLensService.analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_analyze:
            await introspect_logit_lens(logit_lens_args)

            call_args = mock_analyze.call_args[0]
            config = call_args[0]
            assert config.layers == [0, 4, 8, 12]

    @pytest.mark.asyncio
    async def test_logit_lens_with_track(self, logit_lens_args, capsys):
        """Test logit lens with --track argument."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_logit_lens

        logit_lens_args.track = "4,8"

        mock_result = MagicMock()
        mock_result.to_display.return_value = "LOGIT LENS RESULTS\nTracking: 4, 8"

        with patch(
            "chuk_lazarus.introspection.logit_lens.LogitLensService.analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_analyze:
            await introspect_logit_lens(logit_lens_args)

            call_args = mock_analyze.call_args[0]
            config = call_args[0]
            assert config.track_tokens == ["4", "8"]

    @pytest.mark.asyncio
    async def test_logit_lens_with_prompt_arg(self, capsys):
        """Test logit lens with --prompt argument (alternative to --prompts)."""
        from chuk_lazarus.cli.commands.introspect.classifier import introspect_logit_lens

        args = Namespace(
            model="test-model",
            prompts=None,  # Not using --prompts
            prompt="3*3=",  # Using --prompt
            layers=None,
            layer_step=4,
            top_k=5,
            track=None,
        )

        mock_result = MagicMock()
        mock_result.to_display.return_value = "LOGIT LENS RESULTS"

        with patch(
            "chuk_lazarus.introspection.logit_lens.LogitLensService.analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_analyze:
            await introspect_logit_lens(args)

            call_args = mock_analyze.call_args[0]
            config = call_args[0]
            assert config.prompt == "3*3="
