"""Tests for introspect generation CLI commands."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestIntrospectGenerate:
    """Tests for introspect_generate command."""

    @pytest.fixture
    def generate_args(self):
        """Create arguments for generate command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            max_tokens=10,
            temperature=0.0,
            top_k=5,
            layer_step=4,
            track=None,
            chat_template=None,
            raw=False,
            expected=None,
            find_answer=None,
            no_find_answer=False,
            output=None,
        )

    @pytest.fixture
    def mock_generation_service(self):
        """Create mock generation service."""
        with patch("chuk_lazarus.introspection.generation.GenerationService") as mock_service:
            mock_result = MagicMock()
            mock_result.to_display.return_value = (
                "GENERATION ANALYSIS\nModel: test-model\nPrompt: 2+2=\nGenerated: 4"
            )
            mock_result.save = MagicMock()

            mock_service.generate = AsyncMock(return_value=mock_result)

            yield mock_service, mock_result

    def test_generate_basic(self, generate_args, mock_generation_service, capsys):
        """Test basic generation."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert "GENERATION" in captured.out

    def test_generate_with_max_tokens(self, generate_args, mock_generation_service, capsys):
        """Test generation with custom max tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.max_tokens = 50

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_generate_with_temperature(self, generate_args, mock_generation_service, capsys):
        """Test generation with temperature."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.temperature = 0.7

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_generate_raw_mode(self, generate_args, mock_generation_service, capsys):
        """Test raw mode (no chat template)."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.raw = True

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_generate_with_expected_answer(self, generate_args, mock_generation_service, capsys):
        """Test generation with expected answer."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.expected = "4"

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_generate_with_output(self, generate_args, mock_generation_service, tmp_path, capsys):
        """Test generation with output file."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        output_file = tmp_path / "generation_results.json"
        generate_args.output = str(output_file)

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert "saved to" in captured.out

    def test_generate_with_track_tokens(self, generate_args, mock_generation_service, capsys):
        """Test generation with tracked tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_generate

        generate_args.track = "4,5,6"

        asyncio.run(introspect_generate(generate_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestIntrospectLogitEvolution:
    """Tests for introspect_logit_evolution command."""

    @pytest.fixture
    def evolution_args(self):
        """Create arguments for logit evolution command."""
        return Namespace(
            model="test-model",
            prompt="2+2=",
            track="4,5",
            layer_step=4,
            top_k=5,
        )

    @pytest.fixture
    def mock_evolution_service(self):
        """Create mock logit evolution service."""
        with patch("chuk_lazarus.introspection.generation.LogitEvolutionService") as mock_service:
            mock_result = MagicMock()
            mock_result.to_display.return_value = (
                "LOGIT EVOLUTION\nModel: test-model\nPrompt: 2+2=\nTracked tokens: 4, 5"
            )

            mock_service.analyze = AsyncMock(return_value=mock_result)

            yield mock_service, mock_result

    def test_evolution_basic(self, evolution_args, mock_evolution_service, capsys):
        """Test basic logit evolution."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_logit_evolution

        asyncio.run(introspect_logit_evolution(evolution_args))

        captured = capsys.readouterr()
        assert "LOGIT" in captured.out or "test-model" in captured.out

    def test_evolution_custom_layer_step(self, evolution_args, mock_evolution_service, capsys):
        """Test logit evolution with custom layer step."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_logit_evolution

        evolution_args.layer_step = 2

        asyncio.run(introspect_logit_evolution(evolution_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_evolution_no_track(self, evolution_args, mock_evolution_service, capsys):
        """Test logit evolution without track tokens."""
        from chuk_lazarus.cli.commands.introspect.generation import introspect_logit_evolution

        evolution_args.track = None

        asyncio.run(introspect_logit_evolution(evolution_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestGenerationConfig:
    """Tests for GenerationConfig type."""

    def test_generation_config_from_args(self):
        """Test creating generation config from args."""
        # This tests the config model if it exists in _types
        from chuk_lazarus.cli.commands._constants import AnalysisDefaults

        assert AnalysisDefaults.GEN_TOKENS > 0
        assert AnalysisDefaults.TOP_K > 0
