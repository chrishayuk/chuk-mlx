"""Tests for introspect steering CLI commands."""

import tempfile
from argparse import Namespace

import pytest

from chuk_lazarus.cli.commands.introspect._types import SteeringConfig


class TestSteeringConfig:
    """Tests for SteeringConfig."""

    @pytest.fixture
    def basic_args(self):
        """Create basic steering args."""
        return Namespace(
            model="test-model",
            extract=False,
            direction=None,
            neuron=None,
            positive=None,
            negative=None,
            prompts="test prompt",
            layer=None,
            coefficient=1.0,
            compare=None,
            name=None,
            positive_label=None,
            negative_label=None,
            max_tokens=10,
            temperature=0.0,
            output=None,
        )

    def test_from_args(self, basic_args):
        """Test creating config from args."""
        config = SteeringConfig.from_args(basic_args)

        assert config.model == "test-model"
        assert config.extract is False
        assert config.coefficient == 1.0
        assert config.prompts == "test prompt"

    def test_from_args_with_extract(self, basic_args):
        """Test creating config with extract mode."""
        basic_args.extract = True
        basic_args.positive = "good"
        basic_args.negative = "bad"
        config = SteeringConfig.from_args(basic_args)

        assert config.extract is True
        assert config.positive == "good"
        assert config.negative == "bad"

    def test_from_args_with_direction(self, basic_args):
        """Test creating config with direction file."""
        basic_args.direction = "/path/to/direction.npz"
        config = SteeringConfig.from_args(basic_args)

        assert config.direction == "/path/to/direction.npz"

    def test_from_args_with_neuron(self, basic_args):
        """Test creating config with neuron steering."""
        basic_args.neuron = 42
        basic_args.layer = 6
        config = SteeringConfig.from_args(basic_args)

        assert config.neuron == 42
        assert config.layer == 6

    def test_from_args_with_compare(self, basic_args):
        """Test creating config with compare mode."""
        basic_args.compare = "-2,-1,0,1,2"
        config = SteeringConfig.from_args(basic_args)

        assert config.compare == "-2,-1,0,1,2"

    def test_from_args_default_labels(self, basic_args):
        """Test that default labels are applied for None values."""
        # Args have None values for name, positive_label, negative_label
        config = SteeringConfig.from_args(basic_args)

        # Should use defaults from SteeringDefaults
        from chuk_lazarus.cli.commands._constants import SteeringDefaults

        assert config.name == SteeringDefaults.DEFAULT_NAME
        assert config.positive_label == SteeringDefaults.DEFAULT_POSITIVE_LABEL
        assert config.negative_label == SteeringDefaults.DEFAULT_NEGATIVE_LABEL

    def test_from_args_custom_labels(self, basic_args):
        """Test creating config with custom labels."""
        basic_args.name = "emotion"
        basic_args.positive_label = "happy"
        basic_args.negative_label = "sad"
        config = SteeringConfig.from_args(basic_args)

        assert config.name == "emotion"
        assert config.positive_label == "happy"
        assert config.negative_label == "sad"

    def test_from_args_with_output(self, basic_args):
        """Test creating config with output path."""
        basic_args.output = "/path/to/output.npz"
        config = SteeringConfig.from_args(basic_args)

        assert config.output == "/path/to/output.npz"

    def test_config_is_frozen(self, basic_args):
        """Test that config is immutable."""
        from pydantic import ValidationError

        config = SteeringConfig.from_args(basic_args)

        with pytest.raises(ValidationError):
            config.model = "other-model"


class TestSteeringExtractionResult:
    """Tests for SteeringExtractionResult."""

    def test_basic_creation(self):
        """Test creating extraction result."""
        from chuk_lazarus.cli.commands.introspect._types import SteeringExtractionResult

        result = SteeringExtractionResult(
            layer=6,
            norm=1.5,
            cosine_similarity=0.8,
            separation=2.0,
            output_path="/path/to/output.npz",
        )

        assert result.layer == 6
        assert result.norm == 1.5
        assert result.cosine_similarity == 0.8
        assert result.separation == 2.0
        assert result.output_path == "/path/to/output.npz"

    def test_to_display(self):
        """Test display output."""
        from chuk_lazarus.cli.commands.introspect._types import SteeringExtractionResult

        result = SteeringExtractionResult(
            layer=6,
            norm=1.5,
            cosine_similarity=0.8,
            separation=2.0,
        )

        display = result.to_display()
        assert "Layer: 6" in display
        assert "Norm: 1.50" in display


class TestSteeringGenerationResult:
    """Tests for SteeringGenerationResult."""

    def test_basic_creation(self):
        """Test creating generation result."""
        from chuk_lazarus.cli.commands.introspect._types import SteeringGenerationResult

        result = SteeringGenerationResult(
            prompt="test prompt",
            output="generated output",
            layer=6,
            coefficient=1.5,
        )

        assert result.prompt == "test prompt"
        assert result.output == "generated output"
        assert result.layer == 6
        assert result.coefficient == 1.5

    def test_to_display(self):
        """Test display output."""
        from chuk_lazarus.cli.commands.introspect._types import SteeringGenerationResult

        result = SteeringGenerationResult(
            prompt="test",
            output="output",
            layer=6,
            coefficient=1.0,
        )

        display = result.to_display()
        assert "Prompt:" in display
        assert "Output:" in display


class TestIntrospectSteer:
    """Tests for introspect_steer command functionality."""

    @pytest.fixture
    def steer_args(self):
        """Create arguments for steer command."""
        return Namespace(
            model="test-model",
            extract=False,
            direction=None,
            neuron=None,
            positive=None,
            negative=None,
            prompts="test prompt",
            layer=None,
            coefficient=1.0,
            compare=None,
            name=None,
            positive_label=None,
            negative_label=None,
            max_tokens=10,
            temperature=0.0,
            output=None,
        )

    def test_steer_extract_direction(self, steer_args, capsys):
        """Test extracting direction from contrastive prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = "good prompt"
        steer_args.negative = "bad prompt"

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out
        assert "Extracting direction" in captured.out

    def test_steer_extract_and_save(self, steer_args):
        """Test extracting and saving direction."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = "good"
        steer_args.negative = "bad"

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            steer_args.output = f.name

        introspect_steer(steer_args)

        # The mock should have been called to save
        # Note: actual file creation depends on mock behavior

    def test_steer_extract_missing_prompts(self, steer_args):
        """Test that extract mode requires positive/negative prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = None
        steer_args.negative = None

        with pytest.raises(ValueError, match="--extract requires --positive and --negative"):
            introspect_steer(steer_args)

    def test_steer_apply_with_neuron(self, steer_args, capsys):
        """Test steering single neuron."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.neuron = 42

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Steering neuron 42" in captured.out

    def test_steer_compare_coefficients(self, steer_args, capsys):
        """Test comparing multiple steering coefficients."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.compare = "-2,-1,0,1,2"
        steer_args.positive = "good"
        steer_args.negative = "bad"

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Comparing steering" in captured.out

    def test_steer_missing_direction_source(self, steer_args):
        """Test error when no direction source provided."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # No direction, no neuron, no positive/negative
        with pytest.raises(ValueError, match="Must provide --direction, --neuron"):
            introspect_steer(steer_args)

    def test_steer_apply_on_the_fly_direction(self, steer_args, capsys):
        """Test generating direction on-the-fly from positive/negative."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.positive = "happy"
        steer_args.negative = "sad"
        # No direction file or neuron

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "on-the-fly direction" in captured.out
