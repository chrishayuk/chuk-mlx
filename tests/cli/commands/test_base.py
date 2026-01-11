"""Tests for CLI commands base module."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from chuk_lazarus.cli.commands._base import (
    CommandConfig,
    CommandResult,
    CommonFields,
    OutputMixin,
    PathMixin,
)


class ConcreteConfig(CommandConfig):
    """Concrete implementation of CommandConfig for testing."""

    name: str
    value: int = 0

    @classmethod
    def from_args(cls, args: Namespace) -> "ConcreteConfig":
        return cls(name=args.name, value=getattr(args, "value", 0))


class ConcreteResult(CommandResult):
    """Concrete implementation of CommandResult for testing."""

    success: bool
    message: str

    def to_display(self) -> str:
        return f"Success: {self.success}, Message: {self.message}"


class TestCommandConfig:
    """Tests for CommandConfig base class."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        args = Namespace(name="test", value=42)
        config = ConcreteConfig.from_args(args)
        assert config.name == "test"
        assert config.value == 42

    def test_from_args_default(self):
        """Test creating config with defaults."""
        args = Namespace(name="test")
        config = ConcreteConfig.from_args(args)
        assert config.name == "test"
        assert config.value == 0

    def test_config_frozen(self):
        """Test that config is frozen."""
        import pytest
        from pydantic import ValidationError

        config = ConcreteConfig(name="test", value=1)
        with pytest.raises(ValidationError):
            config.name = "changed"


class TestCommandResult:
    """Tests for CommandResult base class."""

    def test_to_display(self):
        """Test result display."""
        result = ConcreteResult(success=True, message="Done")
        display = result.to_display()
        assert "Success: True" in display
        assert "Message: Done" in display

    def test_result_frozen(self):
        """Test that result is frozen."""
        import pytest
        from pydantic import ValidationError

        result = ConcreteResult(success=True, message="Test")
        with pytest.raises(ValidationError):
            result.success = False


class TestOutputMixin:
    """Tests for OutputMixin."""

    def test_format_header(self):
        """Test header formatting."""
        header = OutputMixin.format_header("Test Title")
        assert "Test Title" in header
        assert "=" in header

    def test_format_header_custom_width(self):
        """Test header with custom width."""
        header = OutputMixin.format_header("Title", width=40)
        assert "=" * 40 in header

    def test_format_field(self):
        """Test field formatting."""
        field = OutputMixin.format_field("key", "value")
        assert "key: value" in field
        assert field.startswith("  ")

    def test_format_field_custom_indent(self):
        """Test field with custom indent."""
        field = OutputMixin.format_field("key", "value", indent=4)
        assert "    key: value" in field

    def test_format_table_row(self):
        """Test table row formatting."""
        columns = [("Name", "Alice"), ("Age", 30)]
        row = OutputMixin.format_table_row(columns)
        assert "Alice" in row
        assert "30" in row

    def test_format_table_row_custom_widths(self):
        """Test table row with custom widths."""
        columns = [("Name", "Alice"), ("Age", 30)]
        row = OutputMixin.format_table_row(columns, widths=[10, 5])
        assert "Alice" in row
        assert "30" in row


class TestPathMixin:
    """Tests for PathMixin."""

    def test_ensure_parent_exists(self):
        """Test ensuring parent directory exists."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "file.txt"
            result = PathMixin.ensure_parent_exists(path)

            assert result == path
            assert path.parent.exists()

    def test_resolve_path(self):
        """Test path resolution."""
        result = PathMixin.resolve_path("./relative/path")
        assert result.is_absolute()
        assert "relative/path" in str(result)

    def test_resolve_path_none(self):
        """Test resolving None path."""
        result = PathMixin.resolve_path(None)
        assert result is None

    def test_resolve_path_already_absolute(self):
        """Test resolving absolute path."""
        result = PathMixin.resolve_path("/absolute/path")
        assert result.is_absolute()
        assert str(result) == "/absolute/path"


class TestCommonFields:
    """Tests for CommonFields utility."""

    def test_tokenizer_field(self):
        """Test tokenizer field definition."""
        field = CommonFields.tokenizer_field()
        assert field is not None
        assert field.description is not None

    def test_model_field(self):
        """Test model field definition."""
        field = CommonFields.model_field()
        assert field is not None
        assert field.description is not None

    def test_output_field(self):
        """Test output field definition."""
        field = CommonFields.output_field()
        assert field is not None
        assert field.default is None

    def test_verbose_field(self):
        """Test verbose field definition."""
        field = CommonFields.verbose_field()
        assert field is not None
        assert field.default is False

    def test_seed_field(self):
        """Test seed field definition."""
        field = CommonFields.seed_field()
        assert field is not None
        assert field.default is None
