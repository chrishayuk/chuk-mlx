"""Tests for introspect circuit CLI commands."""

import asyncio
from argparse import Namespace

import pytest

from .conftest import requires_sklearn


class TestIntrospectCircuitCapture:
    """Tests for introspect_circuit_capture command."""

    @pytest.fixture
    def capture_args(self):
        """Create arguments for circuit capture command."""
        return Namespace(
            model="test-model",
            prompts="7*4=|6*8=|9*3=",
            layer=19,
            results=None,
            extract_direction=False,
            output=None,
        )

    def test_capture_requires_layer(self):
        """Test that capture requires --layer."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        args = Namespace(
            model="test-model",
            prompts="7*4=",
            layer=None,
            results=None,
            output=None,
        )

        with pytest.raises(ValueError, match="layer"):
            asyncio.run(introspect_circuit_capture(args))

    def test_capture_basic(self, capture_args, capsys):
        """Test basic circuit capture."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        asyncio.run(introspect_circuit_capture(capture_args))

        captured = capsys.readouterr()
        assert "CIRCUIT" in captured.out

    def test_capture_with_results(self, capture_args, capsys):
        """Test capture with explicit results."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        capture_args.results = "28|48|27"

        asyncio.run(introspect_circuit_capture(capture_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_capture_mismatched_results(self, capture_args):
        """Test error on mismatched results count."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        capture_args.results = "28|48"  # Only 2 results for 3 prompts

        with pytest.raises(ValueError, match="results"):
            asyncio.run(introspect_circuit_capture(capture_args))

    @requires_sklearn
    def test_capture_with_extract_direction(self, capture_args, capsys):
        """Test capture with direction extraction."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        capture_args.extract_direction = True
        capture_args.results = "28|48|27"

        asyncio.run(introspect_circuit_capture(capture_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestIntrospectCircuitInvoke:
    """Tests for introspect_circuit_invoke command."""

    @pytest.fixture
    def invoke_args(self, tmp_path):
        """Create arguments for circuit invoke command."""
        # Create a mock circuit file
        import numpy as np

        circuit_file = tmp_path / "circuit.npz"
        np.savez(
            circuit_file,
            vectors=np.random.randn(3, 768).astype(np.float32),
            prompts=np.array(["7*4=", "6*8=", "9*3="]),
            layer=19,
        )

        return Namespace(
            model="test-model",
            circuit=str(circuit_file),
            prompts="5*5=|8*7=",
            method="steer",
            coefficient=None,
            layer=None,
            top_k=5,
        )

    def test_invoke_basic(self, invoke_args, capsys):
        """Test basic circuit invocation."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_invoke

        asyncio.run(introspect_circuit_invoke(invoke_args))

        captured = capsys.readouterr()
        assert "CIRCUIT" in captured.out or "INVOCATION" in captured.out

    def test_invoke_interpolate_with_coefficient(self, invoke_args, capsys):
        """Test that interpolate method works with --coefficient."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_invoke

        invoke_args.method = "interpolate"
        invoke_args.coefficient = 0.5

        asyncio.run(introspect_circuit_invoke(invoke_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_invoke_extrapolate_with_coefficient(self, invoke_args, capsys):
        """Test that extrapolate method works with --coefficient."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_invoke

        invoke_args.method = "extrapolate"
        invoke_args.coefficient = 1.5

        asyncio.run(introspect_circuit_invoke(invoke_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_invoke_with_coefficient(self, invoke_args, capsys):
        """Test invocation with coefficient."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_invoke

        invoke_args.method = "interpolate"
        invoke_args.coefficient = 0.5

        asyncio.run(introspect_circuit_invoke(invoke_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestIntrospectCircuitDecode:
    """Tests for introspect_circuit_decode command."""

    @pytest.fixture
    def decode_args(self, tmp_path):
        """Create arguments for circuit decode command."""
        import numpy as np

        circuit_file = tmp_path / "circuit.npz"
        np.savez(
            circuit_file,
            vectors=np.random.randn(64, 768).astype(np.float32),
            prompts=np.array(["test"] * 64),
            layer=19,
        )

        return Namespace(
            model="test-model",
            circuit=str(circuit_file),
            prompt="5*5=",
            inject_idx=0,
            max_tokens=10,
            raw=False,
        )

    def test_decode_basic(self, decode_args, capsys):
        """Test basic circuit decode."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_decode

        asyncio.run(introspect_circuit_decode(decode_args))

        captured = capsys.readouterr()
        assert "DECODE" in captured.out or "INJECTION" in captured.out


class TestIntrospectCircuitView:
    """Tests for introspect_circuit_view command."""

    @pytest.fixture
    def view_args(self, tmp_path):
        """Create arguments for circuit view command."""
        import numpy as np

        circuit_file = tmp_path / "circuit.npz"
        np.savez(
            circuit_file,
            vectors=np.random.randn(64, 768).astype(np.float32),
            prompts=np.array(["test"] * 64),
            layer=19,
        )

        return Namespace(
            circuit=str(circuit_file),
            show="table",
            limit=None,
        )

    def test_view_basic(self, view_args, capsys):
        """Test basic circuit view."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_view

        asyncio.run(introspect_circuit_view(view_args))

        captured = capsys.readouterr()
        assert "CIRCUIT" in captured.out or "VIEW" in captured.out or captured.out != ""


class TestIntrospectCircuitTest:
    """Tests for introspect_circuit_test command."""

    @pytest.fixture
    def test_args(self, tmp_path):
        """Create arguments for circuit test command."""
        import numpy as np

        circuit_file = tmp_path / "circuit.npz"
        np.savez(
            circuit_file,
            vectors=np.random.randn(64, 768).astype(np.float32),
            prompts=np.array(["test"] * 64),
            layer=19,
        )

        return Namespace(
            model="test-model",
            circuit=str(circuit_file),
            prompts="5*5=|8*7=",
            expected="25|56",
            max_tokens=10,
        )

    def test_test_basic(self, test_args, capsys):
        """Test basic circuit testing."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_test

        asyncio.run(introspect_circuit_test(test_args))

        captured = capsys.readouterr()
        assert "CIRCUIT" in captured.out or "TEST" in captured.out or captured.out != ""


class TestIntrospectCircuitCompare:
    """Tests for introspect_circuit_compare command."""

    @pytest.fixture
    def compare_args(self, tmp_path):
        """Create arguments for circuit compare command."""
        import numpy as np

        circuit_file1 = tmp_path / "circuit1.npz"
        np.savez(
            circuit_file1,
            vectors=np.random.randn(64, 768).astype(np.float32),
            prompts=np.array(["test"] * 64),
            layer=19,
        )

        circuit_file2 = tmp_path / "circuit2.npz"
        np.savez(
            circuit_file2,
            vectors=np.random.randn(64, 768).astype(np.float32),
            prompts=np.array(["test"] * 64),
            layer=19,
        )

        return Namespace(
            circuit_a=str(circuit_file1),
            circuit_b=str(circuit_file2),
        )

    def test_compare_basic(self, compare_args, capsys):
        """Test basic circuit comparison."""
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_compare

        asyncio.run(introspect_circuit_compare(compare_args))

        captured = capsys.readouterr()
        assert "CIRCUIT" in captured.out or "COMPARE" in captured.out or captured.out != ""


class TestCircuitConfig:
    """Tests for circuit configuration types."""

    def test_circuit_defaults(self):
        """Test circuit default constants."""
        from chuk_lazarus.cli.commands._constants import CircuitDefaults

        assert CircuitDefaults is not None

    def test_output_format_enum(self):
        """Test OutputFormat enum."""
        from chuk_lazarus.cli.commands._constants import OutputFormat

        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.TEXT.value == "text"

    def test_parse_prompts(self):
        """Test prompt parsing."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_prompts

        prompts = parse_prompts("7*4=|6*8=|9*3=")
        assert len(prompts) == 3
        assert prompts[0] == "7*4="

    def test_parse_value_list(self):
        """Test value list parsing."""
        from chuk_lazarus.cli.commands.introspect._utils import parse_value_list

        values = parse_value_list("28|48|27", value_type=int)
        assert values == [28, 48, 27]
