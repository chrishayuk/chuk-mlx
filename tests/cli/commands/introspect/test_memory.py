"""Tests for introspect memory CLI commands."""

import asyncio
import json
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


class TestIntrospectMemory:
    """Tests for introspect_memory command."""

    @pytest.fixture
    def memory_args(self):
        """Create arguments for memory command."""
        return Namespace(
            model="test-model",
            facts="multiplication",
            layer=None,
            top_k=10,
            classify=False,
            save_plot=None,
            output=None,
        )

    def test_memory_basic(self, memory_args, capsys):
        """Test basic memory analysis."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert "MEMORY" in captured.out

    def test_memory_with_layer(self, memory_args, capsys):
        """Test memory analysis with specific layer."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        memory_args.layer = 5

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert "MEMORY" in captured.out

    def test_memory_with_classify(self, memory_args, capsys):
        """Test memory analysis with classification."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        memory_args.classify = True

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_memory_with_output(self, memory_args, tmp_path, capsys):
        """Test memory analysis with output file."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        output_file = tmp_path / "memory_results.json"
        memory_args.output = str(output_file)

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert "saved to" in captured.out

    def test_memory_with_plot(self, memory_args, tmp_path, capsys):
        """Test memory analysis with plot."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        plot_file = tmp_path / "memory_plot.png"
        memory_args.save_plot = str(plot_file)

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        # Should run without error
        assert captured.out != "" or captured.err != ""

    def test_memory_addition_facts(self, memory_args, capsys):
        """Test memory analysis with addition facts."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        memory_args.facts = "addition"

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""

    def test_memory_capitals_facts(self, memory_args, capsys):
        """Test memory analysis with capitals facts."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory

        memory_args.facts = "capitals"

        asyncio.run(introspect_memory(memory_args))

        captured = capsys.readouterr()
        assert captured.out != "" or captured.err != ""


class TestIntrospectMemoryInject:
    """Tests for introspect_memory_inject command."""

    @pytest.fixture
    def inject_args(self):
        """Create arguments for memory inject command."""
        return Namespace(
            model="test-model",
            facts="multiplication",
            query="7*8=",
            queries=None,
            query_layer=None,
            inject_layer=None,
            blend=1.0,
            threshold=0.7,
            save_store=None,
            load_store=None,
            force=False,
            evaluate=False,
        )

    @pytest.fixture
    def mock_external_memory(self):
        """Create mock external memory."""
        from unittest.mock import patch

        with patch("chuk_lazarus.introspection.external_memory.ExternalMemory") as mock_memory_cls:
            mock_memory = MagicMock()
            mock_memory.num_entries = 64

            # Mock query_batch result
            mock_result = MagicMock()
            mock_result.baseline_answer = "56"
            mock_result.baseline_confidence = 0.85
            mock_result.used_injection = False
            mock_result.matched_entry = None
            mock_result.similarity = 0.0
            mock_result.injected_answer = "56"
            mock_result.injected_confidence = 0.95

            mock_memory.query_batch.return_value = [mock_result]
            mock_memory.add_facts = MagicMock()
            mock_memory.save = MagicMock()
            mock_memory.load = MagicMock()

            mock_metrics = {
                "baseline_accuracy": 0.85,
                "injected_accuracy": 0.95,
                "rescued": 8,
                "broken": 1,
            }
            mock_memory.evaluate.return_value = mock_metrics

            mock_memory_cls.from_pretrained.return_value = mock_memory
            yield mock_memory_cls, mock_memory

    def test_inject_basic(self, inject_args, mock_external_memory, capsys):
        """Test basic memory injection."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "EXTERNAL MEMORY INJECTION" in captured.out
        assert "Query:" in captured.out

    def test_inject_no_queries(self, inject_args, mock_external_memory, capsys):
        """Test injection with no queries provided."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        inject_args.query = None
        inject_args.queries = None

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "No queries provided" in captured.out

    def test_inject_multiple_queries(self, inject_args, mock_external_memory, capsys):
        """Test injection with multiple queries."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        # Return multiple results for multiple queries
        mock_result = MagicMock()
        mock_result.baseline_answer = "56"
        mock_result.baseline_confidence = 0.85
        mock_result.used_injection = False
        mock_result.matched_entry = None
        mock_result.similarity = 0.0
        mock_memory.query_batch.return_value = [mock_result, mock_result]

        inject_args.query = None
        inject_args.queries = "7*8=|9*6="

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "Query:" in captured.out

    def test_inject_with_custom_layers(self, inject_args, mock_external_memory, capsys):
        """Test injection with custom query and inject layers."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        inject_args.query_layer = 20
        inject_args.inject_layer = 19

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "Query:" in captured.out

    def test_inject_force_mode(self, inject_args, mock_external_memory, capsys):
        """Test forced injection mode."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        # Set up result for force mode (injection used, answer modified)
        mock_result = MagicMock()
        mock_result.baseline_answer = "55"  # Wrong baseline
        mock_result.baseline_confidence = 0.65
        mock_result.used_injection = True
        mock_result.injected_answer = "56"  # Correct after injection
        mock_result.injected_confidence = 0.95
        mock_result.matched_entry = MagicMock(query="7*8=", answer="56")
        mock_result.similarity = 0.99

        mock_memory.query_batch.return_value = [mock_result]

        inject_args.force = True

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "MODIFIED" in captured.out

    def test_inject_below_threshold(self, inject_args, mock_external_memory, capsys):
        """Test when match is below threshold."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        mock_result = MagicMock()
        mock_result.baseline_answer = "56"
        mock_result.baseline_confidence = 0.85
        mock_result.used_injection = False
        mock_result.matched_entry = MagicMock(query="7*8=", answer="56")
        mock_result.similarity = 0.5  # Below threshold

        mock_memory.query_batch.return_value = [mock_result]

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "Below threshold" in captured.out

    def test_inject_no_match(self, inject_args, mock_external_memory, capsys):
        """Test when no match is found."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        mock_result = MagicMock()
        mock_result.baseline_answer = "unknown"
        mock_result.baseline_confidence = 0.15
        mock_result.used_injection = False
        mock_result.matched_entry = None
        mock_result.similarity = 0.0

        mock_memory.query_batch.return_value = [mock_result]

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "No match found" in captured.out

    def test_inject_save_store(self, inject_args, tmp_path, mock_external_memory, capsys):
        """Test saving memory store."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        store_path = tmp_path / "memory_store.npz"
        inject_args.save_store = str(store_path)

        asyncio.run(introspect_memory_inject(inject_args))

        # Verify save was called
        mock_memory.save.assert_called_once_with(str(store_path))

    def test_inject_load_store(self, inject_args, tmp_path, mock_external_memory, capsys):
        """Test loading memory store."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        mock_memory_cls, mock_memory = mock_external_memory

        store_path = tmp_path / "memory_store.npz"
        inject_args.load_store = str(store_path)

        asyncio.run(introspect_memory_inject(inject_args))

        # Verify load was called
        mock_memory.load.assert_called_once_with(str(store_path))

    def test_inject_evaluate_mode(self, inject_args, mock_external_memory, capsys):
        """Test evaluation mode."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        inject_args.evaluate = True

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "EVALUATION" in captured.out
        assert "Baseline accuracy:" in captured.out
        assert "Rescued:" in captured.out

    def test_inject_from_file(self, inject_args, tmp_path, mock_external_memory, capsys):
        """Test injection with facts from file."""
        from chuk_lazarus.cli.commands.introspect.memory import introspect_memory_inject

        facts_file = tmp_path / "custom_facts.json"
        facts = [
            {"query": "What is 2+2?", "answer": "4"},
            {"query": "What is 3+3?", "answer": "6"},
        ]
        with open(facts_file, "w") as f:
            json.dump(facts, f)

        inject_args.facts = f"@{facts_file}"
        inject_args.query = "What is 2+2?"

        asyncio.run(introspect_memory_inject(inject_args))

        captured = capsys.readouterr()
        assert "Query:" in captured.out


class TestMemoryConfig:
    """Tests for memory config types."""

    def test_layer_depth_ratio(self):
        """Test layer depth ratio calculation."""
        from chuk_lazarus.cli.commands._constants import LayerDepthRatio
        from chuk_lazarus.cli.commands.introspect._utils import get_layer_depth_ratio

        # When layer is specified, ratio is None (use explicit layer)
        ratio = get_layer_depth_ratio(5, LayerDepthRatio.DEEP)
        assert ratio is None

        # When layer is None, use default ratio value
        ratio = get_layer_depth_ratio(None, LayerDepthRatio.DEEP)
        assert ratio == LayerDepthRatio.DEEP.value

    def test_memory_defaults(self):
        """Test memory default constants."""
        from chuk_lazarus.cli.commands._constants import MemoryDefaults

        assert MemoryDefaults.DEFAULT_QUERY_LAYER is not None
        assert MemoryDefaults.DEFAULT_INJECT_LAYER is not None
        assert MemoryDefaults.BLEND >= 0.0
        assert MemoryDefaults.SIMILARITY_THRESHOLD >= 0.0
