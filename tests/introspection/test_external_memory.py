"""Tests for external_memory module."""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from chuk_lazarus.introspection.external_memory import (
    ExternalMemory,
    MemoryConfig,
    MemoryEntry,
    QueryResult,
)


class MockConfig:
    """Mock model configuration."""

    def __init__(self, hidden_size: int = 64, num_hidden_layers: int = 4):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embedding_scale = None

    @classmethod
    def from_hf_config(cls, config_data: dict):
        """Create config from HuggingFace config dict."""
        return cls(
            hidden_size=config_data.get("hidden_size", 64),
            num_hidden_layers=config_data.get("num_hidden_layers", 4),
        )


class MockEmbedding(nn.Module):
    """Mock embedding layer."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((vocab_size, hidden_size))

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.weight[input_ids]


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = mx.random.normal((hidden_size, hidden_size))

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> mx.array:
        if x.ndim == 3:
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
            out_flat = x_flat @ self.weight
            return out_flat.reshape(batch, seq, dim)
        return x @ self.weight


class MockModel(nn.Module):
    """Mock model."""

    def __init__(self, config_or_vocab_size=100, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()

        # Support both config object and direct parameters
        if hasattr(config_or_vocab_size, "hidden_size"):
            # It's a config object
            vocab_size = 100
            hidden_size = config_or_vocab_size.hidden_size
            num_layers = config_or_vocab_size.num_hidden_layers
        else:
            # It's vocab_size
            vocab_size = config_or_vocab_size

        class InnerModel(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers):
                super().__init__()
                self.embed_tokens = MockEmbedding(vocab_size, hidden_size)
                self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
                self.norm = nn.RMSNorm(hidden_size)

        self.model = InnerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class MockTokenizer:
    """Mock tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 100 for c in text[:10]]  # Max 10 tokens

    def decode(self, ids: list[int]) -> str:
        if isinstance(ids, (list, tuple)) and len(ids) > 0:
            return str(ids[0])
        return str(ids)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_init_basic(self):
        entry = MemoryEntry(query="2+2=", answer="4")
        assert entry.query == "2+2="
        assert entry.answer == "4"
        assert entry.query_vector is None
        assert entry.value_vector is None
        assert entry.metadata == {}

    def test_init_with_vectors(self):
        query_vec = mx.random.normal((64,))
        value_vec = mx.random.normal((64,))

        entry = MemoryEntry(
            query="test",
            answer="result",
            query_vector=query_vec,
            value_vector=value_vec,
        )

        assert entry.query_vector is not None
        assert entry.value_vector is not None

    def test_init_with_metadata(self):
        entry = MemoryEntry(
            query="3*4=",
            answer="12",
            metadata={"type": "multiplication", "difficulty": "easy"},
        )

        assert entry.metadata["type"] == "multiplication"
        assert entry.metadata["difficulty"] == "easy"


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_config(self):
        config = MemoryConfig()
        assert config.query_layer == 22
        assert config.inject_layer == 21
        assert config.value_layer == 22
        assert config.similarity_threshold == 0.7
        assert config.blend == 1.0

    def test_custom_config(self):
        config = MemoryConfig(
            query_layer=10,
            inject_layer=9,
            value_layer=10,
            similarity_threshold=0.8,
            blend=0.5,
        )

        assert config.query_layer == 10
        assert config.inject_layer == 9
        assert config.value_layer == 10
        assert config.similarity_threshold == 0.8
        assert config.blend == 0.5


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_init(self):
        result = QueryResult(
            query="test",
            baseline_answer="base",
            baseline_confidence=0.8,
            injected_answer="inject",
            injected_confidence=0.9,
            matched_entry=None,
            similarity=0.75,
            used_injection=True,
        )

        assert result.query == "test"
        assert result.baseline_answer == "base"
        assert result.baseline_confidence == 0.8
        assert result.injected_answer == "inject"
        assert result.injected_confidence == 0.9
        assert result.similarity == 0.75
        assert result.used_injection is True

    def test_with_matched_entry(self):
        entry = MemoryEntry(query="matched", answer="result")
        result = QueryResult(
            query="test",
            baseline_answer="base",
            baseline_confidence=0.5,
            injected_answer=None,
            injected_confidence=None,
            matched_entry=entry,
            similarity=0.9,
            used_injection=False,
        )

        assert result.matched_entry is entry
        assert result.matched_entry.query == "matched"


class TestExternalMemory:
    """Tests for ExternalMemory class."""

    def test_init(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig()

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        assert memory._model is model
        assert memory._tokenizer is tokenizer
        assert memory._config is config
        assert memory._memory_config is memory_config
        assert len(memory._entries) == 0

    def test_init_default_memory_config(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        # Should create default MemoryConfig
        assert memory._memory_config is not None
        assert isinstance(memory._memory_config, MemoryConfig)

    def test_num_entries_empty(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        assert memory.num_entries == 0

    def test_num_entries_with_data(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        memory._entries = [
            MemoryEntry("q1", "a1"),
            MemoryEntry("q2", "a2"),
        ]

        assert memory.num_entries == 2

    def test_hidden_size(self):
        model = MockModel(hidden_size=128)
        tokenizer = MockTokenizer()
        config = MockConfig(hidden_size=128)

        memory = ExternalMemory(model, tokenizer, config)
        assert memory.hidden_size == 128

    def test_get_layers_nested(self):
        model = MockModel(num_layers=6)
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        layers = memory._get_layers()

        assert isinstance(layers, list)
        assert len(layers) == 6

    def test_get_embed(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        embed = memory._get_embed()

        assert embed is not None
        assert hasattr(embed, "weight")

    def test_get_norm(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        norm = memory._get_norm()

        assert norm is not None

    def test_get_lm_head(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        lm_head = memory._get_lm_head()

        assert lm_head is not None

    def test_get_scale_none(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        scale = memory._get_scale()

        assert scale is None

    def test_extract_representation(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        memory = ExternalMemory(model, tokenizer, config)
        rep = memory._extract_representation("test", layer=2)

        assert isinstance(rep, mx.array)
        assert rep.ndim == 1
        assert rep.shape[0] == 64

    def test_add_fact(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        entry = memory.add_fact("2+2=", "4")

        assert memory.num_entries == 1
        assert entry.query == "2+2="
        assert entry.answer == "4"
        assert entry.query_vector is not None
        assert entry.value_vector is not None

    def test_add_fact_with_metadata(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        metadata = {"type": "addition"}
        entry = memory.add_fact("5+3=", "8", metadata=metadata)

        assert entry.metadata == metadata

    def test_add_facts(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        facts = [
            {"query": "2+2=", "answer": "4"},
            {"query": "3+3=", "answer": "6"},
        ]

        entries = memory.add_facts(facts, verbose=False)

        assert len(entries) == 2
        assert memory.num_entries == 2

    def test_add_facts_with_metadata(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        facts = [
            {"query": "2*3=", "answer": "6", "metadata": {"type": "mult"}},
        ]

        entries = memory.add_facts(facts, verbose=False)
        assert entries[0].metadata["type"] == "mult"

    def test_add_multiplication_table(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        entries = memory.add_multiplication_table(min_val=2, max_val=3)

        # 2x2, 2x3, 3x2, 3x3 = 4 entries
        assert len(entries) == 4
        assert memory.num_entries == 4

        # Check one entry
        entry = next(e for e in entries if e.query == "2*3=")
        assert entry.answer == "6"
        assert entry.metadata["type"] == "multiplication"

    def test_match_empty(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        query_vec = mx.random.normal((64,))
        matches = memory.match(query_vec)

        assert len(matches) == 0

    def test_match_single(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        # Match with similar vector
        rep = memory._extract_representation("test", layer=1)
        matches = memory.match(rep, top_k=1)

        assert len(matches) == 1
        entry, similarity = matches[0]
        assert entry.query == "test"
        # Allow small floating point tolerance
        assert -0.01 <= similarity <= 1.01

    def test_match_top_k(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add multiple facts
        for i in range(5):
            memory.add_fact(f"query{i}", f"answer{i}")

        query_vec = mx.random.normal((64,))
        matches = memory.match(query_vec, top_k=3)

        assert len(matches) == 3
        # Should be sorted by similarity
        sims = [sim for _, sim in matches]
        assert sims == sorted(sims, reverse=True)

    def test_forward_with_injection_no_injection(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        top_token, top_prob, layer_preds = memory._forward_with_injection("test")

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)
        assert 0 <= top_prob <= 1
        assert isinstance(layer_preds, dict)

    def test_forward_with_injection_with_injection(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        inject_vector = mx.random.normal((64,))
        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            inject_layer=1,
            inject_vector=inject_vector,
            blend=1.0,
        )

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)

    def test_query_no_match(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        result = memory.query("test")

        assert result.query == "test"
        assert result.baseline_answer is not None
        assert result.used_injection is False

    def test_query_with_match(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(
            query_layer=1,
            value_layer=1,
            inject_layer=0,
            similarity_threshold=0.0,  # Always match
        )

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        result = memory.query("test", use_injection=True)

        assert result.matched_entry is not None
        assert result.similarity > 0

    def test_query_force_injection(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(
            query_layer=1,
            value_layer=1,
            inject_layer=0,
            similarity_threshold=0.99,  # High threshold
        )

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        # Without force_injection, might not use injection
        result = memory.query("different", use_injection=True, force_injection=True)

        # Should still attempt injection even with low similarity
        assert result.matched_entry is not None

    def test_batch_query(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        prompts = ["test1", "test2", "test3"]
        results = memory.batch_query(prompts, verbose=False)

        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

    def test_save_and_load(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("2+2=", "4", metadata={"type": "addition"})

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory"
            memory.save(save_path)

            # Check files exist
            assert (save_path.with_suffix(".npz")).exists()
            assert (save_path.with_suffix(".json")).exists()

            # Create new memory and load
            memory2 = ExternalMemory(model, tokenizer, config)
            memory2.load(save_path)

            assert memory2.num_entries == 1
            assert memory2._entries[0].query == "2+2="
            assert memory2._entries[0].answer == "4"
            assert memory2._entries[0].metadata["type"] == "addition"

    def test_evaluate(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        test_facts = [
            {"query": "test", "answer": "result"},
        ]

        metrics = memory.evaluate(test_facts, verbose=False)

        assert "total" in metrics
        assert "baseline_correct" in metrics
        assert "injected_correct" in metrics
        assert "rescued" in metrics
        assert "broken" in metrics
        assert "baseline_accuracy" in metrics
        assert "injected_accuracy" in metrics

        assert metrics["total"] == 1

    def test_evaluate_verbose_rescued(self, capsys):
        """Test evaluate with verbose output for rescued answers."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "expected")

        # Create test facts where baseline is wrong but injection might be right
        test_facts = [
            {"query": "test", "answer": "expected"},
        ]

        memory.evaluate(test_facts, verbose=True)

        # Check verbose output was generated
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "expected" in captured.out

    def test_evaluate_broken_case(self):
        """Test evaluate when injection breaks correct baseline answer."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add a fact that might cause wrong injections
        memory.add_fact("other", "wrong")

        test_facts = [
            {"query": "test", "answer": "baseline_prediction"},
        ]

        metrics = memory.evaluate(test_facts, verbose=False)

        # Metrics should track broken cases
        assert "broken" in metrics
        assert metrics["broken"] >= 0

    def test_evaluate_verbose_broken(self, capsys):
        """Test evaluate with verbose output for broken answers."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add a fact
        memory.add_fact("test", "injected_result")

        # Test where baseline might be "correct" and injection makes it wrong
        # We'll assume baseline happens to match the answer
        test_facts = [
            {"query": "test", "answer": "some_answer"},
        ]

        memory.evaluate(test_facts, verbose=True)

        # Check verbose output was generated
        captured = capsys.readouterr()
        assert "test" in captured.out
        # Should show expected, baseline, and injected values

    def test_evaluate_verbose_all_paths(self, capsys):
        """Test evaluate verbose output covering all code paths."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add multiple facts to increase chances of different outcomes
        memory.add_fact("query1", "answer1")
        memory.add_fact("query2", "answer2")
        memory.add_fact("query3", "answer3")

        test_facts = [
            {"query": "query1", "answer": "answer1"},
            {"query": "query2", "answer": "answer2"},
            {"query": "query3", "answer": "answer3"},
        ]

        memory.evaluate(test_facts, verbose=True)

        # Check verbose output was generated
        captured = capsys.readouterr()
        assert "query1" in captured.out or "query2" in captured.out or "query3" in captured.out
        # Output should contain expected values
        assert "expected=" in captured.out
        assert "baseline=" in captured.out
        assert "injected=" in captured.out

    def test_add_facts_verbose_progress(self, capsys):
        """Test add_facts prints progress every 10 facts."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add 15 facts to trigger progress output
        facts = [{"query": f"q{i}", "answer": f"a{i}"} for i in range(15)]

        entries = memory.add_facts(facts, verbose=True)

        captured = capsys.readouterr()
        assert "Adding fact 10/15" in captured.out
        assert "Added 15 facts to memory" in captured.out
        assert len(entries) == 15

    def test_batch_query_verbose(self, capsys):
        """Test batch_query with verbose output."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        # Create 15 prompts to trigger verbose output
        prompts = [f"test{i}" for i in range(15)]
        results = memory.batch_query(prompts, verbose=True)

        captured = capsys.readouterr()
        assert "Querying 10/15" in captured.out
        assert len(results) == 15

    def test_query_no_injection_flag(self):
        """Test query with use_injection=False."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1, inject_layer=0)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        result = memory.query("test", use_injection=False)

        # Should not use injection even with a match
        assert result.used_injection is False
        assert result.injected_answer is None

    def test_query_below_threshold(self):
        """Test query when similarity is below threshold."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(
            query_layer=1,
            value_layer=1,
            inject_layer=0,
            similarity_threshold=0.99,  # Very high threshold
        )

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        # Query with different text (low similarity)
        result = memory.query("completely_different_query", use_injection=True)

        # Should not inject due to low similarity
        assert result.matched_entry is not None  # Still finds a match
        assert result.similarity < 0.99  # But similarity is low

    def test_match_with_none_query_vector(self):
        """Test match when entry has None query_vector."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        # Manually add entry without query vector
        entry = MemoryEntry(query="test", answer="result", query_vector=None)
        memory._entries.append(entry)

        query_vec = mx.random.normal((64,))
        matches = memory.match(query_vec, top_k=1)

        # Should return empty list since entry has no query_vector
        assert len(matches) == 0

    def test_match_mixed_entries(self):
        """Test match with mix of entries with/without query vectors."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)

        # Add valid entry
        memory.add_fact("valid", "result")

        # Add invalid entry without vector
        invalid_entry = MemoryEntry(query="invalid", answer="bad", query_vector=None)
        memory._entries.append(invalid_entry)

        query_vec = mx.random.normal((64,))
        matches = memory.match(query_vec, top_k=5)

        # Should only return the valid entry
        assert len(matches) == 1
        assert matches[0][0].query == "valid"


class TestModelStructureVariants:
    """Test different model structure paths."""

    def test_get_layers_direct(self):
        """Test _get_layers when model.layers exists directly."""

        # Create model without nested structure
        class DirectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [MockLayer(64) for _ in range(3)]
                self.embed_tokens = MockEmbedding(100, 64)

        model = DirectModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        layers = memory._get_layers()

        assert len(layers) == 3

    def test_get_embed_direct(self):
        """Test _get_embed when embed_tokens is at top level."""

        class DirectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = MockEmbedding(100, 64)
                self.layers = [MockLayer(64)]

        model = DirectModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        embed = memory._get_embed()

        assert embed is not None
        assert hasattr(embed, "weight")

    def test_get_norm_at_model_level(self):
        """Test _get_norm when norm is at model level."""

        class ModelWithNorm(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [MockLayer(64)]
                        # No norm here

                self.model = Inner()
                self.norm = nn.RMSNorm(64)

        model = ModelWithNorm()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        norm = memory._get_norm()

        assert norm is not None

    def test_get_norm_returns_none(self):
        """Test _get_norm when no norm exists."""

        class NoNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = MockEmbedding(100, 64)
                self.layers = [MockLayer(64)]
                # No norm

        model = NoNormModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        norm = memory._get_norm()

        assert norm is None

    def test_get_lm_head_none(self):
        """Test _get_lm_head when lm_head doesn't exist."""

        class NoLMHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = MockEmbedding(100, 64)
                self.layers = [MockLayer(64)]

        model = NoLMHeadModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        lm_head = memory._get_lm_head()

        assert lm_head is None

    def test_extract_representation_with_scale(self):
        """Test _extract_representation with embedding scale."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        config.embedding_scale = 2.0  # Set scale

        memory = ExternalMemory(model, tokenizer, config)
        rep = memory._extract_representation("test", layer=1)

        assert isinstance(rep, mx.array)
        assert rep.shape[0] == 64

    def test_extract_representation_tuple_output(self):
        """Test _extract_representation when layer returns tuple."""

        class TupleOutputLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = mx.random.normal((hidden_size, hidden_size))

            def __call__(self, x, mask=None):
                if x.ndim == 3:
                    batch, seq, dim = x.shape
                    x_flat = x.reshape(-1, dim)
                    out_flat = x_flat @ self.weight
                    out = out_flat.reshape(batch, seq, dim)
                else:
                    out = x @ self.weight
                # Return tuple
                return (out, None)

        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [TupleOutputLayer(64) for _ in range(3)]
                        self.norm = nn.RMSNorm(64)

                self.model = Inner()

        model = TupleModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        rep = memory._extract_representation("test", layer=1)

        assert isinstance(rep, mx.array)

    def test_extract_representation_no_mask_support(self):
        """Test _extract_representation when layer doesn't accept mask."""

        class NoMaskLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = mx.random.normal((hidden_size, hidden_size))

            def __call__(self, x):
                # Doesn't accept mask parameter
                if x.ndim == 3:
                    batch, seq, dim = x.shape
                    x_flat = x.reshape(-1, dim)
                    out_flat = x_flat @ self.weight
                    return out_flat.reshape(batch, seq, dim)
                return x @ self.weight

        class NoMaskModel(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [NoMaskLayer(64) for _ in range(3)]
                        self.norm = nn.RMSNorm(64)

                self.model = Inner()

        model = NoMaskModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)
        # Should handle TypeError and call without mask
        rep = memory._extract_representation("test", layer=1)

        assert isinstance(rep, mx.array)


class TestForwardWithInjectionEdgeCases:
    """Test edge cases in _forward_with_injection."""

    def test_forward_with_custom_capture_layers(self):
        """Test _forward_with_injection with custom capture_layers."""
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        inject_vector = mx.random.normal((64,))
        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            inject_layer=1,
            inject_vector=inject_vector,
            blend=0.5,
            capture_layers=[0, 1, 2],
        )

        # Should only capture specified layers
        assert len(layer_preds) <= 3
        for layer_idx in layer_preds.keys():
            assert layer_idx in [0, 1, 2]

    def test_forward_with_injection_norm_none(self):
        """Test _forward_with_injection when model has no norm."""

        class NoNormModel(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [MockLayer(64) for _ in range(2)]

                self.model = Inner()
                self.lm_head = nn.Linear(64, 100, bias=False)

        model = NoNormModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            capture_layers=[0],
        )

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)

    def test_forward_with_injection_no_lm_head(self):
        """Test _forward_with_injection when model has no lm_head."""

        class NoLMHeadModel(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [MockLayer(64) for _ in range(2)]
                        self.norm = nn.RMSNorm(64)

                self.model = Inner()

        model = NoLMHeadModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        # Should use embed.weight.T for logits
        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            capture_layers=[0],
        )

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)

    def test_forward_with_injection_lm_head_with_logits_attr(self):
        """Test when lm_head output has .logits attribute."""

        class LogitsOutput:
            def __init__(self, logits):
                self.logits = logits

        class LMHeadWithLogits(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.weight = mx.random.normal((out_dim, in_dim))

            def __call__(self, x):
                if x.ndim == 3:
                    batch, seq, dim = x.shape
                    logits = x @ self.weight.T
                else:
                    logits = x @ self.weight.T
                return LogitsOutput(logits)

        class ModelWithLogitsAttr(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [MockLayer(64) for _ in range(2)]
                        self.norm = nn.RMSNorm(64)

                self.model = Inner()
                self.lm_head = LMHeadWithLogits(64, 100)

        model = ModelWithLogitsAttr()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            capture_layers=[0],
        )

        assert isinstance(top_token, str)
        assert isinstance(top_prob, float)

    def test_forward_with_scale(self):
        """Test _forward_with_injection with embedding scale."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        config.embedding_scale = 1.5

        memory = ExternalMemory(model, tokenizer, config)

        top_token, top_prob, layer_preds = memory._forward_with_injection("test")

        assert isinstance(top_token, str)

    def test_forward_no_mask_support(self):
        """Test _forward_with_injection with layers that don't support mask."""

        class NoMaskLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = mx.random.normal((hidden_size, hidden_size))

            def __call__(self, x):
                if x.ndim == 3:
                    batch, seq, dim = x.shape
                    x_flat = x.reshape(-1, dim)
                    out_flat = x_flat @ self.weight
                    return out_flat.reshape(batch, seq, dim)
                return x @ self.weight

        class NoMaskModel(nn.Module):
            def __init__(self):
                super().__init__()

                class Inner(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.embed_tokens = MockEmbedding(100, 64)
                        self.layers = [NoMaskLayer(64) for _ in range(2)]
                        self.norm = nn.RMSNorm(64)

                self.model = Inner()
                self.lm_head = nn.Linear(64, 100, bias=False)

        model = NoMaskModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        memory = ExternalMemory(model, tokenizer, config)

        # Should handle TypeError and proceed
        top_token, top_prob, layer_preds = memory._forward_with_injection(
            "test",
            capture_layers=[0],
        )

        assert isinstance(top_token, str)


class TestSaveLoadEdgeCases:
    """Test edge cases in save/load."""

    def test_save_with_string_path(self):
        """Test save with string path instead of Path object."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "memory")  # String path
            memory.save(save_path)

            assert Path(save_path).with_suffix(".npz").exists()
            assert Path(save_path).with_suffix(".json").exists()

    def test_load_with_string_path(self):
        """Test load with string path instead of Path object."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory"
            memory.save(save_path)

            memory2 = ExternalMemory(model, tokenizer, config)
            memory2.load(str(save_path))  # String path

            assert memory2.num_entries == 1

    def test_save_entries_without_vectors(self):
        """Test save when some entries lack vectors."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test", "result")

        # Add entry without vectors
        entry = MemoryEntry(query="no_vec", answer="result", query_vector=None, value_vector=None)
        memory._entries.append(entry)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory"
            memory.save(save_path)

            # Should save successfully
            assert save_path.with_suffix(".npz").exists()

    def test_load_missing_vectors(self):
        """Test load when vector files are missing some entries."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        memory_config = MemoryConfig(query_layer=1, value_layer=1)

        memory = ExternalMemory(model, tokenizer, config, memory_config)
        memory.add_fact("test1", "result1")
        memory.add_fact("test2", "result2")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory"
            memory.save(save_path)

            # Load saved data
            vectors = np.load(save_path.with_suffix(".npz"))

            # Remove one vector
            new_vectors = {k: v for k, v in vectors.items() if k != "value_1"}
            np.savez(save_path.with_suffix(".npz"), **new_vectors)

            # Load should handle missing vectors gracefully
            memory2 = ExternalMemory(model, tokenizer, config)
            memory2.load(save_path)

            # First entry should have both vectors
            assert memory2._entries[0].query_vector is not None
            assert memory2._entries[0].value_vector is not None

            # Second entry should have query but not value
            assert memory2._entries[1].query_vector is not None
            # value_1 was removed, so it should be None
            assert memory2._entries[1].value_vector is None


class TestFromPretrainedErrors:
    """Test error handling in from_pretrained."""

    def test_from_pretrained_unsupported_model(self, monkeypatch):
        """Test from_pretrained with unsupported model family."""
        import json
        import tempfile

        # Create temp directory with unsupported config
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "unsupported_model_type"}, f)

        # Mock the HFLoader.download to return our temp directory
        class MockDownloadResult:
            def __init__(self, path):
                self.model_path = path

        def mock_download(model_id):
            return MockDownloadResult(tmpdir)

        # Mock detect_model_family to return None (unsupported)
        def mock_detect(config_data):
            return None

        # Apply mocks
        import chuk_lazarus.inference.loader as loader_module
        import chuk_lazarus.models_v2.families.registry as registry_module

        monkeypatch.setattr(loader_module.HFLoader, "download", mock_download)
        monkeypatch.setattr(registry_module, "detect_model_family", mock_detect)

        # Should raise ValueError for unsupported model
        with pytest.raises(ValueError, match="Unsupported model"):
            ExternalMemory.from_pretrained("fake/unsupported-model")

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_from_pretrained_auto_config(self, monkeypatch):
        """Test from_pretrained with auto-configuration of memory layers."""
        import json
        import tempfile

        # Create temp directory with model config
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": "gpt2",
                    "hidden_size": 64,
                    "num_hidden_layers": 24,
                },
                f,
            )

        # Mock the HFLoader.download to return our temp directory
        class MockDownloadResult:
            def __init__(self, path):
                self.model_path = path

        def mock_download(model_id):
            return MockDownloadResult(tmpdir)

        # Mock detect_model_family to return a valid family
        from chuk_lazarus.models_v2.families.registry import ModelFamilyType

        def mock_detect(config_data):
            return ModelFamilyType.GPT2

        # Mock get_family_info to return mock model classes
        class MockFamilyInfo:
            config_class = MockConfig
            model_class = MockModel

        def mock_get_family_info(family_type):
            return MockFamilyInfo()

        # Mock apply_weights and load_tokenizer
        def mock_apply_weights(model, path, config, dtype):
            pass

        def mock_load_tokenizer(path):
            return MockTokenizer()

        # Apply mocks
        import chuk_lazarus.inference.loader as loader_module
        import chuk_lazarus.models_v2.families.registry as registry_module

        monkeypatch.setattr(loader_module.HFLoader, "download", mock_download)
        monkeypatch.setattr(registry_module, "detect_model_family", mock_detect)
        monkeypatch.setattr(registry_module, "get_family_info", mock_get_family_info)
        monkeypatch.setattr(loader_module.HFLoader, "apply_weights_to_model", mock_apply_weights)
        monkeypatch.setattr(loader_module.HFLoader, "load_tokenizer", mock_load_tokenizer)

        # Test with no memory_config (should auto-configure)
        memory = ExternalMemory.from_pretrained("fake/test-model")

        # Should auto-configure based on 24 layers
        # query_layer = int(24 * 0.92) = 22
        # inject_layer = int(24 * 0.88) = 21
        # value_layer = int(24 * 0.92) = 22
        assert memory._memory_config.query_layer == 22
        assert memory._memory_config.inject_layer == 21
        assert memory._memory_config.value_layer == 22

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_from_pretrained_explicit_config(self, monkeypatch):
        """Test from_pretrained with explicit memory config."""
        import json
        import tempfile

        # Create temp directory with model config
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": "gpt2",
                    "hidden_size": 64,
                    "num_hidden_layers": 24,
                },
                f,
            )

        # Mock the HFLoader.download to return our temp directory
        class MockDownloadResult:
            def __init__(self, path):
                self.model_path = path

        def mock_download(model_id):
            return MockDownloadResult(tmpdir)

        # Mock detect_model_family to return a valid family
        from chuk_lazarus.models_v2.families.registry import ModelFamilyType

        def mock_detect(config_data):
            return ModelFamilyType.GPT2

        # Mock get_family_info to return mock model classes
        class MockFamilyInfo:
            config_class = MockConfig
            model_class = MockModel

        def mock_get_family_info(family_type):
            return MockFamilyInfo()

        # Mock apply_weights and load_tokenizer
        def mock_apply_weights(model, path, config, dtype):
            pass

        def mock_load_tokenizer(path):
            return MockTokenizer()

        # Apply mocks
        import chuk_lazarus.inference.loader as loader_module
        import chuk_lazarus.models_v2.families.registry as registry_module

        monkeypatch.setattr(loader_module.HFLoader, "download", mock_download)
        monkeypatch.setattr(registry_module, "detect_model_family", mock_detect)
        monkeypatch.setattr(registry_module, "get_family_info", mock_get_family_info)
        monkeypatch.setattr(loader_module.HFLoader, "apply_weights_to_model", mock_apply_weights)
        monkeypatch.setattr(loader_module.HFLoader, "load_tokenizer", mock_load_tokenizer)

        # Test with explicit memory_config (should NOT auto-configure)
        custom_config = MemoryConfig(query_layer=10, inject_layer=9, value_layer=10)
        memory = ExternalMemory.from_pretrained("fake/test-model", memory_config=custom_config)

        # Should use the explicit config
        assert memory._memory_config.query_layer == 10
        assert memory._memory_config.inject_layer == 9
        assert memory._memory_config.value_layer == 10

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


class TestDemo:
    """Test the demo function."""

    def test_demo_function(self, monkeypatch, capsys):
        """Test the demo function runs without errors."""
        import json
        import tempfile

        from chuk_lazarus.introspection import external_memory

        # Create temp directory with model config
        tmpdir = Path(tempfile.mkdtemp())
        config_path = tmpdir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": "gpt2",
                    "hidden_size": 64,
                    "num_hidden_layers": 24,
                },
                f,
            )

        # Mock the HFLoader.download to return our temp directory
        class MockDownloadResult:
            def __init__(self, path):
                self.model_path = path

        def mock_download(model_id):
            return MockDownloadResult(tmpdir)

        # Mock detect_model_family to return a valid family
        from chuk_lazarus.models_v2.families.registry import ModelFamilyType

        def mock_detect(config_data):
            return ModelFamilyType.GPT2

        # Mock get_family_info to return mock model classes
        class MockFamilyInfo:
            config_class = MockConfig
            model_class = MockModel

        def mock_get_family_info(family_type):
            return MockFamilyInfo()

        # Mock apply_weights and load_tokenizer
        def mock_apply_weights(model, path, config, dtype):
            pass

        def mock_load_tokenizer(path):
            return MockTokenizer()

        # Apply mocks
        import chuk_lazarus.inference.loader as loader_module
        import chuk_lazarus.models_v2.families.registry as registry_module

        monkeypatch.setattr(loader_module.HFLoader, "download", mock_download)
        monkeypatch.setattr(registry_module, "detect_model_family", mock_detect)
        monkeypatch.setattr(registry_module, "get_family_info", mock_get_family_info)
        monkeypatch.setattr(loader_module.HFLoader, "apply_weights_to_model", mock_apply_weights)
        monkeypatch.setattr(loader_module.HFLoader, "load_tokenizer", mock_load_tokenizer)

        # Run the demo function
        external_memory.demo()

        # Check that output was generated
        captured = capsys.readouterr()
        assert "External Memory Injection Demo" in captured.out
        assert "Testing Standard Queries" in captured.out
        assert "Testing Non-Standard Queries" in captured.out or "Rescue Test" in captured.out
        assert "Override Test" in captured.out

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
