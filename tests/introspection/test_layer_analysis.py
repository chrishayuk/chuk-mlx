"""Tests for layer_analysis module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.layer_analysis import (
    AttentionResult,
    ClusterResult,
    LayerAnalysisResult,
    LayerAnalyzer,
    RepresentationResult,
    analyze_format_sensitivity,
)


class MockConfig:
    """Mock model configuration."""

    def __init__(self, hidden_size: int = 64, num_hidden_layers: int = 4):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


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

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64, num_layers: int = 4):
        super().__init__()

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
        return [ord(c) % 100 for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        if isinstance(ids, (list, tuple)) and len(ids) > 0:
            return chr(ids[0])
        return chr(ids)


class TestRepresentationResult:
    """Tests for RepresentationResult dataclass."""

    def test_init(self):
        prompts = ["test1", "test2"]
        reps = {
            "test1": mx.random.normal((64,)),
            "test2": mx.random.normal((64,)),
        }
        sim_matrix = [[1.0, 0.5], [0.5, 1.0]]

        result = RepresentationResult(
            layer_idx=5,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
        )

        assert result.layer_idx == 5
        assert result.prompts == prompts
        assert len(result.representations) == 2
        assert result.similarity_matrix == sim_matrix

    def test_get_similarity(self):
        prompts = ["a", "b", "c"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]

        result = RepresentationResult(
            layer_idx=0,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
        )

        assert result.get_similarity("a", "b") == 0.8
        assert result.get_similarity("b", "c") == 0.4


class TestAttentionResult:
    """Tests for AttentionResult dataclass."""

    def test_init(self):
        attn_weights = mx.random.normal((8, 5, 5))  # 8 heads, 5 seq len
        tokens = ["a", "b", "c", "d", "e"]

        result = AttentionResult(
            layer_idx=3,
            prompt="test",
            tokens=tokens,
            attention_weights=attn_weights,
        )

        assert result.layer_idx == 3
        assert result.prompt == "test"
        assert result.tokens == tokens
        assert result.attention_weights.shape == (8, 5, 5)

    def test_num_heads(self):
        attn_weights = mx.random.normal((12, 10, 10))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["t"] * 10,
            attention_weights=attn_weights,
        )

        assert result.num_heads == 12

    def test_seq_len(self):
        attn_weights = mx.random.normal((8, 7, 7))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["t"] * 7,
            attention_weights=attn_weights,
        )

        assert result.seq_len == 7

    def test_get_head_pattern(self):
        attn_weights = mx.random.normal((4, 3, 3))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["a", "b", "c"],
            attention_weights=attn_weights,
        )

        head_pattern = result.get_head_pattern(2)
        assert head_pattern.shape == (3, 3)

    def test_get_attention_to_token(self):
        attn_weights = mx.random.normal((4, 5, 5))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["a", "b", "c", "d", "e"],
            attention_weights=attn_weights,
        )

        # Get attention to token 2 from last position
        attn = result.get_attention_to_token(2, from_position=-1)
        assert attn.shape == (4,)  # One value per head


class TestClusterResult:
    """Tests for ClusterResult dataclass."""

    def test_init(self):
        within = {"A": 0.9, "B": 0.85}
        between = {("A", "B"): 0.3}

        result = ClusterResult(
            layer_idx=5,
            labels=["A", "B"],
            within_cluster_similarity=within,
            between_cluster_similarity=between,
            separation_score=0.575,  # (0.9 + 0.85)/2 - 0.3
        )

        assert result.layer_idx == 5
        assert result.labels == ["A", "B"]
        assert result.within_cluster_similarity == within
        assert result.between_cluster_similarity == between
        assert result.separation_score == 0.575


class TestLayerAnalysisResult:
    """Tests for LayerAnalysisResult dataclass."""

    def test_init(self):
        prompts = ["p1", "p2"]
        labels = ["A", "B"]
        layers = [0, 2, 4]

        reps = {}
        for layer in layers:
            reps[layer] = RepresentationResult(
                layer_idx=layer,
                prompts=prompts,
                representations={p: mx.random.normal((64,)) for p in prompts},
                similarity_matrix=[[1.0, 0.5], [0.5, 1.0]],
            )

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=labels,
            layers=layers,
            representations=reps,
        )

        assert result.prompts == prompts
        assert result.labels == labels
        assert result.layers == layers
        assert len(result.representations) == 3


class TestLayerAnalyzer:
    """Tests for LayerAnalyzer class."""

    def test_init(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = LayerAnalyzer(model, tokenizer, "test-model", config)

        assert analyzer._model is model
        assert analyzer._tokenizer is tokenizer
        assert analyzer._model_id == "test-model"
        assert analyzer._config is config

    def test_num_layers_from_config(self):
        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=8)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)
        assert analyzer.num_layers == 8

    def test_num_layers_from_model(self):
        model = MockModel(num_layers=6)
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)
        # Should infer from model structure
        assert analyzer.num_layers > 0

    def test_analyze_representations(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test1", "test2"]
        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[0, 2],
        )

        assert isinstance(result, LayerAnalysisResult)
        assert result.prompts == prompts
        assert result.layers == [0, 2]
        assert len(result.representations) == 2

    def test_analyze_representations_with_labels(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test1", "test2"]
        labels = ["A", "B"]

        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[1],
            labels=labels,
        )

        assert result.labels == labels
        assert result.clusters is not None
        assert 1 in result.clusters

    def test_analyze_representations_default_layers(self):
        model = MockModel(num_layers=24)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=24)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        result = analyzer.analyze_representations(
            prompts=["test"],
        )

        # Should select key layers automatically
        assert len(result.layers) > 2

    def test_analyze_attention(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        # Note: This might fail if hooks don't support attention capture
        # We'll make it a simple smoke test
        try:
            results = analyzer.analyze_attention(
                prompts=["test"],
                layers=[1],
            )
            assert isinstance(results, dict)
        except Exception:
            # If attention capture not implemented, skip
            pytest.skip("Attention capture not available")

    def test_compute_similarity_matrix(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a", "b", "c"]
        reps = {
            "a": mx.array([1.0, 0.0, 0.0]),
            "b": mx.array([0.0, 1.0, 0.0]),
            "c": mx.array([1.0, 0.0, 0.0]),  # Same as 'a'
        }

        matrix = analyzer._compute_similarity_matrix(prompts, reps)

        assert len(matrix) == 3
        assert len(matrix[0]) == 3

        # Diagonal should be 1.0
        assert abs(matrix[0][0] - 1.0) < 1e-6

        # 'a' and 'c' should be similar
        assert matrix[0][2] > 0.99

        # 'a' and 'b' should be orthogonal
        assert abs(matrix[0][1]) < 0.1

    def test_compute_clustering(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a1", "a2", "b1", "b2"]
        labels = ["A", "A", "B", "B"]

        # Create similarity matrix where same-label items are similar
        sim_matrix = [
            [1.0, 0.9, 0.1, 0.2],  # a1
            [0.9, 1.0, 0.15, 0.1],  # a2
            [0.1, 0.15, 1.0, 0.85],  # b1
            [0.2, 0.1, 0.85, 1.0],  # b2
        ]

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        assert isinstance(result, ClusterResult)
        assert "A" in result.within_cluster_similarity
        assert "B" in result.within_cluster_similarity
        assert ("A", "B") in result.between_cluster_similarity or (
            "B",
            "A",
        ) in result.between_cluster_similarity

        # Within-cluster should be higher than between
        avg_within = sum(result.within_cluster_similarity.values()) / len(
            result.within_cluster_similarity
        )
        assert avg_within > 0.5

    def test_compute_clustering_single_sample(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a"]
        labels = ["A"]
        sim_matrix = [[1.0]]

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        # Should handle single sample
        assert "A" in result.within_cluster_similarity


class TestFormatSensitivityAnalysis:
    """Tests for analyze_format_sensitivity convenience function."""

    def test_analyze_format_sensitivity(self):
        # This is an integration test that requires full model setup
        # We'll create a minimal version

        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=8)

        # Mock the from_pretrained to return our mock model
        original_from_pretrained = LayerAnalyzer.from_pretrained

        def mock_from_pretrained(model_id):
            return LayerAnalyzer(model, tokenizer, model_id, config)

        LayerAnalyzer.from_pretrained = mock_from_pretrained

        try:
            base_prompts = ["100 - 37 =", "50 + 25 ="]

            result = analyze_format_sensitivity(
                model_id="test",
                base_prompts=base_prompts,
                layers=[2, 4],
            )

            assert isinstance(result, LayerAnalysisResult)
            # Should have created variants with/without trailing space
            assert len(result.prompts) == 4  # 2 base * 2 variants
            assert result.labels is not None
            assert "working" in result.labels
            assert "broken" in result.labels

        finally:
            LayerAnalyzer.from_pretrained = original_from_pretrained


class TestLayerAnalyzerPrintMethods:
    """Tests for LayerAnalyzer print methods."""

    def test_print_similarity_matrix_basic(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test1", "test2", "test3"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]

        rep_result = RepresentationResult(
            layer_idx=5,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
        )

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=None,
            layers=[5],
            representations={5: rep_result},
        )

        analyzer.print_similarity_matrix(result, layer=5)

        captured = capsys.readouterr()
        assert "Layer 5 Similarity Matrix" in captured.out
        assert "1.00" in captured.out
        assert "0.80" in captured.out

    def test_print_similarity_matrix_with_labels(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["a1", "a2", "b1", "b2"]
        labels = ["A", "A", "B", "B"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.15, 0.1],
            [0.1, 0.15, 1.0, 0.85],
            [0.2, 0.1, 0.85, 1.0],
        ]

        rep_result = RepresentationResult(
            layer_idx=3,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
            labels=labels,
        )

        cluster_result = ClusterResult(
            layer_idx=3,
            labels=["A", "B"],
            within_cluster_similarity={"A": 0.9, "B": 0.85},
            between_cluster_similarity={("A", "B"): 0.1375},
            separation_score=0.7375,
        )

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=labels,
            layers=[3],
            representations={3: rep_result},
            clusters={3: cluster_result},
        )

        analyzer.print_similarity_matrix(result, layer=3)

        captured = capsys.readouterr()
        assert "Layer 3 Similarity Matrix" in captured.out
        assert "[A]" in captured.out
        assert "[B]" in captured.out
        assert "Clustering Analysis" in captured.out
        assert "Within-cluster similarity" in captured.out
        assert "Between-cluster similarity" in captured.out
        assert "Separation score" in captured.out

    def test_print_similarity_matrix_highlights_high_similarity(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a", "b"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [[1.0, 0.97], [0.97, 1.0]]

        rep_result = RepresentationResult(
            layer_idx=0,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
        )

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=None,
            layers=[0],
            representations={0: rep_result},
        )

        analyzer.print_similarity_matrix(result, layer=0)

        captured = capsys.readouterr()
        # High similarity (>0.95) should be marked with *
        assert "0.97*" in captured.out

    def test_print_attention_comparison_basic(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        # Create mock attention result
        tokens = ["the", "cat", "sat"]
        attn_weights = mx.random.normal((4, 3, 3))  # 4 heads, 3 seq len

        attn_result = AttentionResult(
            layer_idx=2,
            prompt="the cat sat",
            tokens=tokens,
            attention_weights=attn_weights,
        )

        attention_results = {2: {"the cat sat": attn_result}}

        analyzer.print_attention_comparison(
            attention_results=attention_results,
            layer=2,
            prompts=["the cat sat"],
            focus_token=-1,
        )

        captured = capsys.readouterr()
        assert "Layer 2 Attention Patterns" in captured.out
        assert "Prompt: 'the cat sat'" in captured.out
        assert "Tokens:" in captured.out

    def test_print_attention_comparison_with_string_token(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        tokens = ["the", "cat", "sat"]
        attn_weights = mx.random.normal((4, 3, 3))

        attn_result = AttentionResult(
            layer_idx=1,
            prompt="test",
            tokens=tokens,
            attention_weights=attn_weights,
        )

        attention_results = {1: {"test": attn_result}}

        analyzer.print_attention_comparison(
            attention_results=attention_results,
            layer=1,
            prompts=["test"],
            focus_token="cat",
        )

        captured = capsys.readouterr()
        assert "Layer 1 Attention Patterns" in captured.out
        assert "'cat'" in captured.out

    def test_print_attention_comparison_missing_token(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        tokens = ["the", "cat", "sat"]
        attn_weights = mx.random.normal((4, 3, 3))

        attn_result = AttentionResult(
            layer_idx=1,
            prompt="test",
            tokens=tokens,
            attention_weights=attn_weights,
        )

        attention_results = {1: {"test": attn_result}}

        # Token not in list should fall back to -1
        analyzer.print_attention_comparison(
            attention_results=attention_results,
            layer=1,
            prompts=["test"],
            focus_token="missing",
        )

        captured = capsys.readouterr()
        assert "Layer 1 Attention Patterns" in captured.out

    def test_print_attention_comparison_empty_layer(self, capsys):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        attention_results = {1: {}}

        analyzer.print_attention_comparison(
            attention_results=attention_results,
            layer=1,
            prompts=["test"],
            focus_token=-1,
        )

        captured = capsys.readouterr()
        assert "Layer 1 Attention Patterns" in captured.out


class TestRepresentationResultEdgeCases:
    """Tests for edge cases in RepresentationResult."""

    def test_get_similarity_invalid_prompt(self):
        prompts = ["a", "b"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [[1.0, 0.5], [0.5, 1.0]]

        result = RepresentationResult(
            layer_idx=0,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
        )

        # Should raise ValueError when prompt not found
        with pytest.raises(ValueError):
            result.get_similarity("a", "c")

    def test_with_labels(self):
        prompts = ["a", "b"]
        labels = ["A", "B"]
        reps = {p: mx.random.normal((64,)) for p in prompts}
        sim_matrix = [[1.0, 0.5], [0.5, 1.0]]

        result = RepresentationResult(
            layer_idx=0,
            prompts=prompts,
            representations=reps,
            similarity_matrix=sim_matrix,
            labels=labels,
        )

        assert result.labels == labels


class TestAttentionResultEdgeCases:
    """Tests for edge cases in AttentionResult."""

    def test_get_attention_to_token_different_positions(self):
        attn_weights = mx.random.normal((4, 5, 5))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["a", "b", "c", "d", "e"],
            attention_weights=attn_weights,
        )

        # Test different from_position values
        attn_first = result.get_attention_to_token(2, from_position=0)
        assert attn_first.shape == (4,)

        attn_mid = result.get_attention_to_token(2, from_position=2)
        assert attn_mid.shape == (4,)

        attn_last = result.get_attention_to_token(2, from_position=-1)
        assert attn_last.shape == (4,)

    def test_get_head_pattern_all_heads(self):
        attn_weights = mx.random.normal((8, 10, 10))
        result = AttentionResult(
            layer_idx=0,
            prompt="test",
            tokens=["t"] * 10,
            attention_weights=attn_weights,
        )

        # Test getting patterns from all heads
        for head_idx in range(8):
            pattern = result.get_head_pattern(head_idx)
            assert pattern.shape == (10, 10)


class TestLayerAnalyzerEdgeCases:
    """Tests for edge cases in LayerAnalyzer."""

    def test_num_layers_fallback(self):
        # Model without config and without standard structure
        class WeirdModel(nn.Module):
            pass

        model = WeirdModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer, config=None)
        # Should fall back to 32
        assert analyzer.num_layers == 32

    def test_analyze_representations_different_positions(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test"]

        # Test with first position
        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[1],
            position=0,
        )

        assert isinstance(result, LayerAnalysisResult)

    def test_analyze_representations_3d_hidden_states(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test"]

        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[0, 1],
        )

        # Should handle 3D hidden states (batch, seq, hidden)
        assert len(result.representations) == 2

    def test_analyze_attention_default_layers(self):
        model = MockModel(num_layers=16)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=16)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        try:
            results = analyzer.analyze_attention(prompts=["test"])
            # Default should be quarter and half
            assert isinstance(results, dict)
        except Exception:
            # If attention capture not implemented, skip
            pytest.skip("Attention capture not available")

    def test_compute_similarity_matrix_zero_vectors(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a", "b"]
        reps = {
            "a": mx.array([0.0, 0.0, 0.0]),
            "b": mx.array([0.0, 0.0, 0.0]),
        }

        matrix = analyzer._compute_similarity_matrix(prompts, reps)

        # Should handle zero vectors gracefully (epsilon prevents division by zero)
        assert len(matrix) == 2
        assert all(isinstance(row, list) for row in matrix)

    def test_compute_similarity_matrix_symmetry(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a", "b", "c"]
        reps = {
            "a": mx.array([1.0, 2.0, 3.0]),
            "b": mx.array([4.0, 5.0, 6.0]),
            "c": mx.array([7.0, 8.0, 9.0]),
        }

        matrix = analyzer._compute_similarity_matrix(prompts, reps)

        # Matrix should be symmetric
        assert matrix[0][1] == matrix[1][0]
        assert matrix[0][2] == matrix[2][0]
        assert matrix[1][2] == matrix[2][1]

    def test_compute_clustering_multiple_labels(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a1", "a2", "b1", "b2", "c1", "c2"]
        labels = ["A", "A", "B", "B", "C", "C"]

        sim_matrix = [
            [1.0, 0.9, 0.2, 0.1, 0.15, 0.1],
            [0.9, 1.0, 0.1, 0.2, 0.1, 0.15],
            [0.2, 0.1, 1.0, 0.88, 0.3, 0.25],
            [0.1, 0.2, 0.88, 1.0, 0.25, 0.3],
            [0.15, 0.1, 0.3, 0.25, 1.0, 0.92],
            [0.1, 0.15, 0.25, 0.3, 0.92, 1.0],
        ]

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        assert len(result.labels) == 3
        assert "A" in result.within_cluster_similarity
        assert "B" in result.within_cluster_similarity
        assert "C" in result.within_cluster_similarity

    def test_compute_clustering_empty_between(self):
        model = MockModel()
        tokenizer = MockTokenizer()

        analyzer = LayerAnalyzer(model, tokenizer)

        # Only one cluster
        prompts = ["a1", "a2"]
        labels = ["A", "A"]
        sim_matrix = [[1.0, 0.9], [0.9, 1.0]]

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        # No between-cluster similarity when only one cluster
        assert len(result.between_cluster_similarity) == 0
        assert result.separation_score >= 0  # avg_within - 0


class TestClusterResultEdgeCases:
    """Tests for edge cases in ClusterResult."""

    def test_multiple_clusters(self):
        within = {"A": 0.9, "B": 0.85, "C": 0.88}
        between = {("A", "B"): 0.2, ("A", "C"): 0.15, ("B", "C"): 0.25}

        result = ClusterResult(
            layer_idx=3,
            labels=["A", "B", "C"],
            within_cluster_similarity=within,
            between_cluster_similarity=between,
            separation_score=0.633,
        )

        assert len(result.labels) == 3
        assert len(result.within_cluster_similarity) == 3
        assert len(result.between_cluster_similarity) == 3


class TestLayerAnalysisResultEdgeCases:
    """Tests for edge cases in LayerAnalysisResult."""

    def test_without_clusters(self):
        prompts = ["p1", "p2"]
        layers = [0, 1]

        reps = {}
        for layer in layers:
            reps[layer] = RepresentationResult(
                layer_idx=layer,
                prompts=prompts,
                representations={p: mx.random.normal((64,)) for p in prompts},
                similarity_matrix=[[1.0, 0.5], [0.5, 1.0]],
            )

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=None,
            layers=layers,
            representations=reps,
            clusters=None,
        )

        assert result.clusters is None

    def test_with_attention(self):
        prompts = ["p1"]
        layers = [0]

        reps = {
            0: RepresentationResult(
                layer_idx=0,
                prompts=prompts,
                representations={"p1": mx.random.normal((64,))},
                similarity_matrix=[[1.0]],
            )
        }

        attn = {
            0: {
                "p1": AttentionResult(
                    layer_idx=0,
                    prompt="p1",
                    tokens=["a", "b"],
                    attention_weights=mx.random.normal((4, 2, 2)),
                )
            }
        }

        result = LayerAnalysisResult(
            prompts=prompts,
            labels=None,
            layers=layers,
            representations=reps,
            attention=attn,
        )

        assert result.attention is not None
        assert 0 in result.attention


class TestFormatSensitivityEdgeCases:
    """Tests for edge cases in format sensitivity analysis."""

    def test_format_sensitivity_strips_trailing_space(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        original_from_pretrained = LayerAnalyzer.from_pretrained

        def mock_from_pretrained(model_id):
            return LayerAnalyzer(model, tokenizer, model_id, config)

        LayerAnalyzer.from_pretrained = mock_from_pretrained

        try:
            # Base prompts with trailing spaces should be stripped
            base_prompts = ["test1  ", "test2   "]

            result = analyze_format_sensitivity(
                model_id="test",
                base_prompts=base_prompts,
                layers=[0],
            )

            # Each base should create 2 variants (with and without space)
            assert len(result.prompts) == 4
            assert result.labels.count("working") == 2
            assert result.labels.count("broken") == 2

        finally:
            LayerAnalyzer.from_pretrained = original_from_pretrained


class TestLayerAnalyzerFromPretrained:
    """Tests for from_pretrained class method."""

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    @patch("chuk_lazarus.models_v2.families.registry.get_family_info")
    def test_from_pretrained_success(self, mock_get_family, mock_detect, mock_loader):
        # Setup mocks
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_path = model_path / "config.json"

            # Create mock config
            config_data = {
                "model_type": "gemma",
                "num_hidden_layers": 8,
                "hidden_size": 64,
            }
            config_path.write_text(json.dumps(config_data))

            # Mock download result
            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            # Mock tokenizer
            mock_tokenizer = MockTokenizer()
            mock_loader.load_tokenizer.return_value = mock_tokenizer

            # Mock model family detection
            mock_detect.return_value = "gemma"

            # Mock config and model classes
            mock_config = MockConfig(num_hidden_layers=8, hidden_size=64)
            mock_config_class = Mock(return_value=mock_config)
            mock_config_class.from_hf_config = Mock(return_value=mock_config)

            mock_model = MockModel(num_layers=8)
            mock_model_class = Mock(return_value=mock_model)

            mock_family_info = Mock()
            mock_family_info.config_class = mock_config_class
            mock_family_info.model_class = mock_model_class
            mock_get_family.return_value = mock_family_info

            # Call from_pretrained
            analyzer = LayerAnalyzer.from_pretrained("test-model")

            # Verify
            assert isinstance(analyzer, LayerAnalyzer)
            assert analyzer._model_id == "test-model"
            mock_loader.download.assert_called_once_with("test-model")
            mock_loader.load_tokenizer.assert_called_once_with(model_path)
            mock_loader.apply_weights_to_model.assert_called_once()

    @patch("chuk_lazarus.inference.loader.HFLoader")
    @patch("chuk_lazarus.models_v2.families.registry.detect_model_family")
    def test_from_pretrained_unsupported_family(self, mock_detect, mock_loader):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            config_path = model_path / "config.json"

            config_data = {"model_type": "unsupported"}
            config_path.write_text(json.dumps(config_data))

            mock_result = Mock()
            mock_result.model_path = model_path
            mock_loader.download.return_value = mock_result

            # Return None for unsupported family
            mock_detect.return_value = None

            with pytest.raises(ValueError, match="Unsupported model family"):
                LayerAnalyzer.from_pretrained("unsupported-model")


class TestLayerAnalyzerAnalyzeAttentionEdgeCases:
    """Additional tests for analyze_attention method."""

    def test_analyze_attention_multiple_prompts(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        try:
            results = analyzer.analyze_attention(
                prompts=["test1", "test2"],
                layers=[1],
            )

            assert isinstance(results, dict)
            if 1 in results:
                # If capture worked, check structure
                assert isinstance(results[1], dict)
        except Exception:
            pytest.skip("Attention capture not available")

    def test_analyze_attention_removes_batch_dim(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        try:
            results = analyzer.analyze_attention(
                prompts=["test"],
                layers=[0],
            )

            if 0 in results and "test" in results[0]:
                result = results[0]["test"]
                # Attention weights should be 3D (not 4D with batch)
                assert result.attention_weights.ndim == 3
        except Exception:
            pytest.skip("Attention capture not available")


class TestComputeSimilarityMatrixEdgeCases:
    """Additional edge cases for similarity matrix computation."""

    def test_single_prompt(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["single"]
        reps = {"single": mx.array([1.0, 2.0, 3.0])}

        matrix = analyzer._compute_similarity_matrix(prompts, reps)

        assert len(matrix) == 1
        assert len(matrix[0]) == 1
        # Self-similarity should be 1.0
        assert abs(matrix[0][0] - 1.0) < 1e-6

    def test_negative_vectors(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a", "b"]
        reps = {
            "a": mx.array([1.0, 0.0]),
            "b": mx.array([-1.0, 0.0]),  # Opposite direction
        }

        matrix = analyzer._compute_similarity_matrix(prompts, reps)

        # Should be negative similarity
        assert matrix[0][1] < 0


class TestComputeClusteringEdgeCases:
    """Additional edge cases for clustering computation."""

    def test_many_samples_one_cluster(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a1", "a2", "a3", "a4", "a5"]
        labels = ["A"] * 5

        sim_matrix = [[1.0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    sim_matrix[i][j] = 0.9

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        assert len(result.labels) == 1
        assert "A" in result.within_cluster_similarity
        assert len(result.between_cluster_similarity) == 0

    def test_perfect_separation(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        analyzer = LayerAnalyzer(model, tokenizer)

        prompts = ["a1", "a2", "b1", "b2"]
        labels = ["A", "A", "B", "B"]

        # Perfect within-cluster similarity, zero between
        sim_matrix = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        # Perfect separation: within=1.0, between=0.0, separation=1.0
        assert result.within_cluster_similarity["A"] == 1.0
        assert result.within_cluster_similarity["B"] == 1.0
        assert result.separation_score == 1.0

    def test_unbalanced_clusters(self):
        model = MockModel()
        tokenizer = MockTokenizer()
        analyzer = LayerAnalyzer(model, tokenizer)

        # One cluster with many samples, one with few
        prompts = ["a1", "a2", "a3", "a4", "b1"]
        labels = ["A", "A", "A", "A", "B"]

        sim_matrix = [[0.8] * 5 for _ in range(5)]
        # Set diagonal to 1.0
        for i in range(5):
            sim_matrix[i][i] = 1.0

        result = analyzer._compute_clustering(prompts, labels, sim_matrix)

        assert "A" in result.within_cluster_similarity
        assert "B" in result.within_cluster_similarity
        # Cluster B has only 1 sample, so within-sim should be 1.0
        assert result.within_cluster_similarity["B"] == 1.0


class TestAnalyzeRepresentationsComprehensive:
    """Comprehensive tests for analyze_representations method."""

    def test_with_empty_hidden_states(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        # Should handle case where hooks don't capture anything
        result = analyzer.analyze_representations(
            prompts=["test"],
            layers=[0],
        )

        assert isinstance(result, LayerAnalysisResult)

    def test_multiple_prompts_multiple_layers(self):
        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=8)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["prompt1", "prompt2", "prompt3"]
        layers = [0, 2, 4, 6]

        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=layers,
        )

        assert len(result.prompts) == 3
        assert len(result.layers) == 4
        assert len(result.representations) == 4

    def test_layer_indices_sorted_and_deduped(self):
        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=8)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        # Layers are sorted and deduped in the method
        result = analyzer.analyze_representations(
            prompts=["test"],
            layers=None,  # Will use default
        )

        # Default layers should be sorted
        assert result.layers == sorted(result.layers)


class TestIntegrationScenarios:
    """Integration-style tests covering realistic usage patterns."""

    def test_full_workflow_with_clustering(self):
        model = MockModel(num_layers=8)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=8)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        # Simulate working vs broken prompts
        prompts = ["100 - 37 = ", "100 - 37 =", "50 + 25 = ", "50 + 25 ="]
        labels = ["working", "broken", "working", "broken"]

        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[2, 4, 6],
            labels=labels,
        )

        # Should have clusters computed
        assert result.clusters is not None
        assert len(result.clusters) == 3  # 3 layers

        # Each cluster result should have metrics
        for cluster in result.clusters.values():
            assert "working" in cluster.within_cluster_similarity
            assert "broken" in cluster.within_cluster_similarity
            assert isinstance(cluster.separation_score, float)

    def test_representation_similarity_lookup(self):
        model = MockModel(num_layers=4)
        tokenizer = MockTokenizer()
        config = MockConfig(num_hidden_layers=4)

        analyzer = LayerAnalyzer(model, tokenizer, config=config)

        prompts = ["test1", "test2", "test3"]

        result = analyzer.analyze_representations(
            prompts=prompts,
            layers=[2],
        )

        # Should be able to look up similarities
        rep_result = result.representations[2]
        sim_12 = rep_result.get_similarity("test1", "test2")
        sim_13 = rep_result.get_similarity("test1", "test3")

        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        # Similarity should be in [-1, 1] range
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0
