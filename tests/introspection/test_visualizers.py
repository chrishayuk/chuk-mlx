"""Tests for introspection visualizers."""

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection import CaptureConfig, CapturedState, ModelHooks
from chuk_lazarus.introspection.attention import AttentionPattern
from chuk_lazarus.introspection.logit_lens import LogitLens
from chuk_lazarus.introspection.visualizers import (
    render_attention_heatmap,
    render_logit_evolution,
)
from chuk_lazarus.introspection.visualizers.attention_heatmap import (
    render_attention_summary,
)
from chuk_lazarus.introspection.visualizers.logit_evolution import render_logit_table


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SimpleMLP(hidden_size)

    def __call__(self, x: mx.array, mask=None, cache=None) -> tuple[mx.array, None]:
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x, None


class SimpleTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [SimpleTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)


class SimpleForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_size: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.model = SimpleTransformerModel(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        h = self.model(input_ids)
        return self.lm_head(h)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i % 26) + 65) for i in ids)


class TestRenderAttentionHeatmap:
    """Tests for attention heatmap rendering."""

    @pytest.fixture
    def sample_pattern(self):
        """Create a sample attention pattern."""
        # [batch, heads, seq, seq]
        weights = mx.random.uniform(shape=(1, 4, 5, 5))
        return AttentionPattern(
            layer_idx=0,
            weights=weights,
            tokens=["The", " cat", " sat", " on", " mat"],
        )

    def test_render_basic(self, sample_pattern):
        html = render_attention_heatmap(sample_pattern)
        assert "<!DOCTYPE html>" in html
        assert "Attention Heatmap" in html
        assert "Layer 0" in html

    def test_render_with_title(self, sample_pattern):
        html = render_attention_heatmap(sample_pattern, title="Custom Title")
        assert "Custom Title" in html

    def test_render_specific_head(self, sample_pattern):
        html = render_attention_heatmap(sample_pattern, head_idx=2)
        assert "Head 2" in html

    def test_render_no_aggregate(self, sample_pattern):
        html = render_attention_heatmap(sample_pattern, aggregate=False)
        assert "Head 0" in html

    def test_render_to_file(self, sample_pattern):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = Path(f.name)

        try:
            render_attention_heatmap(sample_pattern, output_path=output_path)
            assert output_path.exists()
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content
        finally:
            output_path.unlink()

    def test_render_custom_size(self, sample_pattern):
        html = render_attention_heatmap(sample_pattern, width=1200, height=800)
        assert "1300" in html  # width + 100 for container
        assert html  # Should not fail

    def test_render_with_special_chars_in_tokens(self):
        weights = mx.random.uniform(shape=(1, 2, 3, 3))
        pattern = AttentionPattern(
            layer_idx=0,
            weights=weights,
            tokens=["<bos>", "hello's", "world"],
        )
        html = render_attention_heatmap(pattern)
        assert html  # Should handle special characters


class TestRenderAttentionSummary:
    """Tests for attention summary rendering."""

    @pytest.fixture
    def sample_pattern(self):
        weights = mx.zeros((1, 4, 5, 5))
        # Make position 4 attend to position 1
        weights = weights.at[:, :, 4, 1].add(0.5)
        weights = weights.at[:, :, 4, 2].add(0.3)
        weights = weights.at[:, :, 4, 3].add(0.2)
        return AttentionPattern(
            layer_idx=0,
            weights=weights,
            tokens=["The", " cat", " sat", " on", " mat"],
        )

    def test_summary_last_position(self, sample_pattern):
        summary = render_attention_summary(sample_pattern, position=-1)
        assert "position 4" in summary
        assert "mat" in summary

    def test_summary_specific_position(self, sample_pattern):
        summary = render_attention_summary(sample_pattern, position=2)
        assert "position 2" in summary

    def test_summary_top_k(self, sample_pattern):
        summary = render_attention_summary(sample_pattern, position=-1, top_k=3)
        lines = summary.split("\n")
        # Header + separator + 3 items
        data_lines = [line for line in lines if line.strip().startswith("0.")]
        assert len(data_lines) <= 3


class TestRenderLogitEvolution:
    """Tests for logit evolution rendering."""

    @pytest.fixture
    def model_with_lens(self):
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers="all", positions="all"))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)
        lens = LogitLens(hooks, tokenizer)
        return lens

    def test_render_basic(self, model_with_lens):
        html = render_logit_evolution(model_with_lens)
        assert "<!DOCTYPE html>" in html
        assert "Logit Lens" in html

    def test_render_with_tokens(self, model_with_lens):
        # Use token IDs instead of strings (MockTokenizer limitation)
        html = render_logit_evolution(model_with_lens, tokens_to_track=None)
        assert html

    def test_render_with_title(self, model_with_lens):
        html = render_logit_evolution(model_with_lens, title="My Analysis")
        assert "My Analysis" in html

    def test_render_to_file(self, model_with_lens):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = Path(f.name)

        try:
            render_logit_evolution(model_with_lens, output_path=output_path)
            assert output_path.exists()
        finally:
            output_path.unlink()

    def test_render_empty_evolutions(self):
        # Create empty lens
        hooks = ModelHooks.__new__(ModelHooks)
        hooks.state = CapturedState()
        lens = LogitLens(hooks, tokenizer=None)

        html = render_logit_evolution(lens)
        assert "No token evolutions to display" in html


class TestRenderLogitTable:
    """Tests for logit table rendering."""

    @pytest.fixture
    def model_with_lens(self):
        model = SimpleForCausalLM(vocab_size=100, hidden_size=64, num_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        hooks = ModelHooks(model)
        hooks.configure(CaptureConfig(layers="all", positions="all"))
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        hooks.forward(input_ids)
        lens = LogitLens(hooks, tokenizer)
        return lens

    def test_table_basic(self, model_with_lens):
        table = render_logit_table(model_with_lens)
        assert "Logit Lens" in table
        assert "Layer" in table

    def test_table_top_k(self, model_with_lens):
        table = render_logit_table(model_with_lens, top_k=3)
        assert table

    def test_table_position(self, model_with_lens):
        table = render_logit_table(model_with_lens, position=0)
        assert table

    def test_table_empty_lens(self):
        hooks = ModelHooks.__new__(ModelHooks)
        hooks.state = CapturedState()
        lens = LogitLens(hooks, tokenizer=None)

        table = render_logit_table(lens)
        assert "No predictions captured" in table
