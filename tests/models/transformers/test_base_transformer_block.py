"""Tests for BaseTransformerBlock."""

import torch

from chuk_lazarus.models.transformers.transformer_block.base_transformer_block import (
    BaseTransformerBlock,
)


class MockConfig:
    """Basic config for testing."""

    def __init__(
        self,
        hidden_size=512,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="relu",
        mlp_bias=False,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias


def mock_create_mlp(config):
    """Mock MLP creation function."""
    return torch.nn.Identity()


class MockAttention(torch.nn.Module):
    """Mock Attention layer."""

    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, cache=None):
        return hidden_states, cache


def mock_norm_layer(hidden_size, eps=1e-6):
    """Mock normalization layer."""
    return torch.nn.Identity()


class MockTransformerBlock(BaseTransformerBlock):
    """Mock TransformerBlock for testing."""

    def __init__(self, config):
        super().__init__(config, MockAttention, mock_norm_layer)

    def create_mlp(self, config):
        return mock_create_mlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        normed_hidden_states = self.input_layernorm(hidden_states)
        attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)
        hidden_states = hidden_states + attention_output

        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states, cache


def test_transformer_block_initialization():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    assert block.hidden_size == 512
    assert isinstance(block.self_attn, MockAttention)
    assert isinstance(block.input_layernorm, torch.nn.Identity)
    assert isinstance(block.post_attention_layernorm, torch.nn.Identity)
    assert isinstance(block.mlp, torch.nn.Identity)


def test_transformer_block_forward():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    hidden_states = torch.rand(2, 10, 512)

    normed_hidden_states = hidden_states
    attention_output = normed_hidden_states
    expected_after_attention = hidden_states + attention_output

    mlp_output = expected_after_attention
    expected_output = expected_after_attention + mlp_output

    output, _ = block.forward(hidden_states)

    assert torch.allclose(output, expected_output, atol=1e-6), "Output does not match expected!"


def test_transformer_block_forward_with_mask():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    hidden_states = torch.rand(2, 10, 512)
    attention_mask = torch.ones(2, 10, 10)

    output, _ = block.forward(hidden_states, attention_mask)

    assert output is not None
    assert output.shape == hidden_states.shape


def test_transformer_block_forward_with_cache():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    hidden_states = torch.rand(2, 10, 512)
    cache = (torch.rand(2, 10, 512), torch.rand(2, 10, 512))

    output, returned_cache = block.forward(hidden_states, cache=cache)

    assert returned_cache == cache
