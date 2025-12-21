"""Tests for MLX TransformerBlock."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.models.transformers.transformer_block.mlx_transformer_block import MLXTransformerBlock


class MockAttention(nn.Module):
    """Mock attention for MLX transformer block tests."""
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, cache=None):
        if attention_mask is not None:
            attention_mask = mx.expand_dims(attention_mask, axis=-1)
            attention_mask = attention_mask * mx.ones_like(hidden_states)
            hidden_states = hidden_states * attention_mask
        return hidden_states, cache

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def mock_norm_layer(hidden_size, eps=1e-6):
    return nn.Identity()


class MockConfig:
    def __init__(self, hidden_size=512, intermediate_size=1024, rms_norm_eps=1e-6, hidden_act="relu", mlp_bias=False):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias


def test_mlx_transformer_block_forward_with_mask():
    config = MockConfig(hidden_size=512)
    block = MLXTransformerBlock(config, MockAttention, mock_norm_layer)

    block.mlp = nn.Identity()
    block.input_layernorm = nn.Identity()
    block.post_attention_layernorm = nn.Identity()

    hidden_states = mx.random.uniform(shape=(2, 10, 512))
    attention_mask = mx.ones((2, 10))
    attention_mask = attention_mask.at[:, 5:].set(0)

    output, _ = block.forward(hidden_states, attention_mask)

    assert output.shape == hidden_states.shape, f"Output shape mismatch! Expected {hidden_states.shape}, got {output.shape}"

    masked_part = output[:, 5:, :]
    unmasked_part = output[:, :5, :]

    assert np.allclose(np.array(masked_part), np.zeros_like(np.array(masked_part)), atol=1e-6), "Masked part is not zero!"
    assert not np.allclose(np.array(unmasked_part), np.zeros_like(np.array(unmasked_part)), atol=1e-6), "Unmasked part is incorrectly zero!"
