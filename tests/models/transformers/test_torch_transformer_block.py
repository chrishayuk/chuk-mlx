"""Tests for Torch TransformerBlock."""

import torch

from chuk_lazarus.models.transformers.transformer_block.torch_transformer_block import (
    TorchTransformerBlock,
)


def mock_create_mlp(config):
    return torch.nn.Identity()


class MockAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, cache=None):
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * attention_mask
        return hidden_states, cache


def mock_norm_layer(hidden_size, eps=1e-6):
    return torch.nn.Identity()


class MockConfig:
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


def test_torch_transformer_block_initialization():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)
    block.mlp = torch.nn.Identity()

    assert block.hidden_size == 512
    assert isinstance(block.self_attn, MockAttention)
    assert isinstance(block.input_layernorm, torch.nn.Identity)
    assert isinstance(block.post_attention_layernorm, torch.nn.Identity)
    assert isinstance(block.mlp, torch.nn.Identity)


def test_torch_transformer_block_forward():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)

    block.mlp = torch.nn.Identity()
    block.input_layernorm = torch.nn.Identity()
    block.post_attention_layernorm = torch.nn.Identity()

    hidden_states = torch.rand(2, 10, 512)

    normed_hidden_states = block.input_layernorm(hidden_states)
    attention_output, _ = block.self_attn(normed_hidden_states)
    hidden_states = hidden_states + attention_output

    normed_hidden_states = block.post_attention_layernorm(hidden_states)
    mlp_output = block.mlp(normed_hidden_states)
    hidden_states = hidden_states + mlp_output

    expected_output = hidden_states

    assert torch.allclose(hidden_states, expected_output, atol=1e-6), (
        "Output mismatch in TorchTransformerBlock!"
    )


def test_torch_transformer_block_forward_with_mask():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)

    block.mlp = torch.nn.Identity()
    block.input_layernorm = torch.nn.Identity()
    block.post_attention_layernorm = torch.nn.Identity()

    hidden_states = torch.rand(2, 10, 512)
    attention_mask = torch.ones(2, 10)
    attention_mask[:, 5:] = 0

    output, _ = block.forward(hidden_states, attention_mask)

    assert output.shape == hidden_states.shape, (
        f"Output shape mismatch! Expected {hidden_states.shape}, got {output.shape}"
    )

    masked_part = output[:, 5:, :]
    unmasked_part = output[:, :5, :]

    assert torch.allclose(masked_part, torch.zeros_like(masked_part), atol=1e-6), (
        "Masked part is not zero!"
    )
    assert not torch.allclose(unmasked_part, torch.zeros_like(unmasked_part), atol=1e-6), (
        "Unmasked part is incorrectly zero!"
    )
