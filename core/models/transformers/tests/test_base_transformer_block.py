from typing import Optional, Tuple
import torch
from core.models.transformers.base_transformer_block import BaseTransformerBlock

# Basic config for testing
class MockConfig:
    def __init__(self, hidden_size=512, intermediate_size=1024, rms_norm_eps=1e-6, hidden_act="relu", mlp_bias=False):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias  # Add mlp_bias here

# Mock MLP creation function (to avoid using the actual implementation)
def mock_create_mlp(config):
    return torch.nn.Identity()  # Identity for simplicity in tests

# Mock Attention layer (to simulate attention for testing)
class MockAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, cache=None):
        return hidden_states, cache  # Pass-through for testing purposes

# Mock TransformerBlock that inherits BaseTransformerBlock
class MockTransformerBlock(BaseTransformerBlock):
    def __init__(self, config):
        super().__init__(config, MockAttention, mock_norm_layer)

    def create_mlp(self, config):
        return mock_create_mlp(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # Use the logic from the BaseTransformerBlock
        normed_hidden_states = self.input_layernorm(hidden_states)
        attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)
        hidden_states = hidden_states + attention_output  # Residual connection

        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = hidden_states + mlp_output  # Residual connection

        return hidden_states, cache
    
# Mock normalization layer (Identity)
def mock_norm_layer(hidden_size, eps=1e-6):
    return torch.nn.Identity()
      
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

    # Create a dummy input tensor
    hidden_states = torch.rand(2, 10, 512)  # (batch_size, sequence_length, hidden_size)

    # Manually calculate what the output should be
    normed_hidden_states = hidden_states  # Identity normalization
    attention_output = normed_hidden_states  # Identity attention
    expected_after_attention = hidden_states + attention_output  # After residual connection

    # Next, add the MLP output (which is also identity)
    mlp_output = expected_after_attention  # Identity MLP
    expected_output = expected_after_attention + mlp_output  # After residual connection

    # Run the forward pass
    output, _ = block.forward(hidden_states)

    # Compare the manually calculated expected output with the actual output
    assert torch.allclose(output, expected_output, atol=1e-6), "Output does not match expected!"

def test_transformer_block_forward_with_mask():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    # Create a dummy input tensor and attention mask
    hidden_states = torch.rand(2, 10, 512)
    attention_mask = torch.ones(2, 10, 10)

    # Run the forward pass with the mask
    output, _ = block.forward(hidden_states, attention_mask)

    # Even though the mask doesn't change anything in the mock, we ensure the function runs
    assert output is not None
    assert output.shape == hidden_states.shape

def test_transformer_block_forward_with_cache():
    config = MockConfig(hidden_size=512)
    block = MockTransformerBlock(config)

    # Create a dummy input tensor and cache
    hidden_states = torch.rand(2, 10, 512)
    cache = (torch.rand(2, 10, 512), torch.rand(2, 10, 512))

    # Run the forward pass with the cache
    output, returned_cache = block.forward(hidden_states, cache=cache)

    # Check that the cache is returned unchanged (since we are mocking the attention)
    assert returned_cache == cache

