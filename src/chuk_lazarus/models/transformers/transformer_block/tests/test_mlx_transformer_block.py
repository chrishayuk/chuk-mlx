import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.models.transformers.transformer_block.mlx_transformer_block import MLXTransformerBlock

# Mock attention and norm layers for the MLXTransformerBlock
class MockAttention(nn.Module):
    def __init__(self, config):
        super().__init__()  # Properly initialize nn.Module
        # Additional initialization if needed

    def forward(self, hidden_states, attention_mask=None, cache=None):
        if attention_mask is not None:
            # Expand the attention_mask to match the shape of hidden_states
            attention_mask = mx.expand_dims(attention_mask, axis=-1)  # Shape: (batch_size, sequence_length, 1)
            # Use broadcasting to align with hidden_states
            attention_mask = attention_mask * mx.ones_like(hidden_states)  # Shape: (batch_size, sequence_length, hidden_size)
            
            # Apply the mask by element-wise multiplication
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
    # Configuration for MLXTransformerBlock
    config = MockConfig(hidden_size=512)
    block = MLXTransformerBlock(config, MockAttention, mock_norm_layer)

    # Mock the MLP and normalization layers as Identity
    block.mlp = nn.Identity()
    block.input_layernorm = nn.Identity()
    block.post_attention_layernorm = nn.Identity()

    # Step 3.1: Create dummy input tensor and attention mask
    hidden_states = mx.random.uniform(shape=(2, 10, 512))  # (batch_size, sequence_length, hidden_size)
    attention_mask = mx.ones((2, 10))  # Binary mask, shape (batch_size, sequence_length)
    attention_mask[:, 5:] = 0  # Mask out the latter part of the sequence

    # Debugging: Print initial hidden states and attention mask
    print("Initial hidden_states:", hidden_states)
    print("Attention mask shape:", attention_mask.shape)
    print("Attention mask:", attention_mask)

    # Step 3.2: Run the forward pass with the attention mask
    output, _ = block.forward(hidden_states, attention_mask)

    # Step 3.3: Check if the output has the same shape as the input
    assert output.shape == hidden_states.shape, f"Output shape mismatch! Expected {hidden_states.shape}, got {output.shape}"

    # Step 3.4: Check if the masked part of the output is zero
    masked_part = output[:, 5:, :]
    unmasked_part = output[:, :5, :]

    # Debugging: Print masked and unmasked parts
    print("Masked part of the output:", masked_part)
    print("Unmasked part of the output:", unmasked_part)

    # Step 3.5: Ensure that the masked part is zero
    assert np.allclose(masked_part, np.zeros_like(masked_part), atol=1e-6), "Masked part is not zero!"

    # Step 3.6: Ensure that the unmasked part is not zero
    assert not np.allclose(unmasked_part, np.zeros_like(unmasked_part), atol=1e-6), "Unmasked part is incorrectly zero!"
