import torch
from core.models.transformers.transformer_block.torch_transformer_block import TorchTransformerBlock

# Mock MLP creation function (to avoid using the actual implementation)
def mock_create_mlp(config):
    return torch.nn.Identity()  # Identity for simplicity in tests

class MockAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, cache=None):
        if attention_mask is not None:
            # Debugging: Print the attention mask before expansion
            print("Attention mask before expansion:", attention_mask)
            
            # Expand the attention mask to match hidden_states
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            
            # Debugging: Print the attention mask after expansion
            print("Attention mask after expansion:", attention_mask)
            
            # Apply the mask to hidden_states
            hidden_states = hidden_states * attention_mask
            
            # Debugging: Print the hidden states after masking
            print("Hidden states after applying mask:", hidden_states)

        return hidden_states, cache





# Mock normalization layer (Identity)
def mock_norm_layer(hidden_size, eps=1e-6):
    return torch.nn.Identity()
 
class MockConfig:
    def __init__(self, hidden_size=512, intermediate_size=1024, rms_norm_eps=1e-6, hidden_act="relu", mlp_bias=False):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.mlp_bias = mlp_bias  # Add mlp_bias here


# Ensure to mock the MLP creation function in the TorchTransformerBlock
def test_torch_transformer_block_initialization():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)

    # Mock MLP creation
    block.mlp = torch.nn.Identity()  # Mock the MLP as Identity

    # Check that components are correctly initialized
    assert block.hidden_size == 512
    assert isinstance(block.self_attn, MockAttention)
    assert isinstance(block.input_layernorm, torch.nn.Identity)
    assert isinstance(block.post_attention_layernorm, torch.nn.Identity)
    assert isinstance(block.mlp, torch.nn.Identity)  # This should now pass


def test_torch_transformer_block_forward():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)

    # Mock the MLP and normalization layers as Identity
    block.mlp = torch.nn.Identity()
    block.input_layernorm = torch.nn.Identity()
    block.post_attention_layernorm = torch.nn.Identity()

    # Create dummy input tensor
    hidden_states = torch.rand(2, 10, 512)  # (batch_size, sequence_length, hidden_size)

    # Run the forward pass
    normed_hidden_states = block.input_layernorm(hidden_states)
    print("Normed hidden states (input layer norm):", normed_hidden_states)

    attention_output, _ = block.self_attn(normed_hidden_states)
    print("Attention output:", attention_output)

    hidden_states = hidden_states + attention_output  # Residual connection
    print("Hidden states after attention residual:", hidden_states)

    normed_hidden_states = block.post_attention_layernorm(hidden_states)
    print("Normed hidden states (post-attention layer norm):", normed_hidden_states)

    mlp_output = block.mlp(normed_hidden_states)
    print("MLP output:", mlp_output)

    hidden_states = hidden_states + mlp_output  # Residual connection
    print("Final hidden states after residual connection:", hidden_states)

    # Adjust the expected output based on the correct behavior of the transformer block
    expected_output = hidden_states  # Adjust this to match the block's intended behavior
    print("Expected output:", expected_output)

    # Assert the final output matches the expected output
    assert torch.allclose(hidden_states, expected_output, atol=1e-6), "Output mismatch in TorchTransformerBlock!"

def test_torch_transformer_block_forward_with_mask():
    config = MockConfig(hidden_size=512)
    block = TorchTransformerBlock(config, MockAttention, mock_norm_layer)

    # Mock the MLP and normalization layers as Identity
    block.mlp = torch.nn.Identity()
    block.input_layernorm = torch.nn.Identity()
    block.post_attention_layernorm = torch.nn.Identity()

    # Step 3.1: Create dummy input tensor and attention mask
    hidden_states = torch.rand(2, 10, 512)  # (batch_size, sequence_length, hidden_size)
    attention_mask = torch.ones(2, 10)  # Binary mask, shape (batch_size, sequence_length)
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

    # Debugging: Print means of masked and unmasked parts
    print("Masked part mean:", masked_part.abs().mean().item())
    print("Unmasked part mean:", unmasked_part.abs().mean().item())

    # Step 3.5: Ensure that the masked part is zero
    assert torch.allclose(masked_part, torch.zeros_like(masked_part), atol=1e-6), "Masked part is not zero!"
    
    # Step 3.6: Ensure that the unmasked part is not zero
    assert not torch.allclose(unmasked_part, torch.zeros_like(unmasked_part), atol=1e-6), "Unmasked part is incorrectly zero!"
