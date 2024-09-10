import pytest
import mlx.core as mx
from core.models.chuk_loss_function import chukloss

class DummyModel:
    def __call__(self, inputs, attention_mask=None):
        # Simulate a model output
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        vocab_size = 10  # Assuming vocab size of 10

        # Generate random logits, using uniform distribution to simulate continuous values
        logits = mx.random.uniform(low=-1.0, high=1.0, shape=(batch_size, seq_len, vocab_size))
        return logits, None


@pytest.fixture
def dummy_data():
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    inputs = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Use a simple attention mask where all tokens are considered valid
    attention_mask = mx.ones((batch_size, seq_len))

    return inputs, targets, attention_mask


def test_chukloss_valid_tokens(dummy_data):
    model = DummyModel()
    inputs, targets, attention_mask = dummy_data  # Unpack the three values

    loss, ntoks = chukloss(model, inputs, targets, attention_mask)

    # Assert that the loss is an mx.array
    assert isinstance(loss, mx.array)

    # Assert that the loss is a scalar (i.e., it has no dimensions)
    assert loss.size == 1

    # Extract the scalar value from the mx.array
    loss_value = loss.item()

    # Assert that the scalar value is a valid number (i.e., non-negative)
    assert isinstance(loss_value, float)
    assert loss_value >= 0

    # Assert that ntoks matches the expected number of valid tokens
    expected_ntoks = attention_mask.sum().item()  # Sum of the attention mask
    assert ntoks == expected_ntoks, f"Expected {expected_ntoks} tokens, but got {ntoks}"


def test_chukloss_no_valid_tokens():
    model = DummyModel()
    inputs = mx.array([[1, 1, 1], [1, 1, 1]])
    targets = mx.array([[1, 1, 1], [1, 1, 1]])
    
    # Create an attention mask with no valid tokens (all zeros)
    attention_mask = mx.zeros((inputs.shape[0], inputs.shape[1]))

    loss, ntoks = chukloss(model, inputs, targets, attention_mask)

    # If there are no valid tokens, the loss should be handled gracefully
    assert loss == 0
    assert ntoks == 1  # Since we use max(attention_mask.sum().item(), 1) in the function


def test_chukloss_handle_empty_batch():
    model = DummyModel()
    inputs = mx.array([[]])
    targets = mx.array([[]])
    attention_mask = mx.array([[]])

    # Call chukloss and expect it to handle the empty batch without raising an error
    loss, ntoks = chukloss(model, inputs, targets, attention_mask)

    # Assert that the loss is 0 and ntoks is 1 (as a safeguard for empty inputs)
    assert loss == 0
    assert ntoks == 1
