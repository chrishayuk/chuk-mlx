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
    lengths = mx.array([3, 5])
    attention_mask = mx.arange(seq_len)[None, :] < lengths[:, None]

    return inputs, targets, lengths, attention_mask

def test_chukloss_valid_tokens(dummy_data):
    model = DummyModel()
    inputs, targets, lengths, attention_mask = dummy_data

    loss, ntoks = chukloss(model, inputs, targets, attention_mask, lengths)

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
    expected_ntoks = lengths.sum().item()
    assert ntoks == expected_ntoks, f"Expected {expected_ntoks} tokens, but got {ntoks}"

def test_chukloss_no_valid_tokens():
    model = DummyModel()
    inputs = mx.array([[1, 1, 1], [1, 1, 1]])
    targets = mx.array([[1, 1, 1], [1, 1, 1]])
    lengths = mx.array([0, 0])
    attention_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    loss, ntoks = chukloss(model, inputs, targets, lengths, attention_mask)

    # If there are no valid tokens, the loss should be handled gracefully
    assert loss == 0
    assert ntoks == 1  # Since we use max(length_mask.sum().item(), 1) in the function

def test_chukloss_handle_empty_batch():
    model = DummyModel()
    inputs = mx.array([[]])
    targets = mx.array([[]])
    lengths = mx.array([0])
    attention_mask = mx.array([[]])

    with pytest.raises(ValueError):
        loss, ntoks = chukloss(model, inputs, targets, lengths, attention_mask)

    # Adjust this based on how your model is supposed to handle empty batches.
