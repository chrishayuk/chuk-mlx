import numpy as np

def mockloss(model, inputs, targets, lengths):
    inputs = np.array(inputs, dtype=np.int32) 
    targets = np.array(targets, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)

    logits = model(inputs)
    
    # Apply softmax to logits to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Ensure logits are in a valid range to avoid NaNs in log
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    
    length_mask = (np.arange(inputs.shape[1])[None, :] < lengths[:, None]).astype(bool)
    length_mask = length_mask[..., None]
    
    ce = -np.log(probabilities) * targets[..., None]
    ce = ce * length_mask

    # Check for NaN values in the loss components
    if np.isnan(ce).any():
        return (float('nan'), 0)

    ntoks = length_mask.sum().item()
    ce = ce.sum() / ntoks
    
    return (float(ce), int(ntoks))


# Mock value_and_grad function
def mock_value_and_grad(loss_fn):
    def loss_with_grads(model, inputs, targets, lengths):
        loss_value, ntoks = loss_fn(model, inputs, targets, lengths)
        # Mock gradient computation as numpy arrays
        gradients = [np.random.random(p.shape).astype(np.float32) for p in model.trainable_parameters]
        return (loss_value, ntoks), gradients
    return loss_with_grads
