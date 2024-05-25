import mlx.core as mx
import mlx.nn as nn

def chukloss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Create a mask for the padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the cross-entropy loss
    ce = nn.losses.cross_entropy(logits, targets)

    # Apply the mask to exclude padding tokens
    ce = ce * mx.array(length_mask)

    # Calculate the number of valid tokens
    ntoks = length_mask.sum().item()

    # Normalize the loss by the number of valid tokens
    ce = ce.sum() / ntoks
    return ce, ntoks