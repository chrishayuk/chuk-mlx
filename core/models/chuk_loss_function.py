import mlx.core as mx
import mlx.nn as nn

def chukloss(model, inputs, targets, attention_mask, lengths):
    # Run model on inputs with the attention mask
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Calculate the cross-entropy loss
    ce = nn.losses.cross_entropy(logits, targets)

    # Apply the mask to exclude padding tokens (already handled by attention mechanism)
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
    ce = ce * mx.array(length_mask)

    # Calculate the number of valid tokens
    ntoks = max(length_mask.sum().item(), 1)

    # Normalize the loss by the number of valid tokens
    ce = ce.sum() / ntoks

    return ce, ntoks

