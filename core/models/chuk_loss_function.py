import mlx.core as mx
import mlx.nn as nn
import gc
import logging

logger = logging.getLogger(__name__)

def chukloss(model, inputs, targets, attention_mask, lengths):
    # Initialize ce with a default value to ensure it always has a value
    ce = mx.array(0.0)

    # Default to 1 to avoid division by zero in case of error
    ntoks = 1

    try:
        # Run model on inputs with the attention mask
        logits, _ = model(inputs)
        logits = logits.astype(mx.float32)  # Ensure logits are in float32 for numerical stability

        # Calculate the cross-entropy loss
        ce = nn.losses.cross_entropy(logits, targets)

        # Apply the mask to exclude padding tokens
        length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
        length_mask = mx.array(length_mask)  # Ensure length_mask is a tensor

        ce = ce * length_mask  # Apply length mask to the loss

        # Calculate the number of valid tokens
        ntoks = length_mask.sum().item()

        # check ntoks
        if ntoks == 0:
            # Prevent division by zero, handle edge cases
            ntoks = 1  

        # Normalize the loss by the number of valid tokens
        ce = ce.sum() / ntoks

        # return
        return ce, ntoks

    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Error during loss computation: {e}")

        # Ensure ce has a valid value
        ce = mx.array(0.0)  
        ntoks = 1

        # return
        return ce, ntoks

    finally:
        # Explicitly delete intermediate tensors to free up memory
        del logits, length_mask

        # don't believe we need to do garbage collection on loss dunction
        # gc.collect()
