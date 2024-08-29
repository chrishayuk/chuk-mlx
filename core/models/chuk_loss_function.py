import mlx.core as mx
import mlx.nn as nn
import gc
import logging

logger = logging.getLogger(__name__)

def chukloss(model, inputs, targets, attention_mask):
    # Initialize ce with a default value to ensure it always has a value
    ce = mx.array(0.0)

    # Default to 1 to avoid division by zero in case of error
    ntoks = 1

    try:
        # Run model on inputs with the attention mask
        logits, _ = model(inputs)

        # Calculate the cross-entropy loss
        ce = nn.losses.cross_entropy(logits, targets)

        # Apply the attention mask to exclude padding tokens
        ce *= attention_mask

        # Calculate the number of valid tokens based on the attention mask, ensuring it's at least 1
        ntoks = mx.maximum(attention_mask.sum(), 1)

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
        del logits, ce
