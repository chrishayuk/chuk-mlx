import mlx.core as mx
import mlx.nn as nn
import gc
import logging

logger = logging.getLogger(__name__)

def chukloss(model, inputs, targets, attention_mask):
    # Initialize the cross-entropy loss (ce) to a zero array and ntoks (number of tokens) to 1.
    # This initialization is a fallback in case something goes wrong during the computation.
    ce = mx.array(0.0)
    ntoks = 1
    logits = None  # Initialize logits

    try:
        # Call the model's forward method to obtain the logits (predictions before softmax).
        # The model is expected to return logits and optionally a cache (which is ignored here).
        logits, _ = model(inputs)

        # Calculate the cross-entropy loss between the logits and the target labels.
        # Cross-entropy is commonly used for classification tasks, measuring the discrepancy
        # between the predicted and actual labels.
        ce = nn.losses.cross_entropy(logits, targets)

        # Apply the attention mask to the computed loss. The attention mask typically indicates
        # which tokens in the input should contribute to the loss calculation (e.g., non-padding tokens).
        ce = mx.multiply(ce, attention_mask)

        # Calculate the number of tokens that contribute to the loss by summing the attention mask.
        # The use of mx.maximum ensures that ntoks is at least 1 to avoid division by zero.
        ntoks = mx.maximum(attention_mask.sum(), 1)

        # Compute the average loss by dividing the summed cross-entropy by the number of tokens.
        ce = ce.sum() / ntoks

        # Return the computed loss and the number of tokens for reporting or further calculations.
        return ce, ntoks

    except Exception as e:
        # If an error occurs during the loss computation, log the error message for debugging purposes.
        logger.error(f"Error during loss computation: {e}")
        
        # Return a default loss of zero and 1 token, which acts as a safeguard against failure.
        return mx.array(0.0), 1

    finally:
        # Clear references to the logits and ce (cross-entropy) to free memory.
        # This is important to avoid memory leaks, especially when dealing with large tensors.
        del logits, ce

