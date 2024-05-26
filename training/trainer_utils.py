import mlx.core as mx
import logging

# setup the logger
logger = logging.getLogger(__name__)

def schedule_learning_rate(optimizer, iteration_count, warmup_steps):
    # check if we're in warmup
    if iteration_count < warmup_steps:
        # set the warmup factor
        warmup_factor = (iteration_count + 1) / warmup_steps

        # set the learning rate
        current_lr = optimizer.learning_rate * warmup_factor

        # set the optimizer learning reate
        optimizer.learning_rate = current_lr
    else:
        # the current learning rate is the optimizer learning rate
        current_lr = optimizer.learning_rate

    # return the learning rate
    return current_lr