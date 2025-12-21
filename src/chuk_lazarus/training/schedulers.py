import logging
import math

import mlx.core as mx

# Setup the logger
logger = logging.getLogger(__name__)


def schedule_learning_rate(
    optimizer, iteration_count, warmup_steps, scheduler_type="warmup", **kwargs
):
    """
    Schedule the learning rate based on the iteration count, warmup steps, and decay schedule.

    Args:
        optimizer: The optimizer instance containing the learning rate.
        iteration_count: The current iteration count during training.
        warmup_steps: The number of steps to warm up the learning rate.
        scheduler_type: Type of learning rate scheduler. Supported types are:
                        'warmup' (default), 'linear_decay', 'exponential_decay', 'cosine_annealing', 'cosine_decay_with_warmup'.
        kwargs: Additional parameters for other scheduler types (e.g., decay rate, total_steps).

    Returns:
        current_lr: The updated learning rate after applying the schedule.
    """
    # Determine the initial learning rate
    if iteration_count == 0:
        initial_lr = optimizer.learning_rate
        optimizer.initial_lr = initial_lr  # Store the initial learning rate
    else:
        initial_lr = optimizer.initial_lr

    if scheduler_type == "warmup":
        if iteration_count < warmup_steps:
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
        else:
            current_lr = initial_lr

    elif scheduler_type == "linear_decay":
        total_steps = kwargs.get("total_steps", 10000)
        min_lr = kwargs.get("min_lr", 0.0)
        current_lr = max(min_lr, initial_lr * (1 - iteration_count / total_steps))

    elif scheduler_type == "exponential_decay":
        decay_rate = kwargs.get("decay_rate", 0.96)
        decay_steps = kwargs.get("decay_steps", 1000)
        current_lr = initial_lr * (decay_rate ** (iteration_count / decay_steps))

    elif scheduler_type == "cosine_annealing":
        total_steps = kwargs.get("total_steps", 10000)
        min_lr = kwargs.get("min_lr", 0.0)
        current_lr = (
            min_lr
            + (initial_lr - min_lr) * (1 + math.cos(math.pi * iteration_count / total_steps)) / 2
        )

    elif scheduler_type == "cosine_decay_with_warmup":
        total_steps = kwargs.get("total_steps", 10000)
        min_lr = kwargs.get("min_lr", 0.0)

        if iteration_count < warmup_steps:
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
        else:
            current_lr = (
                min_lr
                + (initial_lr - min_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (iteration_count - warmup_steps) / (total_steps - warmup_steps)
                    )
                )
                / 2
            )

    else:
        logger.warning(f"Unsupported scheduler type: {scheduler_type}. Defaulting to warmup.")
        current_lr = schedule_learning_rate(
            optimizer, iteration_count, warmup_steps, scheduler_type="warmup", **kwargs
        )

    # Ensure current_lr is a float
    if isinstance(current_lr, (mx.array, list, tuple)):
        current_lr = float(current_lr.item())  # Convert to float if it's an array

    logger.debug(f"Iter: {iteration_count}, LR: {current_lr:.6f}")

    optimizer.learning_rate = current_lr

    return current_lr
