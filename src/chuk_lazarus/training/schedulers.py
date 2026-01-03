import logging
import math
from enum import Enum

import mlx.core as mx

# Setup the logger
logger = logging.getLogger(__name__)


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""

    WARMUP = "warmup"
    """Linear warmup to initial learning rate."""

    LINEAR_DECAY = "linear_decay"
    """Linear decay from initial to minimum learning rate."""

    EXPONENTIAL_DECAY = "exponential_decay"
    """Exponential decay with configurable rate and steps."""

    COSINE_ANNEALING = "cosine_annealing"
    """Cosine annealing between initial and minimum learning rate."""

    COSINE_DECAY_WITH_WARMUP = "cosine_decay_with_warmup"
    """Warmup followed by cosine decay."""


def schedule_learning_rate(
    optimizer,
    iteration_count: int,
    warmup_steps: int,
    scheduler_type: SchedulerType | str = SchedulerType.WARMUP,
    *,
    total_steps: int = 10000,
    min_lr: float = 0.0,
    decay_rate: float = 0.96,
    decay_steps: int = 1000,
) -> float:
    """
    Schedule the learning rate based on the iteration count, warmup steps, and decay schedule.

    Args:
        optimizer: The optimizer instance containing the learning rate.
        iteration_count: The current iteration count during training.
        warmup_steps: The number of steps to warm up the learning rate.
        scheduler_type: Type of learning rate scheduler (SchedulerType enum or string).
        total_steps: Total training steps (for decay schedulers).
        min_lr: Minimum learning rate (for decay schedulers).
        decay_rate: Decay rate (for exponential decay).
        decay_steps: Steps between decay (for exponential decay).

    Returns:
        current_lr: The updated learning rate after applying the schedule.
    """
    # Normalize scheduler type to string for comparison
    sched_type = scheduler_type.value if isinstance(scheduler_type, SchedulerType) else scheduler_type

    # Determine the initial learning rate
    if iteration_count == 0:
        initial_lr = optimizer.learning_rate
        optimizer.initial_lr = initial_lr  # Store the initial learning rate
    else:
        initial_lr = optimizer.initial_lr

    if sched_type == SchedulerType.WARMUP.value:
        if iteration_count < warmup_steps:
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
        else:
            current_lr = initial_lr

    elif sched_type == SchedulerType.LINEAR_DECAY.value:
        current_lr = max(min_lr, initial_lr * (1 - iteration_count / total_steps))

    elif sched_type == SchedulerType.EXPONENTIAL_DECAY.value:
        current_lr = initial_lr * (decay_rate ** (iteration_count / decay_steps))

    elif sched_type == SchedulerType.COSINE_ANNEALING.value:
        current_lr = (
            min_lr
            + (initial_lr - min_lr) * (1 + math.cos(math.pi * iteration_count / total_steps)) / 2
        )

    elif sched_type == SchedulerType.COSINE_DECAY_WITH_WARMUP.value:
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
            optimizer, iteration_count, warmup_steps, SchedulerType.WARMUP
        )

    # Ensure current_lr is a float
    if isinstance(current_lr, (mx.array, list, tuple)):
        current_lr = float(current_lr.item())  # Convert to float if it's an array

    logger.debug(f"Iter: {iteration_count}, LR: {current_lr:.6f}")

    optimizer.learning_rate = current_lr

    return current_lr
