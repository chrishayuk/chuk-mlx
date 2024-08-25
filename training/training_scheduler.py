import logging
import math

# Setup the logger
logger = logging.getLogger(__name__)

def schedule_learning_rate(optimizer, iteration_count, warmup_steps, scheduler_type='warmup', **kwargs):
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
    initial_lr = optimizer.learning_rate if iteration_count == 0 else optimizer.initial_lr

    if scheduler_type == 'warmup':
        # check if we're during warmup
        if iteration_count < warmup_steps:
            # calculate the warmup factore
            warmup_factor = (iteration_count + 1) / warmup_steps

            # calculatw the learning rate
            current_lr = initial_lr * warmup_factor
            
            # debug
            logger.debug(f"Warmup - Iter: {iteration_count}, LR: {current_lr:.6f}")
        else:
            # after warmup, so learning rate is the initia learning rate
            current_lr = initial_lr

    elif scheduler_type == 'linear_decay':
        # calculate the total steps
        total_steps = kwargs.get('total_steps', 10000)

        # get the minimum learning rate
        min_lr = kwargs.get('min_lr', 0.0)

        # calculate the current learning rate
        current_lr = max(min_lr, initial_lr * (1 - iteration_count / total_steps))

        # debugs
        logger.debug(f"Linear Decay - Iter: {iteration_count}, LR: {current_lr:.6f}")

    elif scheduler_type == 'exponential_decay':
        # set the decay rate
        decay_rate = kwargs.get('decay_rate', 0.96)

        # set the decay steps
        decay_steps = kwargs.get('decay_steps', 1000)

        # set the current learning rate
        current_lr = initial_lr * (decay_rate ** (iteration_count / decay_steps))

        # debug
        logger.debug(f"Exponential Decay - Iter: {iteration_count}, LR: {current_lr:.6f}")

    elif scheduler_type == 'cosine_annealing':
        total_steps = kwargs.get('total_steps', 10000)
        min_lr = kwargs.get('min_lr', 0.0)
        current_lr = min_lr + (initial_lr - min_lr) * \
            (1 + math.cos(math.pi * iteration_count / total_steps)) / 2
        logger.debug(f"Cosine Annealing - Iter: {iteration_count}, LR: {current_lr:.6f}")

    elif scheduler_type == 'cosine_decay_with_warmup':
        total_steps = kwargs.get('total_steps', 10000)
        min_lr = kwargs.get('min_lr', 0.0)
        
        if iteration_count < warmup_steps:
            # Warmup phase
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
            logger.debug(f"Warmup - Iter: {iteration_count}, LR: {current_lr:.6f}")
        else:
            # Cosine decay phase
            current_lr = min_lr + (initial_lr - min_lr) * \
                (1 + math.cos(math.pi * (iteration_count - warmup_steps) / (total_steps - warmup_steps))) / 2
            logger.debug(f"Cosine Decay with Warmup - Iter: {iteration_count}, LR: {current_lr:.6f}")
    
    else:
        logger.warning(f"Unsupported scheduler type: {scheduler_type}. Defaulting to warmup.")
        current_lr = schedule_learning_rate(optimizer, iteration_count, warmup_steps, scheduler_type='warmup', **kwargs)
    
    optimizer.learning_rate = current_lr
    optimizer.initial_lr = initial_lr  # Store the initial learning rate

    return current_lr
