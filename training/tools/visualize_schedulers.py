import argparse
import matplotlib.pyplot as plt
import math

# Mock optimizer class
class MockOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate

# Schedule learning rate function (assuming it's defined or imported)
def schedule_learning_rate(optimizer, iteration_count, warmup_steps, scheduler_type='warmup', **kwargs):
    initial_lr = optimizer.learning_rate if iteration_count == 0 else optimizer.initial_lr

    if scheduler_type == 'warmup':
        if iteration_count < warmup_steps:
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
        else:
            current_lr = initial_lr

    elif scheduler_type == 'linear_decay':
        total_steps = kwargs.get('total_steps', 10000)
        min_lr = kwargs.get('min_lr', 0.0)
        current_lr = max(min_lr, initial_lr * (1 - iteration_count / total_steps))

    elif scheduler_type == 'exponential_decay':
        decay_rate = kwargs.get('decay_rate', 0.96)
        decay_steps = kwargs.get('decay_steps', 1000)
        current_lr = initial_lr * (decay_rate ** (iteration_count / decay_steps))

    elif scheduler_type == 'cosine_decay_with_warmup':
        total_steps = kwargs.get('total_steps', 10000)
        min_lr = kwargs.get('min_lr', 0.0)
        
        if iteration_count < warmup_steps:
            # Warmup phase
            warmup_factor = (iteration_count + 1) / warmup_steps
            current_lr = initial_lr * warmup_factor
        else:
            # Cosine decay phase
            current_lr = min_lr + (initial_lr - min_lr) * \
                (1 + math.cos(math.pi * (iteration_count - warmup_steps) / (total_steps - warmup_steps))) / 2
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    optimizer.learning_rate = current_lr
    optimizer.initial_lr = initial_lr

    return current_lr

# Visualization function
def visualize_scheduler(scheduler_type, total_steps=1000, initial_lr=0.1, warmup_steps=100, **kwargs):
    optimizer = MockOptimizer(learning_rate=initial_lr)
    learning_rates = []

    for step in range(total_steps):
        lr = schedule_learning_rate(
            optimizer,
            iteration_count=step,
            warmup_steps=warmup_steps,
            scheduler_type=scheduler_type,
            **kwargs
        )
        learning_rates.append(lr)
    
    plt.plot(range(total_steps), learning_rates, label=scheduler_type)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule: {scheduler_type}')
    plt.legend()
    plt.show()

# Main function for CLI
def main():
    parser = argparse.ArgumentParser(description="Visualize Learning Rate Schedulers")
    parser.add_argument('--scheduler', type=str, required=True, choices=['warmup', 'linear_decay', 'exponential_decay', 'cosine_annealing','cosine_decay_with_warmup'], help="Type of scheduler")
    parser.add_argument('--total_steps', type=int, default=1000, help="Total training steps")
    parser.add_argument('--initial_lr', type=float, default=0.1, help="Initial learning rate")
    parser.add_argument('--warmup_steps', type=int, default=100, help="Number of warmup steps")
    parser.add_argument('--min_lr', type=float, default=0.0, help="Minimum learning rate (for decay schedulers)")
    parser.add_argument('--decay_rate', type=float, default=0.96, help="Decay rate (for exponential decay)")
    parser.add_argument('--decay_steps', type=int, default=1000, help="Decay steps (for exponential decay)")

    args = parser.parse_args()

    # Call visualization function with parsed arguments
    visualize_scheduler(
        scheduler_type=args.scheduler,
        total_steps=args.total_steps,
        initial_lr=args.initial_lr,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps
    )

if __name__ == "__main__":
    main()
