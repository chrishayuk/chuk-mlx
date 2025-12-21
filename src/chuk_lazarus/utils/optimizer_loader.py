import mlx.optimizers as optim


def linear_warmup_schedule(initial_lr, warmup_steps):
    def schedule(step):
        if step < warmup_steps:
            return initial_lr * (step + 1) / warmup_steps
        else:
            return initial_lr

    return schedule


def piecewise_scheduler(schedulers, milestones):
    def schedule(step):
        index = 0
        for milestone in milestones:
            if step < milestone:
                break
            index += 1
        return schedulers[index](step - milestones[index - 1] if index > 0 else step)

    return schedule


def load_optimizer(optimizer_config, total_iterations):
    optimizer_name = optimizer_config["name"]
    initial_lr = float(optimizer_config["initial_lr"])
    lr_schedule_type = optimizer_config["lr_schedule"]["type"]
    betas = [float(beta) for beta in optimizer_config["betas"]]
    eps = float(optimizer_config["eps"])
    weight_decay = float(optimizer_config["weight_decay"])

    # Calculate decay_steps based on total_iterations
    decay_steps = total_iterations

    # Define the learning rate schedule
    if lr_schedule_type == "cosine_decay":
        lr_schedule_warmup_steps = int(optimizer_config["lr_schedule"].get("warmup_steps", 0))
        lr_schedule_minimum = float(optimizer_config["lr_schedule"].get("minimum", 0.0))

        # Create the cosine decay schedule
        cosine_schedule = optim.cosine_decay(initial_lr, decay_steps, lr_schedule_minimum)

        # Apply warmup to the cosine decay schedule
        if lr_schedule_warmup_steps > 0:
            warmup_schedule = linear_warmup_schedule(initial_lr, lr_schedule_warmup_steps)
            lr_schedule = piecewise_scheduler(
                [warmup_schedule, cosine_schedule], [lr_schedule_warmup_steps]
            )
        else:
            lr_schedule = cosine_schedule

    elif lr_schedule_type == "exponential_decay":
        lr_schedule_decay_rate = float(optimizer_config["lr_schedule"]["decay_rate"])
        lr_schedule = optim.exponential_decay(initial_lr, lr_schedule_decay_rate, decay_steps)
    else:
        raise ValueError(f"Unsupported learning rate schedule type: {lr_schedule_type}")

    # Create the optimizer instance
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            learning_rate=lr_schedule, betas=betas, eps=eps, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer
