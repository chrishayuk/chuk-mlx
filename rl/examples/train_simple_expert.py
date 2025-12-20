"""
Example: Train a tiny RNN expert on a simple task to demonstrate learning.

Task: Learn to output a target value
- Observation: [target_value, current_output, step]
- Action: single continuous value [-1, 1]
- Reward: -abs(action - target_value)

This is simple enough that we should see clear learning within minutes.

Usage:
    python -m rl.examples.train_simple_expert
"""

import logging
import argparse
import random

import mlx.core as mx

from rl.experts.gru_expert import GRUExpert, ExpertConfig
from rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig
from rl.losses.ppo_loss import PPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTargetEnv:
    """
    Dead simple environment: learn to output 0.5 constantly.

    Observation: [target=0.5, last_action, step/max_steps]
    Action: single value in [-1, 1]
    Reward: 1 - abs(action - 0.5)  (max 1.0 when action=0.5)

    This is the simplest possible task - just learn to output 0.5.
    Episode: 10 steps, always same target.
    """

    def __init__(self):
        self.target = 0.5  # Fixed target - always 0.5
        self.step_count = 0
        self.max_steps = 10
        self.last_action = 0.0
        self.total_error = 0.0

    def reset(self, task=None) -> list:
        # Fixed target - always 0.5
        self.target = 0.5
        self.step_count = 0
        self.last_action = 0.0
        self.total_error = 0.0
        return self._get_obs()

    def _get_obs(self) -> list:
        return [
            self.target,
            self.last_action,
            self.step_count / self.max_steps,
        ]

    def step(self, action) -> tuple:
        # Extract scalar action
        if hasattr(action, 'tolist'):
            action_val = float(action.tolist()[0]) if len(action.shape) > 0 else float(action)
        else:
            action_val = float(action[0]) if hasattr(action, '__getitem__') else float(action)

        # Clip to valid range
        action_val = max(-1, min(1, action_val))

        # Reward: how close is action to target (0.5)?
        error = abs(action_val - self.target)
        reward = 1.0 - error  # Range: [0, 1]

        self.last_action = action_val
        self.step_count += 1
        self.total_error += error

        done = self.step_count >= self.max_steps

        info = {
            "target": self.target,
            "action": action_val,
            "error": error,
            "avg_error": self.total_error / self.step_count,
        }

        return self._get_obs(), reward, done, info


def create_simple_expert() -> GRUExpert:
    """Create a tiny GRU expert for the simple task."""
    config = ExpertConfig(
        name="simple_matcher",
        obs_dim=3,      # target, last_action, step
        action_dim=1,   # single output
        hidden_dim=32,  # Small network
        num_layers=1,   # Single layer
        discrete_actions=False,
        action_low=-1.0,
        action_high=1.0,
        use_value_head=True,
    )
    return GRUExpert(config)


def main():
    parser = argparse.ArgumentParser(description="Train simple expert to demonstrate learning")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SIMPLE LEARNING DEMO")
    logger.info("Task: Learn to always output 0.5")
    logger.info("Observation: [target=0.5, last_action, step]")
    logger.info("Action: single value in [-1, 1]")
    logger.info("Reward: 1 - |action - 0.5| (max 1.0 when action=0.5)")
    logger.info("Expected: Mean Reward should climb toward 10.0 (10 steps x 1.0)")
    logger.info("=" * 60)

    env = SimpleTargetEnv()
    expert = create_simple_expert()

    config = PPOTrainerConfig(
        ppo=PPOConfig(
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.05,  # More exploration
        ),
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        rollout_steps=500,       # More steps per rollout
        num_epochs_per_rollout=4,
        batch_size=64,
        checkpoint_dir="./checkpoints/simple_expert",
        log_interval=1,
        checkpoint_interval=20,
        target_reward=9.5,  # 10 steps * 0.95 avg reward
    )

    trainer = PPOTrainer(
        policy=expert,
        env=env,
        config=config,
    )

    logger.info("")
    logger.info("Starting training...")
    logger.info("Watch Mean Reward approach 1.0 (perfect matching)")
    logger.info("")

    trainer.train()

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training complete! Best reward: {trainer.best_reward:.4f}")
    logger.info("=" * 60)

    # Demo the trained agent
    logger.info("")
    logger.info("Testing trained agent (5 trials):")
    logger.info("-" * 50)

    total_error = 0
    expert.reset_hidden(batch_size=1)
    for i in range(5):
        obs = env.reset()
        target = env.target

        # Just take one action and see how close it is
        obs_tensor = mx.array([obs], dtype=mx.float32)
        result = expert(obs_tensor, deterministic=True)
        action_val = float(result["action"][0].tolist()[0]) if hasattr(result["action"][0], 'tolist') else float(result["action"][0])
        error = abs(action_val - target)
        total_error += error

        logger.info(f"  Target: {target:+.3f} | Action: {action_val:+.3f} | Error: {error:.3f}")

    logger.info("-" * 50)
    avg_error = total_error / 5
    logger.info(f"Average Error: {avg_error:.3f}")
    if avg_error < 0.1:
        logger.info("EXCELLENT: Agent learned to match target very well!")
    elif avg_error < 0.3:
        logger.info("GOOD: Agent learned the task reasonably well")
    else:
        logger.info("Learning in progress - may need more training")


if __name__ == "__main__":
    main()
