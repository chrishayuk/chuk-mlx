"""
Example: Train a tiny RNN expert for physics control.

This demonstrates Phase 2 of the hybrid architecture:
- Train a small GRU expert to control projectile physics
- Expert learns to call chuk-mcp-physics to hit targets
- Uses PPO for reinforcement learning

Usage:
    python -m rl.examples.train_physics_expert
"""

import logging
import argparse
from dataclasses import dataclass

import mlx.core as mx

from rl.experts.gru_expert import GRUExpert, ExpertConfig
from rl.trainers.ppo_trainer import PPOTrainer, PPOTrainerConfig
from rl.losses.ppo_loss import PPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock physics simulator (replace with actual MCP call)
def mock_physics_simulate(angle: float, velocity: float, target: float, wind: float = 0) -> dict:
    """
    Simple projectile physics simulation.

    In production, this would call chuk-mcp-physics.
    """
    import math

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Simple physics (no air resistance)
    g = 9.81
    vx = velocity * math.cos(angle_rad)
    vy = velocity * math.sin(angle_rad)

    # Time of flight (when y = 0 again)
    t_flight = 2 * vy / g

    # Horizontal distance
    distance = vx * t_flight + 0.5 * wind * t_flight

    return {
        "distance": distance,
        "angle": angle,
        "velocity": velocity,
        "target": target,
        "error": abs(distance - target),
        "success": abs(distance - target) < 1.0
    }


class SimplePhysicsEnv:
    """
    Simple physics environment that returns numeric observations.

    This is a direct environment for training the RNN expert,
    bypassing the full orchestrator for simplicity.
    """

    def __init__(self):
        self.target = 100.0
        self.wind = 0.0
        self.attempts = 0
        self.max_attempts = 10
        self.last_distance = 0.0
        self.last_error = 100.0
        self.done = False

    def reset(self, task: dict = None) -> list:
        """Reset environment and return initial observation."""
        import random
        self.target = random.uniform(50, 200)
        self.wind = random.uniform(-5, 5)
        self.attempts = 0
        self.last_distance = 0.0
        self.last_error = self.target
        self.done = False
        return self._get_obs()

    def _get_obs(self) -> list:
        """Get current observation as a flat list."""
        return [
            self.target / 200,                    # normalized target
            self.last_error / 200,                # normalized error
            self.last_distance / 200,             # normalized distance
            self.wind / 10,                       # normalized wind
            self.attempts / self.max_attempts,    # normalized attempts
            0.0, 0.0, 0.0, 0.0, 0.0               # padding to obs_dim=10
        ]

    def step(self, action) -> tuple:
        """
        Execute action and return (obs, reward, done, info).

        Action: [angle, velocity] in [-1, 1] range
        """
        import math

        # Convert action from [-1, 1] to actual values
        if hasattr(action, 'tolist'):
            action = action.tolist()

        angle = float(action[0]) * 45 + 45      # Map [-1,1] to [0, 90] degrees
        velocity = (float(action[1]) + 1) * 50  # Map [-1,1] to [0, 100] m/s

        # Simulate physics
        result = mock_physics_simulate(angle, velocity, self.target, self.wind)

        self.last_distance = result["distance"]
        self.last_error = result["error"]
        self.attempts += 1

        # Compute reward
        if result["success"]:
            reward = 1.0
            self.done = True
        else:
            # Reward based on how close we got (normalized)
            reward = max(0, 1.0 - self.last_error / self.target) * 0.1
            # Small penalty for each attempt
            reward -= 0.01

        # Check if max attempts reached
        if self.attempts >= self.max_attempts:
            self.done = True
            if not result["success"]:
                reward -= 0.5  # Penalty for not solving

        obs = self._get_obs()
        info = {"result": result, "attempts": self.attempts}

        return obs, reward, self.done, info


def create_physics_env() -> SimplePhysicsEnv:
    """Create environment for physics control task."""
    return SimplePhysicsEnv()


def create_physics_expert() -> GRUExpert:
    """Create GRU expert for physics control."""
    config = ExpertConfig(
        name="physics_controller",
        obs_dim=10,
        action_dim=2,  # angle, velocity
        hidden_dim=64,
        num_layers=2,
        discrete_actions=False,
        action_low=-1.0,
        action_high=1.0,
        use_value_head=True,
    )
    return GRUExpert(config)


def main():
    parser = argparse.ArgumentParser(description="Train physics control expert")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/physics_expert")
    args = parser.parse_args()

    logger.info("Creating physics environment and expert...")

    # Create environment and expert
    env = create_physics_env()
    expert = create_physics_expert()

    # Training config
    config = PPOTrainerConfig(
        ppo=PPOConfig(
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        ),
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        rollout_steps=512,
        num_epochs_per_rollout=4,
        batch_size=64,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=1,
        checkpoint_interval=10,
        target_reward=0.9,  # High accuracy
    )

    # Create trainer
    trainer = PPOTrainer(
        policy=expert,
        env=env,
        config=config,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")
    logger.info(f"Best reward: {trainer.best_reward:.4f}")


if __name__ == "__main__":
    main()
