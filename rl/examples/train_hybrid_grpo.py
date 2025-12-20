"""
Example: Full hybrid training with GRPO and live MCP tools.

This demonstrates the complete hybrid architecture:
- Mistral-7B as the planner/router
- Tiny RNN experts for control tasks
- Live MCP tool calls during training
- GRPO for training the LLM policy

This is "MCP injected into training" - the full vision.

Usage:
    python -m rl.examples.train_hybrid_grpo \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --physics-expert ./checkpoints/physics_expert/best.npz
"""

import logging
import argparse
import random
from typing import List

import mlx.core as mx

# Import from existing chuk-mlx infrastructure
from core.models.model_loader import load_model
from core.utils.tokenizer_loader import load_tokenizer

from ..experts.gru_expert import GRUExpert, ExpertConfig
from ..experts.registry import ExpertRegistry
from ..env.orchestrator import MCPToolClient, Orchestrator, OrchestratorConfig, Action, ActionType
from ..losses.grpo_loss import GRPOConfig
from ..trainers.grpo_trainer import GRPOTrainer, GRPOTrainerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# MCP Tool Implementations (Mock - replace with real MCP calls)
# ============================================================

def math_solve(expression: str) -> dict:
    """Solve a math expression."""
    try:
        # CAUTION: eval is dangerous - use proper parser in production
        result = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "result": result, "expression": expression}
    except Exception as e:
        return {"success": False, "error": str(e), "expression": expression}


def physics_simulate(angle: float, velocity: float, target: float, wind: float = 0) -> dict:
    """Simulate projectile physics."""
    import math

    angle_rad = math.radians(angle)
    g = 9.81
    vx = velocity * math.cos(angle_rad)
    vy = velocity * math.sin(angle_rad)
    t_flight = 2 * vy / g
    distance = vx * t_flight + 0.5 * wind * t_flight

    return {
        "distance": distance,
        "target": target,
        "error": abs(distance - target),
        "success": abs(distance - target) < 1.0
    }


# ============================================================
# Reward Function (calls MCP tools to verify)
# ============================================================

def compute_reward(prompt: str, response: str) -> float:
    """
    Compute reward for a response by verifying with MCP tools.

    This is where MCP gets "injected into training" - we call
    actual tools to verify the model's outputs.
    """
    reward = 0.0

    # Check if response uses tools appropriately
    if "TOOL:" in response or "DELEGATE:" in response:
        reward += 0.1  # Bonus for attempting tool use

    # Parse and verify tool calls
    if "math_solve" in response:
        # Extract expression
        import re
        match = re.search(r'math_solve\(expression="([^"]+)"\)', response)
        if match:
            expr = match.group(1)
            result = math_solve(expr)
            if result["success"]:
                # Check if the answer in response matches
                expected = str(result["result"])
                if expected in response:
                    reward += 0.5  # Correct math!
                else:
                    reward -= 0.2  # Used tool but wrong answer

    # Check for physics delegation
    if "physics_controller" in response:
        # The fact that it delegated is good
        reward += 0.3

    # Check for final answer
    if "ANSWER:" in response:
        reward += 0.1

    # Penalize hallucination patterns
    if "I think" in response or "probably" in response:
        reward -= 0.1

    return reward


# ============================================================
# Task/Prompt Generation
# ============================================================

def generate_prompts() -> List[str]:
    """Generate training prompts."""
    prompts = []

    # Math problems
    for _ in range(8):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        prompts.append(f"Calculate: {a} {op} {b}")

    # Physics problems
    for _ in range(8):
        target = random.uniform(50, 200)
        wind = random.uniform(-5, 5)
        prompts.append(
            f"Find the launch parameters to hit a target at {target:.1f}m "
            f"with wind speed {wind:.1f}m/s"
        )

    random.shuffle(prompts)
    return prompts


# ============================================================
# Main Training
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train hybrid Mistral + experts with GRPO")
    parser.add_argument("--model", type=str, required=True, help="Mistral model name")
    parser.add_argument("--physics-expert", type=str, help="Pre-trained physics expert checkpoint")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/hybrid_grpo")
    args = parser.parse_args()

    logger.info("Setting up hybrid training environment...")

    # ---- Setup MCP Tools ----
    mcp_client = MCPToolClient()
    mcp_client.register_tool("math_solve", math_solve)
    mcp_client.register_tool("physics_simulate", physics_simulate)

    # ---- Setup RNN Experts ----
    expert_registry = ExpertRegistry()

    # Create physics expert
    physics_config = ExpertConfig(
        name="physics_controller",
        obs_dim=10,
        action_dim=2,
        hidden_dim=64,
        num_layers=2,
    )
    physics_expert = GRUExpert(physics_config)

    # Load pre-trained weights if available
    if args.physics_expert:
        weights = mx.load(args.physics_expert)
        physics_expert.load_weights(list(weights.items()))
        logger.info(f"Loaded physics expert from {args.physics_expert}")

    expert_registry.register(physics_expert)

    # ---- Setup Orchestrator ----
    orchestrator = Orchestrator(
        config=OrchestratorConfig(max_steps=20),
        expert_registry=expert_registry,
        mcp_client=mcp_client,
    )

    # ---- Load Mistral ----
    logger.info(f"Loading policy model: {args.model}")
    policy_model = load_model(args.model, load_weights=True)
    policy_model.set_mode('TRAIN')

    logger.info("Loading reference model...")
    reference_model = load_model(args.model, load_weights=True)
    reference_model.set_mode('INFERENCE')

    tokenizer = load_tokenizer(args.model)

    # ---- Configure GRPO Training ----
    config = GRPOTrainerConfig(
        grpo=GRPOConfig(
            group_size=args.group_size,
            clip_epsilon=0.2,
            kl_coef=0.1,
        ),
        num_iterations=args.iterations,
        prompts_per_iteration=16,
        learning_rate=args.lr,
        max_response_length=256,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=1,
        checkpoint_interval=50,
    )

    # ---- Create Trainer ----
    trainer = GRPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        reward_fn=compute_reward,
        config=config,
    )

    # ---- Train! ----
    logger.info("Starting hybrid GRPO training...")
    logger.info("MCP tools are being called during training to compute rewards!")

    trainer.train(prompt_source=generate_prompts)

    logger.info("Training complete!")
    logger.info(f"Best reward: {trainer.best_reward:.4f}")
    logger.info(f"Total MCP calls: {mcp_client.call_count}")


if __name__ == "__main__":
    main()
