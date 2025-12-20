"""
Reward computation for the hybrid RL system.

Rewards come from multiple sources:
- Task completion (final answer correctness)
- Tool usage efficiency
- Expert performance
- Constraint satisfaction
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Task completion rewards
    correct_answer_reward: float = 1.0
    partial_answer_reward: float = 0.5
    wrong_answer_penalty: float = -0.5
    no_answer_penalty: float = -1.0

    # Step penalties
    step_penalty: float = -0.01
    timeout_penalty: float = -2.0

    # Tool usage
    valid_tool_call_reward: float = 0.0
    invalid_tool_call_penalty: float = -0.1
    tool_error_penalty: float = -0.2

    # Expert usage
    expert_delegation_reward: float = 0.0
    expert_success_bonus: float = 0.2

    # Efficiency bonuses
    fast_completion_bonus: float = 0.5  # Complete in < 5 steps
    efficient_tool_use_bonus: float = 0.2  # Minimal tool calls

    # Constraint penalties
    constraint_violation_penalty: float = -0.5


class RewardComputer:
    """
    Computes rewards for the hybrid system.

    Supports:
    - Step-level rewards (during episode)
    - Episode-level rewards (at completion)
    - Custom reward functions per task type
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.custom_reward_fns: Dict[str, Callable] = {}

    def register_reward_fn(self, task_type: str, reward_fn: Callable):
        """Register a custom reward function for a task type."""
        self.custom_reward_fns[task_type] = reward_fn

    def compute_step_reward(
        self,
        action_type: str,
        action_result: Dict[str, Any],
        state: Dict[str, Any]
    ) -> float:
        """
        Compute reward for a single step.

        Args:
            action_type: Type of action taken
            action_result: Result from executing the action
            state: Current environment state

        Returns:
            Reward for this step
        """
        reward = self.config.step_penalty

        if action_type == "tool_call":
            reward += self._tool_call_reward(action_result)
        elif action_type == "delegate":
            reward += self._delegation_reward(action_result)
        elif action_type == "final_answer":
            reward += self._answer_reward(action_result, state)
        # "think" actions just get step penalty

        # Check constraints
        if self._violated_constraints(state):
            reward += self.config.constraint_violation_penalty

        return reward

    def compute_episode_reward(
        self,
        trajectory: List[Dict],
        final_state: Dict[str, Any],
        task: Dict[str, Any]
    ) -> float:
        """
        Compute total episode reward with bonuses.

        Args:
            trajectory: List of (state, action, reward) tuples
            final_state: State at episode end
            task: The original task

        Returns:
            Total episode reward with bonuses
        """
        # Sum step rewards
        total_reward = sum(t.get("reward", 0) for t in trajectory)

        # Check for custom reward function
        task_type = task.get("type", "default")
        if task_type in self.custom_reward_fns:
            custom_reward = self.custom_reward_fns[task_type](
                trajectory, final_state, task
            )
            total_reward += custom_reward

        # Efficiency bonuses
        episode_length = len(trajectory)
        if episode_length < 5 and self._task_completed(final_state):
            total_reward += self.config.fast_completion_bonus

        tool_calls = sum(1 for t in trajectory
                        if t.get("action_type") == "tool_call")
        if tool_calls <= 3 and self._task_completed(final_state):
            total_reward += self.config.efficient_tool_use_bonus

        return total_reward

    def _tool_call_reward(self, result: Dict) -> float:
        """Compute reward for a tool call."""
        if result.get("error"):
            return self.config.tool_error_penalty
        elif result.get("invalid"):
            return self.config.invalid_tool_call_penalty
        else:
            return self.config.valid_tool_call_reward

    def _delegation_reward(self, result: Dict) -> float:
        """Compute reward for expert delegation."""
        reward = self.config.expert_delegation_reward

        if result.get("success"):
            reward += self.config.expert_success_bonus

        # Add expert's accumulated reward
        reward += result.get("expert_reward", 0)

        return reward

    def _answer_reward(
        self,
        result: Dict,
        state: Dict
    ) -> float:
        """Compute reward for final answer."""
        answer = result.get("answer", "")
        expected = state.get("goal", {}).get("expected_answer")

        if expected is None:
            # No ground truth - small reward for any answer
            return 0.1 if answer else self.config.no_answer_penalty

        # Check correctness
        if self._check_answer(answer, expected):
            return self.config.correct_answer_reward
        elif self._partial_match(answer, expected):
            return self.config.partial_answer_reward
        else:
            return self.config.wrong_answer_penalty

    def _check_answer(self, answer: str, expected: Any) -> bool:
        """Check if answer matches expected."""
        if answer is None:
            return False

        # String comparison (case insensitive)
        if isinstance(expected, str):
            return expected.lower().strip() in answer.lower().strip()

        # Numeric comparison
        if isinstance(expected, (int, float)):
            try:
                # Extract number from answer
                import re
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                for num_str in numbers:
                    num = float(num_str)
                    if abs(num - expected) < 0.01:
                        return True
            except:
                pass

        return False

    def _partial_match(self, answer: str, expected: Any) -> bool:
        """Check for partial match (close but not exact)."""
        if isinstance(expected, (int, float)):
            try:
                import re
                numbers = re.findall(r'-?\d+\.?\d*', answer)
                for num_str in numbers:
                    num = float(num_str)
                    # Within 10% is partial match
                    if abs(num - expected) / abs(expected) < 0.1:
                        return True
            except:
                pass

        return False

    def _violated_constraints(self, state: Dict) -> bool:
        """Check if any constraints are violated."""
        constraints = state.get("constraints", {})

        # Check max tool calls
        if "max_tool_calls" in constraints:
            tool_calls = len([r for r in state.get("results", [])
                            if r.get("type") == "tool_call"])
            if tool_calls > constraints["max_tool_calls"]:
                return True

        # Check required tools
        if "required_tools" in constraints:
            used_tools = set(r.get("tool") for r in state.get("results", [])
                           if r.get("type") == "tool_call")
            if not set(constraints["required_tools"]).issubset(used_tools):
                return True

        return False

    def _task_completed(self, state: Dict) -> bool:
        """Check if task was completed successfully."""
        return state.get("final_answer") is not None


# Pre-built reward functions for common task types

def math_reward_fn(
    trajectory: List[Dict],
    final_state: Dict,
    task: Dict
) -> float:
    """Reward function for math problems."""
    expected = task.get("goal", {}).get("expected_answer")
    answer = final_state.get("final_answer")

    if expected is None or answer is None:
        return 0.0

    try:
        import re
        # Extract numbers and compare
        expected_num = float(expected) if isinstance(expected, str) else expected
        answer_nums = re.findall(r'-?\d+\.?\d*', str(answer))

        for num_str in answer_nums:
            if abs(float(num_str) - expected_num) < 0.001:
                return 0.5  # Bonus for exact match
    except:
        pass

    return 0.0


def physics_reward_fn(
    trajectory: List[Dict],
    final_state: Dict,
    task: Dict
) -> float:
    """Reward function for physics problems."""
    target = task.get("goal", {}).get("target")
    tolerance = task.get("goal", {}).get("tolerance", 1.0)

    # Find best attempt from trajectory
    best_error = float("inf")
    for t in trajectory:
        result = t.get("result", {})
        if isinstance(result, dict) and "distance" in result:
            error = abs(result["distance"] - target)
            best_error = min(best_error, error)

    if best_error < tolerance:
        return 1.0
    elif best_error < tolerance * 5:
        return 0.5
    else:
        return -0.5
