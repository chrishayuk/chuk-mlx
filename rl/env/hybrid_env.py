"""
HybridEnv - Gym-like environment wrapper for the hybrid architecture.

This provides a standard interface for RL training:
- reset() -> observation
- step(action) -> observation, reward, done, info

Works with both:
- Mistral (text actions)
- RNN experts (continuous/discrete actions)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import mlx.core as mx

from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    Action,
    ActionType,
    MCPToolClient
)

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the hybrid environment."""
    orchestrator_config: OrchestratorConfig = field(
        default_factory=OrchestratorConfig
    )

    # Task generation
    task_generator: Optional[callable] = None  # Function to generate tasks
    task_pool: Optional[List[Dict]] = None     # Static pool of tasks

    # Reward shaping
    step_penalty: float = -0.01               # Small penalty per step
    tool_call_reward: float = 0.0             # Reward for valid tool calls
    expert_use_reward: float = 0.0            # Reward for using experts

    # Episode limits
    max_episode_length: int = 50


class HybridEnv:
    """
    Gym-like environment for training the hybrid Mistral + RNN system.

    This wraps the Orchestrator and provides:
    - Standard reset/step interface
    - Task sampling
    - Reward shaping
    - Episode management

    Usage:
        env = HybridEnv(config, expert_registry, mcp_client)

        obs = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        config: EnvConfig,
        expert_registry=None,
        mcp_client: MCPToolClient = None
    ):
        self.config = config
        self.orchestrator = Orchestrator(
            config.orchestrator_config,
            expert_registry=expert_registry,
            mcp_client=mcp_client
        )

        self.current_task: Optional[Dict] = None
        self.episode_reward: float = 0.0
        self.episode_length: int = 0
        self.task_index: int = 0

    def reset(self, task: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reset environment for a new episode.

        Args:
            task: Optional specific task. If None, samples from pool/generator.

        Returns:
            Initial observation
        """
        # Get task
        if task is not None:
            self.current_task = task
        elif self.config.task_generator is not None:
            self.current_task = self.config.task_generator()
        elif self.config.task_pool is not None:
            self.current_task = self.config.task_pool[
                self.task_index % len(self.config.task_pool)
            ]
            self.task_index += 1
        else:
            raise ValueError("No task provided and no task source configured")

        # Reset orchestrator
        obs = self.orchestrator.reset(self.current_task)

        # Reset episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0

        return obs

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and return result.

        Args:
            action: Action to execute

        Returns:
            observation: New observation
            reward: Shaped reward
            done: Whether episode is finished
            info: Additional information
        """
        # Execute via orchestrator
        result = self.orchestrator.step(action)

        # Shape reward
        reward = result.reward
        reward += self.config.step_penalty  # Step penalty

        if action.action_type == ActionType.TOOL_CALL:
            reward += self.config.tool_call_reward
        elif action.action_type == ActionType.DELEGATE_EXPERT:
            reward += self.config.expert_use_reward

        # Update tracking
        self.episode_reward += reward
        self.episode_length += 1

        # Check episode limit
        done = result.done or self.episode_length >= self.config.max_episode_length

        # Build info
        info = result.info.copy()
        info.update({
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "task": self.current_task,
        })

        return result.observation, reward, done, info

    def step_from_text(self, action_text: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Parse action from text (as Mistral would output) and execute.

        Expected formats:
            "TOOL: math_solve(equation='2x+5=17')"
            "DELEGATE: physics_controller(target=100, wind=5)"
            "ANSWER: x = 6"
            "THINK: Let me consider..."
        """
        action = self._parse_action_text(action_text)
        return self.step(action)

    def _parse_action_text(self, text: str) -> Action:
        """Parse action from text format."""
        text = text.strip()

        if text.startswith("TOOL:"):
            return self._parse_tool_call(text[5:].strip())
        elif text.startswith("DELEGATE:"):
            return self._parse_delegate(text[9:].strip())
        elif text.startswith("ANSWER:"):
            return Action(
                action_type=ActionType.FINAL_ANSWER,
                answer=text[7:].strip()
            )
        elif text.startswith("THINK:"):
            return Action(
                action_type=ActionType.THINK,
                thought=text[6:].strip()
            )
        else:
            # Default to thinking
            return Action(
                action_type=ActionType.THINK,
                thought=text
            )

    def _parse_tool_call(self, text: str) -> Action:
        """Parse tool call from text like 'math_solve(equation="2x+5=17")'"""
        import re

        # Match: tool_name(args)
        match = re.match(r'(\w+)\((.*)\)', text)
        if not match:
            return Action(action_type=ActionType.THINK, thought=f"Failed to parse: {text}")

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse args (simplified - in production use proper parsing)
        args = {}
        if args_str:
            # Handle key=value pairs
            for pair in args_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    # Try to convert to number
                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except:
                        pass
                    args[key] = value

        return Action(
            action_type=ActionType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=args
        )

    def _parse_delegate(self, text: str) -> Action:
        """Parse expert delegation from text."""
        import re

        match = re.match(r'(\w+)\((.*)\)', text)
        if not match:
            return Action(action_type=ActionType.THINK, thought=f"Failed to parse: {text}")

        expert_name = match.group(1)
        args_str = match.group(2)

        args = {}
        if args_str:
            for pair in args_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    try:
                        value = float(value)
                        if value.is_integer():
                            value = int(value)
                    except:
                        pass
                    args[key] = value

        return Action(
            action_type=ActionType.DELEGATE_EXPERT,
            expert_name=expert_name,
            expert_args=args
        )

    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        return list(self.orchestrator.mcp_client.tools.keys())

    def get_available_experts(self) -> List[str]:
        """Get list of available RNN experts."""
        if self.orchestrator.expert_registry:
            return self.orchestrator.expert_registry.list_names()
        return []

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for current episode."""
        return {
            "reward": self.episode_reward,
            "length": self.episode_length,
            "tool_calls": self.orchestrator.total_tool_calls,
            "expert_calls": self.orchestrator.total_expert_calls,
            "task": self.current_task,
        }


class VectorizedHybridEnv:
    """
    Vectorized environment for parallel episode collection.

    Runs multiple environments in parallel for efficient rollout collection.
    """

    def __init__(
        self,
        num_envs: int,
        config: EnvConfig,
        expert_registry=None,
        mcp_client: MCPToolClient = None
    ):
        self.num_envs = num_envs
        self.envs = [
            HybridEnv(config, expert_registry, mcp_client)
            for _ in range(num_envs)
        ]

    def reset(self, tasks: List[Dict] = None) -> List[Dict]:
        """Reset all environments."""
        if tasks is None:
            return [env.reset() for env in self.envs]
        else:
            return [env.reset(task) for env, task in zip(self.envs, tasks)]

    def step(self, actions: List[Action]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """Step all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]

        obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]

        return obs, rewards, dones, infos

    def reset_done_envs(self) -> List[Dict]:
        """Reset environments that are done and return new observations."""
        new_obs = []
        for i, env in enumerate(self.envs):
            if env.orchestrator.episode_done:
                new_obs.append(env.reset())
            else:
                new_obs.append(None)
        return new_obs
