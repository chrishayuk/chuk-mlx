"""
Orchestrator - The brain that coordinates Mistral, RNN Experts, and MCP Tools.

This is the central coordinator for the hybrid architecture:
- Routes requests from Mistral to appropriate experts or tools
- Maintains environment state
- Manages episode flow
- Computes observations for each component

Flow:
    User Query
         |
    Mistral (plans, routes)
         |
    Orchestrator (executes actions)
         |-> MCP Tool (direct tool call)
         |-> RNN Expert (delegated control loop)
                  |
              MCP Tools (expert's tool calls)
                  |
    Orchestrator (collects results)
         |
    Mistral (explains, continues, or finishes)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions Mistral can take."""
    TOOL_CALL = "tool_call"           # Direct MCP tool call
    DELEGATE_EXPERT = "delegate"       # Delegate to RNN expert
    FINAL_ANSWER = "final_answer"      # End episode with answer
    THINK = "think"                    # Internal reasoning (no external action)


@dataclass
class Action:
    """An action from the policy (Mistral or RNN expert)."""
    action_type: ActionType
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    expert_name: Optional[str] = None
    expert_args: Optional[Dict] = None
    answer: Optional[str] = None
    thought: Optional[str] = None


@dataclass
class StepResult:
    """Result from executing an action."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Any] = None
    expert_result: Optional[Any] = None


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_steps: int = 50                    # Max steps per episode
    max_expert_steps: int = 20             # Max steps within expert loop
    tool_call_timeout: float = 30.0        # Timeout for MCP calls
    cache_tool_calls: bool = True          # Cache identical tool calls
    verbose: bool = False


class MCPToolClient:
    """
    Interface for calling MCP tools.

    In production, this would connect to actual MCP servers.
    For training, can be mocked or use cached responses.
    """

    def __init__(self, tools: Dict[str, Callable] = None):
        self.tools = tools or {}
        self.cache = {}
        self.call_count = 0

    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler."""
        self.tools[name] = handler

    async def call(
        self,
        tool_name: str,
        args: Dict,
        use_cache: bool = True
    ) -> Any:
        """Call an MCP tool."""
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {tool_name}")
            return self.cache[cache_key]

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        self.call_count += 1
        handler = self.tools[tool_name]

        # Handle async or sync handlers
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**args)
        else:
            result = handler(**args)

        if use_cache:
            self.cache[cache_key] = result

        return result

    def call_sync(self, tool_name: str, args: Dict, use_cache: bool = True) -> Any:
        """Synchronous version for non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.call(tool_name, args, use_cache))


class Orchestrator:
    """
    Main orchestrator for the hybrid Mistral + RNN Expert + MCP system.

    This coordinates:
    - Mistral's high-level planning and routing
    - RNN expert execution for iterative control
    - MCP tool calls for ground truth
    - State management across the episode
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        expert_registry=None,
        mcp_client: MCPToolClient = None
    ):
        self.config = config
        self.expert_registry = expert_registry
        self.mcp_client = mcp_client or MCPToolClient()

        # Episode state
        self.state: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.step_count: int = 0
        self.episode_done: bool = False

        # Metrics
        self.total_tool_calls: int = 0
        self.total_expert_calls: int = 0

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset for a new episode.

        Args:
            task: The task/problem to solve
                {
                    "prompt": "User's question",
                    "goal": {...},  # Optional structured goal
                    "constraints": {...},  # Optional constraints
                }

        Returns:
            Initial observation for Mistral
        """
        self.state = {
            "task": task,
            "prompt": task.get("prompt", ""),
            "goal": task.get("goal", {}),
            "constraints": task.get("constraints", {}),
            "results": [],
            "current_focus": None,
        }
        self.history = []
        self.step_count = 0
        self.episode_done = False

        # Reset all experts
        if self.expert_registry:
            self.expert_registry.reset_all()

        return self._build_observation_for_mistral()

    def step(self, action: Action) -> StepResult:
        """
        Execute an action and return the result.

        Args:
            action: Action from Mistral

        Returns:
            StepResult with observation, reward, done, info
        """
        self.step_count += 1

        if self.step_count > self.config.max_steps:
            return self._timeout_result()

        if action.action_type == ActionType.TOOL_CALL:
            return self._execute_tool_call(action)
        elif action.action_type == ActionType.DELEGATE_EXPERT:
            return self._execute_expert_delegation(action)
        elif action.action_type == ActionType.FINAL_ANSWER:
            return self._execute_final_answer(action)
        elif action.action_type == ActionType.THINK:
            return self._execute_think(action)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

    def _execute_tool_call(self, action: Action) -> StepResult:
        """Execute a direct MCP tool call."""
        self.total_tool_calls += 1

        try:
            result = self.mcp_client.call_sync(
                action.tool_name,
                action.tool_args or {},
                use_cache=self.config.cache_tool_calls
            )

            # Update state
            self.state["results"].append({
                "type": "tool_call",
                "tool": action.tool_name,
                "args": action.tool_args,
                "result": result,
            })

            self.history.append({
                "step": self.step_count,
                "action": action,
                "result": result,
            })

            return StepResult(
                observation=self._build_observation_for_mistral(),
                reward=0.0,  # Intermediate reward
                done=False,
                info={"tool_result": result},
                tool_result=result,
            )

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return StepResult(
                observation=self._build_observation_for_mistral(),
                reward=-0.1,  # Small penalty for failed tool call
                done=False,
                info={"error": str(e)},
            )

    def _execute_expert_delegation(self, action: Action) -> StepResult:
        """Delegate to an RNN expert for iterative control."""
        self.total_expert_calls += 1

        if self.expert_registry is None:
            return StepResult(
                observation=self._build_observation_for_mistral(),
                reward=-0.1,
                done=False,
                info={"error": "No expert registry configured"},
            )

        expert_name = action.expert_name
        if expert_name not in self.expert_registry:
            return StepResult(
                observation=self._build_observation_for_mistral(),
                reward=-0.1,
                done=False,
                info={"error": f"Expert not found: {expert_name}"},
            )

        # Run expert loop
        expert_result = self._run_expert_loop(
            expert_name,
            action.expert_args or {}
        )

        # Update state
        self.state["results"].append({
            "type": "expert_delegation",
            "expert": expert_name,
            "args": action.expert_args,
            "result": expert_result,
        })

        return StepResult(
            observation=self._build_observation_for_mistral(),
            reward=expert_result.get("reward", 0.0),
            done=False,
            info={"expert_result": expert_result},
            expert_result=expert_result,
        )

    def _run_expert_loop(
        self,
        expert_name: str,
        args: Dict
    ) -> Dict:
        """
        Run an RNN expert's control loop.

        The expert takes over for multiple steps, calling MCP tools
        as needed, until it signals completion or hits max steps.
        """
        expert = self.expert_registry[expert_name]
        expert.reset_hidden(batch_size=1)

        # Build initial observation for expert
        obs = self._build_observation_for_expert(expert_name, args)
        obs_tensor = mx.array([obs], dtype=mx.float32)

        expert_steps = 0
        total_reward = 0.0
        trajectory = []

        while expert_steps < self.config.max_expert_steps:
            expert_steps += 1

            # Get expert action
            result = expert(obs_tensor, deterministic=False)
            action = result["action"][0]  # Remove batch dim

            # Execute action via MCP (expert actions are tool parameters)
            tool_result = self._execute_expert_action(expert_name, action, args)

            # Compute reward for this step
            step_reward = self._compute_expert_step_reward(
                expert_name, tool_result, args
            )
            total_reward += step_reward

            trajectory.append({
                "step": expert_steps,
                "obs": obs,
                "action": action.tolist(),
                "reward": step_reward,
                "tool_result": tool_result,
            })

            # Check if done
            if self._expert_done(expert_name, tool_result, args):
                break

            # Update observation
            obs = self._build_observation_for_expert(
                expert_name, args, previous_result=tool_result
            )
            obs_tensor = mx.array([obs], dtype=mx.float32)

        return {
            "expert": expert_name,
            "steps": expert_steps,
            "reward": total_reward,
            "trajectory": trajectory,
            "final_result": trajectory[-1]["tool_result"] if trajectory else None,
        }

    def _execute_expert_action(
        self,
        expert_name: str,
        action: mx.array,
        args: Dict
    ) -> Any:
        """
        Execute an expert's action via MCP.

        Maps the expert's continuous/discrete output to actual tool calls.
        """
        # This is domain-specific - example for physics controller
        if expert_name == "physics_controller":
            # Action: [angle, velocity] in [-1, 1]
            angle = float(action[0]) * 90  # Map to -90 to 90 degrees
            velocity = (float(action[1]) + 1) * 50  # Map to 0-100 m/s

            return self.mcp_client.call_sync(
                "physics_simulate",
                {
                    "angle": angle,
                    "velocity": velocity,
                    "target": args.get("target", 100),
                    "wind": args.get("wind", 0),
                },
                use_cache=False  # Each attempt is different
            )

        # Default: return action as-is
        return {"action": action.tolist()}

    def _compute_expert_step_reward(
        self,
        expert_name: str,
        tool_result: Any,
        args: Dict
    ) -> float:
        """Compute reward for an expert's step."""
        # Domain-specific - example for physics
        if expert_name == "physics_controller":
            target = args.get("target", 100)
            if isinstance(tool_result, dict):
                distance = tool_result.get("distance", 0)
                error = abs(distance - target)
                # Reward inversely proportional to error
                return max(0, 1.0 - error / target)

        return 0.0

    def _expert_done(
        self,
        expert_name: str,
        tool_result: Any,
        args: Dict
    ) -> bool:
        """Check if expert should stop."""
        if expert_name == "physics_controller":
            target = args.get("target", 100)
            tolerance = args.get("tolerance", 0.5)
            if isinstance(tool_result, dict):
                distance = tool_result.get("distance", 0)
                return abs(distance - target) < tolerance

        return False

    def _execute_final_answer(self, action: Action) -> StepResult:
        """Handle final answer from Mistral."""
        self.episode_done = True

        # Compute final reward based on answer correctness
        reward = self._compute_final_reward(action.answer)

        self.state["final_answer"] = action.answer

        return StepResult(
            observation=self._build_observation_for_mistral(),
            reward=reward,
            done=True,
            info={"answer": action.answer},
        )

    def _execute_think(self, action: Action) -> StepResult:
        """Handle internal thinking (no external action)."""
        self.history.append({
            "step": self.step_count,
            "action": action,
            "thought": action.thought,
        })

        return StepResult(
            observation=self._build_observation_for_mistral(),
            reward=0.0,
            done=False,
            info={"thought": action.thought},
        )

    def _timeout_result(self) -> StepResult:
        """Return result when max steps exceeded."""
        self.episode_done = True
        return StepResult(
            observation=self._build_observation_for_mistral(),
            reward=-1.0,  # Penalty for timeout
            done=True,
            info={"timeout": True},
        )

    def _build_observation_for_mistral(self) -> Dict[str, Any]:
        """
        Build observation for Mistral.

        This is a text-friendly representation of current state.
        """
        return {
            "prompt": self.state["prompt"],
            "goal": self.state["goal"],
            "constraints": self.state["constraints"],
            "step": self.step_count,
            "results": self.state["results"][-5:],  # Last 5 results
            "history_summary": self._summarize_history(),
        }

    def _build_observation_for_expert(
        self,
        expert_name: str,
        args: Dict,
        previous_result: Any = None
    ) -> List[float]:
        """
        Build compact numerical observation for RNN expert.

        This is domain-specific and returns a flat vector.
        """
        if expert_name == "physics_controller":
            target = args.get("target", 100)
            wind = args.get("wind", 0)
            attempts = len([r for r in self.state["results"]
                          if r.get("expert") == "physics_controller"])

            if previous_result and isinstance(previous_result, dict):
                distance = previous_result.get("distance", 0)
                error = distance - target
            else:
                distance = 0
                error = target

            return [
                target / 100,           # Normalized target
                error / 100,            # Normalized error
                distance / 100,         # Normalized distance
                wind / 10,              # Normalized wind
                attempts / 10,          # Normalized attempts
                0.0, 0.0, 0.0, 0.0, 0.0  # Padding to obs_dim
            ]

        # Default: zeros
        return [0.0] * 10

    def _summarize_history(self) -> str:
        """Create a brief summary of action history."""
        if not self.history:
            return "No actions taken yet."

        summaries = []
        for h in self.history[-3:]:  # Last 3 actions
            action = h.get("action")
            if action:
                if action.action_type == ActionType.TOOL_CALL:
                    summaries.append(f"Called {action.tool_name}")
                elif action.action_type == ActionType.DELEGATE_EXPERT:
                    summaries.append(f"Delegated to {action.expert_name}")

        return "; ".join(summaries) if summaries else "No significant actions."

    def _compute_final_reward(self, answer: str) -> float:
        """
        Compute final reward for the answer.

        In practice, this would:
        - Call MCP to verify the answer
        - Compare against ground truth
        - Check constraint satisfaction
        """
        goal = self.state.get("goal", {})

        if "expected_answer" in goal:
            expected = str(goal["expected_answer"])
            if answer and expected.lower() in answer.lower():
                return 1.0
            return 0.0

        # Default: small positive reward for providing any answer
        return 0.1 if answer else 0.0
