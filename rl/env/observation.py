"""
Observation builders for different components of the hybrid system.

Different parts of the system need different observation formats:
- Mistral: Text/structured observations
- RNN Experts: Compact numerical vectors
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import mlx.core as mx


@dataclass
class MistralObservation:
    """
    Observation for Mistral (the high-level planner).

    This is text-friendly and includes:
    - The original prompt/task
    - History of actions and results
    - Current state summary
    """
    prompt: str
    available_tools: List[str]
    available_experts: List[str]
    history: List[Dict]
    current_state: Dict
    step: int

    def to_text(self) -> str:
        """Convert to text format for Mistral's context."""
        lines = [
            f"Task: {self.prompt}",
            "",
            f"Available tools: {', '.join(self.available_tools)}",
            f"Available experts: {', '.join(self.available_experts)}",
            "",
            f"Step: {self.step}",
            "",
        ]

        if self.history:
            lines.append("Recent actions:")
            for h in self.history[-3:]:
                lines.append(f"  - {h.get('summary', str(h))}")
            lines.append("")

        if self.current_state:
            lines.append("Current state:")
            for key, value in self.current_state.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "prompt": self.prompt,
            "available_tools": self.available_tools,
            "available_experts": self.available_experts,
            "history": self.history,
            "current_state": self.current_state,
            "step": self.step,
        }


class ObservationBuilder:
    """
    Builds observations for different components.

    This centralizes the logic for converting raw state
    into appropriate observation formats.
    """

    def __init__(
        self,
        available_tools: List[str] = None,
        available_experts: List[str] = None
    ):
        self.available_tools = available_tools or []
        self.available_experts = available_experts or []

    def build_for_mistral(
        self,
        state: Dict[str, Any],
        history: List[Dict],
        step: int
    ) -> MistralObservation:
        """Build observation for Mistral."""
        return MistralObservation(
            prompt=state.get("prompt", ""),
            available_tools=self.available_tools,
            available_experts=self.available_experts,
            history=self._summarize_history(history),
            current_state=self._extract_current_state(state),
            step=step,
        )

    def build_for_expert(
        self,
        expert_name: str,
        state: Dict[str, Any],
        args: Dict[str, Any],
        previous_result: Any = None
    ) -> mx.array:
        """
        Build compact numerical observation for RNN expert.

        Each expert type has its own observation format.
        """
        if expert_name == "physics_controller":
            return self._build_physics_obs(state, args, previous_result)
        elif expert_name == "scheduler":
            return self._build_scheduler_obs(state, args, previous_result)
        elif expert_name == "arc_solver":
            return self._build_arc_obs(state, args, previous_result)
        else:
            # Default: generic observation
            return self._build_generic_obs(state, args, previous_result)

    def _summarize_history(self, history: List[Dict]) -> List[Dict]:
        """Create summaries of history entries."""
        summaries = []
        for h in history[-5:]:  # Last 5 entries
            action = h.get("action")
            result = h.get("result")

            summary = {"step": h.get("step", 0)}

            if action:
                if hasattr(action, "action_type"):
                    summary["action_type"] = action.action_type.value
                    if action.tool_name:
                        summary["tool"] = action.tool_name
                    if action.expert_name:
                        summary["expert"] = action.expert_name

            if result:
                if isinstance(result, dict):
                    summary["result"] = {k: v for k, v in result.items()
                                        if k in ["success", "error", "value", "distance"]}
                else:
                    summary["result"] = str(result)[:100]

            summaries.append(summary)

        return summaries

    def _extract_current_state(self, state: Dict) -> Dict:
        """Extract key current state information."""
        return {
            "goal": state.get("goal", {}),
            "constraints": state.get("constraints", {}),
            "results_count": len(state.get("results", [])),
            "last_result": state.get("results", [{}])[-1] if state.get("results") else None,
        }

    def _build_physics_obs(
        self,
        state: Dict,
        args: Dict,
        previous_result: Any
    ) -> mx.array:
        """Build observation for physics controller."""
        target = args.get("target", 100)
        wind_x = args.get("wind_x", 0)
        wind_y = args.get("wind_y", 0)

        # Extract from previous result
        if previous_result and isinstance(previous_result, dict):
            distance = previous_result.get("distance", 0)
            last_angle = previous_result.get("angle", 0)
            last_velocity = previous_result.get("velocity", 0)
        else:
            distance = 0
            last_angle = 0
            last_velocity = 0

        error = target - distance
        attempts = len([r for r in state.get("results", [])
                       if r.get("type") == "expert_delegation"])

        # Normalize to roughly [-1, 1] range
        obs = [
            target / 1000,              # 0: normalized target
            error / 1000,               # 1: normalized error
            distance / 1000,            # 2: normalized distance
            wind_x / 50,                # 3: normalized wind x
            wind_y / 50,                # 4: normalized wind y
            last_angle / 90,            # 5: normalized last angle
            last_velocity / 100,        # 6: normalized last velocity
            min(attempts / 10, 1.0),    # 7: normalized attempts
            1.0 if error > 0 else -1.0, # 8: direction indicator
            abs(error) / target if target > 0 else 0,  # 9: relative error
        ]

        return mx.array(obs, dtype=mx.float32)

    def _build_scheduler_obs(
        self,
        state: Dict,
        args: Dict,
        previous_result: Any
    ) -> mx.array:
        """Build observation for scheduler expert."""
        tasks = args.get("tasks", [])
        current_schedule = args.get("schedule", [])

        # Flatten task features
        obs = []
        for task in tasks[:10]:  # Max 10 tasks
            obs.extend([
                task.get("duration", 0) / 100,
                task.get("priority", 0) / 10,
                task.get("deadline", 0) / 1000,
            ])

        # Pad if fewer than 10 tasks
        while len(obs) < 30:
            obs.extend([0, 0, 0])

        # Add global state
        obs.extend([
            len(tasks) / 10,
            args.get("total_time", 0) / 1000,
            previous_result.get("score", 0) if previous_result else 0,
            0, 0  # padding
        ])

        return mx.array(obs[:35], dtype=mx.float32)

    def _build_arc_obs(
        self,
        state: Dict,
        args: Dict,
        previous_result: Any
    ) -> mx.array:
        """Build observation for ARC solver."""
        grid = args.get("grid", [[0]])
        goal_grid = args.get("goal_grid", [[0]])

        # Flatten grids (assume max 30x30)
        flat_grid = []
        flat_goal = []

        for row in grid[:30]:
            flat_grid.extend(row[:30])
        for row in goal_grid[:30]:
            flat_goal.extend(row[:30])

        # Pad to 900 elements each
        while len(flat_grid) < 900:
            flat_grid.append(0)
        while len(flat_goal) < 900:
            flat_goal.append(0)

        # Normalize (assuming values 0-9)
        obs = [v / 9 for v in flat_grid[:900]]
        obs.extend([v / 9 for v in flat_goal[:900]])

        # Add metadata
        obs.extend([
            args.get("step", 0) / 20,
            args.get("max_steps", 10) / 20,
            0, 0, 0  # padding
        ])

        return mx.array(obs[:1805], dtype=mx.float32)

    def _build_generic_obs(
        self,
        state: Dict,
        args: Dict,
        previous_result: Any
    ) -> mx.array:
        """Build generic observation when expert type is unknown."""
        obs = []

        # Flatten args (up to 20 values)
        for key, value in list(args.items())[:10]:
            if isinstance(value, (int, float)):
                obs.append(float(value))
            else:
                obs.append(0.0)

        # Pad to 20
        while len(obs) < 20:
            obs.append(0.0)

        return mx.array(obs, dtype=mx.float32)
