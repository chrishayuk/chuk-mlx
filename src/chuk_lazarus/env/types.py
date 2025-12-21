"""
Type definitions for the orchestration system.

This module contains the core data structures used across the environment:
- ActionType: Enum of possible action types
- Action: Represents an action from Mistral or RNN expert
- StepResult: Result from executing an action
- OrchestratorConfig: Configuration for the orchestrator
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


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
