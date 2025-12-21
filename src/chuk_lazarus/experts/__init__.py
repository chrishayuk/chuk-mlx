"""
Tiny RNN Expert modules.

These are small recurrent models (~10-100M params) that handle
specific iterative control tasks. They:
- Take compact numerical observations (not text)
- Maintain hidden state across steps
- Output actions (tool calls or decisions)
- Are trained with RL independently or jointly with the main LLM
"""

from .gru_expert import GRUCell, GRUExpert, create_physics_controller, create_scheduler_expert
from .lstm_expert import LSTMCell, LSTMExpert, create_arc_solver_expert, create_planning_expert
from .registry import ExpertRegistry, create_expert, get_expert, list_experts, register_expert
from .rnn_expert_base import ExpertConfig, RNNExpertBase

__all__ = [
    "GRUCell",
    "GRUExpert",
    "create_physics_controller",
    "create_scheduler_expert",
    "LSTMCell",
    "LSTMExpert",
    "create_arc_solver_expert",
    "create_planning_expert",
    "ExpertRegistry",
    "create_expert",
    "get_expert",
    "list_experts",
    "register_expert",
    "ExpertConfig",
    "RNNExpertBase",
]
