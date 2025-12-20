"""
Tiny RNN Expert modules.

These are small recurrent models (~10-100M params) that handle
specific iterative control tasks. They:
- Take compact numerical observations (not text)
- Maintain hidden state across steps
- Output actions (tool calls or decisions)
- Are trained with RL independently or jointly with the main LLM
"""

from .rnn_expert_base import RNNExpertBase, ExpertConfig
from .gru_expert import GRUExpert, GRUCell, create_physics_controller, create_scheduler_expert
from .lstm_expert import LSTMExpert, LSTMCell, create_planning_expert, create_arc_solver_expert
from .registry import ExpertRegistry, register_expert, get_expert, create_expert, list_experts
