"""
Environment and orchestration for the hybrid architecture.

This module provides:
- Orchestrator: Coordinates Mistral, RNN Experts, and MCP Tools
- HybridEnv: Gym-like environment wrapper
- Action/observation utilities
"""

from .orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    Action,
    ActionType,
    StepResult,
    MCPToolClient,
)
from .hybrid_env import HybridEnv, VectorizedHybridEnv, EnvConfig
