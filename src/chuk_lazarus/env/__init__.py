"""
Environment and orchestration for the hybrid architecture.

This module provides:
- Orchestrator: Coordinates Mistral, RNN Experts, and MCP Tools
- HybridEnv: Gym-like environment wrapper
- Action/observation utilities
- MCP Tool Client
"""

# Types
from .types import (
    Action,
    ActionType,
    OrchestratorConfig,
    StepResult,
)

# MCP Client
from .mcp_client import MCPToolClient

# Orchestrator
from .orchestrator import Orchestrator

# Environment
from .hybrid_env import EnvConfig, HybridEnv, VectorizedHybridEnv

__all__ = [
    "Action",
    "ActionType",
    "OrchestratorConfig",
    "StepResult",
    "MCPToolClient",
    "Orchestrator",
    "EnvConfig",
    "HybridEnv",
    "VectorizedHybridEnv",
]
