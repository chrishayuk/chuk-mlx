# Hybrid RL Architecture for chuk-mlx

## Overview

This module implements a hybrid architecture combining:
- **Mistral-7B** (or similar) as the high-level planner/router
- **Tiny RNN experts** as low-level controllers for iterative tasks
- **MCP tools** as the ground-truth environment
- **RL training** (DPO → PPO/GRPO) for optimization

## Directory Structure

```
rl/
├── __init__.py
├── README.md
│
├── losses/                    # RL loss functions
│   ├── __init__.py
│   ├── dpo_loss.py           # Direct Preference Optimization
│   ├── ppo_loss.py           # Proximal Policy Optimization
│   └── grpo_loss.py          # Group Relative Policy Optimization
│
├── experts/                   # Tiny RNN expert modules
│   ├── __init__.py
│   ├── rnn_expert_base.py    # Base class for RNN experts
│   ├── gru_expert.py         # GRU-based expert
│   ├── lstm_expert.py        # LSTM-based expert
│   └── registry.py           # Expert registration and lookup
│
├── env/                       # Environment and orchestration
│   ├── __init__.py
│   ├── orchestrator.py       # Main orchestrator (glues everything)
│   ├── hybrid_env.py         # Gym-like environment wrapper
│   ├── observation.py        # Observation builders
│   └── reward.py             # Reward computation
│
├── data/                      # RL-specific data handling
│   ├── __init__.py
│   ├── preference_dataset.py # DPO preference pairs
│   ├── rollout_buffer.py     # Episode/trajectory storage
│   └── episode.py            # Episode data structures
│
├── trainers/                  # Training loops
│   ├── __init__.py
│   ├── dpo_trainer.py        # DPO training loop
│   ├── ppo_trainer.py        # PPO training loop
│   └── grpo_trainer.py       # GRPO training loop
│
└── utils/                     # RL utilities
    ├── __init__.py
    ├── advantage.py          # GAE computation
    ├── log_probs.py          # Log probability extraction
    └── kl_divergence.py      # KL penalty computation
```

## Implementation Phases

### Phase 1: DPO Foundation
- Add log-prob extraction to existing models
- Implement DPO loss function
- Create preference pair data loader
- Minimal changes to existing training loop

### Phase 2: Tiny RNN Experts
- Implement GRU/LSTM expert base class
- Train standalone on control tasks
- Expose as callable modules

### Phase 3: Orchestrator
- Build environment wrapper
- Implement observation/action routing
- Connect Mistral ↔ RNN experts ↔ MCP tools

### Phase 4: Full RL Training
- Implement PPO/GRPO
- Rollout buffer and advantage estimation
- Joint training of Mistral + RNN experts

## Quick Start

### 1. Train a Physics Control Expert (RNN)

```bash
python -m rl.examples.train_physics_expert --timesteps 100000
```

This trains a small GRU expert to hit targets using projectile physics.

### 2. Fine-tune Mistral with DPO

```bash
python -m rl.examples.train_mistral_dpo \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --data ./data/tool_preferences.jsonl \
    --create-sample-data
```

This teaches Mistral to prefer tool use over hallucination.

### 3. Full Hybrid Training with GRPO

```bash
python -m rl.examples.train_hybrid_grpo \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --physics-expert ./checkpoints/physics_expert/best.npz
```

This trains the full hybrid system with live MCP tool calls during training.

## Architecture Overview

```
                         ┌──────────────────────────────┐
                         │          USER QUERY          │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       MISTRAL-7B BRAIN       │
                         │   (plans, routes, explains)  │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
            │   TOOL CALL   │   │   DELEGATE    │   │ FINAL ANSWER  │
            │  (direct MCP) │   │  (RNN expert) │   │   (respond)   │
            └───────┬───────┘   └───────┬───────┘   └───────────────┘
                    │                   │
                    ▼                   ▼
            ┌───────────────┐   ┌───────────────┐
            │  MCP SERVERS  │   │  TINY RNN     │
            │ math, physics │   │  EXPERT       │
            │ solver, etc.  │   │  (GRU/LSTM)   │
            └───────────────┘   └───────┬───────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │  MCP TOOLS    │
                                │ (iterative    │
                                │  control)     │
                                └───────────────┘
```

## Key Components

### Loss Functions (`rl/losses/`)

| Loss | Use Case | Complexity |
|------|----------|------------|
| **DPO** | Preference learning without RL | Low |
| **PPO** | Full RL with value function | High |
| **GRPO** | Group-based RL, no value function | Medium |

### Experts (`rl/experts/`)

Small RNN networks for specific control tasks:

```python
from rl.experts import GRUExpert, ExpertConfig

config = ExpertConfig(
    name="physics_controller",
    obs_dim=10,
    action_dim=2,  # angle, velocity
    hidden_dim=64,
)
expert = GRUExpert(config)
```

### Environment (`rl/env/`)

The Orchestrator coordinates all components:

```python
from rl.env import Orchestrator, MCPToolClient

mcp = MCPToolClient()
mcp.register_tool("math_solve", my_math_solver)

orchestrator = Orchestrator(config, expert_registry, mcp)
obs = orchestrator.reset(task)
result = orchestrator.step(action)
```

### Trainers (`rl/trainers/`)

Complete training loops:

```python
from rl.trainers import GRPOTrainer, GRPOTrainerConfig

trainer = GRPOTrainer(
    policy_model=mistral,
    reference_model=ref_mistral,
    tokenizer=tokenizer,
    reward_fn=compute_reward,  # Calls MCP tools!
)
trainer.train(prompt_source)
```

## Integration with chuk-mlx

This RL module integrates with the existing chuk-mlx infrastructure:

- Uses `core.models.model_loader` for loading Mistral
- Uses `core.utils.tokenizer_loader` for tokenizers
- Extends the training loop pattern from `training/`
- Works alongside existing SFT training

## MCP Tool Integration

The key innovation is **live MCP calls during training**:

1. **Data Pipeline Injection**: MCP verifies/generates training labels
2. **Reward Computation**: MCP tools score model outputs
3. **Expert Control**: RNN experts call MCP for iterative tasks

This means your training improves as your MCP servers improve!
