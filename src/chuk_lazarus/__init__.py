"""
CHUK-MLX: A unified MLX-based LLM training framework.

This package provides:
- Model loading with LoRA support
- Multiple training paradigms (SFT, DPO, GRPO, PPO)
- Data generation and preprocessing
- Hybrid LLM + RNN expert architecture
- MCP tool integration

Quick Start:
    from chuk_lazarus.models_v2 import LlamaForCausalLM, LlamaConfig
    from chuk_lazarus.training import SFTTrainer
    from chuk_lazarus.data import SFTDataset

    model = LlamaForCausalLM(LlamaConfig.llama2_7b())
    trainer = SFTTrainer(model, tokenizer)
    trainer.train(dataset)
"""

__version__ = "0.2.0"

# Re-export key components from models_v2 for convenience
from chuk_lazarus.models_v2 import (
    # Models
    CausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    # Adapters
    LoRAConfig,
    LoRALinear,
    MambaConfig,
    MambaForCausalLM,
    Model,
    ModelConfig,
    ModelOutput,
    SequenceClassifier,
    TokenClassifier,
    apply_lora,
    # Training
    compute_lm_loss,
    # Loader
    create_from_preset,
    create_model,
    load_model,
    load_model_async,
)

__all__ = [
    # Models
    "Model",
    "ModelOutput",
    "ModelConfig",
    "CausalLM",
    "SequenceClassifier",
    "TokenClassifier",
    # Families
    "LlamaConfig",
    "LlamaForCausalLM",
    "MambaConfig",
    "MambaForCausalLM",
    # Loader
    "load_model",
    "load_model_async",
    "create_model",
    "create_from_preset",
    # Adapters
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    # Training
    "compute_lm_loss",
]
