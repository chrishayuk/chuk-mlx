"""
Model loading and architectures.

This module provides:
- Unified model loading from HuggingFace or local paths
- LoRA adapter support
- Model configuration
- Architecture implementations (Llama, Mistral, Gemma, Starcoder2, Granite, etc.)
- MLP implementations (SwiGLU, GELU-GLU, ReLU)
- Transformer blocks
"""

# Configuration
from .config import ModelConfig as SimpleModelConfig, LoRAConfig
from .model_config import ModelConfig

# Loading (simple API)
from .loader import load_model, load_tokenizer

# Loading (full API)
from .model_loader import (
    get_model_from_path,
    load_model as load_model_full,
    load_model_tokenizer_and_checkpoint,
)

# LoRA
from .lora import LoRALinear, apply_lora

# Inference (re-exported from chuk_lazarus.inference for convenience)
from chuk_lazarus.inference import generate_sequence, generate_response

# Loss functions
from .chuk_loss_function import chukloss
from .loss_function_loader import load_loss_function

# Weights
from .load_weights import (
    load_model_weights,
    load_checkpoint_weights,
    load_safetensors,
)

# MLX adapter
from .mlx_adapter import MLXAdapter
