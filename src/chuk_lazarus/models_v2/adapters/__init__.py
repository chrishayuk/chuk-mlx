"""
Adapter modules for efficient fine-tuning.

Provides LoRA (Low-Rank Adaptation) and other PEFT methods.
"""

from .lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    merge_lora_weights,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    "merge_lora_weights",
    "count_lora_parameters",
]
