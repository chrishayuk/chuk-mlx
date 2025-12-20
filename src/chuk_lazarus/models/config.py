"""Model and LoRA configuration."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class ModelConfig:
    """
    Unified model configuration.

    Compatible with HuggingFace config.json format.
    """
    # Architecture
    architectures: Optional[List[str]] = None
    model_type: Optional[str] = None

    # Dimensions
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32

    # Attention
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For GQA, defaults to num_attention_heads
    head_dim: Optional[int] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Activation
    hidden_act: str = "silu"
    hidden_activation: Optional[str] = None  # Gemma uses this

    # Normalization
    rms_norm_eps: float = 1e-6
    layer_norm: bool = True

    # Position embeddings
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False

    # Special tokens
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2
    pad_token_id: Optional[int] = None

    # Embeddings
    tie_word_embeddings: bool = True

    # MLP
    mlp_bias: bool = False

    # Dropout
    attention_dropout_prob: Optional[float] = None
    residual_dropout_prob: Optional[float] = None

    def __post_init__(self):
        # Handle GQA: default num_key_value_heads to num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Handle head_dim
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Handle Gemma's hidden_activation
        if self.hidden_activation and not self.hidden_act:
            self.hidden_act = self.hidden_activation

        # Validate rope_scaling
        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'ModelConfig':
        """Load config from JSON file."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"

        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        """Create config from dict, ignoring unknown fields."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dict."""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]):
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def num_layers(self) -> int:
        """Alias for num_hidden_layers."""
        return self.num_hidden_layers
