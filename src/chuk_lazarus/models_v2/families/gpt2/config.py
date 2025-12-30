"""
GPT-2 configuration.

Extends base ModelConfig with GPT-2-specific settings.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import (
    ConfigField,
    DefaultNormEps,
    DefaultPositionEmbeddings,
    DefaultVocabSize,
    HFModelType,
    SpecialTokenId,
)


class GPT2Config(ModelConfig):
    """
    Configuration for GPT-2 models.

    Extends ModelConfig with GPT-2-specific defaults and validation.

    Supports:
    - GPT-2 Small (117M)
    - GPT-2 Medium (345M)
    - GPT-2 Large (762M)
    - GPT-2 XL (1.5B)
    - DistilGPT-2

    Example:
        >>> # GPT-2 Small
        >>> config = GPT2Config(
        ...     vocab_size=DefaultVocabSize.GPT2,
        ...     hidden_size=768,
        ...     num_hidden_layers=12,
        ...     num_attention_heads=12,
        ...     intermediate_size=3072,  # 4 * hidden_size
        ... )
    """

    model_type: str = HFModelType.GPT2.value

    # GPT-2 specific defaults
    hidden_act: str = "gelu_new"  # GPT-2 uses approximate GELU
    layer_norm_eps: float = DefaultNormEps.GPT2.value

    # GPT-2 uses learned positional embeddings (not RoPE)
    use_learned_position_embeddings: bool = True

    # Attention settings
    attention_bias: bool = True  # GPT-2 uses bias
    mlp_bias: bool = True

    @classmethod
    def gpt2_small(cls) -> GPT2Config:
        """Create GPT-2 Small (117M) configuration."""
        return cls(
            vocab_size=DefaultVocabSize.GPT2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,  # MHA
            intermediate_size=3072,  # 4 * hidden_size
            max_position_embeddings=DefaultPositionEmbeddings.GPT2,
        )

    @classmethod
    def gpt2_medium(cls) -> GPT2Config:
        """Create GPT-2 Medium (345M) configuration."""
        return cls(
            vocab_size=DefaultVocabSize.GPT2,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=4096,
            max_position_embeddings=DefaultPositionEmbeddings.GPT2,
        )

    @classmethod
    def gpt2_large(cls) -> GPT2Config:
        """Create GPT-2 Large (762M) configuration."""
        return cls(
            vocab_size=DefaultVocabSize.GPT2,
            hidden_size=1280,
            num_hidden_layers=36,
            num_attention_heads=20,
            num_key_value_heads=20,
            intermediate_size=5120,
            max_position_embeddings=DefaultPositionEmbeddings.GPT2,
        )

    @classmethod
    def gpt2_xl(cls) -> GPT2Config:
        """Create GPT-2 XL (1.5B) configuration."""
        return cls(
            vocab_size=DefaultVocabSize.GPT2,
            hidden_size=1600,
            num_hidden_layers=48,
            num_attention_heads=25,
            num_key_value_heads=25,
            intermediate_size=6400,
            max_position_embeddings=DefaultPositionEmbeddings.GPT2,
        )

    @classmethod
    def distilgpt2(cls) -> GPT2Config:
        """Create DistilGPT-2 configuration."""
        return cls(
            vocab_size=DefaultVocabSize.GPT2,
            hidden_size=768,
            num_hidden_layers=6,  # Reduced from 12
            num_attention_heads=12,
            num_key_value_heads=12,
            intermediate_size=3072,
            max_position_embeddings=DefaultPositionEmbeddings.GPT2,
        )

    @classmethod
    def tiny(cls) -> GPT2Config:
        """Create tiny GPT-2 for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
            max_position_embeddings=256,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> GPT2Config:
        """
        Create config from HuggingFace config.json dict.

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used for GPT-2)

        Returns:
            GPT2Config instance

        Example:
            >>> import json
            >>> with open("config.json") as f:
            ...     hf_config = json.load(f)
            >>> config = GPT2Config.from_hf_config(hf_config)
        """
        # GPT-2 uses n_embd/n_layer/n_head naming
        hidden_size = hf_config.get(ConfigField.N_EMBD, hf_config.get(ConfigField.HIDDEN_SIZE, 768))
        num_layers = hf_config.get(
            ConfigField.N_LAYER, hf_config.get(ConfigField.NUM_HIDDEN_LAYERS, 12)
        )
        num_heads = hf_config.get(
            ConfigField.N_HEAD, hf_config.get(ConfigField.NUM_ATTENTION_HEADS, 12)
        )

        # GPT-2 uses 4 * hidden_size for intermediate (inner) dimension
        intermediate = hf_config.get(
            ConfigField.N_INNER,
            hf_config.get(ConfigField.INTERMEDIATE_SIZE, 4 * hidden_size),
        )

        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE, HFModelType.GPT2.value),
            vocab_size=hf_config.get(ConfigField.VOCAB_SIZE, DefaultVocabSize.GPT2),
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # GPT-2 uses MHA
            intermediate_size=intermediate,
            max_position_embeddings=hf_config.get(
                ConfigField.N_POSITIONS, DefaultPositionEmbeddings.GPT2
            ),
            layer_norm_eps=hf_config.get(ConfigField.LAYER_NORM_EPSILON, DefaultNormEps.GPT2.value),
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS, True),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID, SpecialTokenId.GPT2_BOS),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID, SpecialTokenId.GPT2_EOS),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID, SpecialTokenId.GPT2_EOS),
        )
