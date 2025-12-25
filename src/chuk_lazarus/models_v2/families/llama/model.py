"""
Llama model implementation.

Uses the composable architecture from models_v2.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.attention import GroupedQueryAttention, SlidingWindowAttention
from ...components.embeddings import create_token_embedding
from ...components.ffn import SwiGLU
from ...components.normalization import RMSNorm
from ...core.config import AttentionConfig, FFNConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import LlamaConfig


class LlamaBlock(Block):
    """
    Llama transformer block.

    Standard pre-norm transformer with:
    - RMSNorm
    - GQA or sliding window attention
    - SwiGLU FFN
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int = 0,
    ):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads

        # Pre-attention norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Attention config
        attn_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            sliding_window_size=config.sliding_window,
        )

        # Attention
        if config.sliding_window:
            self.self_attn = SlidingWindowAttention(attn_config)
        else:
            self.self_attn = GroupedQueryAttention(attn_config)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # FFN (SwiGLU)
        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.mlp = SwiGLU(ffn_config)

    @property
    def block_type(self):
        from ...core.enums import BlockType

        return BlockType.TRANSFORMER

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> BlockOutput:
        """Forward pass."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)


class LlamaModel(Backbone):
    """
    Llama backbone (without LM head).

    Just the embeddings + transformer blocks + final norm.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Transformer blocks
        self.layers = [LlamaBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        if attention_mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)
        else:
            mask = attention_mask

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            output = layer(hidden_states, mask=mask, cache=layer_cache)
            hidden_states = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


@register_model(
    model_type="llama",
    architectures=["LlamaForCausalLM", "MistralForCausalLM"],
)
class LlamaForCausalLM(Model):
    """
    Llama for causal language modeling.

    Complete model with backbone + LM head.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = LlamaModel(config)

        # LM head (optionally tied)
        if config.tie_word_embeddings:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                tied_embeddings=self.model.embed_tokens,
            )
        else:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
            )

    @property
    def config(self) -> LlamaConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass."""
        # Backbone
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        # LM head
        head_output = self.lm_head(
            hidden_states=backbone_output.last_hidden_state,
            labels=labels,
        )

        return ModelOutput(
            loss=head_output.loss,
            logits=head_output.logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
        )

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_tokens: list[int] | None = None,
    ) -> mx.array:
        """
        Generate text autoregressively.

        Args:
            input_ids: Prompt, shape (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: Tokens that stop generation

        Returns:
            Generated sequence, shape (batch, total_len)
        """
        stop_tokens_set = set(stop_tokens or [])

        # Process prompt and evaluate immediately
        output = self(input_ids)
        mx.eval(output.logits)
        cache = output.cache

        # Track generated tokens as list for efficiency
        generated_tokens = [input_ids]

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = output.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Get all generated tokens so far
                all_tokens = mx.concatenate(generated_tokens, axis=1)
                # Get unique tokens using Python set (evaluated once per step)
                unique_tokens = set(all_tokens.flatten().tolist())
                vocab_size = logits.shape[-1]
                # Create penalty mask vectorized
                token_indices = mx.array([t for t in unique_tokens if t < vocab_size])
                if token_indices.size > 0:
                    # One-hot encode and sum to get mask
                    mask = mx.zeros((vocab_size,))
                    for tok in token_indices.tolist():
                        mask = mask.at[tok].add(1.0)
                    penalty_mask = mx.where(mask > 0, repetition_penalty, 1.0)
                    logits = logits / penalty_mask

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None and top_k > 0:
                top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                min_val = top_k_values[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            # Sample next token
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            # Evaluate to avoid graph buildup
            mx.eval(next_token)

            # Append to generated list
            generated_tokens.append(next_token)

            # Check stop condition
            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            # Forward with cache
            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(cls, config: LlamaConfig) -> LlamaForCausalLM:
        """Create from config."""
        return cls(config)

    @classmethod
    async def from_pretrained_async(
        cls,
        model_path: str,
        config: LlamaConfig | None = None,
    ) -> LlamaForCausalLM:
        """Load pretrained model."""
        import json
        from pathlib import Path

        path = Path(model_path)

        # Load config
        if config is None:
            config_path = path / "config.json"
            with open(config_path) as f:
                config_data = json.load(f)
            config = LlamaConfig(**config_data)

        # Create model
        model = cls(config)

        # Load weights
        from .convert import convert_hf_weights

        weights_path = path / "model.safetensors"
        if weights_path.exists():
            try:
                import safetensors.numpy as st

                hf_weights = st.load_file(str(weights_path))
                weights = convert_hf_weights(hf_weights)
                weights = {k: mx.array(v) for k, v in weights.items()}
                model.update(weights)
            except ImportError:
                pass

        return model
