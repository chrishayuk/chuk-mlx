"""
StarCoder and StarCoder2 model implementations.

Uses the composable architecture from models_v2.

StarCoder (original) differences from Llama:
- LayerNorm instead of RMSNorm
- GELU activation instead of SiLU/SwiGLU
- Standard MLP instead of gated MLP
- Bias in linear layers
- Learned positional embeddings (not RoPE)
- Multi-Query Attention (MQA)

StarCoder2 differences from StarCoder:
- Uses RoPE positional embeddings
- Uses Grouped Query Attention (GQA)
- Sliding window attention
- Larger context (16K vs 8K)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.attention import GroupedQueryAttention, SlidingWindowAttention
from ...components.embeddings import create_token_embedding
from ...components.ffn import MLP
from ...components.normalization import LayerNorm
from ...core.config import AttentionConfig, FFNConfig, PositionConfig, RoPEConfig
from ...core.enums import ActivationType, PositionEmbeddingType
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import StarCoder2Config, StarCoderConfig


class StarCoder2Block(Block):
    """
    StarCoder2 transformer block.

    Standard pre-norm transformer with:
    - LayerNorm (not RMSNorm)
    - GQA or sliding window attention
    - Standard MLP with GELU (not SwiGLU)
    - Bias in projections
    """

    def __init__(
        self,
        config: StarCoder2Config,
        layer_idx: int = 0,
    ):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        norm_eps = getattr(config, "layer_norm_eps", config.rms_norm_eps)

        # Pre-attention norm (LayerNorm for StarCoder2)
        self.input_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=norm_eps,
        )

        # RoPE config with proper theta
        rope_config = RoPEConfig(
            theta=config.rope_theta,
            traditional=False,
            max_position_embeddings=config.max_position_embeddings,
        )
        position_config = PositionConfig(
            rope=rope_config,
            max_position_embeddings=config.max_position_embeddings,
        )

        # Attention config with proper RoPE
        attn_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            sliding_window_size=config.sliding_window,
            attention_bias=config.attention_bias,
            position=position_config,
        )

        # Attention (with sliding window support)
        if config.sliding_window:
            self.self_attn = SlidingWindowAttention(attn_config)
        else:
            self.self_attn = GroupedQueryAttention(attn_config)

        # Post-attention norm (LayerNorm)
        self.post_attention_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=norm_eps,
        )

        # FFN (standard MLP with GELU_TANH, not SwiGLU)
        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=ActivationType.GELU_TANH,
            bias=config.mlp_bias,
        )
        self.mlp = MLP(ffn_config)

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


class StarCoder2Model(Backbone):
    """
    StarCoder2 backbone (without LM head).

    Just the embeddings + transformer blocks + final norm.
    """

    def __init__(self, config: StarCoder2Config):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        norm_eps = getattr(config, "layer_norm_eps", config.rms_norm_eps)

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Transformer blocks
        self.layers = [
            StarCoder2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]

        # Final norm (LayerNorm for StarCoder2)
        self.norm = LayerNorm(dims=config.hidden_size, eps=norm_eps)

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
        _, seq_len = input_ids.shape

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
    model_type="starcoder2",
    architectures=["Starcoder2ForCausalLM"],
)
class StarCoder2ForCausalLM(Model):
    """
    StarCoder2 for causal language modeling.

    Complete model with backbone + LM head.
    """

    def __init__(self, config: StarCoder2Config):
        super().__init__()

        self._config = config

        # Backbone
        self.model = StarCoder2Model(config)

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
    def config(self) -> StarCoder2Config:
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
        Generate code autoregressively.

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
    def from_config(cls, config: StarCoder2Config) -> StarCoder2ForCausalLM:
        """Create from config."""
        return cls(config)

    @classmethod
    async def from_pretrained_async(
        cls,
        model_path: str,
        config: StarCoder2Config | None = None,
    ) -> StarCoder2ForCausalLM:
        """Load pretrained model."""
        import json
        from pathlib import Path

        path = Path(model_path)

        # Load config
        if config is None:
            config_path = path / "config.json"
            with open(config_path) as f:
                config_data = json.load(f)
            config = StarCoder2Config(**config_data)

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


# =============================================================================
# StarCoder (Original) - GPT-BigCode Architecture
# =============================================================================


class StarCoderBlock(Block):
    """
    StarCoder (original) transformer block.

    GPT-2 style pre-norm transformer with:
    - LayerNorm (not RMSNorm)
    - Multi-Query Attention (MQA)
    - Standard MLP with GELU (not SwiGLU)
    - Bias in projections
    - NO RoPE (uses learned position embeddings)
    """

    def __init__(
        self,
        config: StarCoderConfig,
        layer_idx: int = 0,
    ):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads or 1  # MQA default
        norm_eps = getattr(config, "layer_norm_eps", config.rms_norm_eps)

        # Pre-attention norm (LayerNorm)
        self.input_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=norm_eps,
        )

        # Position config with LEARNED type (no RoPE - learned embeddings at model level)
        position_config = PositionConfig(
            position_type=PositionEmbeddingType.LEARNED,
            max_position_embeddings=config.max_position_embeddings,
            rope=None,
        )

        # Attention config WITHOUT RoPE (learned positions added at embedding level)
        attn_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            attention_bias=config.attention_bias,
            position=position_config,
        )

        # Multi-Query Attention
        self.self_attn = GroupedQueryAttention(attn_config)

        # Post-attention norm (LayerNorm)
        self.post_attention_layernorm = LayerNorm(
            dims=config.hidden_size,
            eps=norm_eps,
        )

        # FFN (standard MLP with GELU_TANH)
        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=ActivationType.GELU_TANH,
            bias=config.mlp_bias,
        )
        self.mlp = MLP(ffn_config)

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


class StarCoderModel(Backbone):
    """
    StarCoder (original) backbone (without LM head).

    GPT-2 style model with learned position embeddings.
    """

    def __init__(self, config: StarCoderConfig):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        norm_eps = getattr(config, "layer_norm_eps", config.rms_norm_eps)

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Learned position embeddings (GPT-2 style)
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        # Transformer blocks
        self.layers = [StarCoderBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]

        # Final norm (LayerNorm)
        self.norm = LayerNorm(dims=config.hidden_size, eps=norm_eps)

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
        position_ids: mx.array | None = None,
    ) -> BackboneOutput:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Get position IDs
        if position_ids is None:
            if cache is not None and len(cache) > 0 and cache[0] is not None:
                # When using cache, we're generating one token at a time
                # Position is the length of the cached sequence
                past_length = cache[0][0].shape[2] if cache[0] is not None else 0
                position_ids = mx.arange(past_length, past_length + seq_len)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
            else:
                position_ids = mx.arange(seq_len)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))

        # Token + position embeddings
        token_embeddings = self.embed_tokens(input_ids)
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = token_embeddings + position_embeddings

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
    model_type="gpt_bigcode",
    architectures=["GPTBigCodeForCausalLM"],
)
class StarCoderForCausalLM(Model):
    """
    StarCoder (original) for causal language modeling.

    Complete model with backbone + LM head.
    """

    def __init__(self, config: StarCoderConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = StarCoderModel(config)

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
    def config(self) -> StarCoderConfig:
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
        position_ids: mx.array | None = None,
    ) -> ModelOutput:
        """Forward pass."""
        # Backbone
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
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
        Generate code autoregressively.

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
                all_tokens = mx.concatenate(generated_tokens, axis=1)
                unique_tokens = set(all_tokens.flatten().tolist())
                vocab_size = logits.shape[-1]
                token_indices = mx.array([t for t in unique_tokens if t < vocab_size])
                if token_indices.size > 0:
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
    def from_config(cls, config: StarCoderConfig) -> StarCoderForCausalLM:
        """Create from config."""
        return cls(config)
