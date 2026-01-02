"""
Jamba model implementation.

Jamba is a hybrid Mamba-Transformer MoE model from AI21 Labs.

Architecture:
- Most layers are Mamba blocks (SSM for O(n) complexity)
- Every 8th layer is a Transformer block (attention for recall)
- Every 2nd layer uses MoE instead of dense FFN
- All layers use pre-norm with RMSNorm
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...components.attention import GroupedQueryAttention
from ...components.embeddings import create_token_embedding
from ...components.ffn import MoE, SwiGLU
from ...components.normalization import RMSNorm
from ...components.ssm import Mamba
from ...core.config import AttentionConfig, FFNConfig, PositionConfig, RoPEConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import JambaConfig


class JambaMambaBlock(nn.Module):
    """
    Jamba Mamba block (non-attention layer).

    Pre-norm -> Mamba -> Residual -> Pre-norm -> FFN/MoE -> Residual
    """

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-Mamba norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Mamba layer
        self.mamba = Mamba(
            d_model=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            bias=config.mamba_proj_bias,
            conv_bias=config.mamba_conv_bias,
            use_mamba2_norms=getattr(config, "use_mamba2_norms", False),
            rms_norm_eps=config.rms_norm_eps,
        )

        # Pre-FFN norm
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN or MoE
        if config.is_moe_layer(layer_idx):
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
            self.feed_forward = MoE(ffn_config)
            self.is_moe = True
        else:
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
            self.feed_forward = SwiGLU(ffn_config)
            self.is_moe = False

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,  # Ignored for Mamba
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass."""
        # Mamba with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.mamba(x, cache)
        x = residual + x

        # FFN/MoE with residual
        residual = x
        x = self.pre_ff_layernorm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, new_cache


class JambaAttentionBlock(nn.Module):
    """
    Jamba Attention block (attention layer).

    Pre-norm -> GQA -> Residual -> Pre-norm -> FFN/MoE -> Residual
    """

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Position config with RoPE
        rope_config = RoPEConfig(
            theta=getattr(config, "rope_theta", 10000.0),
            max_position_embeddings=config.max_position_embeddings,
        )
        position_config = PositionConfig(rope=rope_config)

        # Attention
        attn_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            position=position_config,
        )
        self.self_attn = GroupedQueryAttention(attn_config)

        # Pre-FFN norm
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN or MoE
        if config.is_moe_layer(layer_idx):
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
            self.feed_forward = MoE(ffn_config)
            self.is_moe = True
        else:
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
            self.feed_forward = SwiGLU(ffn_config)
            self.is_moe = False

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass."""
        # Attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # FFN/MoE with residual
        residual = x
        x = self.pre_ff_layernorm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, new_cache


class JambaModel(Backbone):
    """
    Jamba backbone (without LM head).

    Hybrid Mamba-Transformer with MoE:
    - Token embeddings
    - Stack of hybrid blocks (Mamba or Attention based on layer pattern)
    - Final normalization
    """

    def __init__(self, config: JambaConfig):
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

        # Build hybrid layers
        self.layers = []
        for i in range(config.num_hidden_layers):
            if config.is_attention_layer(i):
                self.layers.append(JambaAttentionBlock(config, layer_idx=i))
            else:
                self.layers.append(JambaMambaBlock(config, layer_idx=i))

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
        _, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask for attention layers
        # When using cache, we need to create a mask that allows the new tokens
        # to attend to all cached positions. For single token generation with cache,
        # no mask is needed since we can attend to everything.
        if attention_mask is not None:
            mask = attention_mask
        elif cache is not None:
            # During incremental generation, new tokens can attend to all past tokens
            # For single token, we just need no masking (or a 1x(cache_len+1) mask of 0s)
            mask = None
        else:
            # Initial forward pass: use causal mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden_states, layer_new_cache = layer(hidden_states, mask=mask, cache=layer_cache)
            new_cache.append(layer_new_cache)

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
    model_type="jamba",
    architectures=["JambaForCausalLM"],
)
class JambaForCausalLM(Model):
    """
    Jamba for causal language modeling.

    Complete hybrid Mamba-Transformer MoE model with LM head.
    """

    def __init__(self, config: JambaConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = JambaModel(config)

        # LM head
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
    def config(self) -> JambaConfig:
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

        Jamba generation benefits from:
        - O(1) memory per step for Mamba layers (most layers)
        - Standard KV cache for sparse attention layers
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

    def sanitize(self, weights: dict) -> dict:
        """
        Convert HuggingFace weights to our format.

        This handles:
        - Mamba weight naming (mamba.in_proj -> mamba.ssm.in_proj)
        - MoE router naming (feed_forward.router -> feed_forward.router.gate)
        - Final norm naming (final_layernorm -> norm)
        - LM head naming (lm_head -> lm_head.lm_head)
        - Conv1d weight transposition (HF: out,1,k -> MLX: out,k,1)
        - Tied embeddings (skip lm_head.weight if tie_word_embeddings=True)
        """
        from .convert import _map_weight_name

        converted = {}
        for name, weight in weights.items():
            # Skip lm_head for tied embeddings
            if name == "lm_head.weight" and self._config.tie_word_embeddings:
                continue

            new_name = _map_weight_name(name)
            if new_name is not None:
                # Transpose conv1d weights from HF format (out, 1, kernel) to MLX format (out, kernel, 1)
                if "conv1d.weight" in new_name and weight.ndim == 3:
                    # HF: (out_channels, 1, kernel_size) -> MLX: (out_channels, kernel_size, 1)
                    weight = mx.transpose(weight, (0, 2, 1))
                converted[new_name] = weight

        return converted

    @classmethod
    def from_config(cls, config: JambaConfig) -> JambaForCausalLM:
        """Create from config."""
        return cls(config)

    @classmethod
    async def from_pretrained_async(
        cls,
        model_path: str,
        config: JambaConfig | None = None,
    ) -> JambaForCausalLM:
        """Load pretrained model."""
        import json
        from pathlib import Path

        path = Path(model_path)

        # Load config
        if config is None:
            config_path = path / "config.json"
            with open(config_path) as f:
                config_data = json.load(f)
            config = JambaConfig(**config_data)

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
