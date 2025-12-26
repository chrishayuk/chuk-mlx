"""
Granite 4.x hybrid model implementation.

Hybrid Mamba-2/Transformer architecture with optional MoE.

Reference: https://www.ibm.com/granite/docs/models/granite
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.embeddings import create_token_embedding
from ...components.ffn import SwiGLU
from ...components.normalization import RMSNorm
from ...core.config import FFNConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import GraniteHybridConfig


class GraniteMamba2Block(nn.Module):
    """
    Mamba-2 block for Granite 4.x.

    Simplified Mamba-2 implementation using the selective scan mechanism.
    """

    def __init__(self, config: GraniteHybridConfig, layer_idx: int = 0):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Mamba-2 parameters
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.n_heads = config.mamba_n_heads
        self.d_head = config.mamba_d_head

        # Expanded dimension - ensure it's divisible for groups
        self.d_inner = self.expand * self.hidden_size

        # Input projection (x -> x, z for gating)
        self.in_proj = nn.Linear(
            self.hidden_size,
            self.d_inner * 2,
            bias=config.mamba_proj_bias,
        )

        # Convolution - use groups=1 for depthwise to work with any size
        # For true depthwise conv, d_inner must be divisible
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=1,  # Standard conv instead of depthwise for simplicity
            bias=config.mamba_conv_bias,
            padding=self.d_conv - 1,
        )

        # SSM projections
        # For Mamba-2, we use a simpler multi-head structure
        self.dt_proj = nn.Linear(self.d_inner, self.n_heads, bias=True)
        self.A = mx.ones((self.n_heads,)) * -1.0  # Learnable decay
        self.D = mx.ones((self.n_heads,))  # Skip connection

        # B and C projections
        self.B_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.n_heads * self.d_state, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=config.mamba_proj_bias)

        # Norm
        self.norm = RMSNorm(self.d_inner, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: dict[str, mx.array] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        """Forward pass through Mamba-2 block."""
        batch_size, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)
        x_proj, z = mx.split(xz, 2, axis=-1)

        # Convolution (causal) - MLX Conv1d expects (B, L, C) format
        # Apply 1D conv along sequence dimension
        # Use manual sliding window for simplicity
        x_conv = x_proj
        if self.d_conv > 1:
            # Pad for causal conv
            padding = mx.zeros((batch_size, self.d_conv - 1, self.d_inner))
            x_padded = mx.concatenate([padding, x_proj], axis=1)
            # Simple causal conv via linear combination of shifted inputs
            conv_out = mx.zeros_like(x_proj)
            for i in range(self.d_conv):
                shift = self.d_conv - 1 - i
                conv_out = conv_out + x_padded[:, shift : shift + seq_len, :]
            x_conv = conv_out / self.d_conv  # Normalize

        # Apply SiLU
        x_conv = nn.silu(x_conv)

        # SSM parameters
        dt = nn.softplus(self.dt_proj(x_conv))  # (B, L, n_heads)
        B = self.B_proj(x_conv).reshape(batch_size, seq_len, self.n_heads, self.d_state)
        C = self.C_proj(x_conv).reshape(batch_size, seq_len, self.n_heads, self.d_state)

        # Reshape x for multi-head
        x_heads = x_conv.reshape(batch_size, seq_len, self.n_heads, -1)  # (B, L, H, D/H)

        # Selective scan (simplified)
        # For each position, compute: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t
        y = self._selective_scan(x_heads, dt, B, C, cache)

        # Reshape back
        y = y.reshape(batch_size, seq_len, self.d_inner)

        # Normalize
        y = self.norm(y)

        # Gate with z
        y = y * nn.silu(z)

        # Output projection
        y = self.out_proj(y)

        # Update cache
        new_cache = None  # Simplified - no cache for now

        return y, new_cache

    def _selective_scan(
        self,
        x: mx.array,
        dt: mx.array,
        B: mx.array,
        C: mx.array,
        cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        """
        Simplified selective scan.

        For full efficiency, this should use the chunked algorithm from Mamba-2.
        This is a reference implementation.
        """
        batch_size, seq_len, n_heads, d_per_head = x.shape

        # Initialize state
        h = mx.zeros((batch_size, n_heads, self.d_state))

        outputs = []
        for t in range(seq_len):
            # Get inputs at time t
            x_t = x[:, t, :, :]  # (B, H, D/H)
            dt_t = dt[:, t, :]  # (B, H)
            B_t = B[:, t, :, :]  # (B, H, N)
            C_t = C[:, t, :, :]  # (B, H, N)

            # Discretize A
            A_bar = mx.exp(self.A * dt_t)  # (B, H)

            # Update state: h = A_bar * h + B * x
            # Simplified: use mean of x across d_per_head
            x_mean = mx.mean(x_t, axis=-1, keepdims=True)  # (B, H, 1)
            h = A_bar[:, :, None] * h + B_t * x_mean

            # Output: y = C * h + D * x
            y_t = mx.sum(C_t * h, axis=-1)  # (B, H)
            y_t = y_t + self.D * mx.mean(x_t, axis=-1)

            # Expand back to d_per_head
            y_t = mx.expand_dims(y_t, axis=-1)
            y_t = mx.broadcast_to(y_t, (batch_size, n_heads, d_per_head))

            outputs.append(y_t)

        # Stack outputs
        y = mx.stack(outputs, axis=1)  # (B, L, H, D/H)

        return y


class GraniteHybridAttention(nn.Module):
    """
    Attention block for Granite 4.x hybrid.

    Similar to standard GQA but with Granite-specific multipliers.
    """

    def __init__(self, config: GraniteHybridConfig, layer_idx: int = 0):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        self.attention_multiplier = config.attention_multiplier
        self.n_rep = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # RoPE (only if using rope position embeddings)
        self.use_rope = config.position_embedding_type == "rope"
        if self.use_rope:
            from ...components.embeddings.rope import RoPE
            from ...core.config import RoPEConfig

            rope_config = RoPEConfig(
                theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
            )
            self.rope = RoPE(rope_config, dims=self.head_dim)

        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE
        if self.use_rope:
            offset = 0
            if cache is not None:
                offset = cache[0].shape[2]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        # Cache
        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = (k, v)

        # Repeat KV
        if self.n_rep > 1:
            batch, num_kv_heads, kv_seq_len, head_dim = k.shape
            k = mx.expand_dims(k, axis=2)
            k = mx.repeat(k, self.n_rep, axis=2)
            k = k.reshape(batch, num_kv_heads * self.n_rep, kv_seq_len, head_dim)
            v = mx.expand_dims(v, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)
            v = v.reshape(batch, num_kv_heads * self.n_rep, kv_seq_len, head_dim)

        # Attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output with multiplier
        output = self.o_proj(output)
        output = output * self.attention_multiplier

        return output, new_cache


class GraniteHybridMoE(nn.Module):
    """
    MoE layer for Granite 4.x with shared expert.

    Similar to Llama 4 MoE but with Granite-specific settings.
    """

    def __init__(self, config: GraniteHybridConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.shared_intermediate_size = config.shared_intermediate_size

        # Router
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)

        # Shared expert (if configured)
        self.has_shared_expert = config.shared_intermediate_size > 0
        if self.has_shared_expert:
            shared_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_intermediate_size,
            )
            self.shared_expert = SwiGLU(shared_config)

        # Routed experts
        expert_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.experts = [SwiGLU(expert_config) for _ in range(config.num_local_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through MoE."""
        batch_size, seq_len, hidden_size = x.shape

        # Shared expert output
        if self.has_shared_expert:
            shared_output = self.shared_expert(x)
        else:
            shared_output = mx.zeros_like(x)

        # Router
        router_logits = self.router(x)
        router_scores = mx.sigmoid(router_logits)

        # Top-k selection
        sorted_indices = mx.argsort(-router_logits, axis=-1)
        top_k_indices = sorted_indices[:, :, : self.num_experts_per_tok]
        top_k_scores = mx.take_along_axis(router_scores, top_k_indices, axis=-1)

        # Compute routed outputs
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = top_k_indices.reshape(-1, self.num_experts_per_tok)
        scores_flat = top_k_scores.reshape(-1, self.num_experts_per_tok)

        routed_output = mx.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            expert_mask = indices_flat == expert_idx
            expert_weights = mx.sum(
                scores_flat * expert_mask.astype(scores_flat.dtype), axis=-1, keepdims=True
            )
            if mx.any(expert_weights > 0):
                expert_out = expert(x_flat)
                routed_output = routed_output + expert_out * expert_weights

        routed_output = routed_output.reshape(batch_size, seq_len, hidden_size)

        return shared_output + routed_output


class GraniteHybridBlock(Block):
    """
    Granite 4.x hybrid block.

    Can be either a Mamba-2 block or an attention block based on layer_type.
    """

    def __init__(
        self,
        config: GraniteHybridConfig,
        layer_idx: int = 0,
        layer_type: str = "attention",
    ):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.residual_multiplier = config.residual_multiplier

        # Pre-block norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Block type
        if layer_type == "mamba":
            self.block = GraniteMamba2Block(config, layer_idx=layer_idx)
        else:
            self.block = GraniteHybridAttention(config, layer_idx=layer_idx)

        # Post-block norm
        self.post_block_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN (dense or MoE)
        if config.is_moe:
            self.mlp = GraniteHybridMoE(config)
        else:
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size
                if config.shared_intermediate_size == 0
                else config.shared_intermediate_size,
            )
            self.mlp = SwiGLU(ffn_config)

    @property
    def block_type(self):
        from ...core.enums import BlockType

        if self.layer_type == "mamba":
            return BlockType.MAMBA
        return BlockType.TRANSFORMER

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> BlockOutput:
        """Forward pass."""
        # Block (Mamba or Attention)
        residual = x
        x = self.input_layernorm(x)

        if self.layer_type == "mamba":
            x, new_cache = self.block(x, cache=cache)
        else:
            x, new_cache = self.block(x, mask=mask, cache=cache)

        x = residual + x * self.residual_multiplier

        # FFN
        residual = x
        x = self.post_block_layernorm(x)
        x = self.mlp(x)
        x = residual + x * self.residual_multiplier

        return BlockOutput(hidden_states=x, cache=new_cache)


class GraniteHybridModel(Backbone):
    """
    Granite 4.x hybrid backbone.

    Interleaved Mamba-2 and Transformer blocks with optional MoE.
    """

    def __init__(self, config: GraniteHybridConfig):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers
        self.embedding_multiplier = config.embedding_multiplier

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Hybrid blocks
        self.layers = [
            GraniteHybridBlock(
                config,
                layer_idx=i,
                layer_type=config.layer_types[i] if i < len(config.layer_types) else "attention",
            )
            for i in range(config.num_hidden_layers)
        ]

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

        # Embeddings with multiplier
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embedding_multiplier

        # Create causal mask (for attention layers)
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
    model_type="granitemoehybrid",
    architectures=["GraniteMoeHybridForCausalLM"],
)
class GraniteHybridForCausalLM(Model):
    """
    Granite 4.x hybrid for causal language modeling.

    Supports:
    - Dense hybrid (Micro): All attention layers
    - MoE hybrid (Tiny, Small): Mixed Mamba-2 + Attention with MoE
    """

    def __init__(self, config: GraniteHybridConfig):
        super().__init__()

        self._config = config
        self.logits_scaling = config.logits_scaling

        # Backbone
        self.model = GraniteHybridModel(config)

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
    def config(self) -> GraniteHybridConfig:
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
        """Forward pass with logits scaling."""
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        head_output = self.lm_head(
            hidden_states=backbone_output.last_hidden_state,
            labels=labels,
        )

        logits = head_output.logits
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling

        return ModelOutput(
            loss=head_output.loss,
            logits=logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
        )

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_tokens: list[int] | None = None,
    ) -> mx.array:
        """Generate text autoregressively."""
        stop_tokens_set = set(stop_tokens or [])

        output = self(input_ids)
        mx.eval(output.logits)
        cache = output.cache

        generated_tokens = [input_ids]

        for _ in range(max_new_tokens):
            logits = output.logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                min_val = top_k_values[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            mx.eval(next_token)
            generated_tokens.append(next_token)

            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(cls, config: GraniteHybridConfig) -> GraniteHybridForCausalLM:
        """Create from config."""
        return cls(config)


# Convenience aliases
GraniteHybrid = GraniteHybridForCausalLM
Granite4 = GraniteHybridForCausalLM
