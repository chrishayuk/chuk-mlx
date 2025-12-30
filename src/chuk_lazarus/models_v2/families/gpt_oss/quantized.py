"""
Quantized GPT-OSS model implementation.

This module provides 8-bit quantized versions of GPT-OSS components
for use with the lmstudio-community/gpt-oss-20b-MLX-8bit model.

The quantized model uses:
- QuantizedLinear for all projections
- Batched experts (single tensor with num_experts as first dimension)
- Attention sinks for streaming inference
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...components.normalization import RMSNorm
from ...models.base import Model, ModelOutput
from .config import GptOssConfig


class BatchedQuantizedExperts(nn.Module):
    """
    Batched quantized MoE experts.

    Stores all expert weights in batched tensors for efficient loading
    from the MLX format which has shape (num_experts, out_features, in_features).

    Uses separate gate_proj and up_proj (not fused) to match the weight format.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # These will be set during weight loading
        # Shapes: (num_experts, intermediate_size, hidden_size) for gate/up
        #         (num_experts, hidden_size, intermediate_size) for down
        self.gate_proj = {
            "weight": None,
            "scales": None,
            "biases": None,
            "bias": None,
        }
        self.up_proj = {
            "weight": None,
            "scales": None,
            "biases": None,
            "bias": None,
        }
        self.down_proj = {
            "weight": None,
            "scales": None,
            "biases": None,
            "bias": None,
        }

    def __call__(self, x: mx.array, expert_idx: int) -> mx.array:
        """
        Apply a single expert to the input.

        Args:
            x: Input tensor (batch, hidden_size)
            expert_idx: Index of expert to use

        Returns:
            Output tensor (batch, hidden_size)
        """
        # Get expert weights
        gate_w = self.gate_proj["weight"][expert_idx]
        gate_s = self.gate_proj["scales"][expert_idx]
        gate_b = self.gate_proj["biases"][expert_idx]
        gate_bias = (
            self.gate_proj["bias"][expert_idx] if self.gate_proj["bias"] is not None else None
        )

        up_w = self.up_proj["weight"][expert_idx]
        up_s = self.up_proj["scales"][expert_idx]
        up_b = self.up_proj["biases"][expert_idx]
        up_bias = self.up_proj["bias"][expert_idx] if self.up_proj["bias"] is not None else None

        down_w = self.down_proj["weight"][expert_idx]
        down_s = self.down_proj["scales"][expert_idx]
        down_b = self.down_proj["biases"][expert_idx]
        down_bias = (
            self.down_proj["bias"][expert_idx] if self.down_proj["bias"] is not None else None
        )

        # Dequantize and apply gate projection
        gate_out = mx.quantized_matmul(x, gate_w, scales=gate_s, biases=gate_b)
        if gate_bias is not None:
            gate_out = gate_out + gate_bias

        # Dequantize and apply up projection
        up_out = mx.quantized_matmul(x, up_w, scales=up_s, biases=up_b)
        if up_bias is not None:
            up_out = up_out + up_bias

        # SwiGLU
        hidden = nn.silu(gate_out) * up_out

        # Dequantize and apply down projection
        out = mx.quantized_matmul(hidden, down_w, scales=down_s, biases=down_b)
        if down_bias is not None:
            out = out + down_bias

        return out


class QuantizedMoERouter(nn.Module):
    """Quantized expert router."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 8,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Quantized router gate
        self.gate = nn.QuantizedLinear(
            hidden_size, num_experts, bias=bias, group_size=group_size, bits=bits
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute routing weights and expert indices."""
        logits = self.gate(x)

        # Get top-k experts
        indices = mx.argsort(logits, axis=-1)[:, :, -self.num_experts_per_tok :]
        indices = indices[:, :, ::-1]  # Descending order

        # Get weights and normalize
        weights = mx.softmax(logits, axis=-1)
        top_k_weights = mx.take_along_axis(weights, indices, axis=-1)
        top_k_weights = top_k_weights / (mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6)

        return top_k_weights, indices


class QuantizedMoE(nn.Module):
    """
    Quantized Mixture of Experts with batched expert storage.

    Uses BatchedQuantizedExperts for efficient weight loading and storage.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 8,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Quantized router
        self.router = QuantizedMoERouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            bias=bias,
            group_size=group_size,
            bits=bits,
        )

        # Batched experts
        self.experts = BatchedQuantizedExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through quantized MoE."""
        batch_size, seq_len, hidden_size = x.shape

        # Get routing
        weights, indices = self.router(x)

        # Flatten
        x_flat = x.reshape(-1, hidden_size)
        indices_flat = indices.reshape(-1, self.num_experts_per_tok)
        weights_flat = weights.reshape(-1, self.num_experts_per_tok)

        # Compute weighted expert outputs
        output = mx.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = indices_flat == expert_idx
            expert_weights = mx.sum(weights_flat * expert_mask.astype(weights_flat.dtype), axis=-1)

            if mx.any(expert_weights > 0):
                expert_out = self.experts(x_flat, expert_idx)
                output = output + expert_out * expert_weights[:, None]

        return output.reshape(batch_size, seq_len, hidden_size)


class QuantizedGptOssAttention(nn.Module):
    """
    Quantized GPT-OSS attention with optional sliding window.

    Uses QuantizedLinear for Q/K/V/O projections.
    """

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int = 0,
        attention_type: str = "full_attention",
        group_size: int = 64,
        bits: int = 8,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.attention_type = attention_type

        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads

        # Quantized projections
        self.q_proj = nn.QuantizedLinear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
            group_size=group_size,
            bits=bits,
        )
        self.k_proj = nn.QuantizedLinear(
            config.hidden_size,
            num_kv_heads * head_dim,
            bias=config.attention_bias,
            group_size=group_size,
            bits=bits,
        )
        self.v_proj = nn.QuantizedLinear(
            config.hidden_size,
            num_kv_heads * head_dim,
            bias=config.attention_bias,
            group_size=group_size,
            bits=bits,
        )
        self.o_proj = nn.QuantizedLinear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            group_size=group_size,
            bits=bits,
        )

        # Attention sinks (learned per-layer) - will be set during weight loading
        self.sinks = None

        # Attention parameters
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.sliding_window = (
            config.sliding_window if attention_type == "sliding_attention" else None
        )

        # RoPE
        self.rope = nn.RoPE(
            dims=head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with optional sliding window."""
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq, dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            q = self.rope(q, offset=cache[0].shape[2])
            k = self.rope(k, offset=cache[0].shape[2])
        else:
            q = self.rope(q)
            k = self.rope(k)

        # Update cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # GQA: repeat K, V if needed
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply sliding window mask if applicable
        if self.sliding_window is not None and mask is None:
            kv_len = k.shape[2]
            q_len = q.shape[2]
            q_pos = mx.arange(kv_len - q_len, kv_len)
            k_pos = mx.arange(kv_len)
            distances = q_pos[:, None] - k_pos[None, :]
            sliding_mask = mx.where(
                (distances >= 0) & (distances < self.sliding_window),
                0.0,
                float("-inf"),
            )
            scores = scores + sliding_mask
        elif mask is not None:
            scores = scores + mask

        # Softmax and output
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape back
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch, seq_len, -1)

        return self.o_proj(output), new_cache


class QuantizedGptOssBlock(nn.Module):
    """Quantized GPT-OSS transformer block."""

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int = 0,
        group_size: int = 64,
        bits: int = 8,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        attention_type = config.get_layer_type(layer_idx)

        # Norms (not quantized)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Quantized attention
        self.self_attn = QuantizedGptOssAttention(
            config,
            layer_idx=layer_idx,
            attention_type=attention_type,
            group_size=group_size,
            bits=bits,
        )

        # Quantized MoE
        self.mlp = QuantizedMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=True,
            group_size=group_size,
            bits=bits,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass."""
        # Attention
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # MoE
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


class QuantizedGptOssModel(Backbone):
    """Quantized GPT-OSS backbone."""

    def __init__(self, config: GptOssConfig, group_size: int = 64, bits: int = 8):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Embeddings (quantized in the MLX model)
        self.embed_tokens = nn.QuantizedEmbedding(
            config.vocab_size,
            config.hidden_size,
            group_size=group_size,
            bits=bits,
        )

        # Quantized transformer blocks
        self.layers = [
            QuantizedGptOssBlock(config, layer_idx=i, group_size=group_size, bits=bits)
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

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Causal mask
        if attention_mask is None and cache is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)
        else:
            mask = attention_mask

        # Process layers
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden_states, layer_new_cache = layer(hidden_states, mask=mask, cache=layer_cache)
            new_cache.append(layer_new_cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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


class QuantizedGptOssForCausalLM(Model):
    """Quantized GPT-OSS for causal language modeling."""

    def __init__(self, config: GptOssConfig, group_size: int = 64, bits: int = 8):
        super().__init__()

        self._config = config
        self.group_size = group_size
        self.bits = bits

        # Quantized backbone
        self.model = QuantizedGptOssModel(config, group_size=group_size, bits=bits)

        # Quantized LM head
        self.lm_head = nn.QuantizedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            group_size=group_size,
            bits=bits,
        )

    @property
    def config(self) -> GptOssConfig:
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
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        logits = self.lm_head(backbone_output.last_hidden_state)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, logits.shape[-1]),
                shift_labels.reshape(-1),
            )

        return ModelOutput(
            loss=loss,
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

            if temperature == 0.0:
                next_token = mx.argmax(logits, axis=-1)
                next_token = mx.expand_dims(next_token, axis=-1)
            else:
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

            if int(next_token[0, 0]) in stop_tokens_set:
                break

            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(
        cls, config: GptOssConfig, group_size: int = 64, bits: int = 8
    ) -> QuantizedGptOssForCausalLM:
        """Create from config."""
        return cls(config, group_size=group_size, bits=bits)

    @staticmethod
    def sanitize(weights: dict[str, Any]) -> dict[str, Any]:
        """
        Map HuggingFace weight names to our model structure.

        The main transformations:
        - Router: mlp.router.* -> mlp.router.gate.*
        - Experts: mlp.experts.{gate,up,down}_proj.* are batched tensors
        """
        result = {}

        for key, value in weights.items():
            new_key = key

            # Router gate mapping
            if ".mlp.router." in key and ".mlp.router.gate." not in key:
                new_key = key.replace(".mlp.router.", ".mlp.router.gate.")

            # Expert weights stay as-is (batched format)
            # They'll be loaded directly into BatchedQuantizedExperts

            result[new_key] = value

        return result
