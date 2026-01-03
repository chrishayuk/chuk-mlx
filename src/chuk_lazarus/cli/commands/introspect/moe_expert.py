"""MoE Expert CLI commands.

Direct manipulation of MoE expert routing for analysis and experimentation.

Usage:
    # Chat with a specific expert (force all routing to expert N)
    lazarus introspect moe-expert chat -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = "

    # Compare multiple experts on the same prompt
    lazarus introspect moe-expert compare -m openai/gpt-oss-20b --experts 6,7,20 -p "def fibonacci(n):"

    # Ablate an expert (remove it from routing)
    lazarus introspect moe-expert ablate -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = "

    # Vary top-k (normally 4, try 1, 2, 8)
    lazarus introspect moe-expert topk -m openai/gpt-oss-20b --k 1 -p "Hello world"

    # Analyze expert co-activation patterns
    lazarus introspect moe-expert collab -m openai/gpt-oss-20b -p "127 * 89 = "

    # Test specific expert pairs/groups
    lazarus introspect moe-expert pairs -m openai/gpt-oss-20b --experts 6,7 -p "127 * 89 = "

    # Interactive expert explorer
    lazarus introspect moe-expert interactive -m openai/gpt-oss-20b
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def introspect_moe_expert(args):
    """MoE expert command dispatcher."""
    action = getattr(args, "action", "chat")

    if action == "analyze":
        _analyze_experts(args)
    elif action == "chat":
        _chat_with_expert(args)
    elif action == "compare":
        _compare_experts(args)
    elif action == "ablate":
        _ablate_expert(args)
    elif action == "topk":
        _vary_topk(args)
    elif action == "collab":
        _analyze_collaboration(args)
    elif action == "pairs":
        _test_expert_pairs(args)
    elif action == "interactive":
        _interactive_mode(args)
    elif action == "weights":
        _show_router_weights(args)
    elif action == "tokenizer":
        _analyze_tokenizer(args)
    elif action == "control-tokens":
        _analyze_control_token_experts(args)
    elif action == "trace":
        _trace_token_experts(args)
    elif action == "entropy":
        _analyze_routing_entropy(args)
    elif action == "divergence":
        _analyze_layer_divergence(args)
    elif action == "role":
        _analyze_layer_role(args)
    elif action == "context-test":
        _test_context_independence(args)
    elif action == "vocab-map":
        _test_vocab_mapping(args)
    elif action == "router-probe":
        _probe_router_inputs(args)
    elif action == "pattern-discovery":
        _discover_expert_patterns(args)
    elif action == "full-taxonomy":
        _full_expert_taxonomy(args)
    elif action == "layer-sweep":
        _sweep_all_layers(args)
    else:
        print(f"Unknown action: {action}")


def _load_model(model_id: str):
    """Load model and tokenizer."""
    from ....inference.loader import DType, HFLoader
    from ....models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {model_id}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    return model, tokenizer


def _get_moe_info(model) -> dict:
    """Get MoE layer information."""
    layers = list(model.model.layers)
    moe_layers = []
    num_experts = 0
    num_experts_per_tok = 0

    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            moe_layers.append(i)
            router = layer.mlp.router
            num_experts = router.num_experts
            num_experts_per_tok = router.num_experts_per_tok

    return {
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "total_layers": len(layers),
    }


class ExpertRouter:
    """Utility class for manipulating expert routing."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.info = _get_moe_info(model)

        if not self.info["moe_layers"]:
            raise ValueError("Model has no MoE layers")

        self._backbone = model.model
        self._layers = list(model.model.layers)
        self._embed = getattr(self._backbone, "embed_tokens", None)
        self._norm = getattr(self._backbone, "norm", None)
        self._lm_head = getattr(model, "lm_head", None)

        if hasattr(model, "config"):
            self._embed_scale = getattr(model.config, "embedding_scale", None)
        else:
            self._embed_scale = None

        # Detect MoE architecture type
        self._moe_type = self._detect_moe_type()

    def _detect_moe_type(self) -> str:
        """Detect which MoE architecture is used."""
        layer = self._layers[self.info["moe_layers"][0]]
        moe = layer.mlp

        # Check for GPT-OSS batched experts (has gate_up_proj_blocks)
        if hasattr(moe, "experts") and hasattr(moe.experts, "gate_up_proj_blocks"):
            return "gpt_oss_batched"
        # Check for standard MoE with list of experts
        elif hasattr(moe, "experts") and isinstance(moe.experts, list):
            return "standard_list"
        else:
            return "unknown"

    def _apply_single_expert_gpt_oss(
        self,
        x: mx.array,
        moe,
        expert_idx: int,
    ) -> mx.array:
        """Apply a single expert from GPT-OSS batched format."""
        experts = moe.experts
        num_tokens = x.shape[0]

        # Get expert weights for this specific expert
        gate_up_blocks = experts.gate_up_proj_blocks[expert_idx]
        gate_up_scales = experts.gate_up_proj_scales[expert_idx]
        gate_up_bias = experts.gate_up_proj_bias[expert_idx]

        down_blocks = experts.down_proj_blocks[expert_idx]
        down_scales = experts.down_proj_scales[expert_idx]
        down_bias = experts.down_proj_bias[expert_idx]

        # Apply fused gate+up projection with MXFP4 dequantization
        gate_up_out = mx.quantized_matmul(
            x,
            gate_up_blocks,
            scales=gate_up_scales,
            biases=None,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        gate_up_out = gate_up_out + gate_up_bias

        # Split into gate and up (interleaved)
        gate_out = gate_up_out[:, 0::2]
        up_out = gate_up_out[:, 1::2]

        # GPT-OSS custom SwiGLU
        hidden = self._gpt_oss_swiglu(up_out, gate_out)

        # Apply down projection
        expert_out = mx.quantized_matmul(
            hidden,
            down_blocks,
            scales=down_scales,
            biases=None,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        expert_out = expert_out + down_bias

        return expert_out

    def _gpt_oss_swiglu(
        self,
        x_linear: mx.array,
        x_glu: mx.array,
        alpha: float = 1.702,
        limit: float = 7.0,
    ) -> mx.array:
        """GPT-OSS custom SwiGLU activation."""
        x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
        x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
        glu_scaled = alpha * x_glu
        sig = mx.sigmoid(glu_scaled)
        out_glu = x_glu * sig
        return out_glu * (x_linear + 1)

    def _apply_single_expert(self, h_norm: mx.array, moe, expert_idx: int) -> mx.array:
        """Apply a single expert based on MoE architecture type."""
        batch_size, seq_len, hidden_size = h_norm.shape
        x_flat = h_norm.reshape(-1, hidden_size)

        if self._moe_type == "gpt_oss_batched":
            expert_out = self._apply_single_expert_gpt_oss(x_flat, moe, expert_idx)
        elif self._moe_type == "standard_list":
            expert = moe.experts[expert_idx]
            expert_out = expert(x_flat)
        else:
            raise ValueError(f"Unsupported MoE type: {self._moe_type}")

        return expert_out.reshape(batch_size, seq_len, hidden_size)

    def generate_with_forced_expert(
        self,
        prompt: str,
        expert_idx: int,
        max_tokens: int = 20,
        layers: list[int] | None = None,
    ) -> tuple[str, dict]:
        """Generate forcing all routing to a specific expert.

        Args:
            prompt: Input prompt
            expert_idx: Expert index to force (0-based)
            max_tokens: Maximum tokens to generate
            layers: Which MoE layers to modify (None = all)

        Returns:
            (generated_text, stats)
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for step in range(max_tokens):
            h = self._embed(input_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for idx, layer in enumerate(self._layers):
                if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    # Apply attention first
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    # Force routing to specific expert
                    h_norm = layer.post_attention_layernorm(h_attn)
                    moe = layer.mlp

                    # Apply single expert using appropriate method
                    expert_out = self._apply_single_expert(h_norm, moe, expert_idx)

                    h = h_attn + expert_out
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        stats = {
            "expert_idx": expert_idx,
            "tokens_generated": len(generated),
            "layers_modified": len(layers),
            "moe_type": self._moe_type,
        }

        return text, stats

    def _apply_moe_with_ablation(
        self,
        h_norm: mx.array,
        moe,
        ablate_idx: int,
    ) -> tuple[mx.array, bool]:
        """Apply MoE with one expert ablated. Returns output and whether ablated expert would have been selected."""
        batch_size, seq_len, hidden_size = h_norm.shape
        x_flat = h_norm.reshape(-1, hidden_size)

        num_experts = self.info["num_experts"]
        k = self.info["num_experts_per_tok"]
        router = moe.router

        # Compute router logits
        router_logits = x_flat @ router.weight.T + router.bias

        # Check if ablated expert would have been selected
        orig_topk = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        would_select_ablated = bool(mx.any(orig_topk == ablate_idx))

        # Mask out ablated expert with large negative value
        mask = mx.zeros((num_experts,))
        mask = mask.at[ablate_idx].add(1.0)
        router_logits = router_logits - mask * 1e9

        # Select top-k from remaining experts
        topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
        weights = mx.softmax(topk_logits, axis=-1)

        # Compute weighted expert outputs
        expert_outs = mx.zeros_like(x_flat)

        for exp_idx in range(num_experts):
            if exp_idx == ablate_idx:
                continue

            # Find where this expert is selected and its weight
            exp_mask = (topk_indices == exp_idx)
            exp_weights = mx.sum(weights * exp_mask.astype(weights.dtype), axis=-1, keepdims=True)

            if mx.any(exp_weights > 0):
                if self._moe_type == "gpt_oss_batched":
                    exp_out = self._apply_single_expert_gpt_oss(x_flat, moe, exp_idx)
                else:
                    exp_out = moe.experts[exp_idx](x_flat)
                expert_outs = expert_outs + exp_out * exp_weights

        return expert_outs.reshape(batch_size, seq_len, hidden_size), would_select_ablated

    def _apply_moe_with_multi_ablation(
        self,
        h_norm: mx.array,
        moe,
        ablate_indices: list[int],
    ) -> tuple[mx.array, int]:
        """Apply MoE with multiple experts ablated. Returns output and count of would-be-selected ablated experts."""
        batch_size, seq_len, hidden_size = h_norm.shape
        x_flat = h_norm.reshape(-1, hidden_size)

        num_experts = self.info["num_experts"]
        k = self.info["num_experts_per_tok"]
        router = moe.router

        # Compute router logits
        router_logits = x_flat @ router.weight.T + router.bias

        # Check how many ablated experts would have been selected
        orig_topk = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        would_select_count = 0
        for ablate_idx in ablate_indices:
            if bool(mx.any(orig_topk == ablate_idx)):
                would_select_count += 1

        # Mask out all ablated experts with large negative values
        ablate_set = set(ablate_indices)
        mask = mx.zeros((num_experts,))
        for ablate_idx in ablate_indices:
            mask = mask.at[ablate_idx].add(1.0)
        router_logits = router_logits - mask * 1e9

        # Select top-k from remaining experts
        topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
        weights = mx.softmax(topk_logits, axis=-1)

        # Compute weighted expert outputs
        expert_outs = mx.zeros_like(x_flat)

        for exp_idx in range(num_experts):
            if exp_idx in ablate_set:
                continue

            # Find where this expert is selected and its weight
            exp_mask = (topk_indices == exp_idx)
            exp_weights = mx.sum(weights * exp_mask.astype(weights.dtype), axis=-1, keepdims=True)

            if mx.any(exp_weights > 0):
                if self._moe_type == "gpt_oss_batched":
                    exp_out = self._apply_single_expert_gpt_oss(x_flat, moe, exp_idx)
                else:
                    exp_out = moe.experts[exp_idx](x_flat)
                expert_outs = expert_outs + exp_out * exp_weights

        return expert_outs.reshape(batch_size, seq_len, hidden_size), would_select_count

    def generate_with_multi_ablation(
        self,
        prompt: str,
        ablate_indices: list[int],
        max_tokens: int = 20,
        layers: list[int] | None = None,
    ) -> tuple[str, dict]:
        """Generate with multiple experts ablated (removed from routing).

        Args:
            prompt: Input prompt
            ablate_indices: List of expert indices to ablate
            max_tokens: Maximum tokens to generate
            layers: Which MoE layers to modify (None = all)

        Returns:
            (generated_text, stats)
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []
        total_would_select = 0

        for step in range(max_tokens):
            h = self._embed(input_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for idx, layer in enumerate(self._layers):
                if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    # Apply attention first
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    # Apply MoE with multi-ablation
                    h_norm = layer.post_attention_layernorm(h_attn)
                    expert_outs, would_count = self._apply_moe_with_multi_ablation(
                        h_norm, layer.mlp, ablate_indices
                    )

                    total_would_select += would_count
                    h = h_attn + expert_outs
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        stats = {
            "ablated_experts": ablate_indices,
            "num_ablated": len(ablate_indices),
            "tokens_generated": len(generated),
            "total_would_select": total_would_select,
            "layers_modified": len(layers),
            "moe_type": self._moe_type,
        }

        return text, stats

    def generate_with_ablated_expert(
        self,
        prompt: str,
        ablate_idx: int,
        max_tokens: int = 20,
        layers: list[int] | None = None,
    ) -> tuple[str, dict]:
        """Generate with one expert ablated (removed from routing).

        Args:
            prompt: Input prompt
            ablate_idx: Expert index to ablate
            max_tokens: Maximum tokens to generate
            layers: Which MoE layers to modify (None = all)

        Returns:
            (generated_text, stats)
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []
        ablation_activations = 0

        for step in range(max_tokens):
            h = self._embed(input_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for idx, layer in enumerate(self._layers):
                if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    # Apply attention first
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    # Apply MoE with ablation
                    h_norm = layer.post_attention_layernorm(h_attn)
                    expert_outs, would_select = self._apply_moe_with_ablation(h_norm, layer.mlp, ablate_idx)

                    if would_select:
                        ablation_activations += 1

                    h = h_attn + expert_outs
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        stats = {
            "ablated_expert": ablate_idx,
            "tokens_generated": len(generated),
            "ablation_activations": ablation_activations,
            "layers_modified": len(layers),
            "moe_type": self._moe_type,
        }

        return text, stats

    def _apply_moe_with_custom_k(
        self,
        h_norm: mx.array,
        moe,
        k: int,
    ) -> mx.array:
        """Apply MoE with custom top-k value."""
        batch_size, seq_len, hidden_size = h_norm.shape
        x_flat = h_norm.reshape(-1, hidden_size)

        num_experts = self.info["num_experts"]
        router = moe.router

        # Compute router logits
        router_logits = x_flat @ router.weight.T + router.bias

        # Select top-k
        topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
        weights = mx.softmax(topk_logits, axis=-1)

        # Compute weighted expert outputs
        expert_outs = mx.zeros_like(x_flat)

        for exp_idx in range(num_experts):
            # Find where this expert is selected and its weight
            exp_mask = (topk_indices == exp_idx)
            exp_weights = mx.sum(weights * exp_mask.astype(weights.dtype), axis=-1, keepdims=True)

            if mx.any(exp_weights > 0):
                if self._moe_type == "gpt_oss_batched":
                    exp_out = self._apply_single_expert_gpt_oss(x_flat, moe, exp_idx)
                else:
                    exp_out = moe.experts[exp_idx](x_flat)
                expert_outs = expert_outs + exp_out * exp_weights

        return expert_outs.reshape(batch_size, seq_len, hidden_size)

    def generate_with_topk(
        self,
        prompt: str,
        k: int,
        max_tokens: int = 20,
        layers: list[int] | None = None,
    ) -> tuple[str, dict]:
        """Generate with modified top-k expert selection.

        Args:
            prompt: Input prompt
            k: Number of experts to use per token
            max_tokens: Maximum tokens to generate
            layers: Which MoE layers to modify (None = all)

        Returns:
            (generated_text, stats)
        """
        if layers is None:
            layers = self.info["moe_layers"]

        num_experts = self.info["num_experts"]
        k = min(k, num_experts)  # Can't use more experts than exist

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for step in range(max_tokens):
            h = self._embed(input_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for idx, layer in enumerate(self._layers):
                if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    # Apply attention first
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    # Apply MoE with custom k
                    h_norm = layer.post_attention_layernorm(h_attn)
                    expert_outs = self._apply_moe_with_custom_k(h_norm, layer.mlp, k)

                    h = h_attn + expert_outs
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        stats = {
            "top_k": k,
            "default_k": self.info["num_experts_per_tok"],
            "tokens_generated": len(generated),
            "layers_modified": len(layers),
            "moe_type": self._moe_type,
        }

        return text, stats

    def generate_normal(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate with normal routing (baseline)."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        return self.tokenizer.decode(generated).strip()

    def capture_router_weights(
        self,
        prompt: str,
        layers: list[int] | None = None,
    ) -> dict[int, list[tuple[list[int], list[float]]]]:
        """Capture expert selections AND their routing weights.

        Returns:
            Dict mapping layer_idx -> list of (expert_indices, weights) per token
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        results = {layer_idx: [] for layer_idx in layers}

        h = self._embed(input_ids)
        if self._embed_scale:
            h = h * self._embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        k = self.info["num_experts_per_tok"]

        for idx, layer in enumerate(self._layers):
            if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                # Apply attention first
                attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h_attn = h + attn_out

                # Get router weights
                h_norm = layer.post_attention_layernorm(h_attn)
                moe = layer.mlp

                batch_size, seq_len, hidden_size = h_norm.shape
                x_flat = h_norm.reshape(-1, hidden_size)
                router = moe.router
                router_logits = x_flat @ router.weight.T + router.bias

                # Get top-k with weights
                topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
                topk_logits = mx.take_along_axis(router_logits, topk_indices, axis=-1)
                weights = mx.softmax(topk_logits, axis=-1)

                # Store per-token results
                topk_indices_list = topk_indices.tolist()
                weights_list = weights.tolist()

                for token_idx in range(len(topk_indices_list)):
                    indices = topk_indices_list[token_idx]
                    wts = weights_list[token_idx]
                    # Sort by weight descending
                    sorted_pairs = sorted(zip(indices, wts), key=lambda x: x[1], reverse=True)
                    sorted_indices = [p[0] for p in sorted_pairs]
                    sorted_weights = [p[1] for p in sorted_pairs]
                    results[idx].append((sorted_indices, sorted_weights))

                # Still need to apply MoE for next layer
                moe_out = moe(h_norm)
                h = h_attn + moe_out
            else:
                try:
                    out = layer(h, mask=mask)
                except TypeError:
                    out = layer(h)

                if hasattr(out, "hidden_states"):
                    h = out.hidden_states
                elif isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out

        return results

    def capture_expert_selections(
        self,
        prompt: str,
        layers: list[int] | None = None,
    ) -> dict[int, list[list[int]]]:
        """Capture which experts are selected at each layer for a prompt.

        Returns:
            Dict mapping layer_idx -> list of selected expert indices per token
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        selections = {layer_idx: [] for layer_idx in layers}

        h = self._embed(input_ids)
        if self._embed_scale:
            h = h * self._embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        k = self.info["num_experts_per_tok"]

        for idx, layer in enumerate(self._layers):
            if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                # Apply attention first
                attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                h_attn = h + attn_out

                # Get router selections
                h_norm = layer.post_attention_layernorm(h_attn)
                moe = layer.mlp

                batch_size, seq_len, hidden_size = h_norm.shape
                x_flat = h_norm.reshape(-1, hidden_size)
                router = moe.router
                router_logits = x_flat @ router.weight.T + router.bias

                # Get top-k selections
                topk_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
                topk_indices = topk_indices.tolist()

                # Store selections per token
                for token_idx, token_experts in enumerate(topk_indices):
                    selections[idx].append(sorted(token_experts))

                # Still need to apply MoE for next layer
                moe_out = moe(h_norm)
                h = h_attn + moe_out
            else:
                try:
                    out = layer(h, mask=mask)
                except TypeError:
                    out = layer(h)

                if hasattr(out, "hidden_states"):
                    h = out.hidden_states
                elif isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out

        return selections

    def analyze_coactivation(
        self,
        prompts: list[str],
        target_layer: int | None = None,
    ) -> dict:
        """Analyze which experts frequently co-activate together.

        Returns:
            Dict with coactivation matrix and pair frequencies
        """
        from collections import defaultdict

        if target_layer is None:
            target_layer = self.info["moe_layers"][len(self.info["moe_layers"]) // 2]

        num_experts = self.info["num_experts"]

        # Track pair frequencies
        pair_counts = defaultdict(int)
        expert_counts = defaultdict(int)
        total_activations = 0

        for prompt in prompts:
            selections = self.capture_expert_selections(prompt, layers=[target_layer])

            for token_experts in selections[target_layer]:
                total_activations += 1

                # Count individual experts
                for exp in token_experts:
                    expert_counts[exp] += 1

                # Count pairs (unordered)
                for i, exp1 in enumerate(token_experts):
                    for exp2 in token_experts[i + 1:]:
                        pair = tuple(sorted([exp1, exp2]))
                        pair_counts[pair] += 1

        # Build coactivation matrix
        coactivation = [[0] * num_experts for _ in range(num_experts)]
        for (exp1, exp2), count in pair_counts.items():
            coactivation[exp1][exp2] = count
            coactivation[exp2][exp1] = count

        # Find top pairs
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "layer": target_layer,
            "total_activations": total_activations,
            "expert_counts": dict(expert_counts),
            "pair_counts": dict(sorted_pairs[:50]),  # Top 50 pairs
            "coactivation_matrix": coactivation,
            "top_pairs": sorted_pairs[:20],
        }

    def _apply_expert_group(
        self,
        h_norm: mx.array,
        moe,
        expert_indices: list[int],
    ) -> mx.array:
        """Apply a specific group of experts with equal weighting."""
        batch_size, seq_len, hidden_size = h_norm.shape
        x_flat = h_norm.reshape(-1, hidden_size)

        # Equal weights for all experts in group
        weight = 1.0 / len(expert_indices)

        expert_outs = mx.zeros_like(x_flat)

        for exp_idx in expert_indices:
            if self._moe_type == "gpt_oss_batched":
                exp_out = self._apply_single_expert_gpt_oss(x_flat, moe, exp_idx)
            else:
                exp_out = moe.experts[exp_idx](x_flat)
            expert_outs = expert_outs + exp_out * weight

        return expert_outs.reshape(batch_size, seq_len, hidden_size)

    def generate_with_expert_group(
        self,
        prompt: str,
        expert_indices: list[int],
        max_tokens: int = 20,
        layers: list[int] | None = None,
    ) -> tuple[str, dict]:
        """Generate using only a specific group of experts.

        Args:
            prompt: Input prompt
            expert_indices: List of expert indices to use
            max_tokens: Maximum tokens to generate
            layers: Which MoE layers to modify (None = all)

        Returns:
            (generated_text, stats)
        """
        if layers is None:
            layers = self.info["moe_layers"]

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for step in range(max_tokens):
            h = self._embed(input_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = input_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            for idx, layer in enumerate(self._layers):
                if idx in layers and hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    # Apply attention first
                    attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_attn = h + attn_out

                    # Force routing to specific expert group
                    h_norm = layer.post_attention_layernorm(h_attn)
                    moe = layer.mlp

                    # Apply expert group
                    expert_out = self._apply_expert_group(h_norm, moe, expert_indices)

                    h = h_attn + expert_out
                else:
                    try:
                        out = layer(h, mask=mask)
                    except TypeError:
                        out = layer(h)

                    if hasattr(out, "hidden_states"):
                        h = out.hidden_states
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        h = out

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if len(generated) > 3 and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        stats = {
            "expert_group": expert_indices,
            "group_size": len(expert_indices),
            "tokens_generated": len(generated),
            "layers_modified": len(layers),
            "moe_type": self._moe_type,
        }

        return text, stats


def _extract_answer(text: str) -> float | None:
    """Extract numeric answer from text."""
    import re
    match = re.search(r'-?\d+(?:,\d+)*(?:\.\d+)?', text.replace(",", ""))
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    return None


def _get_correct_answer(prompt: str) -> float | None:
    """Compute correct answer for arithmetic prompt."""
    import re
    match = re.match(r'^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=', prompt)
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if op == '+':
            return float(a + b)
        elif op == '-':
            return float(a - b)
        elif op == '*':
            return float(a * b)
        elif op == '/' and b != 0:
            return float(a / b)
    return None


def _chat_with_expert(args):
    """Chat with a specific expert."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    expert_idx = args.expert
    prompt = args.prompt

    if expert_idx >= router.info["num_experts"]:
        print(f"Error: Expert {expert_idx} doesn't exist. Model has {router.info['num_experts']} experts (0-{router.info['num_experts']-1})")
        return

    print(f"\n{'=' * 70}")
    print(f"CHAT WITH EXPERT {expert_idx}")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"MoE type: {router._moe_type}")
    print(f"MoE layers: {len(router.info['moe_layers'])} ({router.info['moe_layers'][0]} to {router.info['moe_layers'][-1]})")
    print(f"Experts: {router.info['num_experts']} (normally {router.info['num_experts_per_tok']} active)")
    print(f"{'=' * 70}")

    # Get normal response for comparison
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    normal_response = router.generate_normal(prompt)
    expert_response, stats = router.generate_with_forced_expert(prompt, expert_idx)

    print(f"Normal (top-{router.info['num_experts_per_tok']}): {normal_response}")
    print(f"Expert {expert_idx} only:      {expert_response}")

    # Check correctness for arithmetic
    correct = _get_correct_answer(prompt)
    if correct is not None:
        normal_ans = _extract_answer(normal_response)
        expert_ans = _extract_answer(expert_response)

        print(f"\nCorrect answer: {int(correct)}")
        print(f"Normal correct: {normal_ans == correct if normal_ans else False}")
        print(f"Expert correct: {expert_ans == correct if expert_ans else False}")

    print(f"{'=' * 70}")


def _compare_experts(args):
    """Compare multiple experts on the same prompt."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    experts = [int(x.strip()) for x in args.experts.split(",")]
    prompt = args.prompt

    print(f"\n{'=' * 70}")
    print(f"EXPERT COMPARISON")
    print(f"{'=' * 70}")
    print(f"Prompt: {prompt}")
    print(f"Comparing experts: {experts}")
    print(f"{'=' * 70}")

    # Normal response
    normal_response = router.generate_normal(prompt)
    print(f"\n{'Expert':<12} {'Response':<40} {'Correct'}")
    print("-" * 60)

    correct = _get_correct_answer(prompt)
    normal_ans = _extract_answer(normal_response)
    normal_correct = "Y" if (correct and normal_ans == correct) else ("N" if correct else "-")
    print(f"{'Normal':<12} {normal_response:<40} {normal_correct}")

    # Each expert
    for expert_idx in experts:
        if expert_idx >= router.info["num_experts"]:
            print(f"Expert {expert_idx:<4}  [INVALID - max is {router.info['num_experts']-1}]")
            continue

        response, _ = router.generate_with_forced_expert(prompt, expert_idx)
        ans = _extract_answer(response)
        is_correct = "Y" if (correct and ans == correct) else ("N" if correct else "-")
        print(f"Expert {expert_idx:<4}  {response:<40} {is_correct}")

    print(f"{'=' * 70}")


def _ablate_expert(args):
    """Ablate (remove) one or more experts and see the effect."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    # Support both --expert (single) and --experts (multiple)
    if args.experts:
        ablate_indices = [int(x.strip()) for x in args.experts.split(",")]
    elif args.expert is not None:
        ablate_indices = [args.expert]
    else:
        print("Error: Must specify --expert or --experts")
        return

    prompt = args.prompt
    is_multi = len(ablate_indices) > 1

    print(f"\n{'=' * 70}")
    if is_multi:
        print(f"MULTI-EXPERT ABLATION - Removing Experts {ablate_indices}")
    else:
        print(f"EXPERT ABLATION - Removing Expert {ablate_indices[0]}")
    print(f"{'=' * 70}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 70}")

    # Normal response
    normal_response = router.generate_normal(prompt)

    # Ablated response (single or multi)
    if is_multi:
        ablated_response, stats = router.generate_with_multi_ablation(prompt, ablate_indices)
        ablation_count = stats['total_would_select']
    else:
        ablated_response, stats = router.generate_with_ablated_expert(prompt, ablate_indices[0])
        ablation_count = stats['ablation_activations']

    ablate_str = ",".join(str(e) for e in ablate_indices)
    print(f"\nNormal (with Experts):      {normal_response}")
    print(f"Ablated (without {ablate_str}): {ablated_response}")

    correct = _get_correct_answer(prompt)
    if correct is not None:
        normal_ans = _extract_answer(normal_response)
        ablated_ans = _extract_answer(ablated_response)

        print(f"\nCorrect answer: {int(correct)}")
        print(f"Normal correct:  {normal_ans == correct if normal_ans else False}")
        print(f"Ablated correct: {ablated_ans == correct if ablated_ans else False}")

        if normal_ans == correct and ablated_ans != correct:
            print(f"\n** ABLATION BROKE THE MODEL **")
            print(f"   Expert(s) {ablate_str} appear critical for this task!")
        elif normal_ans != correct and ablated_ans == correct:
            print(f"\n** ABLATION FIXED THE MODEL **")
            print(f"   Expert(s) {ablate_str} were interfering!")
        elif normal_ans == ablated_ans:
            print(f"\n** NO EFFECT **")
            print(f"   Expert(s) {ablate_str} not critical for this task.")

    print(f"\nStats: Ablated experts would have been selected {ablation_count} times")
    print(f"{'=' * 70}")

    # Optionally run on multiple problems
    if args.benchmark:
        problems = [
            "2 + 2 = ", "5 * 5 = ", "10 - 3 = ",
            "23 * 17 = ", "127 * 89 = ", "456 + 789 = ",
            "999 * 888 = ", "1234 + 5678 = ",
        ]

        print(f"\n{'=' * 70}")
        if is_multi:
            print(f"ABLATION BENCHMARK - Experts {ablate_indices}")
        else:
            print(f"ABLATION BENCHMARK - Expert {ablate_indices[0]}")
        print(f"{'=' * 70}")

        normal_correct = 0
        ablated_correct = 0

        for prob in problems:
            normal = router.generate_normal(prob)
            if is_multi:
                ablated, _ = router.generate_with_multi_ablation(prob, ablate_indices)
            else:
                ablated, _ = router.generate_with_ablated_expert(prob, ablate_indices[0])

            correct = _get_correct_answer(prob)
            n_ans = _extract_answer(normal)
            a_ans = _extract_answer(ablated)

            n_ok = n_ans == correct if (correct and n_ans) else False
            a_ok = a_ans == correct if (correct and a_ans) else False

            if n_ok:
                normal_correct += 1
            if a_ok:
                ablated_correct += 1

            status = ""
            if n_ok and not a_ok:
                status = "<- BROKEN"
            elif not n_ok and a_ok:
                status = "<- FIXED"

            print(f"{prob:<20} Normal: {normal:<12} Ablated: {ablated:<12} {status}")

        print(f"\nNormal accuracy:  {normal_correct}/{len(problems)} ({100*normal_correct/len(problems):.0f}%)")
        print(f"Ablated accuracy: {ablated_correct}/{len(problems)} ({100*ablated_correct/len(problems):.0f}%)")

        if ablated_correct < normal_correct:
            print(f"\nRemoving {len(ablate_indices)} expert(s) caused {normal_correct - ablated_correct} additional failures")
        elif ablated_correct > normal_correct:
            print(f"\nRemoving {len(ablate_indices)} expert(s) actually improved {ablated_correct - normal_correct} cases!")
        else:
            print(f"\nNo change in accuracy!")

        print(f"{'=' * 70}")


def _vary_topk(args):
    """Experiment with different top-k values."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    k = args.k
    prompt = args.prompt
    default_k = router.info["num_experts_per_tok"]

    print(f"\n{'=' * 70}")
    print(f"TOP-K EXPERIMENT - Using k={k} (default: {default_k})")
    print(f"{'=' * 70}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 70}")

    # Normal response
    normal_response = router.generate_normal(prompt)

    # Modified top-k response
    topk_response, stats = router.generate_with_topk(prompt, k)

    print(f"\nNormal (k={default_k}): {normal_response}")
    print(f"Modified (k={k}):  {topk_response}")

    correct = _get_correct_answer(prompt)
    if correct is not None:
        normal_ans = _extract_answer(normal_response)
        topk_ans = _extract_answer(topk_response)

        print(f"\nCorrect answer: {int(correct)}")
        print(f"Normal correct:   {normal_ans == correct if normal_ans else False}")
        print(f"Modified correct: {topk_ans == correct if topk_ans else False}")

    print(f"{'=' * 70}")

    # Compare multiple k values if requested
    if args.compare_k:
        k_values = [int(x.strip()) for x in args.compare_k.split(",")]

        print(f"\n{'=' * 70}")
        print(f"TOP-K COMPARISON")
        print(f"{'=' * 70}")

        print(f"\n{'k':<6} {'Response':<40} {'Correct'}")
        print("-" * 55)

        for k_val in k_values:
            if k_val == default_k:
                response = normal_response
            else:
                response, _ = router.generate_with_topk(prompt, k_val)

            ans = _extract_answer(response)
            is_correct = "Y" if (correct and ans == correct) else ("N" if correct else "-")

            label = f"k={k_val}" + (" (default)" if k_val == default_k else "")
            print(f"{label:<12} {response:<40} {is_correct}")

        print(f"{'=' * 70}")


def _analyze_collaboration(args):
    """Analyze which experts frequently collaborate (co-activate) together."""
    from ....introspection.moe.datasets import get_grouped_prompts, CATEGORY_GROUPS, PromptCategoryGroup

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    print(f"\n{'=' * 70}")
    print(f"EXPERT COLLABORATION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Experts: {router.info['num_experts']} (top-{router.info['num_experts_per_tok']} per token)")
    print(f"{'=' * 70}")

    # Load prompts from JSON dataset instead of hardcoding
    categories = get_grouped_prompts()

    # Analyze each category
    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = router.info["moe_layers"][len(router.info["moe_layers"]) // 2]

    print(f"\nAnalyzing layer: {target_layer}")
    print()

    category_results = {}

    for category, prompts in categories.items():
        print(f"Analyzing {category}...", end=" ", flush=True)
        analysis = router.analyze_coactivation(prompts, target_layer=target_layer)
        category_results[category] = analysis
        print(f"({analysis['total_activations']} tokens)")

    # Display top pairs per category
    print(f"\n{'=' * 70}")
    print("TOP EXPERT PAIRS BY CATEGORY")
    print(f"{'=' * 70}")

    for category, analysis in category_results.items():
        print(f"\n{category}:")
        print("-" * 40)

        top_pairs = analysis["top_pairs"][:10]
        for (exp1, exp2), count in top_pairs:
            pct = 100 * count / analysis["total_activations"] if analysis["total_activations"] > 0 else 0
            print(f"  Expert {exp1:2d} + Expert {exp2:2d}: {count:3d} times ({pct:5.1f}%)")

    # Analyze specialization patterns
    print(f"\n{'=' * 70}")
    print("SPECIALIZATION ANALYSIS")
    print(f"{'=' * 70}")

    # Group categories by type (from moe.datasets module)
    category_groups = {
        group.name: [cat.name.upper() for cat in cats]
        for group, cats in CATEGORY_GROUPS.items()
    }

    # Find top experts per category group
    group_top_experts = {}
    for group_name, cat_list in category_groups.items():
        expert_counts = {}
        for cat in cat_list:
            if cat in category_results:
                for exp, count in category_results[cat]["expert_counts"].items():
                    expert_counts[exp] = expert_counts.get(exp, 0) + count
        if expert_counts:
            sorted_experts = sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)
            group_top_experts[group_name] = sorted_experts[:10]

    # Display top experts per group
    print("\nTop experts by category group:")
    for group_name, top_experts in group_top_experts.items():
        experts_str = ", ".join(f"E{e}({c})" for e, c in top_experts[:5])
        print(f"  {group_name:<12}: {experts_str}")

    # Find specialists (experts that appear strongly in one group but not others)
    print("\n" + "-" * 50)
    print("POTENTIAL SPECIALISTS:")
    all_groups = list(group_top_experts.keys())
    for group_name, top_experts in group_top_experts.items():
        top_exp_set = set(e for e, _ in top_experts[:5])
        other_top = set()
        for other_group in all_groups:
            if other_group != group_name and other_group in group_top_experts:
                other_top.update(e for e, _ in group_top_experts[other_group][:5])
        specialists = top_exp_set - other_top
        if specialists:
            print(f"  {group_name} specialists: Experts {sorted(specialists)}")

    # Find generalists (experts that appear in many groups)
    print("\n" + "-" * 50)
    print("GENERALIST EXPERTS (appear across multiple groups):")
    all_top_experts = {}
    for group_name, top_experts in group_top_experts.items():
        for exp, count in top_experts:
            if exp not in all_top_experts:
                all_top_experts[exp] = []
            all_top_experts[exp].append(group_name)

    generalists = [(exp, groups) for exp, groups in all_top_experts.items() if len(groups) >= 4]
    generalists.sort(key=lambda x: len(x[1]), reverse=True)
    for exp, groups in generalists[:5]:
        print(f"  Expert {exp:2d}: appears in {', '.join(groups)}")

    # Token-level analysis: check if PUNCTUATION/PROPER_NOUNS experts differ from content
    if "PUNCTUATION" in category_results and "ARITHMETIC" in category_results:
        punct_experts = set(e for e, _ in category_results["PUNCTUATION"]["top_pairs"][:10])
        arith_experts = set(e for e, _ in category_results["ARITHMETIC"]["top_pairs"][:10])
        punct_specific = punct_experts - arith_experts
        if punct_specific:
            print("\n" + "-" * 50)
            print("TOKEN-LEVEL SPECIALIZATION:")
            print(f"  PUNCTUATION-specific pairs (not in arithmetic): {len(punct_specific)} pairs")
            print("  -> Suggests experts specialize by TOKEN TYPE, not domain!")

    print(f"\n{'=' * 70}")

    # If a specific prompt was provided, show detailed breakdown
    if args.prompt:
        print(f"\nDETAILED ANALYSIS FOR: {args.prompt}")
        print("-" * 50)

        selections = router.capture_expert_selections(args.prompt, layers=[target_layer])

        print(f"\nToken-by-token expert selections (layer {target_layer}):")
        tokens = tokenizer.encode(args.prompt)
        for i, (token_id, experts) in enumerate(zip(tokens, selections[target_layer])):
            token_str = tokenizer.decode([token_id])
            experts_str = ", ".join(str(e) for e in experts)
            print(f"  Token {i}: '{token_str}' -> Experts [{experts_str}]")

    print(f"{'=' * 70}")

    # Save results if output specified
    if getattr(args, "output", None):
        import json
        output_data = {
            "model": args.model,
            "layer": target_layer,
            "categories": {
                cat: {
                    "total_activations": res["total_activations"],
                    "top_pairs": [(list(p), c) for p, c in res["top_pairs"]],
                    "expert_counts": res["expert_counts"],
                }
                for cat, res in category_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def _test_expert_pairs(args):
    """Test generation with specific expert pairs/groups."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    experts = [int(x.strip()) for x in args.experts.split(",")]
    prompt = args.prompt

    print(f"\n{'=' * 70}")
    print(f"EXPERT PAIR/GROUP TEST")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Testing expert group: {experts}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 70}")

    # Validate expert indices
    for exp in experts:
        if exp >= router.info["num_experts"]:
            print(f"Error: Expert {exp} doesn't exist. Model has {router.info['num_experts']} experts.")
            return

    # Normal response
    normal_response = router.generate_normal(prompt)

    # Group response
    group_response, stats = router.generate_with_expert_group(prompt, experts)

    print(f"\n{'Configuration':<25} {'Response':<40}")
    print("-" * 65)
    print(f"{'Normal (top-' + str(router.info['num_experts_per_tok']) + ')':<25} {normal_response:<40}")
    print(f"{'Experts ' + str(experts):<25} {group_response:<40}")

    # Check correctness for arithmetic
    correct = _get_correct_answer(prompt)
    if correct is not None:
        normal_ans = _extract_answer(normal_response)
        group_ans = _extract_answer(group_response)

        print(f"\nCorrect answer: {int(correct)}")
        print(f"Normal correct: {normal_ans == correct if normal_ans else False}")
        print(f"Group correct:  {group_ans == correct if group_ans else False}")

    print(f"{'=' * 70}")

    # Compare with individual experts and other pairs if requested
    if getattr(args, "compare_singles", False):
        print(f"\n{'=' * 70}")
        print("COMPARISON: Group vs Individual Experts")
        print(f"{'=' * 70}")

        print(f"\n{'Configuration':<25} {'Response':<40} {'Correct'}")
        print("-" * 75)

        # Individual experts
        for exp in experts:
            response, _ = router.generate_with_forced_expert(prompt, exp)
            ans = _extract_answer(response)
            is_correct = "Y" if (correct and ans == correct) else ("N" if correct else "-")
            print(f"{'Expert ' + str(exp) + ' only':<25} {response:<40} {is_correct}")

        # The group
        group_ans = _extract_answer(group_response)
        is_correct = "Y" if (correct and group_ans == correct) else ("N" if correct else "-")
        print(f"{'Group ' + str(experts):<25} {group_response:<40} {is_correct}")

        print(f"{'=' * 70}")

    # Benchmark across multiple problems
    if getattr(args, "benchmark", False):
        problems = [
            "2 + 2 = ", "5 * 5 = ", "10 - 3 = ",
            "23 * 17 = ", "127 * 89 = ", "456 + 789 = ",
            "999 * 888 = ", "1234 + 5678 = ",
        ]

        print(f"\n{'=' * 70}")
        print(f"BENCHMARK: Expert Group {experts}")
        print(f"{'=' * 70}")

        normal_correct = 0
        group_correct = 0

        for prob in problems:
            normal = router.generate_normal(prob)
            group, _ = router.generate_with_expert_group(prob, experts)

            correct = _get_correct_answer(prob)
            n_ans = _extract_answer(normal)
            g_ans = _extract_answer(group)

            n_ok = n_ans == correct if (correct and n_ans) else False
            g_ok = g_ans == correct if (correct and g_ans) else False

            if n_ok:
                normal_correct += 1
            if g_ok:
                group_correct += 1

            status = ""
            if n_ok and not g_ok:
                status = "<- WORSE"
            elif not n_ok and g_ok:
                status = "<- BETTER"

            print(f"{prob:<20} Normal: {normal:<12} Group: {group:<12} {status}")

        print(f"\nNormal accuracy: {normal_correct}/{len(problems)} ({100*normal_correct/len(problems):.0f}%)")
        print(f"Group accuracy:  {group_correct}/{len(problems)} ({100*group_correct/len(problems):.0f}%)")

        print(f"{'=' * 70}")


def _interactive_mode(args):
    """Interactive expert explorer."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    print(f"\n{'=' * 70}")
    print(f"MOE EXPERT EXPLORER - Interactive Mode")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Experts: {router.info['num_experts']} (default top-{router.info['num_experts_per_tok']})")
    print(f"MoE layers: {len(router.info['moe_layers'])}")
    print(f"{'=' * 70}")
    print("Commands:")
    print("  <prompt>              - Generate with normal routing")
    print("  !expert N <prompt>    - Force routing to expert N")
    print("  !compare N,M <prompt> - Compare experts N and M")
    print("  !ablate N <prompt>    - Ablate expert N")
    print("  !topk K <prompt>      - Use top-K experts")
    print("  !pair N,M <prompt>    - Use only experts N,M together")
    print("  !collab <prompt>      - Show token-by-token expert selections")
    print("  !info                 - Show model info")
    print("  !quit                 - Exit")
    print(f"{'=' * 70}")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.startswith("!"):
            parts = line[1:].split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd == "quit":
                print("Goodbye!")
                break

            elif cmd == "info":
                print(f"Model: {args.model}")
                print(f"Experts: {router.info['num_experts']}")
                print(f"Top-k: {router.info['num_experts_per_tok']}")
                print(f"MoE layers: {router.info['moe_layers']}")

            elif cmd == "expert" and len(parts) >= 3:
                try:
                    expert_idx = int(parts[1])
                    prompt = parts[2]
                    if not prompt.endswith("= "):
                        prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

                    normal = router.generate_normal(prompt)
                    forced, _ = router.generate_with_forced_expert(prompt, expert_idx)

                    print(f"Normal:     {normal}")
                    print(f"Expert {expert_idx}:   {forced}")
                except (ValueError, IndexError) as e:
                    print(f"Usage: !expert N <prompt>")

            elif cmd == "compare" and len(parts) >= 3:
                try:
                    experts = [int(x.strip()) for x in parts[1].split(",")]
                    prompt = parts[2]
                    if not prompt.endswith("= "):
                        prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

                    normal = router.generate_normal(prompt)
                    print(f"Normal:     {normal}")

                    for exp in experts:
                        forced, _ = router.generate_with_forced_expert(prompt, exp)
                        print(f"Expert {exp}:   {forced}")
                except (ValueError, IndexError):
                    print("Usage: !compare N,M,... <prompt>")

            elif cmd == "ablate" and len(parts) >= 3:
                try:
                    ablate_idx = int(parts[1])
                    prompt = parts[2]
                    if not prompt.endswith("= "):
                        prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

                    normal = router.generate_normal(prompt)
                    ablated, stats = router.generate_with_ablated_expert(prompt, ablate_idx)

                    print(f"Normal:           {normal}")
                    print(f"Without E{ablate_idx}:      {ablated}")
                    print(f"E{ablate_idx} activations: {stats['ablation_activations']}")
                except (ValueError, IndexError):
                    print("Usage: !ablate N <prompt>")

            elif cmd == "topk" and len(parts) >= 3:
                try:
                    k = int(parts[1])
                    prompt = parts[2]
                    if not prompt.endswith("= "):
                        prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

                    normal = router.generate_normal(prompt)
                    topk_result, _ = router.generate_with_topk(prompt, k)

                    print(f"Normal (k={router.info['num_experts_per_tok']}): {normal}")
                    print(f"k={k}:          {topk_result}")
                except (ValueError, IndexError):
                    print("Usage: !topk K <prompt>")

            elif cmd == "pair" and len(parts) >= 3:
                try:
                    expert_group = [int(x.strip()) for x in parts[1].split(",")]
                    prompt = parts[2]
                    if not prompt.endswith("= "):
                        prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

                    normal = router.generate_normal(prompt)
                    group_result, _ = router.generate_with_expert_group(prompt, expert_group)

                    print(f"Normal:        {normal}")
                    print(f"Pair {expert_group}: {group_result}")
                except (ValueError, IndexError):
                    print("Usage: !pair N,M,... <prompt>")

            elif cmd == "collab" and len(parts) >= 2:
                try:
                    prompt = parts[1]
                    target_layer = router.info["moe_layers"][len(router.info["moe_layers"]) // 2]
                    selections = router.capture_expert_selections(prompt, layers=[target_layer])

                    print(f"\nToken-by-token expert selections (layer {target_layer}):")
                    tokens = tokenizer.encode(prompt)
                    for i, (token_id, experts) in enumerate(zip(tokens, selections[target_layer])):
                        token_str = tokenizer.decode([token_id])
                        experts_str = ", ".join(str(e) for e in experts)
                        print(f"  Token {i}: '{token_str}' -> Experts [{experts_str}]")

                    # Show most common pairs
                    from collections import Counter
                    pairs = []
                    for token_experts in selections[target_layer]:
                        for i, e1 in enumerate(token_experts):
                            for e2 in token_experts[i+1:]:
                                pairs.append(tuple(sorted([e1, e2])))
                    if pairs:
                        common = Counter(pairs).most_common(5)
                        print(f"\nMost common pairs:")
                        for (e1, e2), count in common:
                            print(f"  Expert {e1} + Expert {e2}: {count}x")
                except (ValueError, IndexError):
                    print("Usage: !collab <prompt>")

            else:
                print("Unknown command. Try !expert, !compare, !ablate, !topk, !pair, !collab, !info, or !quit")

        else:
            # Normal generation
            prompt = line
            if not prompt.endswith("= "):
                prompt = prompt + " = " if any(c in prompt for c in "+-*/") else prompt

            response = router.generate_normal(prompt)
            print(f"Response: {response}")


def _show_router_weights(args):
    """Show router confidence/weights for each expert selection."""
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    prompt = args.prompt

    print(f"\n{'=' * 70}")
    print(f"ROUTER CONFIDENCE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 70}")

    # Target layer
    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = router.info["moe_layers"][len(router.info["moe_layers"]) // 2]

    print(f"\nAnalyzing layer: {target_layer}")
    print()

    # Get weights
    weights_data = router.capture_router_weights(prompt, layers=[target_layer])
    tokens = tokenizer.encode(prompt)

    print("Token-by-token routing weights (sorted by confidence):")
    print("-" * 70)

    total_confidence_per_expert = {}

    for i, (token_id, (experts, weights)) in enumerate(zip(tokens, weights_data[target_layer])):
        token_str = tokenizer.decode([token_id])
        expert_str = ", ".join(f"E{e}:{w:.2f}" for e, w in zip(experts, weights))
        print(f"  Token {i}: '{token_str:12s}' -> {expert_str}")

        # Track confidence per expert
        for exp, wt in zip(experts, weights):
            if exp not in total_confidence_per_expert:
                total_confidence_per_expert[exp] = {"total_weight": 0, "selections": 0}
            total_confidence_per_expert[exp]["total_weight"] += wt
            total_confidence_per_expert[exp]["selections"] += 1

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("EXPERT CONFIDENCE SUMMARY")
    print(f"{'=' * 70}")
    print("\nAverage routing weight when selected (higher = more confident):")
    print("-" * 50)

    sorted_experts = sorted(
        total_confidence_per_expert.items(),
        key=lambda x: x[1]["total_weight"] / x[1]["selections"] if x[1]["selections"] > 0 else 0,
        reverse=True,
    )

    for exp_idx, data in sorted_experts:
        avg_weight = data["total_weight"] / data["selections"] if data["selections"] > 0 else 0
        print(f"  Expert {exp_idx:2d}: avg_weight={avg_weight:.3f} (selected {data['selections']} times)")

    # Look for "weak" selections (low confidence)
    print(f"\n{'=' * 70}")
    print("WEAK SELECTIONS (barely made top-4)")
    print(f"{'=' * 70}")

    weak_threshold = 0.15  # Less than 15% of routing weight
    weak_selections = []

    for i, (token_id, (experts, weights)) in enumerate(zip(tokens, weights_data[target_layer])):
        token_str = tokenizer.decode([token_id])
        for exp, wt in zip(experts, weights):
            if wt < weak_threshold:
                weak_selections.append((token_str, exp, wt))

    if weak_selections:
        print(f"\nExperts selected with <{weak_threshold:.0%} weight:")
        for token_str, exp, wt in weak_selections[:10]:
            print(f"  '{token_str}' -> E{exp} with weight {wt:.3f}")
    else:
        print(f"\nNo experts selected with <{weak_threshold:.0%} weight")
        print("All selections were made with reasonable confidence.")

    # Compare math experts on math tokens
    if "=" in prompt or "*" in prompt or "+" in prompt:
        print(f"\n{'=' * 70}")
        print("MATH TOKEN ANALYSIS")
        print(f"{'=' * 70}")

        math_experts = [6, 7, 11, 19]  # Known math-heavy experts
        print(f"\nHow confidently are 'math experts' {math_experts} selected on math tokens?")

        for i, (token_id, (experts, weights)) in enumerate(zip(tokens, weights_data[target_layer])):
            token_str = tokenizer.decode([token_id])
            if any(c in token_str for c in "0123456789*+=-"):
                math_expert_weights = []
                for exp, wt in zip(experts, weights):
                    if exp in math_experts:
                        math_expert_weights.append((exp, wt))

                if math_expert_weights:
                    exp_str = ", ".join(f"E{e}:{w:.2f}" for e, w in math_expert_weights)
                    print(f"  '{token_str}': {exp_str}")
                else:
                    print(f"  '{token_str}': NO math expert selected!")

    print(f"\n{'=' * 70}")


def _analyze_experts(args):
    """Analyze expert specialization patterns across all prompt categories.

    This reveals whether experts specialize by:
    - Domain (math vs code vs facts) - high-level
    - Language (Python vs Rust) - mid-level
    - Token type (punctuation, proper nouns) - low-level

    Based on ST-MoE paper findings that experts often specialize by token type
    rather than semantic domain.
    """
    from collections import defaultdict
    from ....introspection.moe.datasets import (
        get_grouped_prompts,
        CATEGORY_GROUPS,
        PromptCategoryGroup,
    )

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    print(f"\n{'=' * 70}")
    print(f"EXPERT SPECIALIZATION ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Experts: {router.info['num_experts']} (top-{router.info['num_experts_per_tok']} per token)")
    print(f"MoE layers: {len(router.info['moe_layers'])}")
    print(f"{'=' * 70}")

    # Load all prompts from JSON dataset
    categories = get_grouped_prompts()

    # Target a representative layer for analysis
    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = router.info["moe_layers"][len(router.info["moe_layers"]) // 2]

    print(f"\nAnalyzing layer: {target_layer}")
    print(f"Categories: {len(categories)}")
    print()

    # Track activations per expert per category
    expert_category_counts = defaultdict(lambda: defaultdict(int))
    category_total_tokens = defaultdict(int)

    # Process each category
    for category, prompts in categories.items():
        print(f"  Processing {category}...", end=" ", flush=True)

        for prompt in prompts:
            selections = router.capture_expert_selections(prompt, layers=[target_layer])

            for token_experts in selections[target_layer]:
                category_total_tokens[category] += 1
                for exp in token_experts:
                    expert_category_counts[exp][category] += 1

        print(f"({category_total_tokens[category]} tokens)")

    num_experts = router.info["num_experts"]

    # === PART 1: Expert-Centric View ===
    print(f"\n{'=' * 70}")
    print("EXPERT-CENTRIC VIEW: What does each expert specialize in?")
    print(f"{'=' * 70}")

    # For each expert, find their top categories
    expert_profiles = {}
    for exp_idx in range(num_experts):
        cat_counts = expert_category_counts[exp_idx]
        if not cat_counts:
            continue

        total = sum(cat_counts.values())
        sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)

        # Compute concentration (how specialized vs. generalist)
        if total > 0:
            probs = [c / total for _, c in sorted_cats]
            # Entropy-based concentration (lower = more specialized)
            import math
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = math.log(len(sorted_cats)) if len(sorted_cats) > 1 else 1
            concentration = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
        else:
            concentration = 0

        expert_profiles[exp_idx] = {
            "total_activations": total,
            "top_categories": sorted_cats[:5],
            "concentration": concentration,
            "all_counts": dict(cat_counts),
        }

    # Sort experts by total activations (most active first)
    sorted_experts = sorted(
        expert_profiles.items(),
        key=lambda x: x[1]["total_activations"],
        reverse=True,
    )

    # Show top 10 most active experts
    print("\nTop 10 Most Active Experts:")
    print("-" * 60)
    for exp_idx, profile in sorted_experts[:10]:
        top_cats = ", ".join(f"{cat}({cnt})" for cat, cnt in profile["top_categories"][:3])
        spec = "SPECIALIST" if profile["concentration"] > 0.5 else "GENERALIST"
        print(f"  Expert {exp_idx:2d}: {profile['total_activations']:5d} activations | {spec:<10} | {top_cats}")

    # === PART 2: Category-Centric View ===
    print(f"\n{'=' * 70}")
    print("CATEGORY-CENTRIC VIEW: Which experts handle each category?")
    print(f"{'=' * 70}")

    # Group by PromptCategoryGroup
    category_groups = {
        group.name: [cat.name.upper() for cat in cats]
        for group, cats in CATEGORY_GROUPS.items()
    }

    for group_name, cat_list in category_groups.items():
        print(f"\n{group_name}:")
        print("-" * 50)

        for cat in cat_list:
            if cat not in category_total_tokens:
                continue

            # Find top experts for this category
            cat_experts = []
            for exp_idx in range(num_experts):
                if cat in expert_category_counts[exp_idx]:
                    cat_experts.append((exp_idx, expert_category_counts[exp_idx][cat]))

            cat_experts.sort(key=lambda x: x[1], reverse=True)
            top_exp_str = ", ".join(f"E{e}({c})" for e, c in cat_experts[:5])
            print(f"  {cat:<18}: {top_exp_str}")

    # === PART 3: Find Specialists and Generalists ===
    print(f"\n{'=' * 70}")
    print("SPECIALIZATION PATTERNS")
    print(f"{'=' * 70}")

    # Find true specialists (concentrated in 1-2 categories)
    specialists = [(exp, prof) for exp, prof in expert_profiles.items()
                   if prof["concentration"] > 0.6 and prof["total_activations"] > 50]
    specialists.sort(key=lambda x: x[1]["concentration"], reverse=True)

    print("\nTrue Specialists (>60% concentration):")
    print("-" * 50)
    if specialists:
        for exp_idx, profile in specialists[:10]:
            top_cat = profile["top_categories"][0][0] if profile["top_categories"] else "?"
            pct = 100 * profile["top_categories"][0][1] / profile["total_activations"]
            print(f"  Expert {exp_idx:2d}: {pct:5.1f}% {top_cat}")
    else:
        print("  None found - experts may specialize by token type, not domain!")

    # Find generalists (appear across many categories)
    generalists = [(exp, prof) for exp, prof in expert_profiles.items()
                   if prof["concentration"] < 0.3 and prof["total_activations"] > 100]
    generalists.sort(key=lambda x: x[1]["total_activations"], reverse=True)

    print("\nGeneralists (appear across many categories):")
    print("-" * 50)
    for exp_idx, profile in generalists[:5]:
        num_cats = len([c for c in profile["all_counts"].values() if c > 5])
        print(f"  Expert {exp_idx:2d}: {profile['total_activations']:5d} activations across {num_cats} categories")

    # === PART 4: Cross-Group Analysis ===
    print(f"\n{'=' * 70}")
    print("CROSS-GROUP ANALYSIS: Do experts specialize by domain or token type?")
    print(f"{'=' * 70}")

    # For each expert, check if they're domain-specific or appear across domains
    group_expert_counts = defaultdict(lambda: defaultdict(int))
    for exp_idx, profile in expert_profiles.items():
        for cat, count in profile["all_counts"].items():
            # Find which group this category belongs to
            for group_name, cat_list in category_groups.items():
                if cat in cat_list:
                    group_expert_counts[exp_idx][group_name] += count
                    break

    # Categorize experts by their group spread
    domain_specialists = []  # Appear in 1-2 groups
    cross_domain = []  # Appear in 4+ groups

    for exp_idx, group_counts in group_expert_counts.items():
        active_groups = [g for g, c in group_counts.items() if c > 10]
        if len(active_groups) <= 2 and sum(group_counts.values()) > 50:
            domain_specialists.append((exp_idx, active_groups, sum(group_counts.values())))
        elif len(active_groups) >= 4:
            cross_domain.append((exp_idx, active_groups, sum(group_counts.values())))

    print("\nDomain Specialists (1-2 groups only):")
    print("-" * 50)
    domain_specialists.sort(key=lambda x: x[2], reverse=True)
    for exp_idx, groups, total in domain_specialists[:10]:
        print(f"  Expert {exp_idx:2d}: {', '.join(groups)}")

    print("\nCross-Domain Experts (4+ groups):")
    print("-" * 50)
    cross_domain.sort(key=lambda x: x[2], reverse=True)
    for exp_idx, groups, total in cross_domain[:10]:
        print(f"  Expert {exp_idx:2d}: {', '.join(groups)}")

    # === PART 5: Token-Type vs Domain Analysis ===
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT: Token-Type vs Domain Specialization")
    print(f"{'=' * 70}")

    # Check if STRUCTURE category experts are different from content experts
    structure_cats = category_groups.get("STRUCTURE", [])
    content_cats = (
        category_groups.get("MATH", []) +
        category_groups.get("CODE", []) +
        category_groups.get("FACTS", [])
    )

    structure_experts = set()
    content_experts = set()

    for exp_idx in range(num_experts):
        struct_count = sum(expert_category_counts[exp_idx].get(cat, 0) for cat in structure_cats)
        content_count = sum(expert_category_counts[exp_idx].get(cat, 0) for cat in content_cats)

        if struct_count > 20:
            structure_experts.add(exp_idx)
        if content_count > 20:
            content_experts.add(exp_idx)

    structure_only = structure_experts - content_experts
    content_only = content_experts - structure_experts
    both = structure_experts & content_experts

    print(f"\nStructure-focused experts (punctuation, pronouns, etc.): {len(structure_only)}")
    if structure_only:
        print(f"  Experts: {sorted(structure_only)[:10]}")

    print(f"\nContent-focused experts (math, code, facts): {len(content_only)}")
    if content_only:
        print(f"  Experts: {sorted(content_only)[:10]}")

    print(f"\nMixed experts (both structure and content): {len(both)}")

    if len(both) > len(structure_only) + len(content_only):
        print("\n** FINDING: Most experts are MIXED - they don't specialize by domain! **")
        print("   This suggests experts specialize by TOKEN TYPE within domains,")
        print("   not by semantic meaning. (Consistent with ST-MoE findings)")
    elif len(structure_only) > 5 or len(content_only) > 5:
        print("\n** FINDING: Some domain specialization detected **")
        print("   But experts likely still specialize by token type within domains.")

    # === PART 6: Candidate Experts Summary ===
    print(f"\n{'=' * 70}")
    print("CANDIDATE 'SPECIALIZED' EXPERTS (for video demo)")
    print(f"{'=' * 70}")

    # Find best candidates for "math expert", "code expert", etc.
    candidates = {}

    # Math expert candidate
    math_cats = category_groups.get("MATH", [])
    math_scores = []
    for exp_idx in range(num_experts):
        math_count = sum(expert_category_counts[exp_idx].get(cat, 0) for cat in math_cats)
        other_count = sum(
            expert_category_counts[exp_idx].get(cat, 0)
            for cat in categories if cat not in math_cats
        )
        if math_count > 0:
            purity = math_count / (math_count + other_count + 1e-10)
            math_scores.append((exp_idx, math_count, purity))

    math_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
    if math_scores:
        candidates["MATH"] = math_scores[0]
        print(f"\n'Math Expert' candidate: Expert {math_scores[0][0]}")
        print(f"  Math activations: {math_scores[0][1]}, Purity: {math_scores[0][2]:.1%}")

    # Code expert candidate
    code_cats = category_groups.get("CODE", [])
    code_scores = []
    for exp_idx in range(num_experts):
        code_count = sum(expert_category_counts[exp_idx].get(cat, 0) for cat in code_cats)
        other_count = sum(
            expert_category_counts[exp_idx].get(cat, 0)
            for cat in categories if cat not in code_cats
        )
        if code_count > 0:
            purity = code_count / (code_count + other_count + 1e-10)
            code_scores.append((exp_idx, code_count, purity))

    code_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
    if code_scores:
        candidates["CODE"] = code_scores[0]
        print(f"\n'Code Expert' candidate: Expert {code_scores[0][0]}")
        print(f"  Code activations: {code_scores[0][1]}, Purity: {code_scores[0][2]:.1%}")

    # Logic expert candidate
    logic_cats = ["LOGIC", "ANALOGIES", "CAUSATION"]
    logic_scores = []
    for exp_idx in range(num_experts):
        logic_count = sum(expert_category_counts[exp_idx].get(cat, 0) for cat in logic_cats)
        other_count = sum(
            expert_category_counts[exp_idx].get(cat, 0)
            for cat in categories if cat not in logic_cats
        )
        if logic_count > 0:
            purity = logic_count / (logic_count + other_count + 1e-10)
            logic_scores.append((exp_idx, logic_count, purity))

    logic_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
    if logic_scores:
        candidates["LOGIC"] = logic_scores[0]
        print(f"\n'Logic Expert' candidate: Expert {logic_scores[0][0]}")
        print(f"  Logic activations: {logic_scores[0][1]}, Purity: {logic_scores[0][2]:.1%}")

    print(f"\n{'=' * 70}")
    print("DEMO COMMANDS")
    print(f"{'=' * 70}")

    # Generate demo commands for the video
    if "MATH" in candidates:
        exp = candidates["MATH"][0]
        print(f"\n# The 'math expert' (Expert {exp})")
        print(f"lazarus introspect moe-expert chat -m {args.model} --expert {exp} -p \"127 * 89 = \"")

    if "CODE" in candidates:
        exp = candidates["CODE"][0]
        print(f"\n# The 'code expert' (Expert {exp})")
        print(f"lazarus introspect moe-expert chat -m {args.model} --expert {exp} -p \"def fibonacci(n):\"")

    if "LOGIC" in candidates:
        exp = candidates["LOGIC"][0]
        print(f"\n# The 'logic expert' (Expert {exp})")
        print(f"lazarus introspect moe-expert chat -m {args.model} --expert {exp} -p \"If A implies B, then\"")

    # Comparison command
    expert_list = ",".join(str(c[0]) for c in candidates.values() if c)
    if expert_list:
        print(f"\n# Compare all 'specialists'")
        print(f"lazarus introspect moe-expert compare -m {args.model} --experts {expert_list} -p \"127 * 89 = \"")

    print(f"\n{'=' * 70}")

    # Save results if output specified
    if getattr(args, "output", None):
        output_data = {
            "model": args.model,
            "layer": target_layer,
            "num_experts": num_experts,
            "expert_profiles": {
                str(k): {
                    "total_activations": v["total_activations"],
                    "concentration": v["concentration"],
                    "top_categories": [(cat, cnt) for cat, cnt in v["top_categories"]],
                }
                for k, v in expert_profiles.items()
            },
            "candidates": {
                k: {"expert": v[0], "count": v[1], "purity": v[2]}
                for k, v in candidates.items()
            },
            "category_totals": dict(category_total_tokens),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


# =============================================================================
# TOKENIZER ANALYSIS FUNCTIONS
# =============================================================================

def extract_all_control_tokens(tokenizer) -> dict:
    """
    Extract ALL control tokens from a tokenizer - no hardcoding, pure discovery.

    Returns dict with:
    - special_tokens: From tokenizer config (bos, eos, pad, etc.)
    - added_tokens: Explicitly added post-training
    - chat_template_tokens: Extracted from Jinja2 template
    - structural_tokens: Pattern-matched from vocab
    - high_id_tokens: Late additions (high token IDs)
    - byte_tokens: Byte-level fallbacks
    """
    import re

    results = {
        "special_tokens": {},
        "added_tokens": {},
        "chat_template_tokens": {},
        "structural_tokens": {},
        "high_id_tokens": {},
        "byte_tokens": [],
    }

    # =========================================
    # 1. EXPLICIT SPECIAL TOKENS (from config)
    # =========================================
    if hasattr(tokenizer, "special_tokens_map"):
        special_map = tokenizer.special_tokens_map
        for role, token in special_map.items():
            if isinstance(token, str):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    results["special_tokens"][role] = {
                        "token": token,
                        "id": token_id,
                    }
                except Exception:
                    pass
            elif isinstance(token, list):
                for t in token:
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(t)
                        results["special_tokens"][f"{role}_{t}"] = {
                            "token": t,
                            "id": token_id,
                        }
                    except Exception:
                        pass

    # =========================================
    # 2. ADDED TOKENS (explicitly added post-training)
    # =========================================
    if hasattr(tokenizer, "added_tokens_encoder"):
        for token, idx in tokenizer.added_tokens_encoder.items():
            is_special = True
            if hasattr(tokenizer, "added_tokens_decoder"):
                decoder_info = tokenizer.added_tokens_decoder.get(idx)
                if decoder_info and hasattr(decoder_info, "special"):
                    is_special = decoder_info.special
            results["added_tokens"][token] = {
                "id": idx,
                "special": is_special,
            }

    # =========================================
    # 3. CHAT TEMPLATE PARSING (extract from Jinja2)
    # =========================================
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        template = tokenizer.chat_template
        results["chat_template_raw"] = template

        # Extract string literals that look like control tokens
        string_literals = re.findall(r'["\'](<[^>]+>|<\|[^|]+\|>|\[[^\]]+\])["\']', template)

        vocab = tokenizer.get_vocab()
        for lit in string_literals:
            if lit in vocab:
                results["chat_template_tokens"][lit] = {
                    "id": vocab[lit],
                    "source": "template_literal",
                }

        # Try to parse Jinja AST for variable usage
        try:
            from jinja2 import Environment, meta
            env = Environment()
            ast = env.parse(template)
            variables = meta.find_undeclared_variables(ast)
            results["chat_template_variables"] = list(variables)
        except ImportError:
            results["chat_template_variables"] = []
        except Exception:
            results["chat_template_variables"] = []

    # =========================================
    # 4. STRUCTURAL DETECTION (pattern-based discovery)
    # =========================================
    vocab = tokenizer.get_vocab()

    patterns = {
        "angle_bracket": r"^<[^>]+>$",           # <token>
        "pipe_delimited": r"^<\|[^|]+\|>$",      # <|token|>
        "square_bracket": r"^\[[^\]]+\]$",       # [TOKEN]
        "double_bracket": r"^<<[^>]+>>$",        # <<token>>
        "hash_prefix": r"^##",                   # ##continuation (BERT)
        "underscore_prefix": r"^▁",              # ▁word (SentencePiece)
    }

    for token in vocab:
        for pattern_name, pattern in patterns.items():
            if re.match(pattern, token):
                if pattern_name not in results["structural_tokens"]:
                    results["structural_tokens"][pattern_name] = []
                results["structural_tokens"][pattern_name].append({
                    "token": token,
                    "id": vocab[token],
                })

    # =========================================
    # 5. HIGH-ID TOKENS (added late = likely control)
    # =========================================
    vocab_size = len(vocab)
    threshold = vocab_size - 500  # Last 500 tokens

    for token, idx in vocab.items():
        if idx >= threshold:
            results["high_id_tokens"][token] = {
                "id": idx,
                "percentile": idx / vocab_size,
            }

    # =========================================
    # 6. BYTE-LEVEL FALLBACKS
    # =========================================
    results["byte_tokens"] = [
        {"token": t, "id": vocab[t]}
        for t in vocab
        if re.match(r"^<0x[0-9A-Fa-f]{2}>$", t)
    ]

    return results


def categorize_discovered_tokens(discovered: dict) -> dict:
    """
    Group discovered tokens by likely function.
    """
    all_tokens = {}

    # Merge all sources
    for source in ["special_tokens", "added_tokens", "chat_template_tokens"]:
        for name, info in discovered.get(source, {}).items():
            token = info.get("token", name)
            all_tokens[token] = {
                **info,
                "source": source,
            }

    for pattern, tokens in discovered.get("structural_tokens", {}).items():
        for t in tokens:
            if t["token"] not in all_tokens:
                all_tokens[t["token"]] = {
                    **t,
                    "source": f"structural:{pattern}",
                }

    # Now categorize by likely function
    categories = {
        "GENERATION_CONTROL": [],    # eos, bos, pad
        "CHAT_ROLES": [],            # user, assistant, system
        "TOOL_USE": [],              # function, tool, code
        "REASONING": [],             # think, step, reflection
        "STRUCTURED_OUTPUT": [],     # json, xml, output
        "RETRIEVAL": [],             # context, document, source
        "CODE_INFILL": [],           # fim, prefix, suffix, middle
        "UNKNOWN_CONTROL": [],       # Detected but uncategorized
    }

    for token, info in all_tokens.items():
        token_lower = token.lower()

        if any(x in token_lower for x in ["eos", "bos", "end", "start", "pad", "sep", "mask", "endoftext"]):
            categories["GENERATION_CONTROL"].append((token, info))
        elif any(x in token_lower for x in ["user", "assistant", "system", "human", "bot", "inst", "im_start", "im_end"]):
            categories["CHAT_ROLES"].append((token, info))
        elif any(x in token_lower for x in ["function", "tool", "python", "call", "ipython"]):
            categories["TOOL_USE"].append((token, info))
        elif any(x in token_lower for x in ["think", "reason", "step", "reflect", "scratch"]):
            categories["REASONING"].append((token, info))
        elif any(x in token_lower for x in ["json", "xml", "output", "result"]):
            categories["STRUCTURED_OUTPUT"].append((token, info))
        elif any(x in token_lower for x in ["context", "document", "source", "retriev", "cite"]):
            categories["RETRIEVAL"].append((token, info))
        elif any(x in token_lower for x in ["fim", "prefix", "suffix", "middle", "infill"]):
            categories["CODE_INFILL"].append((token, info))
        else:
            categories["UNKNOWN_CONTROL"].append((token, info))

    return categories


def _analyze_tokenizer(args):
    """Analyze tokenizer to discover all control tokens programmatically."""
    from transformers import AutoTokenizer

    print(f"\n{'=' * 70}")
    print("TOKENIZER CONTROL TOKEN DISCOVERY")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")

    # Extract all control tokens
    discovered = extract_all_control_tokens(tokenizer)

    print(f"\n{'=' * 70}")
    print("DISCOVERED SOURCES")
    print(f"{'=' * 70}")

    print(f"\n  special_tokens_map: {len(discovered['special_tokens'])} tokens")
    for name, info in list(discovered["special_tokens"].items())[:5]:
        print(f"    {name}: {info['token']} (id={info['id']})")

    print(f"\n  added_tokens: {len(discovered['added_tokens'])} tokens")
    for token, info in list(discovered["added_tokens"].items())[:10]:
        print(f"    {token} (id={info['id']}, special={info.get('special', '?')})")

    if discovered.get("chat_template_raw"):
        print(f"\n  chat_template: Found (Jinja2)")
        ct_tokens = discovered.get("chat_template_tokens", {})
        print(f"    Extracted tokens: {len(ct_tokens)}")
        for token, info in ct_tokens.items():
            print(f"      {token} (id={info['id']})")
    else:
        print(f"\n  chat_template: Not found")

    for pattern, tokens in discovered.get("structural_tokens", {}).items():
        print(f"\n  structural ({pattern}): {len(tokens)} tokens")
        for t in tokens[:5]:
            print(f"    {t['token']} (id={t['id']})")
        if len(tokens) > 5:
            print(f"    ... and {len(tokens) - 5} more")

    print(f"\n  high_id tokens (>99.5%): {len(discovered['high_id_tokens'])} tokens")
    sorted_high = sorted(discovered["high_id_tokens"].items(), key=lambda x: x[1]["id"])
    for token, info in sorted_high[:10]:
        print(f"    {token} (id={info['id']}, pct={info['percentile']:.3f})")

    print(f"\n  byte_tokens: {len(discovered['byte_tokens'])} tokens")

    # Categorize tokens
    categories = categorize_discovered_tokens(discovered)

    print(f"\n{'=' * 70}")
    print("CATEGORIZED CONTROL TOKENS")
    print(f"{'=' * 70}")

    for category, tokens in categories.items():
        if tokens:
            print(f"\n{category} ({len(tokens)} tokens):")
            for token, info in tokens[:10]:
                source = info.get("source", "unknown")
                token_id = info.get("id", "?")
                print(f"  {token:<20} id={token_id:<8} source={source}")
            if len(tokens) > 10:
                print(f"  ... and {len(tokens) - 10} more")

    # Show chat template if present
    if discovered.get("chat_template_raw"):
        print(f"\n{'=' * 70}")
        print("CHAT TEMPLATE ANALYSIS")
        print(f"{'=' * 70}")
        template = discovered["chat_template_raw"]
        # Truncate if too long
        if len(template) > 500:
            print(f"\n{template[:500]}...")
        else:
            print(f"\n{template}")

    print(f"\n{'=' * 70}")
    print("NEXT STEP: Analyze which experts handle these tokens")
    print(f"{'=' * 70}")
    print(f"\nRun: lazarus introspect moe-expert control-tokens -m {args.model}")
    print(f"{'=' * 70}")

    # Save results if output specified
    if getattr(args, "output", None):
        output_data = {
            "model": args.model,
            "vocab_size": len(vocab),
            "discovered": {
                "special_tokens": discovered["special_tokens"],
                "added_tokens": discovered["added_tokens"],
                "chat_template_tokens": discovered.get("chat_template_tokens", {}),
                "high_id_count": len(discovered["high_id_tokens"]),
                "byte_token_count": len(discovered["byte_tokens"]),
            },
            "categories": {
                cat: [(t, {"id": info.get("id"), "source": info.get("source")})
                      for t, info in tokens]
                for cat, tokens in categories.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


def _analyze_control_token_experts(args):
    """Analyze which experts handle control tokens vs regular tokens."""
    import math
    from collections import defaultdict

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    print(f"\n{'=' * 70}")
    print("CONTROL TOKEN → EXPERT MAPPING")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Experts: {router.info['num_experts']} (top-{router.info['num_experts_per_tok']} per token)")
    print(f"{'=' * 70}")

    # Extract control tokens
    discovered = extract_all_control_tokens(tokenizer)
    categories = categorize_discovered_tokens(discovered)

    # Target layer
    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = router.info["moe_layers"][len(router.info["moe_layers"]) // 2]

    print(f"\nAnalyzing layer: {target_layer}")

    # Analyze control tokens
    control_results = []

    for category, tokens in categories.items():
        if not tokens:
            continue

        print(f"\nProcessing {category}...", end=" ", flush=True)

        for token, info in tokens[:20]:  # Limit to 20 per category
            token_id = info.get("id")
            if token_id is None:
                continue

            # Create minimal prompt with just this token
            # We need context, so prepend a common token
            prompt = f"The {token}"

            try:
                # Get router weights for this token
                weights_data = router.capture_router_weights(prompt, layers=[target_layer])

                if target_layer in weights_data and len(weights_data[target_layer]) > 0:
                    # Find the position of our control token
                    # Usually it's the last token in the prompt
                    token_data = weights_data[target_layer][-1]
                    experts, weights = token_data

                    top_expert = experts[0]
                    top_weight = weights[0]

                    # Compute entropy
                    entropy = -sum(w * math.log(w + 1e-10) for w in weights if w > 0)
                    max_entropy = math.log(len(weights))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                    control_results.append({
                        "token": token,
                        "category": category,
                        "top_expert": top_expert,
                        "top_weight": top_weight,
                        "entropy": entropy,
                        "normalized_entropy": normalized_entropy,
                        "is_specialist": top_weight > 0.5,
                        "all_experts": list(zip(experts, weights)),
                    })
            except Exception as e:
                # Skip tokens that cause issues
                continue

        print(f"({len([r for r in control_results if r['category'] == category])} analyzed)")

    # Analyze some regular tokens for comparison
    print(f"\nProcessing REGULAR tokens for comparison...", end=" ", flush=True)

    regular_tokens = ["the", "a", "is", "and", "of", "to", "in", "that", "it", "for",
                      "127", "42", "100", "def", "class", "return", "if", "else"]

    regular_results = []
    vocab = tokenizer.get_vocab()

    for token in regular_tokens:
        if token not in vocab:
            continue

        prompt = f"The {token}"

        try:
            weights_data = router.capture_router_weights(prompt, layers=[target_layer])

            if target_layer in weights_data and len(weights_data[target_layer]) > 0:
                token_data = weights_data[target_layer][-1]
                experts, weights = token_data

                top_expert = experts[0]
                top_weight = weights[0]

                entropy = -sum(w * math.log(w + 1e-10) for w in weights if w > 0)
                max_entropy = math.log(len(weights))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                regular_results.append({
                    "token": token,
                    "category": "REGULAR",
                    "top_expert": top_expert,
                    "top_weight": top_weight,
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "is_specialist": top_weight > 0.5,
                })
        except Exception:
            continue

    print(f"({len(regular_results)} analyzed)")

    # Display results
    print(f"\n{'=' * 70}")
    print("CONTROL TOKEN ROUTING ANALYSIS")
    print(f"{'=' * 70}")

    print("\n{:<20} | {:<12} | {:<10} | {:<6} | {:<8} | {}".format(
        "Token", "Category", "Top Expert", "Prob", "Entropy", "Specialist?"
    ))
    print("-" * 80)

    # Sort by top_weight descending (most specialized first)
    sorted_results = sorted(control_results, key=lambda x: x["top_weight"], reverse=True)

    for r in sorted_results[:30]:
        specialist = "YES" if r["is_specialist"] else "no"
        print("{:<20} | {:<12} | E{:<9} | {:.2f}  | {:.2f}    | {}".format(
            r["token"][:20],
            r["category"][:12],
            r["top_expert"],
            r["top_weight"],
            r["entropy"],
            specialist
        ))

    # Show regular tokens for comparison
    print(f"\n{'=' * 70}")
    print("REGULAR TOKEN COMPARISON (baseline)")
    print(f"{'=' * 70}")

    print("\n{:<20} | {:<12} | {:<10} | {:<6} | {:<8} | {}".format(
        "Token", "Category", "Top Expert", "Prob", "Entropy", "Specialist?"
    ))
    print("-" * 80)

    for r in regular_results:
        specialist = "YES" if r["is_specialist"] else "no"
        print("{:<20} | {:<12} | E{:<9} | {:.2f}  | {:.2f}    | {}".format(
            r["token"][:20],
            r["category"][:12],
            r["top_expert"],
            r["top_weight"],
            r["entropy"],
            specialist
        ))

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 70}")

    if control_results:
        avg_control_weight = sum(r["top_weight"] for r in control_results) / len(control_results)
        avg_control_entropy = sum(r["entropy"] for r in control_results) / len(control_results)
        control_specialists = sum(1 for r in control_results if r["is_specialist"])

        print(f"\nControl tokens ({len(control_results)} analyzed):")
        print(f"  Average top weight: {avg_control_weight:.3f}")
        print(f"  Average entropy: {avg_control_entropy:.3f}")
        print(f"  Specialists (>50% to one expert): {control_specialists} ({100*control_specialists/len(control_results):.0f}%)")

    if regular_results:
        avg_regular_weight = sum(r["top_weight"] for r in regular_results) / len(regular_results)
        avg_regular_entropy = sum(r["entropy"] for r in regular_results) / len(regular_results)
        regular_specialists = sum(1 for r in regular_results if r["is_specialist"])

        print(f"\nRegular tokens ({len(regular_results)} analyzed):")
        print(f"  Average top weight: {avg_regular_weight:.3f}")
        print(f"  Average entropy: {avg_regular_entropy:.3f}")
        print(f"  Specialists (>50% to one expert): {regular_specialists} ({100*regular_specialists/len(regular_results):.0f}%)")

    # Find expert specialists by category
    print(f"\n{'=' * 70}")
    print("EXPERT SPECIALIZATION BY CATEGORY")
    print(f"{'=' * 70}")

    expert_categories = defaultdict(list)
    for r in control_results:
        if r["is_specialist"]:
            expert_categories[r["top_expert"]].append(r["category"])

    for exp, cats in sorted(expert_categories.items()):
        cat_counts = defaultdict(int)
        for c in cats:
            cat_counts[c] += 1
        cat_str = ", ".join(f"{c}({n})" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]))
        print(f"  Expert {exp}: {cat_str}")

    # Key finding
    print(f"\n{'=' * 70}")
    print("KEY FINDING")
    print(f"{'=' * 70}")

    if control_results and regular_results:
        if avg_control_weight > avg_regular_weight + 0.1:
            print(f"\n** Control tokens route to specialists more often! **")
            print(f"   Control avg weight: {avg_control_weight:.3f}")
            print(f"   Regular avg weight: {avg_regular_weight:.3f}")
            print(f"   Difference: +{avg_control_weight - avg_regular_weight:.3f}")
            print(f"\n   'The only true specialists handle protocol tokens, not content.'")
        else:
            print(f"\n** No strong difference between control and regular tokens. **")
            print(f"   Both route to generalists.")

    print(f"\n{'=' * 70}")

    # Save results if output specified
    if getattr(args, "output", None):
        output_data = {
            "model": args.model,
            "layer": target_layer,
            "control_tokens": control_results,
            "regular_tokens": regular_results,
            "summary": {
                "control_avg_weight": avg_control_weight if control_results else 0,
                "regular_avg_weight": avg_regular_weight if regular_results else 0,
                "control_specialists": control_specialists if control_results else 0,
                "regular_specialists": regular_specialists if regular_results else 0,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")


def _trace_token_experts(args):
    """Trace which experts handle each token across ALL layers.

    This reveals whether experts are shared or independent across layers.
    """
    from collections import defaultdict

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    prompt = args.prompt

    print(f"\n{'=' * 70}")
    print("TOKEN → EXPERT TRACE (ALL LAYERS)")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Prompt: {prompt}")
    print(f"MoE layers: {len(router.info['moe_layers'])}")
    print(f"{'=' * 70}")

    # Check architecture
    print(f"\n{'=' * 70}")
    print("ARCHITECTURE CHECK")
    print(f"{'=' * 70}")

    # Check if experts are shared or independent
    layers = router._layers
    if len(layers) >= 2:
        mlp0 = layers[0].mlp if hasattr(layers[0], 'mlp') else None
        mlp1 = layers[1].mlp if hasattr(layers[1], 'mlp') else None
        if mlp0 is not None and mlp1 is not None:
            shared = mlp0 is mlp1
            if shared:
                print(">>> SHARED EXPERTS: Same 32 experts reused at every layer <<<")
                print("    Expert 6 at layer 0 IS Expert 6 at layer 12")
            else:
                print(">>> INDEPENDENT EXPERTS: Each layer has its own 32 experts <<<")
                print("    Expert 6 at layer 0 is DIFFERENT from Expert 6 at layer 12")
                print(f"    Total expert MLPs: 32 × {len(router.info['moe_layers'])} = {32 * len(router.info['moe_layers'])}")

    # Get all MoE layers
    all_layers = router.info["moe_layers"]

    # Capture expert selections at ALL layers
    print(f"\n{'=' * 70}")
    print("EXPERT SELECTIONS PER LAYER")
    print(f"{'=' * 70}")

    selections = router.capture_expert_selections(prompt, layers=all_layers)
    tokens = tokenizer.encode(prompt)

    # For each token, show which experts were selected at each layer
    print(f"\nTokens: {[tokenizer.decode([t]) for t in tokens]}")

    for token_idx, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])
        print(f"\n{'=' * 50}")
        print(f"Token {token_idx}: '{token_str}'")
        print(f"{'=' * 50}")

        # Track expert frequency across layers
        expert_freq = defaultdict(int)

        for layer_idx in all_layers:
            if layer_idx in selections and token_idx < len(selections[layer_idx]):
                experts = selections[layer_idx][token_idx]
                expert_str = ", ".join(f"E{e}" for e in sorted(experts))
                print(f"  Layer {layer_idx:2d}: [{expert_str}]")
                for e in experts:
                    expert_freq[e] += 1

        # Show most common experts for this token
        if expert_freq:
            sorted_experts = sorted(expert_freq.items(), key=lambda x: -x[1])
            print(f"\n  Most frequent experts for '{token_str}':")
            for exp, count in sorted_experts[:5]:
                pct = 100 * count / len(all_layers)
                print(f"    E{exp}: {count}/{len(all_layers)} layers ({pct:.0f}%)")

    # Analyze consistency
    print(f"\n{'=' * 70}")
    print("ROUTING CONSISTENCY ANALYSIS")
    print(f"{'=' * 70}")

    for token_idx, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])

        # Get all expert sets for this token
        expert_sets = []
        for layer_idx in all_layers:
            if layer_idx in selections and token_idx < len(selections[layer_idx]):
                expert_sets.append(set(selections[layer_idx][token_idx]))

        if expert_sets:
            # Check if same experts selected at all layers
            first_set = expert_sets[0]
            all_same = all(s == first_set for s in expert_sets)

            # Count unique expert sets
            unique_sets = len(set(frozenset(s) for s in expert_sets))

            if all_same:
                print(f"  '{token_str}': IDENTICAL routing at all {len(expert_sets)} layers")
            else:
                print(f"  '{token_str}': {unique_sets} DIFFERENT routings across {len(expert_sets)} layers")

    # Key insight
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print(f"{'=' * 70}")

    # Check if routing is consistent or varies
    total_unique = 0
    total_layers = 0
    for token_idx, token_id in enumerate(tokens):
        expert_sets = []
        for layer_idx in all_layers:
            if layer_idx in selections and token_idx < len(selections[layer_idx]):
                expert_sets.append(frozenset(selections[layer_idx][token_idx]))
        if expert_sets:
            total_unique += len(set(expert_sets))
            total_layers += len(expert_sets)

    avg_unique = total_unique / len(tokens) if tokens else 0

    if avg_unique > len(all_layers) * 0.8:
        print(f"\n** DIFFERENT EXPERTS AT EACH LAYER **")
        print(f"   Average unique routing sets per token: {avg_unique:.1f} out of {len(all_layers)} layers")
        print(f"\n   This means routing is LAYER-SPECIFIC:")
        print(f"   - Early layers route to different experts than late layers")
        print(f"   - Expert 'specialization' is a per-layer phenomenon")
        print(f"   - 'Expert 6' at layer 0 does different work than 'Expert 6' at layer 12")
    else:
        print(f"\n** CONSISTENT EXPERTS ACROSS LAYERS **")
        print(f"   Average unique routing sets per token: {avg_unique:.1f} out of {len(all_layers)} layers")
        print(f"\n   This means routing is TOKEN-SPECIFIC:")
        print(f"   - Same experts handle the same token at all layers")
        print(f"   - Expert 'specialization' is global across the model")

    print(f"\n{'=' * 70}")


def _analyze_routing_entropy(args):
    """Analyze routing entropy/confidence by layer - where does the model 'decide'?"""
    import math
    from collections import defaultdict

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    prompt = args.prompt or "127 * 89 = "

    print(f"\n{'=' * 70}")
    print("ROUTING ENTROPY BY LAYER")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 70}")

    all_layers = router.info["moe_layers"]

    # Capture weights at all layers
    weights = router.capture_router_weights(prompt, layers=all_layers)
    tokens = tokenizer.encode(prompt)

    # Calculate entropy for each token at each layer
    layer_entropies = defaultdict(list)

    for layer_idx in all_layers:
        if layer_idx in weights:
            for token_idx, (experts, probs) in enumerate(weights[layer_idx]):
                # Shannon entropy: -sum(p * log(p))
                entropy = 0.0
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log2(p)
                layer_entropies[layer_idx].append(entropy)

    # Calculate average entropy per layer
    print(f"\n{'=' * 70}")
    print("AVERAGE ENTROPY BY LAYER")
    print(f"{'=' * 70}")
    print("(Lower entropy = more confident routing)")
    print()

    layer_avg_entropy = {}
    max_possible_entropy = math.log2(router.info["num_experts_per_tok"])  # Maximum entropy for top-k

    for layer_idx in all_layers:
        if layer_idx in layer_entropies:
            avg = sum(layer_entropies[layer_idx]) / len(layer_entropies[layer_idx])
            layer_avg_entropy[layer_idx] = avg

            # Visual bar
            confidence = 1 - (avg / max_possible_entropy)
            bar_len = int(confidence * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)

            print(f"  Layer {layer_idx:2d}: entropy={avg:.3f}  [{bar}] {confidence*100:.0f}% confident")

    # Find most and least confident layers
    sorted_layers = sorted(layer_avg_entropy.items(), key=lambda x: x[1])

    print(f"\n{'=' * 70}")
    print("MOST CONFIDENT LAYERS (lowest entropy)")
    print(f"{'=' * 70}")
    for layer, entropy in sorted_layers[:5]:
        print(f"  Layer {layer}: entropy {entropy:.3f}")

    print(f"\n{'=' * 70}")
    print("LEAST CONFIDENT LAYERS (highest entropy)")
    print(f"{'=' * 70}")
    for layer, entropy in sorted_layers[-5:]:
        print(f"  Layer {layer}: entropy {entropy:.3f}")

    # Token-level analysis
    print(f"\n{'=' * 70}")
    print("TOKEN-BY-TOKEN ENTROPY")
    print(f"{'=' * 70}")

    for token_idx, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])
        print(f"\nToken {token_idx}: '{token_str}'")

        # Show entropy at key layers (early, middle, late, calculator)
        key_layers = [0, len(all_layers)//2, 19, 20, 21, len(all_layers)-1]
        for layer in key_layers:
            if layer in layer_entropies and token_idx < len(layer_entropies[layer]):
                ent = layer_entropies[layer][token_idx]
                confidence = 1 - (ent / max_possible_entropy)
                print(f"  L{layer:2d}: entropy={ent:.3f} ({confidence*100:.0f}% confident)")

    # Calculator layer analysis
    print(f"\n{'=' * 70}")
    print("CALCULATOR LAYERS (L19-21) CONFIDENCE")
    print(f"{'=' * 70}")

    calc_layers = [19, 20, 21]
    for layer in calc_layers:
        if layer in layer_avg_entropy:
            avg = layer_avg_entropy[layer]
            rank = sorted_layers.index((layer, avg)) + 1
            print(f"  Layer {layer}: entropy {avg:.3f} (rank {rank}/{len(all_layers)})")


def _analyze_layer_divergence(args):
    """Analyze routing divergence between two sets of prompts across layers.

    Use --prompt for group A prompts (comma-separated) and --compare for group B.
    Shows which layers route differently for different domains.
    """
    from collections import defaultdict
    import math as math_module

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    # Get prompts from args or use defaults
    prompts_a = args.prompt.split(",") if args.prompt else [
        "127 * 89 = ",
        "2 + 2 = ",
        "456 + 789 = ",
    ]

    compare_prompts = getattr(args, "compare_prompts", None)
    prompts_b = compare_prompts.split(",") if compare_prompts else [
        "The quick brown fox",
        "Hello, how are you?",
        "Once upon a time",
    ]

    print(f"\n{'=' * 70}")
    print("LAYER ROUTING DIVERGENCE")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Group A: {prompts_a[:3]}{'...' if len(prompts_a) > 3 else ''}")
    print(f"Group B: {prompts_b[:3]}{'...' if len(prompts_b) > 3 else ''}")
    print(f"{'=' * 70}")

    all_layers = router.info["moe_layers"]

    # Track routing patterns for both groups
    routing_a = {layer: defaultdict(int) for layer in all_layers}
    routing_b = {layer: defaultdict(int) for layer in all_layers}

    for prompt in prompts_a:
        prompt = prompt.strip()
        selections = router.capture_expert_selections(prompt, layers=all_layers)
        for layer_idx in all_layers:
            if layer_idx in selections:
                for token_idx in range(len(selections[layer_idx])):
                    for expert in selections[layer_idx][token_idx]:
                        routing_a[layer_idx][expert] += 1

    for prompt in prompts_b:
        prompt = prompt.strip()
        selections = router.capture_expert_selections(prompt, layers=all_layers)
        for layer_idx in all_layers:
            if layer_idx in selections:
                for token_idx in range(len(selections[layer_idx])):
                    for expert in selections[layer_idx][token_idx]:
                        routing_b[layer_idx][expert] += 1

    # Calculate Jensen-Shannon divergence at each layer
    print(f"\n{'=' * 70}")
    print("ROUTING DIVERGENCE BY LAYER")
    print(f"{'=' * 70}")
    print("Higher divergence = layer routes group A differently than group B")
    print()

    layer_divergence = {}

    for layer_idx in all_layers:
        total_a = sum(routing_a[layer_idx].values()) or 1
        total_b = sum(routing_b[layer_idx].values()) or 1

        all_experts = set(routing_a[layer_idx].keys()) | set(routing_b[layer_idx].keys())

        divergence = 0.0
        for expert in all_experts:
            p = routing_a[layer_idx].get(expert, 0) / total_a
            q = routing_b[layer_idx].get(expert, 0) / total_b
            m = (p + q) / 2

            if p > 0 and m > 0:
                divergence += p * math_module.log2(p / m) / 2
            if q > 0 and m > 0:
                divergence += q * math_module.log2(q / m) / 2

        layer_divergence[layer_idx] = divergence

        # Visual bar
        bar_len = int(divergence * 50)
        bar = '█' * bar_len + '░' * (50 - bar_len)

        print(f"  Layer {layer_idx:2d}: {divergence:.4f} [{bar}]")

    # Identify most divergent layers
    sorted_layers = sorted(layer_divergence.items(), key=lambda x: -x[1])

    print(f"\n{'=' * 70}")
    print("MOST DIVERGENT LAYERS")
    print(f"{'=' * 70}")

    for layer, div in sorted_layers[:5]:
        # Show which experts differ most between groups
        top_a = sorted(routing_a[layer].items(), key=lambda x: -x[1])[:3]
        top_b = sorted(routing_b[layer].items(), key=lambda x: -x[1])[:3]

        print(f"\n  Layer {layer} (divergence: {div:.4f}):")
        print(f"    Group A experts: {', '.join(f'E{e}:{c}' for e, c in top_a)}")
        print(f"    Group B experts: {', '.join(f'E{e}:{c}' for e, c in top_b)}")

    print(f"\n{'=' * 70}")


def _analyze_layer_role(args):
    """Investigate what a specific MoE layer is actually doing.

    Analyzes:
    1. What token types trigger confident routing
    2. Expert output magnitude vs residual stream
    3. What changes when we ablate confident experts
    """
    from collections import defaultdict
    import math as math_module

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9  # Default to highest-confidence layer

    print(f"\n{'=' * 70}")
    print(f"LAYER {target_layer} ROLE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"{'=' * 70}")

    # Test prompts covering different token types
    test_cases = {
        "Numbers": ["127", "45", "1000", "3.14159"],
        "Operators": ["* 89 =", "+ 2 =", "- 10 ="],
        "Words": ["hello", "the", "beautiful"],
        "Punctuation": [".", ",", "?", "!"],
        "Code keywords": ["def", "return", "if", "for"],
        "Proper nouns": ["Paris", "Einstein", "Python"],
        "Formatting": ["  ", "\n", "    "],
        "Mixed": ["Hello, world!", "x = 5", "The answer is 42."],
    }

    print(f"\n{'=' * 70}")
    print(f"TOKEN TYPE ROUTING CONFIDENCE AT LAYER {target_layer}")
    print(f"{'=' * 70}")

    category_confidences = {}

    for category, prompts in test_cases.items():
        confidences = []
        expert_counts = defaultdict(int)

        for prompt in prompts:
            try:
                weights = router.capture_router_weights(prompt, layers=[target_layer])
                if target_layer in weights:
                    for token_idx, (experts, probs) in enumerate(weights[target_layer]):
                        # Calculate entropy as inverse confidence
                        entropy = 0.0
                        for p in probs:
                            if p > 0:
                                entropy -= p * math_module.log2(p)

                        max_entropy = math_module.log2(router.info["num_experts_per_tok"])
                        confidence = 1 - (entropy / max_entropy)
                        confidences.append(confidence)

                        # Track which experts
                        for exp in experts:
                            expert_counts[exp] += 1
            except Exception:
                pass

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            category_confidences[category] = avg_conf
            top_experts = sorted(expert_counts.items(), key=lambda x: -x[1])[:3]
            expert_str = ", ".join(f"E{e}" for e, _ in top_experts)

            # Visual bar
            bar_len = int(avg_conf * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)

            print(f"  {category:15s}: [{bar}] {avg_conf*100:5.1f}%  Top: {expert_str}")

    # Find what this layer is most confident about
    sorted_categories = sorted(category_confidences.items(), key=lambda x: -x[1])

    print(f"\n{'=' * 70}")
    print(f"LAYER {target_layer} IS MOST CONFIDENT ABOUT:")
    print(f"{'=' * 70}")

    for category, conf in sorted_categories[:3]:
        print(f"  1. {category}: {conf*100:.1f}% confident")

    print(f"\n{'=' * 70}")
    print(f"LAYER {target_layer} IS LEAST CONFIDENT ABOUT:")
    print(f"{'=' * 70}")

    for category, conf in sorted_categories[-3:]:
        print(f"  - {category}: {conf*100:.1f}% confident")

    # Analyze expert specialization at this layer
    print(f"\n{'=' * 70}")
    print(f"EXPERT SPECIALIZATION AT LAYER {target_layer}")
    print(f"{'=' * 70}")

    # Track which experts handle which categories
    expert_category_counts = defaultdict(lambda: defaultdict(int))

    for category, prompts in test_cases.items():
        for prompt in prompts:
            try:
                weights = router.capture_router_weights(prompt, layers=[target_layer])
                if target_layer in weights:
                    for token_idx, (experts, probs) in enumerate(weights[target_layer]):
                        for exp, prob in zip(experts, probs):
                            expert_category_counts[exp][category] += prob
            except Exception:
                pass

    # Find most specialized experts
    print("\nExperts with strongest category preferences:")

    for exp_idx in sorted(expert_category_counts.keys()):
        categories = expert_category_counts[exp_idx]
        if categories:
            total = sum(categories.values())
            top_category = max(categories.items(), key=lambda x: x[1])
            specialization = top_category[1] / total if total > 0 else 0

            if specialization > 0.3:  # More than 30% on one category
                print(f"  E{exp_idx:2d}: {top_category[0]} ({specialization*100:.0f}%)")

    # Hypothesis testing
    print(f"\n{'=' * 70}")
    print(f"HYPOTHESES FOR LAYER {target_layer}")
    print(f"{'=' * 70}")

    # Check if it's a "formatting" layer
    formatting_conf = category_confidences.get("Formatting", 0)
    punctuation_conf = category_confidences.get("Punctuation", 0)
    if formatting_conf > 0.15 or punctuation_conf > 0.15:
        print(f"  [?] FORMATTING/STRUCTURE: High confidence on whitespace/punctuation")

    # Check if it's a "syntax" layer
    code_conf = category_confidences.get("Code keywords", 0)
    if code_conf > 0.15:
        print(f"  [?] SYNTAX: High confidence on code keywords")

    # Check if it's a "semantic" layer
    word_conf = category_confidences.get("Words", 0)
    noun_conf = category_confidences.get("Proper nouns", 0)
    if word_conf > 0.15 or noun_conf > 0.15:
        print(f"  [?] SEMANTIC: High confidence on words/nouns")

    # Check if it's a "numeric" layer
    num_conf = category_confidences.get("Numbers", 0)
    op_conf = category_confidences.get("Operators", 0)
    if num_conf > 0.15 or op_conf > 0.15:
        print(f"  [?] NUMERIC: High confidence on numbers/operators")

    # General pattern
    avg_all = sum(category_confidences.values()) / len(category_confidences) if category_confidences else 0
    print(f"\n  Average confidence across all categories: {avg_all*100:.1f}%")

    if avg_all > 0.18:
        print(f"  → This layer routes CONFIDENTLY overall (infrastructure layer)")
    elif avg_all < 0.08:
        print(f"  → This layer routes UNCERTAINLY overall (attention-dominated layer)")
    else:
        print(f"  → This layer has MIXED routing confidence")


def _test_context_independence(args):
    """Test if the same token routes to the same expert regardless of context.

    This tests the hypothesis: "MoE experts are single-token transformers"

    If TRUE: same token → same expert, always
    If FALSE: context changes routing
    """
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    # Get target token and contexts
    target_token = args.token if hasattr(args, 'token') and args.token else "the"
    contexts_str = args.contexts if hasattr(args, 'contexts') and args.contexts else None

    if contexts_str:
        contexts = [c.strip() for c in contexts_str.split(",")]
    else:
        # Default contexts for common tokens
        default_contexts = {
            "the": ["the", "the cat", "the dog sat", "under the bridge", "127 * the", "def the()"],
            "def": ["def", "def fib(n):", "def add(a,b):", "the def comedy", "x = def"],
            "127": ["127", "127 + 3", "127 * 89", "the 127", "x = 127"],
            "+": ["+", "2 + 2", "127 + 89", "a + b", "x = y +"],
            "hello": ["hello", "hello world", "say hello", "hello, how are you?"],
        }
        contexts = default_contexts.get(target_token, [target_token, f"{target_token} test", f"prefix {target_token}"])

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9  # Use highest-confidence layer

    print(f"\n{'=' * 70}")
    print(f"CONTEXT INDEPENDENCE TEST")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Target token: '{target_token}'")
    print(f"Layer: {target_layer}")
    print(f"Contexts: {len(contexts)}")
    print(f"{'=' * 70}")

    # Find the token ID for the target
    target_ids = tokenizer.encode(target_token)
    if len(target_ids) > 1:
        print(f"\nWarning: '{target_token}' tokenizes to multiple tokens: {target_ids}")
        print(f"Using first token only.")
    target_id = target_ids[0] if target_ids else None

    if target_id is None:
        print("Error: Could not tokenize target token")
        return

    print(f"\nTarget token ID: {target_id}")
    print(f"Target token decoded: '{tokenizer.decode([target_id])}'")

    # Test each context
    print(f"\n{'=' * 70}")
    print(f"TOKEN '{target_token}' ROUTING ACROSS CONTEXTS")
    print(f"{'=' * 70}")

    results = []

    for context in contexts:
        tokens = tokenizer.encode(context)

        # Find position of target token in this context
        target_positions = [i for i, t in enumerate(tokens) if t == target_id]

        if not target_positions:
            print(f"  '{context}': target token not found in tokenization")
            continue

        # Get routing for this context
        weights = router.capture_router_weights(context, layers=[target_layer])

        for pos in target_positions:
            if target_layer in weights and pos < len(weights[target_layer]):
                experts, probs = weights[target_layer][pos]
                top_expert = experts[0]
                top_prob = probs[0]

                results.append({
                    "context": context,
                    "position": pos,
                    "top_expert": top_expert,
                    "top_prob": top_prob,
                    "all_experts": list(zip(experts, probs)),
                })

                expert_str = ", ".join(f"E{e}:{p:.2f}" for e, p in zip(experts[:3], probs[:3]))
                print(f"  '{context:30s}' pos={pos} → {expert_str}")

    # Analyze consistency
    print(f"\n{'=' * 70}")
    print(f"CONSISTENCY ANALYSIS")
    print(f"{'=' * 70}")

    if not results:
        print("No results to analyze")
        return

    top_experts = [r["top_expert"] for r in results]
    unique_experts = set(top_experts)

    # Calculate consistency metrics
    most_common_expert = max(set(top_experts), key=top_experts.count)
    consistency = top_experts.count(most_common_expert) / len(top_experts)

    print(f"\nTop expert selections: {top_experts}")
    print(f"Unique experts used: {sorted(unique_experts)}")
    print(f"Most common: E{most_common_expert} ({top_experts.count(most_common_expert)}/{len(top_experts)} = {consistency*100:.0f}%)")

    # Check if routing weights are similar
    if len(results) > 1:
        probs_for_common = [r["top_prob"] for r in results if r["top_expert"] == most_common_expert]
        if probs_for_common:
            avg_prob = sum(probs_for_common) / len(probs_for_common)
            prob_variance = sum((p - avg_prob) ** 2 for p in probs_for_common) / len(probs_for_common)
            print(f"Weight consistency: avg={avg_prob:.3f}, variance={prob_variance:.4f}")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"VERDICT")
    print(f"{'=' * 70}")

    if consistency >= 0.9:
        print(f"\n✓ CONTEXT-INDEPENDENT: Token '{target_token}' routes to E{most_common_expert}")
        print(f"  in {consistency*100:.0f}% of contexts tested.")
        print(f"\n  The expert is reading the TOKEN, not the context.")
    elif consistency >= 0.7:
        print(f"\n~ MOSTLY INDEPENDENT: Token '{target_token}' usually routes to E{most_common_expert}")
        print(f"  but shows some context sensitivity ({consistency*100:.0f}% consistency).")
    else:
        print(f"\n✗ CONTEXT-DEPENDENT: Token '{target_token}' routing varies significantly")
        print(f"  Only {consistency*100:.0f}% consistency across contexts.")
        print(f"\n  The expert IS reading context, not just the token.")


def _test_vocab_mapping(args):
    """Map which expert 'owns' which tokens in the vocabulary.

    Tests vocabulary partitioning hypothesis.
    """
    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9

    # Get vocabulary
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    sample_size = min(getattr(args, 'sample_size', 1000), vocab_size)

    print(f"\n{'=' * 70}")
    print(f"VOCABULARY → EXPERT MAPPING")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Layer: {target_layer}")
    print(f"Sampling: {sample_size} tokens")
    print(f"{'=' * 70}")

    # Sample tokens and get their routing
    from collections import defaultdict
    expert_tokens = defaultdict(list)

    # Sample from vocabulary
    import random
    sample_ids = random.sample(range(min(vocab_size, 50000)), sample_size)

    for token_id in sample_ids:
        try:
            token_str = tokenizer.decode([token_id])
            if not token_str or token_str.isspace():
                continue

            # Get routing for this single token
            weights = router.capture_router_weights(token_str, layers=[target_layer])

            if target_layer in weights and weights[target_layer]:
                experts, probs = weights[target_layer][0]
                top_expert = experts[0]
                top_prob = probs[0]

                expert_tokens[top_expert].append({
                    "token": token_str,
                    "prob": top_prob,
                    "token_id": token_id,
                })
        except Exception:
            pass

    # Report findings
    print(f"\n{'=' * 70}")
    print(f"EXPERT TOKEN OWNERSHIP")
    print(f"{'=' * 70}")

    total_tokens = sum(len(tokens) for tokens in expert_tokens.values())

    for expert_idx in sorted(expert_tokens.keys()):
        tokens = expert_tokens[expert_idx]
        pct = 100 * len(tokens) / total_tokens if total_tokens > 0 else 0

        # Categorize tokens
        examples = [t["token"][:10] for t in sorted(tokens, key=lambda x: -x["prob"])[:5]]
        example_str = ", ".join(f"'{e}'" for e in examples)

        print(f"  E{expert_idx:2d}: {len(tokens):4d} tokens ({pct:5.1f}%)  Examples: {example_str}")

    # Look for patterns
    print(f"\n{'=' * 70}")
    print(f"TOKEN TYPE PATTERNS")
    print(f"{'=' * 70}")

    for expert_idx in sorted(expert_tokens.keys(), key=lambda x: -len(expert_tokens[x]))[:10]:
        tokens = expert_tokens[expert_idx]

        # Categorize
        numbers = [t for t in tokens if t["token"].strip().replace('.', '').replace(',', '').isdigit()]
        letters = [t for t in tokens if t["token"].strip().isalpha()]
        punct = [t for t in tokens if t["token"].strip() and not t["token"].strip().isalnum()]

        if len(tokens) >= 10:
            print(f"\n  E{expert_idx}: {len(tokens)} tokens")
            if numbers:
                print(f"    Numbers: {len(numbers)} ({100*len(numbers)/len(tokens):.0f}%)")
            if letters:
                print(f"    Letters: {len(letters)} ({100*len(letters)/len(tokens):.0f}%)")
            if punct:
                print(f"    Punctuation: {len(punct)} ({100*len(punct)/len(tokens):.0f}%)")


def _probe_router_inputs(args):
    """Probe what the router is actually looking at.

    Decomposes the router input into:
    1. Token embedding contribution
    2. Position-dependent contribution (everything else)
    3. Attention-computed context

    This identifies the mechanism driving routing decisions.
    """
    model, tokenizer = _load_model(args.model)

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9

    # Test prompts: same token at same position, different contexts
    test_prompts = [
        ("111 127", "number context"),
        ("222 127", "number context"),
        ("abc 127", "word context"),
        ("xyz 127", "word context"),
    ]

    print(f"\n{'=' * 70}")
    print(f"ROUTER INPUT DECOMPOSITION")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Layer: {target_layer}")
    print(f"Target token: '127' at position 2")
    print(f"{'=' * 70}")

    # Get the model internals
    backbone = model.model
    layers = list(backbone.layers)
    embed = getattr(backbone, "embed_tokens", None)
    embed_scale = getattr(model.config, "embedding_scale", None) if hasattr(model, "config") else None

    # Track hidden states at each step
    print(f"\n{'=' * 70}")
    print(f"STEP 1: Token Embedding (before any layers)")
    print(f"{'=' * 70}")

    for prompt, ctx_type in test_prompts:
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        h = embed(input_ids)
        if embed_scale:
            h = h * embed_scale

        # Get embedding for position 2 (the "127" token)
        token_embed = h[0, 2, :]  # [hidden_dim]

        # Compute what the router would see if we only used token embedding
        layer = layers[target_layer]
        router = layer.mlp.router

        # Router input is normally: layernorm(h + attention_output)
        # But here we just use raw embedding (no attention yet)
        router_logits_embed_only = token_embed @ router.weight.T + router.bias
        top_expert = int(mx.argmax(router_logits_embed_only))
        top_prob = float(mx.softmax(router_logits_embed_only)[top_expert])

        print(f"  '{prompt:15s}' ({ctx_type:15s}): embed → E{top_expert} ({top_prob:.2f})")

    print(f"\n{'=' * 70}")
    print(f"STEP 2: After Attention (layers 0 to {target_layer-1})")
    print(f"{'=' * 70}")

    # Now run through layers and capture the full hidden state
    results = []

    for prompt, ctx_type in test_prompts:
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        h = embed(input_ids)
        if embed_scale:
            h = h * embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Track contributions
        original_embed = mx.array(h[0, 2, :])  # Token embedding at position 2

        # Run through layers up to target
        for idx, layer in enumerate(layers):
            if idx > target_layer:
                break

            # Apply attention
            attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h = h + attn_out

            if idx < target_layer:
                # Apply MLP for layers before target
                mlp_out = layer.mlp(layer.post_attention_layernorm(h))
                h = h + mlp_out

        # At target layer, get the router input
        h_norm = layer.post_attention_layernorm(h)
        router_input = h_norm[0, 2, :]  # Position 2

        # Get router decision
        router = layer.mlp.router
        router_logits = router_input @ router.weight.T + router.bias
        top_expert = int(mx.argmax(router_logits))
        top_logit = float(router_logits[top_expert])
        probs = mx.softmax(router_logits)
        top_prob = float(probs[top_expert])

        # Compute delta from original embedding
        delta = router_input - original_embed
        delta_norm = float(mx.sqrt(mx.sum(delta ** 2)))
        embed_norm = float(mx.sqrt(mx.sum(original_embed ** 2)))
        router_input_norm = float(mx.sqrt(mx.sum(router_input ** 2)))

        results.append({
            "prompt": prompt,
            "ctx_type": ctx_type,
            "expert": top_expert,
            "prob": top_prob,
            "logit": top_logit,
            "embed_norm": embed_norm,
            "delta_norm": delta_norm,
            "router_input_norm": router_input_norm,
            "delta_ratio": delta_norm / embed_norm if embed_norm > 0 else 0,
        })

        print(f"  '{prompt:15s}' ({ctx_type:15s}): → E{top_expert} ({top_prob:.2f})")
        print(f"      embed_norm={embed_norm:.1f}, delta_norm={delta_norm:.1f}, ratio={delta_norm/embed_norm:.2f}")

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS: What's driving the routing decision?")
    print(f"{'=' * 70}")

    # Group by context type
    number_results = [r for r in results if r["ctx_type"] == "number context"]
    word_results = [r for r in results if r["ctx_type"] == "word context"]

    number_experts = [r["expert"] for r in number_results]
    word_experts = [r["expert"] for r in word_results]

    number_deltas = [r["delta_norm"] for r in number_results]
    word_deltas = [r["delta_norm"] for r in word_results]

    print(f"\nNumber context: experts = {number_experts}, avg_delta = {sum(number_deltas)/len(number_deltas):.1f}")
    print(f"Word context:   experts = {word_experts}, avg_delta = {sum(word_deltas)/len(word_deltas):.1f}")

    # Check if delta magnitude correlates with routing
    print(f"\n{'=' * 70}")
    print(f"STEP 3: Router Weight Analysis")
    print(f"{'=' * 70}")

    # Get router weights and analyze which directions matter
    layer = layers[target_layer]
    router_weight = layer.mlp.router.weight  # [num_experts, hidden_dim]

    # Compute which expert directions are activated by different contexts
    for prompt, ctx_type in test_prompts[:2]:  # Just show 2 examples
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        h = embed(input_ids)
        if embed_scale:
            h = h * embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for idx, layer in enumerate(layers):
            if idx > target_layer:
                break
            attn_out = layer.self_attn(layer.input_layernorm(h), mask=mask)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h = h + attn_out
            if idx < target_layer:
                mlp_out = layer.mlp(layer.post_attention_layernorm(h))
                h = h + mlp_out

        h_norm = layer.post_attention_layernorm(h)
        router_input = h_norm[0, 2, :]

        # Get top-5 router logits
        layer = layers[target_layer]
        router = layer.mlp.router
        router_logits = router_input @ router.weight.T + router.bias
        top5 = mx.argpartition(router_logits, kth=-5)[-5:]
        top5_logits = router_logits[top5]

        # Sort by logit
        sorted_idx = mx.argsort(top5_logits)[::-1]
        top5_sorted = [int(top5[i]) for i in sorted_idx.tolist()]
        top5_logits_sorted = [float(top5_logits[i]) for i in sorted_idx.tolist()]

        print(f"\n  '{prompt}' ({ctx_type}):")
        print(f"    Top-5 experts: {top5_sorted}")
        print(f"    Logits:        {[f'{l:.2f}' for l in top5_logits_sorted]}")

    print(f"\n{'=' * 70}")
    print(f"CONCLUSION")
    print(f"{'=' * 70}")

    # Determine mechanism
    if len(set(number_experts)) == 1 and len(set(word_experts)) > 1:
        print("\n✓ CONTEXT MATTERS: Same token routes differently based on context.")
        print("  Number contexts produce consistent routing.")
        print("  Word contexts produce variable routing.")
        print("\n  Mechanism: Router reads attention-computed features,")
        print("  not just token embeddings.")
    elif all(e == number_experts[0] for e in number_experts + word_experts):
        print("\n✓ TOKEN DOMINATES: Same expert regardless of context.")
        print("  The router primarily reads token identity.")
    else:
        print("\n~ MIXED: Both token and context affect routing.")
        print(f"  Number experts: {number_experts}")
        print(f"  Word experts: {word_experts}")


def _discover_expert_patterns(args):
    """Discover what context patterns activate each expert.

    For each expert, collect examples that route to it, then analyze:
    - Position distribution
    - Preceding token types
    - Current token types
    - Sequence patterns
    """
    from collections import defaultdict

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9

    print(f"\n{'=' * 70}")
    print(f"EXPERT PATTERN DISCOVERY")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Layer: {target_layer}")
    print(f"{'=' * 70}")

    # Generate diverse test contexts
    test_contexts = []

    # Position 0 tests (single tokens)
    single_tokens = [
        "the", "a", "an", "hello", "world", "def", "class", "return",
        "127", "42", "3", "1000", "0", "256",
        "+", "-", "*", "/", "=", "(", ")", "{", "}",
        "The", "Hello", "What", "How", "Why",
    ]
    for tok in single_tokens:
        test_contexts.append((tok, "single", tok, 0))

    # Number after number
    num_num = [
        ("111 127", "num_num", "127", 2),
        ("222 42", "num_num", "42", 2),
        ("999 3", "num_num", "3", 2),
        ("42 127", "num_num", "127", 2),
        ("1 2 3", "num_num", "3", 4),
        ("100 200 300", "num_num", "300", 4),
    ]
    test_contexts.extend(num_num)

    # Number after word
    word_num = [
        ("abc 127", "word_num", "127", 2),
        ("xyz 42", "word_num", "42", 2),
        ("the 3", "word_num", "3", 2),
        ("def 256", "word_num", "256", 2),
        ("hello 100", "word_num", "100", 2),
        ("x = 42", "word_num", "42", 4),
    ]
    test_contexts.extend(word_num)

    # Word after word
    word_word = [
        ("the cat", "word_word", "cat", 1),
        ("hello world", "word_word", "world", 1),
        ("def fibonacci", "word_word", "fibonacci", 1),
        ("a b c", "word_word", "c", 3),
        ("the quick brown", "word_word", "brown", 2),
    ]
    test_contexts.extend(word_word)

    # Word after number
    num_word = [
        ("127 abc", "num_word", "abc", 2),
        ("42 hello", "num_word", "hello", 2),
        ("3 items", "num_word", "items", 2),
    ]
    test_contexts.extend(num_word)

    # Punctuation contexts
    punct_contexts = [
        ("hello,", "punct", ",", 1),
        ("world.", "punct", ".", 1),
        ("what?", "punct", "?", 1),
        ("x = y", "punct", "=", 2),
        ("(x)", "punct", ")", 2),
    ]
    test_contexts.extend(punct_contexts)

    # Code patterns
    code_contexts = [
        ("def foo():", "code", ":", 4),
        ("return x", "code", "x", 1),
        ("if x:", "code", ":", 2),
        ("for i in", "code", "in", 2),
        ("import os", "code", "os", 1),
    ]
    test_contexts.extend(code_contexts)

    # Collect routing data per expert
    expert_activations = defaultdict(list)

    print(f"\nAnalyzing {len(test_contexts)} test contexts...")

    for prompt, pattern_type, target_token, expected_pos in test_contexts:
        try:
            weights = router.capture_router_weights(prompt, layers=[target_layer])

            if target_layer not in weights or not weights[target_layer]:
                continue

            # Get tokens
            tokens = tokenizer.encode(prompt)

            for pos, (experts, probs) in enumerate(weights[target_layer]):
                top_expert = experts[0]
                top_prob = probs[0]

                # Determine token type
                if pos < len(tokens):
                    token_id = tokens[pos]
                    token_str = tokenizer.decode([token_id])
                else:
                    token_str = "?"

                # Classify token
                if token_str.strip().replace('.', '').replace(',', '').isdigit():
                    token_type = "NUMBER"
                elif token_str.strip().isalpha():
                    token_type = "WORD"
                elif token_str.strip() and not token_str.strip().isalnum():
                    token_type = "PUNCT"
                else:
                    token_type = "OTHER"

                # Classify preceding token
                if pos == 0:
                    prev_type = "START"
                elif pos - 1 < len(tokens):
                    prev_id = tokens[pos - 1]
                    prev_str = tokenizer.decode([prev_id])
                    if prev_str.strip().replace('.', '').replace(',', '').isdigit():
                        prev_type = "NUMBER"
                    elif prev_str.strip().isalpha():
                        prev_type = "WORD"
                    elif prev_str.strip() and not prev_str.strip().isalnum():
                        prev_type = "PUNCT"
                    else:
                        prev_type = "OTHER"
                else:
                    prev_type = "?"

                expert_activations[top_expert].append({
                    "prompt": prompt,
                    "position": pos,
                    "token": token_str,
                    "token_type": token_type,
                    "prev_type": prev_type,
                    "prob": top_prob,
                    "pattern": pattern_type,
                })
        except Exception as e:
            pass

    # Analyze patterns for each expert
    print(f"\n{'=' * 70}")
    print(f"EXPERT PATTERN ANALYSIS")
    print(f"{'=' * 70}")

    expert_patterns = {}

    for expert_id in sorted(expert_activations.keys()):
        activations = expert_activations[expert_id]
        if len(activations) < 3:
            continue

        # Position distribution
        positions = [a["position"] for a in activations]
        pos_0_pct = 100 * positions.count(0) / len(positions)

        # Token type distribution
        token_types = [a["token_type"] for a in activations]
        number_pct = 100 * token_types.count("NUMBER") / len(token_types)
        word_pct = 100 * token_types.count("WORD") / len(token_types)
        punct_pct = 100 * token_types.count("PUNCT") / len(token_types)

        # Previous type distribution
        prev_types = [a["prev_type"] for a in activations]
        prev_start_pct = 100 * prev_types.count("START") / len(prev_types)
        prev_num_pct = 100 * prev_types.count("NUMBER") / len(prev_types)
        prev_word_pct = 100 * prev_types.count("WORD") / len(prev_types)

        # Average probability
        avg_prob = sum(a["prob"] for a in activations) / len(activations)

        # Determine pattern
        pattern = "UNKNOWN"
        if pos_0_pct > 80:
            pattern = "SEQUENCE_START"
        elif prev_num_pct > 60 and number_pct > 50:
            pattern = "NUMBER_CONTINUATION"
        elif prev_word_pct > 50 and number_pct > 50:
            pattern = "NUMBER_AFTER_WORD"
        elif prev_word_pct > 50 and word_pct > 50:
            pattern = "WORD_CONTINUATION"
        elif punct_pct > 50:
            pattern = "PUNCTUATION"
        elif number_pct > 60:
            pattern = "NUMBER_GENERAL"
        elif word_pct > 60:
            pattern = "WORD_GENERAL"

        expert_patterns[expert_id] = {
            "pattern": pattern,
            "count": len(activations),
            "avg_prob": avg_prob,
            "pos_0_pct": pos_0_pct,
            "number_pct": number_pct,
            "word_pct": word_pct,
            "punct_pct": punct_pct,
            "prev_num_pct": prev_num_pct,
            "prev_word_pct": prev_word_pct,
            "examples": [a["prompt"] for a in activations[:5]],
        }

        print(f"\nE{expert_id}: {pattern}")
        print(f"  Count: {len(activations)}, Avg prob: {avg_prob:.2f}")
        print(f"  Position 0: {pos_0_pct:.0f}%")
        print(f"  Token type: NUMBER={number_pct:.0f}%, WORD={word_pct:.0f}%, PUNCT={punct_pct:.0f}%")
        print(f"  Prev type:  NUMBER={prev_num_pct:.0f}%, WORD={prev_word_pct:.0f}%, START={prev_start_pct:.0f}%")
        print(f"  Examples: {[a['prompt'][:20] for a in activations[:3]]}")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"PATTERN SUMMARY")
    print(f"{'=' * 70}")

    pattern_groups = defaultdict(list)
    for expert_id, info in expert_patterns.items():
        pattern_groups[info["pattern"]].append(expert_id)

    for pattern, experts in sorted(pattern_groups.items()):
        print(f"\n{pattern}:")
        for e in experts:
            info = expert_patterns[e]
            print(f"  E{e}: {info['count']} activations, {info['avg_prob']:.2f} avg prob")

    # Save results
    output_path = getattr(args, "output", None)
    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump({
                "layer": target_layer,
                "expert_patterns": expert_patterns,
                "pattern_groups": {k: list(v) for k, v in pattern_groups.items()},
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


def _full_expert_taxonomy(args):
    """Complete taxonomy of all 32 experts across all context variables.

    Tests systematic combinations of:
    - Position (0, 1, 2, 3+)
    - Current token type (number, word, punct, operator, code_kw)
    - Preceding token type (start, number, word, punct, operator)
    - Sequence pattern (num_seq, word_seq, code, math, mixed)
    """
    from collections import defaultdict
    import json

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    target_layer = getattr(args, "layer", None)
    if target_layer is None:
        target_layer = 9

    num_experts = router.info["num_experts"]

    print(f"\n{'=' * 70}")
    print(f"COMPLETE EXPERT TAXONOMY")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Layer: {target_layer}")
    print(f"Experts: {num_experts}")
    print(f"{'=' * 70}")

    # Comprehensive test prompts organized by pattern
    test_prompts = {
        # Pure number sequences
        "num_seq": [
            "1", "42", "127", "999", "3.14",
            "1 2", "42 127", "100 200", "1 2 3", "10 20 30 40",
            "1 + 2", "42 * 3", "100 - 50", "10 / 2",
        ],
        # Pure word sequences
        "word_seq": [
            "the", "hello", "world", "cat", "dog",
            "the cat", "hello world", "big dog", "red car",
            "the quick brown", "hello my friend", "a b c d",
        ],
        # Code patterns
        "code": [
            "def", "class", "return", "import", "if",
            "def foo", "class Bar", "return x", "import os",
            "def foo():", "if x:", "for i in", "while True:",
            "def foo(x):", "class Foo(Bar):", "return x + y",
        ],
        # Math expressions
        "math": [
            "x", "y", "x + y", "a * b", "n - 1",
            "x = 1", "y = 2", "x = y + 1",
            "sum = a + b", "result = x * y",
            "f(x)", "g(x, y)", "max(a, b)",
        ],
        # Punctuation heavy
        "punct": [
            "(", ")", "{", "}", "[", "]",
            "(x)", "{x}", "[x]",
            "x, y", "a; b", "x: y",
            "(a, b)", "{x: y}", "[1, 2, 3]",
        ],
        # Mixed patterns
        "mixed": [
            "the 42", "hello 127", "x = 42", "item 1",
            "42 items", "127 things", "3 cats",
            "chapter 1", "page 42", "version 2.0",
            "test 1 2 3", "a 1 b 2", "x 1 y 2 z 3",
        ],
        # Operators
        "operators": [
            "+", "-", "*", "/", "=", "==", "!=",
            "1 + 2", "x - y", "a * b", "n / 2",
            "x == y", "a != b", "x += 1", "y -= 2",
        ],
        # Whitespace patterns
        "whitespace": [
            "a  b", "x   y", "1    2",
            "the  cat", "hello   world",
        ],
    }

    # Collect all activations per expert
    expert_activations = defaultdict(list)
    total_tokens = 0

    print(f"\nAnalyzing test prompts...")

    for pattern_type, prompts in test_prompts.items():
        for prompt in prompts:
            try:
                weights = router.capture_router_weights(prompt, layers=[target_layer])
                if target_layer not in weights:
                    continue

                tokens = tokenizer.encode(prompt)

                for pos, (experts, probs) in enumerate(weights[target_layer]):
                    if pos >= len(tokens):
                        continue

                    total_tokens += 1
                    top_expert = experts[0]
                    top_prob = probs[0]

                    # Get token info
                    token_id = tokens[pos]
                    token_str = tokenizer.decode([token_id]).strip()

                    # Classify current token
                    if token_str.replace('.', '').replace(',', '').isdigit():
                        curr_type = "NUMBER"
                    elif token_str.isalpha():
                        if token_str in ('def', 'class', 'return', 'import', 'if', 'for', 'while', 'else', 'elif', 'try', 'except', 'with', 'as', 'in', 'is', 'not', 'and', 'or', 'True', 'False', 'None'):
                            curr_type = "CODE_KW"
                        else:
                            curr_type = "WORD"
                    elif token_str in ('+', '-', '*', '/', '=', '==', '!=', '+=', '-=', '*=', '/=', '<', '>', '<=', '>='):
                        curr_type = "OPERATOR"
                    elif token_str and not token_str.isalnum():
                        curr_type = "PUNCT"
                    else:
                        curr_type = "OTHER"

                    # Classify preceding token
                    if pos == 0:
                        prev_type = "START"
                    else:
                        prev_id = tokens[pos - 1]
                        prev_str = tokenizer.decode([prev_id]).strip()
                        if prev_str.replace('.', '').replace(',', '').isdigit():
                            prev_type = "NUMBER"
                        elif prev_str.isalpha():
                            if prev_str in ('def', 'class', 'return', 'import', 'if', 'for', 'while', 'else', 'elif'):
                                prev_type = "CODE_KW"
                            else:
                                prev_type = "WORD"
                        elif prev_str in ('+', '-', '*', '/', '=', '==', '!='):
                            prev_type = "OPERATOR"
                        elif prev_str and not prev_str.isalnum():
                            prev_type = "PUNCT"
                        else:
                            prev_type = "OTHER"

                    expert_activations[top_expert].append({
                        "prompt": prompt,
                        "position": pos,
                        "token": token_str,
                        "curr_type": curr_type,
                        "prev_type": prev_type,
                        "pattern": pattern_type,
                        "prob": top_prob,
                    })
            except Exception:
                pass

    print(f"Analyzed {total_tokens} tokens across {sum(len(p) for p in test_prompts.values())} prompts")

    # Analyze each expert
    print(f"\n{'=' * 70}")
    print(f"EXPERT PROFILES")
    print(f"{'=' * 70}")

    expert_profiles = {}

    for expert_id in range(num_experts):
        activations = expert_activations.get(expert_id, [])

        if not activations:
            expert_profiles[expert_id] = {
                "pattern": "UNUSED",
                "count": 0,
                "avg_prob": 0,
                "description": "No activations observed",
            }
            continue

        # Position distribution
        positions = [a["position"] for a in activations]
        pos_0_pct = 100 * positions.count(0) / len(positions) if positions else 0
        pos_1_pct = 100 * positions.count(1) / len(positions) if positions else 0
        pos_2plus_pct = 100 * sum(1 for p in positions if p >= 2) / len(positions) if positions else 0

        # Current token type distribution
        curr_types = [a["curr_type"] for a in activations]
        curr_dist = {t: 100 * curr_types.count(t) / len(curr_types) for t in set(curr_types)}

        # Previous token type distribution
        prev_types = [a["prev_type"] for a in activations]
        prev_dist = {t: 100 * prev_types.count(t) / len(prev_types) for t in set(prev_types)}

        # Pattern distribution
        patterns = [a["pattern"] for a in activations]
        pattern_dist = {p: 100 * patterns.count(p) / len(patterns) for p in set(patterns)}

        # Average probability
        avg_prob = sum(a["prob"] for a in activations) / len(activations)

        # Determine primary pattern
        if pos_0_pct > 80:
            primary = "SEQUENCE_START"
            desc = "First token handler"
        elif prev_dist.get("NUMBER", 0) > 60:
            primary = "AFTER_NUMBER"
            desc = "Tokens following numbers"
        elif prev_dist.get("WORD", 0) > 60:
            primary = "AFTER_WORD"
            desc = "Tokens following words"
        elif prev_dist.get("CODE_KW", 0) > 40:
            primary = "AFTER_CODE_KW"
            desc = "Tokens following code keywords"
        elif prev_dist.get("OPERATOR", 0) > 40:
            primary = "AFTER_OPERATOR"
            desc = "Tokens following operators"
        elif prev_dist.get("PUNCT", 0) > 40:
            primary = "AFTER_PUNCT"
            desc = "Tokens following punctuation"
        elif curr_dist.get("NUMBER", 0) > 60:
            primary = "NUMBER_TOKEN"
            desc = "Number tokens"
        elif curr_dist.get("WORD", 0) > 60:
            primary = "WORD_TOKEN"
            desc = "Word tokens"
        elif curr_dist.get("CODE_KW", 0) > 40:
            primary = "CODE_KEYWORD"
            desc = "Code keyword tokens"
        elif curr_dist.get("OPERATOR", 0) > 40:
            primary = "OPERATOR_TOKEN"
            desc = "Operator tokens"
        elif curr_dist.get("PUNCT", 0) > 40:
            primary = "PUNCT_TOKEN"
            desc = "Punctuation tokens"
        elif pattern_dist.get("code", 0) > 50:
            primary = "CODE_CONTEXT"
            desc = "Code patterns"
        elif pattern_dist.get("math", 0) > 50:
            primary = "MATH_CONTEXT"
            desc = "Math expressions"
        else:
            primary = "MIXED"
            desc = "Mixed patterns"

        expert_profiles[expert_id] = {
            "pattern": primary,
            "count": len(activations),
            "avg_prob": avg_prob,
            "description": desc,
            "pos_0_pct": pos_0_pct,
            "pos_1_pct": pos_1_pct,
            "pos_2plus_pct": pos_2plus_pct,
            "curr_dist": curr_dist,
            "prev_dist": prev_dist,
            "pattern_dist": pattern_dist,
            "examples": [a["prompt"][:25] for a in activations[:5]],
        }

        # Print profile
        print(f"\nE{expert_id}: {primary}")
        print(f"  Count: {len(activations)}, Avg prob: {avg_prob:.2f}")
        print(f"  Position: 0={pos_0_pct:.0f}%, 1={pos_1_pct:.0f}%, 2+={pos_2plus_pct:.0f}%")

        # Top current types
        top_curr = sorted(curr_dist.items(), key=lambda x: -x[1])[:3]
        print(f"  Current: {', '.join(f'{t}={v:.0f}%' for t, v in top_curr)}")

        # Top prev types
        top_prev = sorted(prev_dist.items(), key=lambda x: -x[1])[:3]
        print(f"  Previous: {', '.join(f'{t}={v:.0f}%' for t, v in top_prev)}")

        # Examples
        examples = [a["prompt"][:20] for a in activations[:3]]
        print(f"  Examples: {examples}")

    # Summary by pattern
    print(f"\n{'=' * 70}")
    print(f"PATTERN SUMMARY")
    print(f"{'=' * 70}")

    pattern_groups = defaultdict(list)
    for expert_id, profile in expert_profiles.items():
        pattern_groups[profile["pattern"]].append(expert_id)

    for pattern in sorted(pattern_groups.keys()):
        experts = pattern_groups[pattern]
        total_count = sum(expert_profiles[e]["count"] for e in experts)
        avg_prob = sum(expert_profiles[e]["avg_prob"] * expert_profiles[e]["count"] for e in experts) / max(total_count, 1)
        print(f"\n{pattern}: {len(experts)} experts, {total_count} activations, {avg_prob:.2f} avg prob")
        for e in experts:
            p = expert_profiles[e]
            print(f"  E{e}: {p['count']} acts, {p['avg_prob']:.2f} prob - {p['description']}")

    # Activation distribution
    print(f"\n{'=' * 70}")
    print(f"ACTIVATION DISTRIBUTION")
    print(f"{'=' * 70}")

    sorted_by_count = sorted(expert_profiles.items(), key=lambda x: -x[1]["count"])
    print(f"\nMost active experts:")
    for expert_id, profile in sorted_by_count[:10]:
        pct = 100 * profile["count"] / total_tokens if total_tokens > 0 else 0
        print(f"  E{expert_id}: {profile['count']} ({pct:.1f}%) - {profile['pattern']}")

    unused = [e for e, p in expert_profiles.items() if p["count"] == 0]
    if unused:
        print(f"\nUnused experts: {unused}")

    rare = [e for e, p in expert_profiles.items() if 0 < p["count"] < 5]
    if rare:
        print(f"Rarely used (<5 activations): {rare}")

    # Save results
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w") as f:
            # Convert to JSON-serializable format
            serializable = {}
            for k, v in expert_profiles.items():
                serializable[k] = {
                    "pattern": v["pattern"],
                    "count": v["count"],
                    "avg_prob": v["avg_prob"],
                    "description": v["description"],
                    "pos_0_pct": v.get("pos_0_pct", 0),
                    "examples": v.get("examples", []),
                }
            json.dump({
                "layer": target_layer,
                "total_tokens": total_tokens,
                "expert_profiles": serializable,
                "pattern_groups": {k: list(v) for k, v in pattern_groups.items()},
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")


def _sweep_all_layers(args):
    """Sweep all MoE layers to analyze expert taxonomy across the entire model.

    One command, all 24 layers. Shows:
    1. Layer-by-layer routing confidence
    2. Which experts are "workhorses" vs "spectators" per layer
    3. How expert patterns evolve across layers
    4. Expert lifecycle tracking (same expert ID, different roles)
    """
    from collections import defaultdict
    import json

    model, tokenizer = _load_model(args.model)
    router = ExpertRouter(model, tokenizer)

    all_moe_layers = router.info["moe_layers"]
    num_experts = router.info["num_experts"]
    num_experts_per_tok = router.info["num_experts_per_tok"]

    print(f"\n{'=' * 70}")
    print(f"LAYER SWEEP: EXPERT TAXONOMY ACROSS ALL {len(all_moe_layers)} LAYERS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"MoE Layers: {all_moe_layers[0]}-{all_moe_layers[-1]}")
    print(f"Experts per layer: {num_experts}")
    print(f"Top-k routing: {num_experts_per_tok}")
    print(f"Total expert MLPs: {num_experts} × {len(all_moe_layers)} = {num_experts * len(all_moe_layers)}")
    print(f"{'=' * 70}")

    # Test prompts covering diverse patterns
    test_prompts = {
        "num_seq": ["1", "42", "127", "1 2", "42 127", "10 20 30"],
        "word_seq": ["the", "hello", "the cat", "hello world", "the quick brown"],
        "code": ["def", "class", "def foo", "class Bar", "def foo():"],
        "math": ["x", "x + y", "x = 1", "a * b", "f(x)"],
        "punct": ["(", ")", "(x)", "{x: y}", "[1, 2]"],
        "mixed": ["the 42", "x = 42", "127 things", "chapter 1"],
        "operators": ["+", "-", "1 + 2", "x - y", "=="],
    }

    # Collect data per layer
    layer_results = {}
    expert_lifecycle = defaultdict(list)  # expert_id -> [(layer, pattern, count), ...]

    print(f"\nSweeping {len(all_moe_layers)} layers...")

    for layer_idx in all_moe_layers:
        expert_activations = defaultdict(list)
        total_tokens = 0

        for pattern_type, prompts in test_prompts.items():
            for prompt in prompts:
                try:
                    weights = router.capture_router_weights(prompt, layers=[layer_idx])
                    if layer_idx not in weights:
                        continue

                    tokens = tokenizer.encode(prompt)

                    for pos, (experts, probs) in enumerate(weights[layer_idx]):
                        if pos >= len(tokens):
                            continue

                        total_tokens += 1
                        top_expert = experts[0]
                        top_prob = probs[0]

                        token_id = tokens[pos]
                        token_str = tokenizer.decode([token_id]).strip()

                        # Classify current token
                        if token_str.replace('.', '').replace(',', '').isdigit():
                            curr_type = "NUMBER"
                        elif token_str.isalpha():
                            if token_str in ('def', 'class', 'return', 'import', 'if', 'for', 'while', 'else', 'elif', 'try', 'except', 'with', 'as', 'in', 'is', 'not', 'and', 'or', 'True', 'False', 'None'):
                                curr_type = "CODE_KW"
                            else:
                                curr_type = "WORD"
                        elif token_str in ('+', '-', '*', '/', '=', '==', '!=', '+=', '-=', '*=', '/=', '<', '>', '<=', '>='):
                            curr_type = "OPERATOR"
                        elif token_str and not token_str.isalnum():
                            curr_type = "PUNCT"
                        else:
                            curr_type = "OTHER"

                        # Classify preceding token
                        if pos == 0:
                            prev_type = "START"
                        else:
                            prev_id = tokens[pos - 1]
                            prev_str = tokenizer.decode([prev_id]).strip()
                            if prev_str.replace('.', '').replace(',', '').isdigit():
                                prev_type = "NUMBER"
                            elif prev_str.isalpha():
                                if prev_str in ('def', 'class', 'return', 'import', 'if', 'for', 'while', 'else', 'elif'):
                                    prev_type = "CODE_KW"
                                else:
                                    prev_type = "WORD"
                            elif prev_str in ('+', '-', '*', '/', '=', '==', '!='):
                                prev_type = "OPERATOR"
                            elif prev_str and not prev_str.isalnum():
                                prev_type = "PUNCT"
                            else:
                                prev_type = "OTHER"

                        expert_activations[top_expert].append({
                            "position": pos,
                            "curr_type": curr_type,
                            "prev_type": prev_type,
                            "prob": top_prob,
                        })
                except Exception:
                    pass

        # Analyze this layer
        expert_profiles = {}
        workhorses = []  # >5% of activations
        spectators = []  # <1% of activations

        for expert_id in range(num_experts):
            activations = expert_activations.get(expert_id, [])

            if not activations:
                expert_profiles[expert_id] = {"pattern": "UNUSED", "count": 0, "avg_prob": 0}
                spectators.append(expert_id)
                continue

            positions = [a["position"] for a in activations]
            pos_0_pct = 100 * positions.count(0) / len(positions) if positions else 0

            prev_types = [a["prev_type"] for a in activations]
            prev_dist = {t: 100 * prev_types.count(t) / len(prev_types) for t in set(prev_types)}

            curr_types = [a["curr_type"] for a in activations]
            curr_dist = {t: 100 * curr_types.count(t) / len(curr_types) for t in set(curr_types)}

            avg_prob = sum(a["prob"] for a in activations) / len(activations)

            # Determine pattern
            if pos_0_pct > 80:
                pattern = "SEQUENCE_START"
            elif prev_dist.get("NUMBER", 0) > 60:
                pattern = "AFTER_NUMBER"
            elif prev_dist.get("WORD", 0) > 60:
                pattern = "AFTER_WORD"
            elif prev_dist.get("CODE_KW", 0) > 40:
                pattern = "AFTER_CODE_KW"
            elif prev_dist.get("OPERATOR", 0) > 40:
                pattern = "AFTER_OPERATOR"
            elif prev_dist.get("PUNCT", 0) > 40:
                pattern = "AFTER_PUNCT"
            elif curr_dist.get("NUMBER", 0) > 60:
                pattern = "NUMBER_TOKEN"
            elif curr_dist.get("WORD", 0) > 60:
                pattern = "WORD_TOKEN"
            elif curr_dist.get("CODE_KW", 0) > 40:
                pattern = "CODE_KEYWORD"
            elif curr_dist.get("OPERATOR", 0) > 40:
                pattern = "OPERATOR_TOKEN"
            elif curr_dist.get("PUNCT", 0) > 40:
                pattern = "PUNCT_TOKEN"
            else:
                pattern = "MIXED"

            expert_profiles[expert_id] = {
                "pattern": pattern,
                "count": len(activations),
                "avg_prob": avg_prob,
                "pos_0_pct": pos_0_pct,
            }

            pct_of_total = 100 * len(activations) / total_tokens if total_tokens > 0 else 0
            if pct_of_total >= 5:
                workhorses.append(expert_id)
            elif pct_of_total < 1:
                spectators.append(expert_id)

            # Track lifecycle
            expert_lifecycle[expert_id].append((layer_idx, pattern, len(activations)))

        # Group by pattern
        pattern_groups = defaultdict(list)
        for expert_id, profile in expert_profiles.items():
            pattern_groups[profile["pattern"]].append(expert_id)

        # Calculate layer-wide confidence
        all_probs = []
        for expert_id, activations in expert_activations.items():
            all_probs.extend([a["prob"] for a in activations])
        avg_confidence = sum(all_probs) / len(all_probs) if all_probs else 0

        layer_results[layer_idx] = {
            "total_tokens": total_tokens,
            "expert_profiles": expert_profiles,
            "pattern_groups": {k: list(v) for k, v in pattern_groups.items()},
            "workhorses": workhorses,
            "spectators": spectators,
            "avg_confidence": avg_confidence,
        }

        # Progress indicator
        print(f"  Layer {layer_idx:2d}: {len(workhorses):2d} workhorses, {len(spectators):2d} spectators, {avg_confidence:.2f} avg confidence")

    # Summary Report
    print(f"\n{'=' * 70}")
    print(f"LAYER-BY-LAYER SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Layer':<6} | {'Workhorses':<12} | {'Spectators':<12} | {'Confidence':<12} | {'Top Patterns'}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 30}")

    for layer_idx in sorted(layer_results.keys()):
        result = layer_results[layer_idx]
        patterns = result["pattern_groups"]
        # Get top patterns by expert count
        top_patterns = sorted(
            [(p, len(e)) for p, e in patterns.items() if p != "UNUSED"],
            key=lambda x: -x[1]
        )[:3]
        pattern_str = ", ".join(f"{p}({c})" for p, c in top_patterns)

        conf_bar = "#" * int(result["avg_confidence"] * 20)
        print(f"{layer_idx:5d} | {len(result['workhorses']):10d} | {len(result['spectators']):10d} | {result['avg_confidence']:.2f} [{conf_bar:<8}] | {pattern_str}")

    # Pattern Evolution
    print(f"\n{'=' * 70}")
    print(f"PATTERN EVOLUTION ACROSS LAYERS")
    print(f"{'=' * 70}")

    all_patterns = set()
    for result in layer_results.values():
        all_patterns.update(result["pattern_groups"].keys())
    all_patterns.discard("UNUSED")

    print(f"\n{'Pattern':<20} | {'Layers Active':<30} | {'Peak Layer':<12} | {'Peak Experts'}")
    print(f"{'-' * 20}-+-{'-' * 30}-+-{'-' * 12}-+-{'-' * 12}")

    for pattern in sorted(all_patterns):
        layers_with_pattern = []
        peak_layer = None
        peak_count = 0
        for layer_idx, result in layer_results.items():
            count = len(result["pattern_groups"].get(pattern, []))
            if count > 0:
                layers_with_pattern.append(layer_idx)
                if count > peak_count:
                    peak_count = count
                    peak_layer = layer_idx

        if layers_with_pattern:
            # Summarize layer ranges
            if len(layers_with_pattern) > 5:
                layers_str = f"{min(layers_with_pattern)}-{max(layers_with_pattern)} ({len(layers_with_pattern)} layers)"
            else:
                layers_str = ", ".join(str(l) for l in layers_with_pattern)
            print(f"{pattern:<20} | {layers_str:<30} | {peak_layer:<12} | {peak_count}")

    # Expert Lifecycle Analysis
    print(f"\n{'=' * 70}")
    print(f"EXPERT LIFECYCLE (ROLE CHANGES ACROSS LAYERS)")
    print(f"{'=' * 70}")

    # Find experts with interesting lifecycles (different roles at different layers)
    interesting_experts = []
    for expert_id, lifecycle in expert_lifecycle.items():
        patterns_seen = set(pattern for _, pattern, _ in lifecycle if pattern != "UNUSED")
        if len(patterns_seen) > 1:
            interesting_experts.append((expert_id, lifecycle))

    if interesting_experts:
        print(f"\nExperts with CHANGING roles across layers:")
        for expert_id, lifecycle in sorted(interesting_experts, key=lambda x: -len(set(p for _, p, _ in x[1] if p != "UNUSED")))[:10]:
            patterns_seen = set(pattern for _, pattern, _ in lifecycle if pattern != "UNUSED")
            print(f"\n  E{expert_id}: {len(patterns_seen)} different roles")
            # Show layer -> pattern mapping
            layer_patterns = []
            for layer, pattern, count in lifecycle:
                if pattern != "UNUSED" and count > 0:
                    layer_patterns.append(f"L{layer}:{pattern[:10]}")
            # Group consecutive same patterns
            if layer_patterns:
                print(f"    {' -> '.join(layer_patterns[:8])}")
                if len(layer_patterns) > 8:
                    print(f"    ... and {len(layer_patterns) - 8} more")
    else:
        print(f"\nNo experts found with changing roles across layers.")

    # Find consistent specialists
    print(f"\nExperts with CONSISTENT roles (same pattern across 5+ layers):")
    consistent_experts = []
    for expert_id, lifecycle in expert_lifecycle.items():
        # Count pattern occurrences
        pattern_counts = defaultdict(int)
        for _, pattern, count in lifecycle:
            if pattern != "UNUSED" and count > 0:
                pattern_counts[pattern] += 1
        # Check for dominant pattern
        for pattern, layer_count in pattern_counts.items():
            if layer_count >= 5:
                consistent_experts.append((expert_id, pattern, layer_count))

    if consistent_experts:
        for expert_id, pattern, layer_count in sorted(consistent_experts, key=lambda x: -x[2])[:10]:
            print(f"  E{expert_id}: {pattern} at {layer_count} layers")
    else:
        print(f"  No consistent specialists found.")

    # SEQUENCE_START expert tracking (the key finding)
    print(f"\n{'=' * 70}")
    print(f"SEQUENCE_START HANDLER BY LAYER")
    print(f"{'=' * 70}")
    print(f"\n(Shows which expert(s) handle position-0 tokens at each layer)")

    for layer_idx in sorted(layer_results.keys()):
        result = layer_results[layer_idx]
        start_experts = result["pattern_groups"].get("SEQUENCE_START", [])
        if start_experts:
            experts_str = ", ".join(f"E{e}" for e in start_experts)
            print(f"  Layer {layer_idx:2d}: {experts_str}")
        else:
            # Check if any expert has high pos_0 even if not classified as SEQUENCE_START
            high_pos0 = []
            for expert_id, profile in result["expert_profiles"].items():
                if profile.get("pos_0_pct", 0) > 50 and profile["count"] > 0:
                    high_pos0.append((expert_id, profile["pos_0_pct"]))
            if high_pos0:
                experts_str = ", ".join(f"E{e}({p:.0f}%)" for e, p in high_pos0)
                print(f"  Layer {layer_idx:2d}: {experts_str} (partial)")
            else:
                print(f"  Layer {layer_idx:2d}: (no dominant handler)")

    # Concentration analysis
    print(f"\n{'=' * 70}")
    print(f"EXPERT CONCENTRATION")
    print(f"{'=' * 70}")

    for layer_idx in sorted(layer_results.keys()):
        result = layer_results[layer_idx]
        total = result["total_tokens"]
        if total == 0:
            continue

        # Get top 3 experts by activation count
        sorted_experts = sorted(
            [(e, p["count"]) for e, p in result["expert_profiles"].items() if p["count"] > 0],
            key=lambda x: -x[1]
        )[:3]

        top_3_count = sum(c for _, c in sorted_experts)
        top_3_pct = 100 * top_3_count / total if total > 0 else 0

        experts_str = ", ".join(f"E{e}({100*c/total:.0f}%)" for e, c in sorted_experts)
        print(f"  Layer {layer_idx:2d}: Top 3 = {top_3_pct:.0f}% | {experts_str}")

    # Save results
    output_path = getattr(args, "output", None)
    if output_path:
        with open(output_path, "w") as f:
            # Convert to JSON-serializable format
            serializable_results = {}
            for layer_idx, result in layer_results.items():
                serializable_results[str(layer_idx)] = {
                    "total_tokens": result["total_tokens"],
                    "workhorses": result["workhorses"],
                    "spectators": result["spectators"],
                    "avg_confidence": result["avg_confidence"],
                    "pattern_groups": result["pattern_groups"],
                    "expert_profiles": {
                        str(e): {
                            "pattern": p["pattern"],
                            "count": p["count"],
                            "avg_prob": p["avg_prob"],
                        }
                        for e, p in result["expert_profiles"].items()
                    },
                }

            serializable_lifecycle = {}
            for expert_id, lifecycle in expert_lifecycle.items():
                serializable_lifecycle[str(expert_id)] = [
                    {"layer": l, "pattern": p, "count": c}
                    for l, p, c in lifecycle
                ]

            json.dump({
                "model": args.model,
                "num_layers": len(all_moe_layers),
                "num_experts": num_experts,
                "layer_results": serializable_results,
                "expert_lifecycle": serializable_lifecycle,
            }, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Results saved to {output_path}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total expert MLPs analyzed: {num_experts * len(all_moe_layers)}")

    # Count unique active experts
    active_at_any_layer = set()
    for expert_id, lifecycle in expert_lifecycle.items():
        if any(count > 0 for _, _, count in lifecycle):
            active_at_any_layer.add(expert_id)
    print(f"Expert IDs active at any layer: {len(active_at_any_layer)}/{num_experts}")

    # Average workhorses/spectators
    avg_workhorses = sum(len(r["workhorses"]) for r in layer_results.values()) / len(layer_results)
    avg_spectators = sum(len(r["spectators"]) for r in layer_results.values()) / len(layer_results)
    print(f"Average workhorses per layer: {avg_workhorses:.1f}")
    print(f"Average spectators per layer: {avg_spectators:.1f}")

    # Confidence range
    confidences = [r["avg_confidence"] for r in layer_results.values()]
    print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")

    # High/low confidence layers
    high_conf_layers = [l for l, r in layer_results.items() if r["avg_confidence"] > 0.4]
    low_conf_layers = [l for l, r in layer_results.items() if r["avg_confidence"] < 0.2]
    if high_conf_layers:
        print(f"High-confidence layers (>0.4): {high_conf_layers}")
    if low_conf_layers:
        print(f"Low-confidence layers (<0.2): {low_conf_layers}")


__all__ = [
    "introspect_moe_expert",
]
