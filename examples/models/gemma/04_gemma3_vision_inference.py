#!/usr/bin/env python3
"""
Gemma 3 Vision Inference Example

Multimodal inference with Gemma 3 using the chuk_lazarus framework.
Demonstrates loading pretrained weights and running vision-language tasks.

Architecture Overview:
    Gemma 3 multimodal models consist of three main components:

    1. SigLIP Vision Encoder (27 layers, 1152 hidden dim)
       - Patches image into 14x14 patches -> 64x64 = 4096 tokens for 896x896 images
       - Each patch goes through vision transformer layers
       - Uses standard multi-head attention with GELU(precise) activation
       - Pre-norm architecture with LayerNorm

    2. Multi-Modal Projector
       - Average pooling: 64x64 -> 16x16 (4096 -> 256 tokens)
       - Gemma-style RMSNorm (1+weight scaling) + Linear projection
       - Output scaled by 1/sqrt(hidden_size) to match text embedding magnitude

    3. Gemma 3 Language Model (34 layers for 4B)
       - Image embeddings replace <image_soft_token> placeholders (256 tokens)
       - Alternating sliding window (1024) / global attention layers
       - GQA with 8 query heads, 4 KV heads

Supported models from mlx-community (bf16 recommended):
- mlx-community/gemma-3-4b-it-bf16 (4B params, multimodal)
- mlx-community/gemma-3-12b-it-bf16 (12B params, multimodal)
- mlx-community/gemma-3-27b-it-bf16 (27B params, multimodal)

Note: The 1B model is text-only. Use 4B+ for vision capabilities.

Requirements:
    pip install huggingface_hub safetensors transformers pillow

Usage:
    # Basic usage with local image
    python 04_gemma3_vision_inference.py --image path/to/image.jpg

    # Custom prompt
    python 04_gemma3_vision_inference.py --image path/to/image.jpg --prompt "Describe this image"

    # URL-based image
    python 04_gemma3_vision_inference.py --url https://example.com/image.jpg

References:
    - For text-only inference: 03_gemma3_inference.py
    - mlx-vlm (alternative implementation): https://github.com/Blaizzy/mlx-vlm
    - Gemma 3 blog: https://huggingface.co/blog/gemma3
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SigLIPVisionConfig:
    """Configuration for SigLIP vision encoder."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 896
    patch_size: int = 14
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class Gemma3VisionConfig:
    """Configuration for Gemma 3 multimodal model."""

    vision_config: SigLIPVisionConfig
    text_config: GemmaConfig
    mm_tokens_per_image: int = 256  # After pooling: 16x16 = 256 tokens
    mm_tokens_per_side: int = 16  # Output spatial dimension
    image_token_index: int = 262144
    boi_token_index: int = 255999  # Beginning of image
    eoi_token_index: int = 256000  # End of image


# =============================================================================
# SigLIP Vision Encoder
# =============================================================================


class SigLIPMLP(nn.Module):
    """MLP block for SigLIP."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        # SigLIP uses GELU with precise approximation and biased linear layers
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class SigLIPAttention(nn.Module):
    """Multi-head attention for SigLIP."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # SigLIP uses biased linear layers in attention
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (B, num_heads, L, head_dim)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Use MLX optimized scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.out_proj(output)


class SigLIPEncoderLayer(nn.Module):
    """Transformer encoder layer for SigLIP."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.self_attn = SigLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm architecture
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SigLIPVisionEmbeddings(nn.Module):
    """Patch embedding + position embedding for SigLIP."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (B, H, W, C) - MLX uses channels-last
        B = pixel_values.shape[0]

        # Patch embedding: (B, H, W, C) -> (B, H', W', hidden_size)
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions: (B, H', W', hidden_size) -> (B, num_patches, hidden_size)
        patch_embeds = patch_embeds.reshape(B, -1, self.config.hidden_size)

        # Add position embeddings
        position_ids = mx.arange(patch_embeds.shape[1])
        embeddings = patch_embeds + self.position_embedding(position_ids)

        return embeddings


class SigLIPVisionEncoder(nn.Module):
    """SigLIP Vision Transformer encoder."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.layers = [SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class SigLIPVisionModel(nn.Module):
    """Complete SigLIP vision model."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """
        Args:
            pixel_values: (B, H, W, C) image tensor, values in [0, 1]

        Returns:
            (B, num_patches, hidden_size) vision features
        """
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


# =============================================================================
# Multi-Modal Projector
# =============================================================================


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm that uses (1 + weight) as scale factor.

    This matches the HuggingFace/mlx-vlm implementation where the weight
    parameter is an offset from 1.0, not the direct scale.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dims,))  # Offset from 1.0

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class MultiModalProjector(nn.Module):
    """Projects vision features to language model space with pooling.

    The projector performs:
    1. Average pooling: 64x64 tokens -> 16x16 tokens (4096 -> 256)
    2. RMSNorm normalization (Gemma-style with 1+weight scaling)
    3. Linear projection to text embedding dimension
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        patches_per_image: int = 4096,  # 64x64
        tokens_per_side: int = 16,  # Output: 16x16 = 256 tokens
    ):
        super().__init__()
        self.patches_per_image = patches_per_image
        self.tokens_per_side = tokens_per_side
        self.kernel_size = int(math.sqrt(patches_per_image)) // tokens_per_side  # 64 // 16 = 4

        # Use Gemma-style RMSNorm with (1 + weight) scaling
        self.mm_soft_emb_norm = GemmaRMSNorm(vision_hidden_size)
        # Note: weight shape is (vision, text) for projection
        self.mm_input_projection_weight = mx.zeros((vision_hidden_size, text_hidden_size))

    def __call__(self, image_features: mx.array) -> mx.array:
        """
        Args:
            image_features: (B, num_patches, vision_hidden_size) where num_patches = 4096

        Returns:
            (B, 256, text_hidden_size) - pooled and projected features
        """
        batch_size, num_patches, hidden_size = image_features.shape

        # Reshape to spatial: (B, 64, 64, hidden)
        spatial_size = int(math.sqrt(num_patches))
        image_features = image_features.reshape(batch_size, spatial_size, spatial_size, hidden_size)

        # Average pool: 64x64 -> 16x16 using kernel_size=4
        # MLX doesn't have AvgPool2d, so we do it manually
        k = self.kernel_size
        new_h = spatial_size // k
        new_w = spatial_size // k

        # Reshape for pooling: (B, new_h, k, new_w, k, hidden)
        image_features = image_features.reshape(batch_size, new_h, k, new_w, k, hidden_size)
        # Average over the k x k windows
        image_features = mx.mean(image_features, axis=(2, 4))  # (B, new_h, new_w, hidden)

        # Flatten back to sequence: (B, 256, hidden)
        image_features = image_features.reshape(batch_size, new_h * new_w, hidden_size)

        # Apply soft embedding norm
        image_features = self.mm_soft_emb_norm(image_features)
        # Project to text space
        projected = image_features @ self.mm_input_projection_weight
        return projected


# =============================================================================
# Gemma 3 Vision Model
# =============================================================================


class VisionTower(nn.Module):
    """Wrapper to match HF weight structure (vision_tower.vision_model.*)."""

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.vision_model = SigLIPVisionModel(config)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.vision_model(pixel_values)


class Gemma3ForConditionalGeneration(nn.Module):
    """Gemma 3 multimodal model for vision-language tasks."""

    def __init__(self, config: Gemma3VisionConfig):
        super().__init__()
        self.config = config

        # Vision encoder (wrapped to match HF structure)
        self.vision_tower = VisionTower(config.vision_config)

        # Multi-modal projector
        self.multi_modal_projector = MultiModalProjector(
            config.vision_config.hidden_size, config.text_config.hidden_size
        )

        # Language model
        self.language_model = GemmaForCausalLM(config.text_config)

    def get_image_features(self, pixel_values: mx.array) -> mx.array:
        """Extract and project image features.

        Scale by 1/sqrt(hidden_size) following mlx-vlm approach.
        This ensures image features have the right magnitude when combined
        with text embeddings before the backbone's sqrt(hidden_size) scaling.
        """
        vision_outputs = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(vision_outputs)

        # Scale image features by 1/sqrt(hidden_size) - standard Gemma 3 multimodal scaling
        hidden_size = self.config.text_config.hidden_size
        image_features = image_features / (hidden_size**0.5)

        # Cast to match the typical model dtype
        image_features = image_features.astype(mx.bfloat16)

        return image_features

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass with optional image input.

        Args:
            input_ids: (B, seq_len) token IDs
            pixel_values: (B, H, W, C) image tensor or None

        Returns:
            logits: (B, seq_len, vocab_size)
        """
        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            # Get image features
            image_features = self.get_image_features(pixel_values)

            # Find image token positions and replace with image features
            # For simplicity, we insert image features at the beginning
            # In practice, you'd find the image token positions
            batch_size = input_ids.shape[0]

            # Concatenate: [image_features, text_embeddings]
            inputs_embeds = mx.concatenate([image_features, inputs_embeds], axis=1)

        # Forward through language model with embeddings
        output = self.language_model.model(
            input_ids=None,
            input_embeddings=inputs_embeds,
        )

        # Get logits
        logits = self.language_model.lm_head(output.last_hidden_state)

        return logits

    def generate(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        image_positions: list[int] | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        stop_tokens: list[int] | None = None,
    ) -> mx.array:
        """
        Generate text given optional image input with KV-cache for efficiency.

        Args:
            input_ids: Tokenized text prompt (batch_size, seq_len)
            pixel_values: Preprocessed image (batch_size, H, W, C) or None
            image_positions: List of positions where image soft tokens are (to be replaced)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_k: Top-k sampling (filter to top k tokens)
            top_p: Nucleus sampling (filter to tokens with cumulative prob <= top_p)
            stop_tokens: Token IDs that stop generation

        Returns:
            Generated token IDs including the prompt tokens
        """
        # Step 1: Build initial embeddings
        print("  Embedding text tokens...", flush=True)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None and image_positions is not None and len(image_positions) > 0:
            print("  Processing image through vision encoder...", flush=True)
            image_features = self.get_image_features(pixel_values)
            mx.eval(image_features)
            print(f"  Image features shape: {image_features.shape}", flush=True)

            # Replace embeddings at image positions with image features
            # This follows the mlx-vlm approach: scatter image features into placeholder positions
            batch_size, seq_len, embed_dim = inputs_embeds.shape
            num_image_tokens = image_features.shape[1]

            if len(image_positions) == num_image_tokens:
                # Replace the embeddings at image positions with scaled image features
                # Use MLX operations to scatter image features at the correct positions
                # The image soft tokens are contiguous starting at image_positions[0]
                start_pos = image_positions[0]
                end_pos = start_pos + num_image_tokens

                # Build new embeddings: [before][image_features][after]
                before = inputs_embeds[:, :start_pos, :]
                after = inputs_embeds[:, end_pos:, :]
                inputs_embeds = mx.concatenate([before, image_features, after], axis=1)
                print(
                    f"  Replaced {num_image_tokens} image soft token embeddings at positions {start_pos}-{end_pos - 1}",
                    flush=True,
                )
            else:
                print(
                    f"  Warning: image_positions ({len(image_positions)}) != num_image_tokens ({num_image_tokens})",
                    flush=True,
                )

            print(f"  Final embeddings shape: {inputs_embeds.shape}", flush=True)

        # Step 2: Process initial sequence through backbone to get KV-cache
        print("  Building KV-cache (this may take a moment)...", flush=True)
        output = self.language_model.model(
            input_ids=None,
            input_embeddings=inputs_embeds,
            cache=None,  # No cache on first pass
        )
        cache = output.cache
        mx.eval(cache)
        print("  KV-cache ready!", flush=True)

        # Track generated tokens
        generated_ids = list(input_ids[0].tolist())

        # Get first token from initial pass
        logits = self.language_model.lm_head(output.last_hidden_state[:, -1:, :])
        logits = logits[:, 0, :]

        print("  Generating tokens:", end=" ", flush=True)
        for i in range(max_new_tokens):
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Apply top-k sampling
            if top_k is not None and top_k > 0:
                top_k_logits = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                threshold = top_k_logits[:, -1:]
                logits = mx.where(logits < threshold, float("-inf"), logits)

            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
                cutoff_idx = mx.sum(cumsum_probs < top_p, axis=-1, keepdims=True)
                cutoff_logit = mx.take_along_axis(sorted_logits, cutoff_idx, axis=-1)
                logits = mx.where(logits < cutoff_logit, float("-inf"), logits)

            # Sample or greedy decode
            if temperature == 0:
                next_token = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                probs = mx.softmax(logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=-1)

            next_token_id = int(next_token[0, 0])
            generated_ids.append(next_token_id)
            print(".", end="", flush=True)

            # Check stop tokens
            if stop_tokens and next_token_id in stop_tokens:
                break

            # Forward pass with cache - only process the new token!
            # This is the key to efficiency: we reuse the cached K/V tensors
            output = self.language_model.model(
                input_ids=next_token,  # Just the new token
                cache=cache,  # Reuse cached K/V from previous tokens
            )
            cache = output.cache
            logits = self.language_model.lm_head(output.last_hidden_state[:, -1:, :])
            logits = logits[:, 0, :]
            mx.eval(logits)

        print(" done!", flush=True)
        return mx.array([generated_ids])


# =============================================================================
# Model Loading
# =============================================================================


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace Hub."""

    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...", flush=True)
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors"],
    )
    return Path(path)


def load_config(model_path: Path) -> Gemma3VisionConfig:
    """Load configuration from HuggingFace format."""
    with open(model_path / "config.json") as f:
        hf_config = json.load(f)

    # Vision config
    vc = hf_config.get("vision_config", {})
    vision_config = SigLIPVisionConfig(
        hidden_size=vc.get("hidden_size", 1152),
        intermediate_size=vc.get("intermediate_size", 4304),
        num_hidden_layers=vc.get("num_hidden_layers", 27),
        num_attention_heads=vc.get("num_attention_heads", 16),
        image_size=vc.get("image_size", 896),
        patch_size=vc.get("patch_size", 14),
    )

    # Text config - need to infer from weights
    tc = hf_config.get("text_config", {})
    text_config = GemmaConfig(
        vocab_size=tc.get("vocab_size", 262144),
        hidden_size=tc.get("hidden_size", 2560),
        num_hidden_layers=tc.get("num_hidden_layers", 34),
        num_attention_heads=tc.get("num_attention_heads", 8),  # Will be inferred
        num_key_value_heads=tc.get("num_key_value_heads", 4),  # Will be inferred
        intermediate_size=tc.get("intermediate_size", 10240),
        head_dim=tc.get("head_dim", 256),
        sliding_window=tc.get("sliding_window", 1024),
    )

    return Gemma3VisionConfig(
        vision_config=vision_config,
        text_config=text_config,
        mm_tokens_per_image=hf_config.get("mm_tokens_per_image", 256),
        image_token_index=hf_config.get("image_token_index", 262144),
        boi_token_index=hf_config.get("boi_token_index", 255999),
        eoi_token_index=hf_config.get("eoi_token_index", 256000),
    )


def load_weights(model_path: Path) -> dict:
    """Load weights from safetensors files."""
    weights = {}
    for sf_path in model_path.glob("*.safetensors"):
        print(f"  Loading {sf_path.name}...", flush=True)
        file_weights = mx.load(str(sf_path))
        weights.update(file_weights)
    return weights


def infer_text_config_from_weights(weights: dict, config: Gemma3VisionConfig) -> Gemma3VisionConfig:
    """Infer missing text config values from weights."""
    head_dim = config.text_config.head_dim

    # Find q_proj and k_proj to infer num_heads
    for k, v in weights.items():
        if "language_model.model.layers.0.self_attn.q_proj.weight" in k:
            num_attention_heads = v.shape[0] // head_dim
            print(f"  Inferred num_attention_heads={num_attention_heads}")
            config.text_config = GemmaConfig(
                vocab_size=config.text_config.vocab_size,
                hidden_size=config.text_config.hidden_size,
                num_hidden_layers=config.text_config.num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=config.text_config.num_key_value_heads,
                intermediate_size=config.text_config.intermediate_size,
                head_dim=head_dim,
                sliding_window=config.text_config.sliding_window,
            )
        if "language_model.model.layers.0.self_attn.k_proj.weight" in k:
            num_key_value_heads = v.shape[0] // head_dim
            print(f"  Inferred num_key_value_heads={num_key_value_heads}")
            config.text_config = GemmaConfig(
                vocab_size=config.text_config.vocab_size,
                hidden_size=config.text_config.hidden_size,
                num_hidden_layers=config.text_config.num_hidden_layers,
                num_attention_heads=config.text_config.num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=config.text_config.intermediate_size,
                head_dim=head_dim,
                sliding_window=config.text_config.sliding_window,
            )
            break

    return config


def load_gemma3_vision_model(
    model_id: str,
) -> tuple[Gemma3ForConditionalGeneration, any, Gemma3VisionConfig]:
    """Load Gemma 3 vision model from HuggingFace Hub."""
    from transformers import AutoTokenizer

    # Download
    model_path = download_model(model_id)

    # Load weights
    print("Loading weights...", flush=True)
    weights = load_weights(model_path)
    print(f"  Loaded {len(weights)} tensors", flush=True)

    # Load config and infer missing values
    print("Loading config...", flush=True)
    config = load_config(model_path)
    config = infer_text_config_from_weights(weights, config)
    print(
        f"  Vision: {config.vision_config.num_hidden_layers} layers, {config.vision_config.hidden_size} dim",
        flush=True,
    )
    print(
        f"  Text: {config.text_config.num_hidden_layers} layers, {config.text_config.hidden_size} dim",
        flush=True,
    )

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Create model
    print("Creating model...", flush=True)
    model = Gemma3ForConditionalGeneration(config)

    # Load weights
    print("Applying weights...", flush=True)
    nested_weights = tree_unflatten(list(weights.items()))
    model.update(nested_weights)

    return model, tokenizer, config


# =============================================================================
# Image Processing
# =============================================================================


def load_image(path_or_url: str, size: int = 896) -> mx.array:
    """Load and preprocess an image."""
    import numpy as np
    from PIL import Image

    if path_or_url.startswith(("http://", "https://")):
        import urllib.request
        from io import BytesIO

        with urllib.request.urlopen(path_or_url) as response:
            image = Image.open(BytesIO(response.read()))
    else:
        image = Image.open(path_or_url)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to target size
    image = image.resize((size, size), Image.Resampling.BILINEAR)

    # Convert to numpy array and normalize to [0, 1]
    pixel_values = np.array(image).astype(np.float32) / 255.0

    # SigLIP normalization (mean=0.5, std=0.5 for all channels)
    pixel_values = (pixel_values - 0.5) / 0.5

    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    pixel_values = np.expand_dims(pixel_values, axis=0)

    return mx.array(pixel_values)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Vision Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-4b-it-bf16",
        help="HuggingFace model ID (must be 4B+ for vision)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to local image file",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL of image to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt for image description",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    if not args.image and not args.url:
        print("Error: Please provide --image or --url")
        return

    image_source = args.image or args.url

    print("=" * 60)
    print("Gemma 3 Vision Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Image: {image_source}")
    print("-" * 60)

    # Load model
    model, tokenizer, config = load_gemma3_vision_model(args.model)
    print("\nModel loaded successfully!")

    # Evaluate model parameters
    mx.eval(model.parameters())

    # Load and preprocess image
    print("\nLoading image...")
    pixel_values = load_image(image_source, config.vision_config.image_size)
    print(f"  Image shape: {pixel_values.shape}")

    # Format prompt with proper image tokens
    # Gemma 3 expects: <bos><start_of_turn>user\n<start_of_image>[256 soft tokens]<end_of_image>prompt<end_of_turn>\n<start_of_turn>model\n
    # The 256 <image_soft_token> (262144) placeholders get replaced with actual image embeddings
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    # Build prompt with 256 image soft token placeholders
    num_image_tokens = 256
    image_soft_token = "<image_soft_token>"
    text_prompt = f"<bos><start_of_turn>user\n<start_of_image>{image_soft_token * num_image_tokens}<end_of_image>{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer.encode(text_prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Find the image soft token positions (262144)
    image_soft_token_id = 262144
    input_ids_list = input_ids[0].tolist()
    image_positions = [i for i, tid in enumerate(input_ids_list) if tid == image_soft_token_id]
    print(
        f"  Found {len(image_positions)} image soft tokens starting at position {image_positions[0] if image_positions else 'N/A'}"
    )

    # Generate
    print("\nGenerating response...")
    output_ids = model.generate(
        input_ids,
        pixel_values=pixel_values,
        image_positions=image_positions,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=40,
        top_p=0.95,
        stop_tokens=[tokenizer.eos_token_id, 106],  # 106 is <end_of_turn>
    )

    # Decode only the newly generated tokens (skip the prompt tokens)
    prompt_len = input_ids.shape[1]
    generated_tokens = output_ids[0, prompt_len:].tolist()
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up the response
    response = response.strip()

    # Remove any trailing special tokens that might remain
    for suffix in ["<end_of_turn>", "<eos>", "</s>"]:
        if response.endswith(suffix):
            response = response[: -len(suffix)].strip()

    print(f"\nResponse:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
