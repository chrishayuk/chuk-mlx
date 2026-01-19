"""
Unified inference pipeline for any HuggingFace model.

Provides a single entry point that:
1. Auto-detects model family from config.json
2. Loads config and weights using family-specific converters
3. Provides unified chat/generate interface
4. Supports introspection hooks for all families

Usage:
    from chuk_lazarus.inference import UnifiedPipeline

    # Auto-detect and load any supported model
    pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Or async
    pipeline = await UnifiedPipeline.from_pretrained_async("model-id")

    # Chat
    result = pipeline.chat("What is the capital of France?")
    print(result.text)

    # Generate with introspection
    result = pipeline.generate(
        "The capital of France is",
        introspect=True,
    )
    print(result.hidden_states)  # Layer activations
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from chuk_lazarus.models_v2.families import (
    FamilyInfo,
    ModelFamilyType,
    detect_model_family,
    get_family_info,
)

from .chat import ChatHistory, format_chat_prompt, format_history
from .generation import GenerationConfig, GenerationResult, generate
from .loader import DType, HFLoader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class UnifiedPipelineConfig(BaseModel):
    """Configuration for the unified pipeline."""

    dtype: DType = Field(DType.BFLOAT16, description="Weight dtype")
    cache_dir: Path | None = Field(None, description="Model cache directory")
    default_system_message: str | None = Field(
        "You are a helpful assistant.", description="Default system prompt"
    )
    default_max_tokens: int = Field(256, ge=1, description="Default max tokens")
    default_temperature: float = Field(0.7, ge=0.0, description="Default temperature")

    # Introspection
    enable_introspection: bool = Field(True, description="Enable layer hooks")
    introspection_layers: list[int] | None = Field(None, description="Layers to track (None = all)")


class UnifiedPipelineState(BaseModel):
    """Internal state of the unified pipeline."""

    model_id: str
    model_path: Path
    family_type: ModelFamilyType
    tensor_count: int
    is_loaded: bool = False


class IntrospectionResult(BaseModel):
    """Results from introspection during generation."""

    # Layer-wise hidden states (if captured)
    hidden_states: list[Any] | None = None

    # Attention patterns (if captured)
    attention_patterns: list[Any] | None = None

    # Pre-head activations
    pre_head_activations: Any | None = None


class UnifiedPipeline:
    """Unified inference pipeline for any supported model family.

    This is the primary interface for model inference. It:
    1. Auto-detects model family from HuggingFace config
    2. Uses family-specific loading and weight conversion
    3. Provides unified chat/generate interface
    4. Supports introspection hooks for interpretability research

    Example:
        # One-liner for any supported model
        pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Chat
        result = pipeline.chat("Hello!")
        print(result.text)

        # With introspection
        result, intro = pipeline.generate_with_introspection(
            "The capital of France is",
            capture_hidden_states=True,
        )
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        model_config: BaseModel,
        family_info: FamilyInfo,
        pipeline_config: UnifiedPipelineConfig | None = None,
        state: UnifiedPipelineState | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._family_info = family_info
        self._pipeline_config = pipeline_config or UnifiedPipelineConfig()
        self._state = state

        # Introspection hooks (mutable state for capturing)
        self._introspection_data: IntrospectionResult | None = None

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def config(self) -> BaseModel:
        """Access the model config."""
        return self._model_config

    @property
    def family(self) -> FamilyInfo:
        """Access the model family info."""
        return self._family_info

    @property
    def family_type(self) -> ModelFamilyType:
        """Get the model family type."""
        return self._family_info.family_type

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a model from HuggingFace, auto-detecting the family.

        Args:
            model_id: HuggingFace model ID or local path
            pipeline_config: Pipeline configuration
            verbose: Print loading progress

        Returns:
            Configured UnifiedPipeline instance
        """
        pipeline_config = pipeline_config or UnifiedPipelineConfig()

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        log(f"Loading {model_id}...")
        log("=" * 60)

        # Download
        log("\n1. Downloading model...")
        result = HFLoader.download(model_id, cache_dir=pipeline_config.cache_dir, verbose=verbose)
        log(f"   Path: {result.model_path}")

        # Load raw config for family detection
        log("\n2. Detecting model family...")
        config_path = result.model_path / "config.json"
        with open(config_path) as f:
            hf_config = json.load(f)

        family_type = detect_model_family(hf_config)
        if family_type is None:
            model_type = hf_config.get("model_type", "unknown")
            archs = hf_config.get("architectures", [])
            raise ValueError(
                f"Unable to detect model family. model_type={model_type}, "
                f"architectures={archs}. Model may not be supported yet."
            )

        family_info = get_family_info(family_type)
        if family_info is None:
            raise ValueError(f"No family info registered for {family_type}")

        log(f"   Detected: {family_type.value}")

        # Load config using family-specific class
        log("\n3. Loading configuration...")
        config_class = family_info.config_class

        # Try from_hf_config first, then fall back to direct construction
        if hasattr(config_class, "from_hf_config"):
            model_config = config_class.from_hf_config(hf_config)
        else:
            model_config = config_class(**hf_config)

        log(f"   Hidden: {model_config.hidden_size}, Layers: {model_config.num_hidden_layers}")

        # Load tokenizer
        log("\n4. Loading tokenizer...")
        tokenizer = HFLoader.load_tokenizer(result.model_path)
        log(f"   Vocab size: {len(tokenizer)}")

        # Create model
        log("\n5. Creating model...")
        model_class = family_info.model_class
        model = model_class(model_config)

        # Load and apply weights using unified loader
        log("\n6. Loading and applying weights...")
        HFLoader.apply_weights_to_model(
            model,
            result.model_path,
            model_config,
            dtype=pipeline_config.dtype,
            verbose=verbose,
        )
        log("   Weights applied!")

        log("\n" + "=" * 60)
        log(f"Model loaded successfully! ({family_type.value})")

        # Count parameters (flatten nested structure)
        def count_params(params):
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += count_params(v)
                elif hasattr(v, "size"):
                    total += v.size
            return total

        param_count = count_params(model.parameters())

        state = UnifiedPipelineState(
            model_id=model_id,
            model_path=result.model_path,
            family_type=family_type,
            tensor_count=param_count,
            is_loaded=True,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            family_info=family_info,
            pipeline_config=pipeline_config,
            state=state,
        )

    @classmethod
    async def from_pretrained_async(
        cls,
        model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a model from HuggingFace asynchronously.

        Runs in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.from_pretrained(model_id, pipeline_config, verbose),
        )

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a response to a chat message.

        Args:
            user_message: The user's message
            system_message: Optional system prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerationResult with text and stats
        """
        system = system_message or self._pipeline_config.default_system_message
        prompt = format_chat_prompt(self._tokenizer, user_message, system)

        config = GenerationConfig(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self._pipeline_config.default_max_tokens,
            temperature=temperature if temperature is not None else self._pipeline_config.default_temperature,
        )

        return generate(self._model, self._tokenizer, prompt, config)

    def chat_with_history(
        self,
        history: ChatHistory,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a response using chat history.

        Args:
            history: ChatHistory with conversation
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerationResult with text and stats
        """
        prompt = format_history(self._tokenizer, history)

        config = GenerationConfig(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self._pipeline_config.default_max_tokens,
            temperature=temperature if temperature is not None else self._pipeline_config.default_temperature,
        )

        return generate(self._model, self._tokenizer, prompt, config)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text from a raw prompt.

        Args:
            prompt: Input prompt (no formatting applied)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            config: Full generation config

        Returns:
            GenerationResult with text and stats
        """
        if config is None:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self._pipeline_config.default_max_tokens,
                temperature=temperature if temperature is not None else self._pipeline_config.default_temperature,
            )

        return generate(self._model, self._tokenizer, prompt, config)

    def list_supported_families(self) -> list[str]:
        """List all supported model families."""
        from chuk_lazarus.models_v2.families import list_model_families

        return [f.value for f in list_model_families()]

    @classmethod
    def from_compressed(
        cls,
        compressed_path: str | Path,
        original_model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a compressed MoE model from overlay format.

        This loads a model that was compressed using the moe-overlay-compress command.
        The compressed model uses low-rank approximations of expert weights.

        Args:
            compressed_path: Path to compressed overlay directory
            original_model_id: Original model ID (for non-expert weights)
            pipeline_config: Pipeline configuration
            verbose: Print loading progress

        Returns:
            Configured UnifiedPipeline with compressed experts

        Example:
            pipeline = UnifiedPipeline.from_compressed(
                "gpt-oss-20b-overlay",
                "openai/gpt-oss-20b",
            )
            result = pipeline.chat("What is the capital of France?")
        """
        from chuk_lazarus.introspection.moe import OverlayExperts

        pipeline_config = pipeline_config or UnifiedPipelineConfig()
        compressed_path = Path(compressed_path)

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        log(f"Loading compressed model from: {compressed_path}")
        log("=" * 60)

        # Load overlay
        log("\n1. Loading compressed overlay...")
        overlay = OverlayExperts.load(compressed_path)
        log(f"   Layers: {overlay.num_layers}, Experts: {overlay.num_experts}")
        log(f"   Compression: {overlay.config.compression_ratio:.1f}x")
        log(f"   Memory: {overlay.memory_usage_mb():.1f} MB")

        # Load original model using standard pipeline
        log("\n2. Loading base model...")
        pipeline = cls.from_pretrained(
            original_model_id,
            pipeline_config=pipeline_config,
            verbose=verbose,
        )

        # Patch experts to use overlay
        log("\n3. Patching experts with compressed weights...")
        cls._patch_experts_with_overlay(pipeline.model, overlay)
        log("   Experts patched!")

        # Update state
        if pipeline._state:
            pipeline._state.model_id = f"{original_model_id} (compressed)"

        log("\n" + "=" * 60)
        log(f"Compressed model loaded! ({overlay.config.compression_ratio:.1f}x compression)")

        return pipeline

    @staticmethod
    def _patch_experts_with_overlay(model, overlay) -> None:
        """Patch model to use compressed expert weights.

        Replaces expert forward passes with overlay-based reconstruction.
        """
        import mlx.core as mx
        import mlx.nn as nn

        # GPT-OSS custom SwiGLU activation
        def gpt_oss_swiglu(
            x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0
        ) -> mx.array:
            """GPT-OSS custom SwiGLU: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)"""
            x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
            x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
            out_glu = x_glu * mx.sigmoid(alpha * x_glu)
            return out_glu * (x_linear + 1)

        # Create overlay-based batched experts class for GPT-OSS style models
        class OverlayBatchedExperts(nn.Module):
            """Batched experts using overlay reconstruction."""

            def __init__(self, layer_idx: int, overlay, hidden_size: int, num_experts: int):
                super().__init__()
                self.layer_idx = layer_idx
                self.overlay = overlay
                self.hidden_size = hidden_size
                self.num_experts = num_experts

            def __call__(
                self, x: mx.array, expert_indices: mx.array, expert_weights: mx.array
            ) -> mx.array:
                """Apply experts using overlay reconstruction."""
                num_tokens = x.shape[0]
                output = mx.zeros((num_tokens, self.hidden_size), dtype=x.dtype)

                for expert_idx in range(self.num_experts):
                    # Find which tokens use this expert
                    expert_mask = expert_indices == expert_idx
                    token_weights = mx.sum(
                        expert_weights * expert_mask.astype(expert_weights.dtype), axis=-1
                    )

                    if not mx.any(token_weights > 0):
                        continue

                    # Apply expert via overlay (efficient low-rank)
                    # Gate projection (even indices in original gate_up)
                    gate_out = self.overlay.apply_expert(self.layer_idx, "gate", expert_idx, x)
                    # Up projection (odd indices in original gate_up)
                    up_out = self.overlay.apply_expert(self.layer_idx, "up", expert_idx, x)
                    # GPT-OSS uses custom SwiGLU: (gate * sigmoid(alpha * gate)) * (up + 1)
                    hidden = gpt_oss_swiglu(up_out, gate_out)
                    expert_out = self.overlay.apply_expert(
                        self.layer_idx, "down", expert_idx, hidden
                    )

                    # Weight and accumulate
                    output = output + expert_out * token_weights[:, None]

                return output

        # Find layers and replace experts
        for layer_idx in overlay.moe_layer_indices:
            # Find the layer
            layer = None
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_idx]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer = model.transformer.h[layer_idx]
            elif hasattr(model, "layers"):
                layer = model.layers[layer_idx]

            if layer is None:
                continue

            # Find MoE block - for GPT-OSS it's the mlp
            moe_block = None
            for attr in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
                if hasattr(layer, attr):
                    block = getattr(layer, attr)
                    if hasattr(block, "experts"):
                        moe_block = block
                        moe_attr = attr
                        break

            if moe_block is None:
                continue

            # Get hidden size from down projection output dimension
            # down_shape = (hidden_size, intermediate_size), so down_shape[0] = hidden_size
            hidden_size = overlay.config.down_shape[0]
            num_experts = overlay.num_experts

            # Replace the experts module with overlay version
            new_experts = OverlayBatchedExperts(layer_idx, overlay, hidden_size, num_experts)
            getattr(layer, moe_attr).experts = new_experts
