"""
High-level inference pipeline for simplified model usage.

Provides a single-import, minimal-code API for loading and
running inference with any supported model family.

Design principles:
- One-liner setup where possible
- Async native
- Pydantic for configuration
- No dictionary goop
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

import mlx.core as mx
from pydantic import BaseModel, Field

from .chat import ChatHistory, format_chat_prompt, format_history
from .generation import GenerationConfig, GenerationResult, generate
from .loader import DType, HFLoader, WeightConverter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


# Type variables for model and config
ConfigT = TypeVar("ConfigT", bound=BaseModel)
ModelT = TypeVar("ModelT")


@runtime_checkable
class CausalLMProtocol(Protocol):
    """Protocol for causal language models."""

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> mx.array: ...

    def update(self, weights: dict) -> None: ...

    def parameters(self) -> dict: ...


class PipelineConfig(BaseModel):
    """Configuration for the inference pipeline."""

    dtype: DType = Field(DType.BFLOAT16, description="Weight dtype")
    cache_dir: Path | None = Field(None, description="Model cache directory")
    default_system_message: str | None = Field(
        "You are a helpful assistant.", description="Default system prompt"
    )
    default_max_tokens: int = Field(100, ge=1, description="Default max tokens")
    default_temperature: float = Field(0.7, ge=0.0, description="Default temperature")


class PipelineState(BaseModel):
    """Internal state of the pipeline."""

    model_id: str
    model_path: Path
    tensor_count: int
    is_loaded: bool = False


class InferencePipeline(Generic[ConfigT, ModelT]):
    """High-level inference pipeline for any model family.

    Example usage:

        # One-liner setup
        pipeline = InferencePipeline.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            LlamaForCausalLM,
            LlamaConfig,
        )

        # Simple chat
        response = pipeline.chat("What is the capital of France?")
        print(response.text)

        # With custom settings
        response = pipeline.generate(
            "Write a poem about AI",
            max_new_tokens=200,
            temperature=0.9,
        )
    """

    def __init__(
        self,
        model: ModelT,
        tokenizer: PreTrainedTokenizer,
        config: ConfigT,
        pipeline_config: PipelineConfig | None = None,
        state: PipelineState | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._pipeline_config = pipeline_config or PipelineConfig()
        self._state = state

    @property
    def model(self) -> ModelT:
        """Access the underlying model."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def config(self) -> ConfigT:
        """Access the model config."""
        return self._config

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_class: type[ModelT],
        config_class: type[ConfigT],
        converter: WeightConverter | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> InferencePipeline[ConfigT, ModelT]:
        """Load a model from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            model_class: Model class to instantiate
            config_class: Config class for model
            converter: Optional weight name converter
            pipeline_config: Pipeline configuration

        Returns:
            Configured InferencePipeline instance
        """
        pipeline_config = pipeline_config or PipelineConfig()

        print(f"Loading {model_id}...")
        print("=" * 60)

        # Download
        print("\n1. Downloading model...")
        result = HFLoader.download(model_id, cache_dir=pipeline_config.cache_dir)
        print(f"   Path: {result.model_path}")

        # Load config
        print("\n2. Loading configuration...")
        config = _load_config(result.model_path, config_class)

        # Create model
        print("\n3. Creating model...")
        model = model_class(config)

        # Load weights
        print("\n4. Loading weights...")
        loaded = HFLoader.load_weights(
            result.model_path,
            dtype=pipeline_config.dtype,
            converter=converter,
        )
        print(f"   Loaded {loaded.tensor_count} tensors")

        # Apply weights
        nested = HFLoader.build_nested_weights(loaded)
        model.update(nested)
        mx.eval(model.parameters())
        print("   Weights applied!")

        # Load tokenizer
        print("\n5. Loading tokenizer...")
        tokenizer = HFLoader.load_tokenizer(result.model_path)
        print(f"   Vocab size: {len(tokenizer)}")

        print("\n" + "=" * 60)
        print("Model loaded successfully!")

        state = PipelineState(
            model_id=model_id,
            model_path=result.model_path,
            tensor_count=loaded.tensor_count,
            is_loaded=True,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            pipeline_config=pipeline_config,
            state=state,
        )

    @classmethod
    async def from_pretrained_async(
        cls,
        model_id: str,
        model_class: type[ModelT],
        config_class: type[ConfigT],
        converter: WeightConverter | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> InferencePipeline[ConfigT, ModelT]:
        """Load a model from HuggingFace asynchronously.

        Runs in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.from_pretrained(
                model_id, model_class, config_class, converter, pipeline_config
            ),
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
            system_message: Optional system prompt (uses default if not provided)
            max_new_tokens: Max tokens to generate (uses default if not provided)
            temperature: Sampling temperature (uses default if not provided)

        Returns:
            GenerationResult with text and stats
        """
        system = system_message or self._pipeline_config.default_system_message
        prompt = format_chat_prompt(self._tokenizer, user_message, system)

        config = GenerationConfig(
            max_new_tokens=max_new_tokens or self._pipeline_config.default_max_tokens,
            temperature=temperature or self._pipeline_config.default_temperature,
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
            max_new_tokens=max_new_tokens or self._pipeline_config.default_max_tokens,
            temperature=temperature or self._pipeline_config.default_temperature,
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
            config: Full generation config (overrides other params)

        Returns:
            GenerationResult with text and stats
        """
        if config is None:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens or self._pipeline_config.default_max_tokens,
                temperature=temperature or self._pipeline_config.default_temperature,
            )

        return generate(self._model, self._tokenizer, prompt, config)


def _load_config(model_path: Path, config_class: type[ConfigT]) -> ConfigT:
    """Load and parse model config from HuggingFace format."""
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Handle list-valued token IDs (common in newer models)
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if key in config_data and isinstance(config_data[key], list):
            config_data[key] = config_data[key][0] if config_data[key] else None

    return config_class(**config_data)
