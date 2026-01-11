"""
Text generation utilities with typed outputs.

Provides high-level generation functions with proper
type safety and statistics tracking.

Supports virtual expert plugins for tool-augmented generation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from chuk_lazarus.inference.virtual_experts.base import VirtualExpertPlugin
    from chuk_lazarus.models_v2.base import CausalLMProtocol


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = Field(100, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int | None = Field(None, ge=1, description="Top-k sampling")
    stop_tokens: list[int] = Field(default_factory=list, description="Token IDs to stop on")
    use_plugins: bool = Field(True, description="Whether to use virtual expert plugins")


class GenerationStats(BaseModel):
    """Statistics from a generation run."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of generated tokens")
    total_time_seconds: float = Field(..., description="Total generation time")
    tokens_per_second: float = Field(..., description="Generation speed")

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Generated {self.output_tokens} tokens in {self.total_time_seconds:.2f}s "
            f"({self.tokens_per_second:.1f} tok/s)"
        )


class GenerationResult(BaseModel):
    """Result of text generation."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Generated text")
    stats: GenerationStats = Field(..., description="Generation statistics")
    stop_reason: str = Field("max_tokens", description="Why generation stopped")
    plugin_used: str | None = Field(None, description="Name of plugin used, if any")


def get_stop_tokens(tokenizer: PreTrainedTokenizer) -> list[int]:
    """Extract stop token IDs from tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        List of token IDs to stop on
    """
    stop_tokens = []

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_tokens.extend(tokenizer.eos_token_id)
        else:
            stop_tokens.append(tokenizer.eos_token_id)

    return stop_tokens


def generate(
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
    plugins: list[VirtualExpertPlugin] | None = None,
) -> GenerationResult:
    """Generate text from a prompt.

    Args:
        model: Causal language model with generate() method
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        config: Generation configuration
        plugins: Optional list of virtual expert plugins to try before model generation

    Returns:
        GenerationResult with text and stats
    """
    if config is None:
        config = GenerationConfig()

    start_time = time.time()
    plugin_used: str | None = None

    # Check if any plugin can handle this prompt
    if config.use_plugins and plugins:
        for plugin in sorted(plugins, key=lambda p: p.priority, reverse=True):
            if plugin.can_handle(prompt):
                result = plugin.execute(prompt)
                if result is not None:
                    gen_time = time.time() - start_time
                    input_ids = tokenizer.encode(prompt)
                    output_ids = tokenizer.encode(result)

                    stats = GenerationStats(
                        input_tokens=len(input_ids),
                        output_tokens=len(output_ids),
                        total_time_seconds=gen_time,
                        tokens_per_second=(len(output_ids) / gen_time if gen_time > 0 else 0),
                    )

                    return GenerationResult(
                        text=result,
                        stats=stats,
                        stop_reason="plugin",
                        plugin_used=plugin.name,
                    )

    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    input_length = input_ids.shape[1]

    # Get stop tokens
    stop_tokens = config.stop_tokens or get_stop_tokens(tokenizer)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        stop_tokens=stop_tokens,
    )
    mx.eval(output_ids)
    gen_time = time.time() - start_time

    # Decode generated tokens only
    new_tokens = output_ids[0, input_length:]
    output_length = new_tokens.shape[0]
    generated_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    # Determine stop reason
    stop_reason = "max_tokens"
    if output_length < config.max_new_tokens:
        if new_tokens.size > 0 and int(new_tokens[-1]) in stop_tokens:
            stop_reason = "eos"
        else:
            stop_reason = "stop_token"

    # Build stats
    stats = GenerationStats(
        input_tokens=input_length,
        output_tokens=output_length,
        total_time_seconds=gen_time,
        tokens_per_second=output_length / gen_time if gen_time > 0 else 0,
    )

    return GenerationResult(
        text=generated_text,
        stats=stats,
        stop_reason=stop_reason,
        plugin_used=plugin_used,
    )


def generate_stream(
    model: CausalLMProtocol,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
):
    """Generate text with streaming output.

    Yields text chunks as they're generated.

    Args:
        model: Causal language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        config: Generation configuration

    Yields:
        Text chunks as they're generated
    """
    if config is None:
        config = GenerationConfig()

    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Get stop tokens
    stop_tokens = set(config.stop_tokens or get_stop_tokens(tokenizer))

    # Generate token by token
    tokens: list[int] = []
    cache = None
    y = input_ids

    for _ in range(config.max_new_tokens):
        logits, cache = model(y, cache=cache)
        if logits is None or logits.shape[1] == 0:
            break

        logits = logits[:, -1, :]

        # Sample
        if config.temperature == 0:
            next_token = mx.argmax(logits, axis=-1)
        else:
            probs = mx.softmax(logits / config.temperature, axis=-1)
            next_token = mx.random.categorical(probs)

        next_token_id = int(next_token.item())

        # Check stop
        if next_token_id in stop_tokens:
            break

        tokens.append(next_token_id)
        y = next_token[None]

        # Decode incrementally
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if len(text) > 0:
            yield text
            tokens = []  # Reset for next chunk
