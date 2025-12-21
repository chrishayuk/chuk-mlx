"""
Unified model loading.

Supports:
- HuggingFace models via mlx-lm
- Local model architectures
- LoRA adapters
- Weight loading
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..utils.huggingface import load_from_hub
from .config import LoRAConfig, ModelConfig
from .lora import apply_lora

logger = logging.getLogger(__name__)


@dataclass
class LoadOptions:
    """Options for model loading."""

    load_weights: bool = True
    use_lora: bool = False
    lora_config: LoRAConfig | None = None
    adapter_path: str | None = None
    use_4bit: bool = False
    dtype: str | None = None  # "float16", "bfloat16", etc.


class ModelWrapper(nn.Module):
    """
    Wrapper that provides a unified interface for loaded models.

    Handles both:
    - mlx-lm loaded models
    - Custom architecture models
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: ModelConfig):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._lora_layers = {}
        self.training = False

    def __call__(self, input_ids: mx.array, cache: Any = None) -> tuple[mx.array, Any]:
        """Forward pass."""
        return self._model(input_ids, cache=cache)

    @property
    def model(self) -> nn.Module:
        """Access underlying model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Access tokenizer."""
        return self._tokenizer

    @property
    def config(self) -> ModelConfig:
        """Access model config."""
        return self._config

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._config.vocab_size

    def parameters(self):
        """Get model parameters."""
        return self._model.parameters()

    def trainable_parameters(self):
        """Get trainable parameters (LoRA only if enabled)."""
        if self._lora_layers:
            params = {}
            for name, layer in self._lora_layers.items():
                params[f"{name}.lora_A"] = layer.lora_A
                params[f"{name}.lora_B"] = layer.lora_B
            return params
        return self.parameters()

    def freeze(self):
        """Freeze all parameters."""
        self._model.freeze()

    def set_mode(self, mode: str):
        """Set training or inference mode."""
        self.training = mode.upper() == "TRAIN"

    def save_weights(self, path: str):
        """Save model weights."""
        weights = dict(self.parameters())
        flat = _flatten_params(weights)
        mx.save_safetensors(path, flat)
        logger.info(f"Saved weights to {path}")

    def load_weights(self, path: str):
        """Load model weights."""
        weights = mx.load(path)
        self._model.load_weights(list(weights.items()))
        logger.info(f"Loaded weights from {path}")

    def save_adapter(self, path: str):
        """Save LoRA adapter weights."""
        if not self._lora_layers:
            logger.warning("No LoRA layers to save")
            return

        adapter_weights = {}
        for name, layer in self._lora_layers.items():
            adapter_weights[f"{name}.lora_A"] = layer.lora_A
            adapter_weights[f"{name}.lora_B"] = layer.lora_B

        mx.save_safetensors(path, adapter_weights)
        logger.info(f"Saved adapter to {path}")

    def load_adapter(self, path: str):
        """Load LoRA adapter weights."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Adapter not found: {path}")

        weights = mx.load(path)
        for name, layer in self._lora_layers.items():
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in weights:
                layer.lora_A = weights[a_key]
            if b_key in weights:
                layer.lora_B = weights[b_key]

        logger.info(f"Loaded adapter from {path}")

    def generate(
        self, prompt: str, max_tokens: int = 256, temperature: float = 1.0, top_p: float = 0.9
    ) -> str:
        """Generate text from prompt."""
        try:
            from mlx_lm import generate

            return generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )
        except ImportError:
            logger.warning("mlx-lm not available, using basic generation")
            return self._basic_generate(prompt, max_tokens, temperature)

    def _basic_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Basic generation fallback."""
        tokens = self._tokenizer.encode(prompt)
        generated = list(tokens)

        for _ in range(max_tokens):
            input_ids = mx.array([generated])
            logits, _ = self(input_ids)
            next_logits = logits[0, -1, :] / max(temperature, 1e-6)

            probs = mx.softmax(next_logits)
            next_token = int(mx.argmax(probs))

            generated.append(next_token)

            if next_token == self._tokenizer.eos_token_id:
                break

        return self._tokenizer.decode(generated[len(tokens) :])


def load_model(model_name: str, options: LoadOptions | None = None, **kwargs) -> ModelWrapper:
    """
    Load a model from HuggingFace or local path.

    Args:
        model_name: HuggingFace model ID or local path
        options: Loading options (LoRA, weights, etc.)
        **kwargs: Additional options (use_lora, lora_rank, etc.)

    Returns:
        ModelWrapper with model, tokenizer, and config

    Examples:
        # Simple load
        model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # With LoRA
        model = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            options=LoadOptions(use_lora=True, lora_config=LoRAConfig(rank=8))
        )

        # Shorthand
        model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_lora=True)
    """
    # Handle kwargs as shorthand
    if options is None:
        options = LoadOptions()

    if kwargs.get("use_lora"):
        options.use_lora = True
        if "lora_rank" in kwargs:
            options.lora_config = LoRAConfig(rank=kwargs["lora_rank"])

    if "adapter_path" in kwargs:
        options.adapter_path = kwargs["adapter_path"]

    # Load model
    try:
        model, tokenizer, config = _load_from_mlx_lm(model_name)
    except Exception as e:
        logger.info(f"mlx-lm load failed ({e}), trying local architectures")
        model, tokenizer, config = _load_local(model_name)

    # Create wrapper
    wrapper = ModelWrapper(model, tokenizer, config)

    # Apply LoRA if requested
    if options.use_lora:
        lora_config = options.lora_config or LoRAConfig()
        wrapper._lora_layers = apply_lora(model, lora_config)
        logger.info(
            f"Applied LoRA with rank={lora_config.rank} to {len(wrapper._lora_layers)} layers"
        )

    # Load adapter weights if provided
    if options.adapter_path:
        wrapper.load_adapter(options.adapter_path)

    return wrapper


def load_tokenizer(model_name: str) -> Any:
    """Load just the tokenizer."""
    try:
        from mlx_lm import load

        _, tokenizer = load(model_name)
        return tokenizer
    except ImportError:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)


def _load_from_mlx_lm(model_name: str) -> tuple[nn.Module, Any, ModelConfig]:
    """Load model using mlx-lm."""
    from mlx_lm import load

    logger.info(f"Loading model via mlx-lm: {model_name}")
    model, tokenizer = load(model_name)

    # Get config
    model_path = load_from_hub(model_name)
    config = ModelConfig.from_file(model_path / "config.json")

    return model, tokenizer, config


def _load_local(model_name: str) -> tuple[nn.Module, Any, ModelConfig]:
    """Load from local architecture definitions."""
    model_path = load_from_hub(model_name)
    config = ModelConfig.from_file(model_path / "config.json")

    # Determine architecture
    arch = None
    if config.architectures:
        arch = config.architectures[0].lower()

    # Import appropriate model class
    if arch and "llama" in arch:
        from .architectures.llama import LlamaModel

        model = LlamaModel(config)
    elif arch and "mistral" in arch:
        from .architectures.mistral import MistralModel

        model = MistralModel(config)
    elif arch and "gemma" in arch:
        from .architectures.gemma import GemmaModel

        model = GemmaModel(config)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load weights
    model = _load_weights(model, model_path)

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return model, tokenizer, config


def _load_weights(model: nn.Module, model_path: Path) -> nn.Module:
    """Load weights from safetensors or bin files."""
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(model_path.glob("*.bin"))

    if not weight_files:
        logger.warning(f"No weight files found in {model_path}")
        return model

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))

    # Sanitize weight names if needed
    weights = _sanitize_weights(weights)

    model.load_weights(list(weights.items()))
    return model


def _sanitize_weights(weights: dict) -> dict:
    """Sanitize weight names for MLX compatibility."""
    sanitized = {}
    for key, value in weights.items():
        # Remove common prefixes
        new_key = key
        for prefix in ["model.", "transformer."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        sanitized[new_key] = value

    return sanitized


def _flatten_params(params: dict, prefix: str = "") -> dict:
    """Flatten nested parameter dict."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        elif isinstance(v, mx.array):
            flat[key] = v
    return flat
