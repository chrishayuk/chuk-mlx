"""
Model loader using the registry system.

Provides a unified interface for loading models from:
- Local paths
- HuggingFace Hub
- Preset configurations

Uses the registry to find the right model class based on config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import mlx.core as mx

from .core.config import ModelConfig
from .core.registry import get_factory, list_models
from .models.base import Model

M = TypeVar("M", bound=Model)


async def load_model_async(
    path_or_id: str,
    model_type: str | None = None,
    device: str | None = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> Model:
    """
    Load a model asynchronously.

    This is the main entry point for loading models. It:
    1. Loads the config from the path
    2. Uses the registry to find the right model class
    3. Creates the model
    4. Loads the weights

    Args:
        path_or_id: Local path or HuggingFace model ID
        model_type: Optional override for model type
        device: Device to load to (None = default)
        dtype: Data type for weights
        **kwargs: Additional arguments passed to model

    Returns:
        Loaded model instance

    Example:
        >>> model = await load_model_async("meta-llama/Llama-2-7b-hf")
        >>> # Or from local path
        >>> model = await load_model_async("/path/to/model")
    """
    import aiofiles

    path = Path(path_or_id)

    # Check if local path or HuggingFace ID
    if path.exists():
        model_path = path
    else:
        # Download from HuggingFace
        model_path = await download_from_hub_async(path_or_id)

    # Load config
    config_path = model_path / "config.json"
    async with aiofiles.open(config_path) as f:
        config_data = json.loads(await f.read())

    # Determine model type
    if model_type is None:
        model_type = config_data.get("model_type")
        if model_type is None:
            # Try to infer from architectures
            architectures = config_data.get("architectures", [])
            for arch in architectures:
                # Look up in registry by architecture name
                factory = get_factory_by_architecture(arch)
                if factory:
                    break
            else:
                raise ValueError(
                    "Cannot determine model type. Specify model_type or register architecture."
                )

    # Get factory from registry
    factory = get_factory(model_type)
    if factory is None:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list_models()}")

    # Create config
    config = ModelConfig(**config_data)

    # Create model
    model = factory(config)

    # Load weights
    weights = await load_weights_async(model_path, dtype=dtype)
    model.update(weights)

    return model


def load_model(
    path_or_id: str,
    model_type: str | None = None,
    dtype: str = "float16",
    **kwargs: Any,
) -> Model:
    """
    Load a model synchronously.

    Convenience wrapper around load_model_async.

    Args:
        path_or_id: Local path or HuggingFace model ID
        model_type: Optional override for model type
        dtype: Data type for weights
        **kwargs: Additional arguments

    Returns:
        Loaded model instance
    """
    import asyncio

    return asyncio.run(load_model_async(path_or_id, model_type=model_type, dtype=dtype, **kwargs))


async def load_weights_async(
    model_path: Path,
    dtype: str = "float16",
) -> dict[str, mx.array]:
    """
    Load model weights asynchronously.

    Supports:
    - safetensors (preferred)
    - PyTorch .bin files
    - NPZ files

    Args:
        model_path: Path to model directory
        dtype: Target data type

    Returns:
        Dictionary of weights
    """
    # Determine weight format
    safetensor_path = model_path / "model.safetensors"
    pytorch_path = model_path / "pytorch_model.bin"
    npz_path = model_path / "weights.npz"

    if safetensor_path.exists():
        weights = await load_safetensors_async(safetensor_path)
    elif pytorch_path.exists():
        weights = await load_pytorch_async(pytorch_path)
    elif npz_path.exists():
        weights = await load_npz_async(npz_path)
    else:
        # Try sharded safetensors
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            weights = await load_sharded_safetensors_async(model_path, index_path)
        else:
            raise FileNotFoundError(
                f"No weights found in {model_path}. "
                f"Expected: model.safetensors, pytorch_model.bin, or weights.npz"
            )

    # Convert dtype
    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.float16)

    weights = {
        k: v.astype(target_dtype) if v.dtype in (mx.float32, mx.float16, mx.bfloat16) else v
        for k, v in weights.items()
    }

    return weights


async def load_safetensors_async(path: Path) -> dict[str, mx.array]:
    """Load weights from safetensors file."""
    try:
        import safetensors.numpy as st
    except ImportError as err:
        raise ImportError("safetensors not installed. Run: pip install safetensors") from err

    # Load in thread pool to not block
    import asyncio

    def load():
        data = st.load_file(str(path))
        return {k: mx.array(v) for k, v in data.items()}

    return await asyncio.get_event_loop().run_in_executor(None, load)


async def load_pytorch_async(path: Path) -> dict[str, mx.array]:
    """Load weights from PyTorch .bin file."""
    try:
        import torch
    except ImportError as err:
        raise ImportError("torch not installed for loading .bin files") from err

    import asyncio

    def load():
        data = torch.load(str(path), map_location="cpu")
        return {k: mx.array(v.numpy()) for k, v in data.items()}

    return await asyncio.get_event_loop().run_in_executor(None, load)


async def load_npz_async(path: Path) -> dict[str, mx.array]:
    """Load weights from NPZ file."""
    import asyncio

    import numpy as np

    def load():
        data = np.load(str(path))
        return {k: mx.array(data[k]) for k in data.files}

    return await asyncio.get_event_loop().run_in_executor(None, load)


async def load_sharded_safetensors_async(
    model_path: Path,
    index_path: Path,
) -> dict[str, mx.array]:
    """Load sharded safetensors files."""
    import json

    import aiofiles

    async with aiofiles.open(index_path) as f:
        index = json.loads(await f.read())

    weight_map = index.get("weight_map", {})

    # Group by shard file
    shards: dict[str, list[str]] = {}
    for weight_name, shard_file in weight_map.items():
        if shard_file not in shards:
            shards[shard_file] = []
        shards[shard_file].append(weight_name)

    # Load each shard
    all_weights = {}
    for shard_file in shards:
        shard_path = model_path / shard_file
        shard_weights = await load_safetensors_async(shard_path)
        all_weights.update(shard_weights)

    return all_weights


async def download_from_hub_async(
    model_id: str,
    revision: str = "main",
    cache_dir: str | None = None,
) -> Path:
    """
    Download model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
        revision: Git revision to download
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        ) from err

    import asyncio

    def download():
        return Path(
            snapshot_download(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
            )
        )

    return await asyncio.get_event_loop().run_in_executor(None, download)


def get_factory_by_architecture(architecture: str):
    """
    Get model factory by architecture name.

    Architecture names are like "LlamaForCausalLM", "MistralForCausalLM", etc.
    """
    from .core.registry import get_model_class

    return get_model_class(architecture)


def create_model(
    model_type: str,
    config: ModelConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Model:
    """
    Create a model from type and config.

    Args:
        model_type: Model type (e.g., "llama", "mamba")
        config: Model configuration (or dict to create one)
        **kwargs: Override config values

    Returns:
        Model instance

    Example:
        >>> model = create_model("llama", vocab_size=32000, hidden_size=4096)
    """
    factory = get_factory(model_type)
    if factory is None:
        raise ValueError(f"Unknown model type: {model_type}")

    if config is None:
        config = ModelConfig(**kwargs)
    elif isinstance(config, dict):
        config = ModelConfig(**{**config, **kwargs})
    else:
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return factory(config)


def create_from_preset(
    preset: str,
    model_type: str = "llama",
) -> Model:
    """
    Create model from a preset configuration.

    Args:
        preset: Preset name (e.g., "llama2_7b", "mistral_7b", "mamba_130m")
        model_type: Model type if not inferable from preset

    Returns:
        Model instance

    Example:
        >>> model = create_from_preset("llama2_7b")
        >>> model = create_from_preset("mamba_130m")
    """
    # Try to get preset from the family config
    if preset.startswith("llama") or preset.startswith("mistral") or preset.startswith("code"):
        from .families.llama import LlamaConfig, LlamaForCausalLM

        preset_method = getattr(LlamaConfig, preset, None)
        if preset_method:
            config = preset_method()
            return LlamaForCausalLM(config)

    if preset.startswith("mamba"):
        from .families.mamba import MambaConfig, MambaForCausalLM

        preset_method = getattr(MambaConfig, preset, None)
        if preset_method:
            config = preset_method()
            return MambaForCausalLM(config)

    raise ValueError(f"Unknown preset: {preset}")
