"""
HuggingFace model loading utilities.

Consolidates common patterns for downloading, loading tokenizers,
and loading weights from HuggingFace models.

Design principles:
- Async native where applicable
- Pydantic models for configuration
- No dictionary goop - use typed structures
- No magic strings - use enums/constants
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mlx.core as mx
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class DType(str, Enum):
    """Supported data types for model weights."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

    def to_mlx(self) -> mx.Dtype:
        """Convert to MLX dtype."""
        mapping = {
            DType.FLOAT16: mx.float16,
            DType.FLOAT32: mx.float32,
            DType.BFLOAT16: mx.bfloat16,
        }
        return mapping[self]


class DownloadConfig(BaseModel):
    """Configuration for model download."""

    model_id: str = Field(..., description="HuggingFace model ID")
    cache_dir: Path | None = Field(None, description="Local cache directory")
    allow_patterns: list[str] = Field(
        default_factory=lambda: ["*.json", "*.safetensors", "*.model", "tokenizer*"],
        description="File patterns to download",
    )
    prefer_sharded: bool = Field(True, description="Prefer sharded safetensors over consolidated")


class LoadedWeights(BaseModel):
    """Container for loaded model weights with metadata."""

    model_config = {"arbitrary_types_allowed": True}

    weights: dict[str, mx.array] = Field(..., description="Weight tensors by name")
    dtype: DType = Field(DType.BFLOAT16, description="Target dtype")
    source_path: Path = Field(..., description="Path weights were loaded from")
    tensor_count: int = Field(..., description="Number of tensors loaded")

    @property
    def layer_count(self) -> int:
        """Infer number of layers from weight names."""
        max_idx = -1
        for name in self.weights:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        max_idx = max(max_idx, int(parts[i + 1]))
                    except ValueError:
                        pass
        return max_idx + 1 if max_idx >= 0 else 0


@dataclass
class DownloadResult:
    """Result of a model download operation."""

    model_path: Path
    model_id: str
    is_cached: bool = False


@runtime_checkable
class WeightConverter(Protocol):
    """Protocol for weight name converters."""

    def convert(self, hf_name: str) -> str | None:
        """Convert HuggingFace weight name to framework format.

        Returns None to skip the weight.
        """
        ...


class StandardWeightConverter:
    """Standard weight name converter for transformer models."""

    def __init__(self, tie_word_embeddings: bool = False):
        self.tie_word_embeddings = tie_word_embeddings

    def convert(self, hf_name: str) -> str | None:
        """Convert HuggingFace weight name to framework format."""
        # Embeddings
        if hf_name == "model.embed_tokens.weight":
            return "model.embed_tokens.weight.weight"

        # Final layer norm
        if hf_name == "model.norm.weight":
            return "model.norm.weight"

        # LM head
        if hf_name == "lm_head.weight":
            if self.tie_word_embeddings:
                return None
            return "lm_head.lm_head.weight"

        # Layer pattern
        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", hf_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = layer_match.group(2)

            # Skip rotary embeddings - computed dynamically
            if "rotary_emb" in rest:
                return None

            return f"model.layers.{layer_idx}.{rest}"

        return None


class HFLoader:
    """High-level loader for HuggingFace models."""

    def __init__(self, config: DownloadConfig | None = None):
        self._config = config

    @staticmethod
    def download(
        model_id: str,
        cache_dir: Path | str | None = None,
        prefer_sharded: bool = True,
    ) -> DownloadResult:
        """Download model from HuggingFace Hub synchronously.

        Args:
            model_id: HuggingFace model ID
            cache_dir: Optional cache directory
            prefer_sharded: Prefer sharded over consolidated safetensors

        Returns:
            DownloadResult with path and metadata
        """
        try:
            from huggingface_hub import list_repo_files, snapshot_download
        except ImportError as err:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            ) from err

        print(f"Downloading {model_id}...")

        # Determine ignore patterns
        ignore_patterns: list[str] = []
        if prefer_sharded:
            try:
                files = list_repo_files(model_id)
                has_sharded = any("model-0" in f and f.endswith(".safetensors") for f in files)
                has_consolidated = any(f == "consolidated.safetensors" for f in files)

                if has_sharded and has_consolidated:
                    ignore_patterns.append("consolidated.safetensors")
                    print("  (Skipping consolidated.safetensors - using sharded files)")
            except Exception:
                pass

        path = snapshot_download(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
            ignore_patterns=ignore_patterns if ignore_patterns else None,
        )

        return DownloadResult(
            model_path=Path(path),
            model_id=model_id,
        )

    @staticmethod
    async def download_async(
        model_id: str,
        cache_dir: Path | str | None = None,
        prefer_sharded: bool = True,
    ) -> DownloadResult:
        """Download model from HuggingFace Hub asynchronously.

        Runs the download in a thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: HFLoader.download(model_id, cache_dir, prefer_sharded)
        )

    @staticmethod
    def load_tokenizer(model_path: Path | str) -> PreTrainedTokenizer:
        """Load tokenizer from model path.

        Args:
            model_path: Path to model directory

        Returns:
            Tokenizer with pad_token configured
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as err:
            raise ImportError("transformers not installed. Run: pip install transformers") from err

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @staticmethod
    def load_weights(
        model_path: Path,
        dtype: DType = DType.BFLOAT16,
        converter: WeightConverter | None = None,
    ) -> LoadedWeights:
        """Load weights from safetensors files.

        Args:
            model_path: Path to model directory
            dtype: Target dtype for weights
            converter: Optional weight name converter

        Returns:
            LoadedWeights container with tensors and metadata
        """
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        target_dtype = dtype.to_mlx()

        # Use default converter if none provided
        if converter is None:
            converter = StandardWeightConverter()

        # Load and convert weights
        converted_weights: dict[str, mx.array] = {}

        for sf_path in safetensor_files:
            print(f"  Loading {sf_path.name}...")
            raw_weights = mx.load(str(sf_path))

            for hf_name, weight in raw_weights.items():
                # Convert name
                our_name = converter.convert(hf_name)
                if our_name is None:
                    continue

                # Convert dtype
                if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
                    weight = weight.astype(target_dtype)

                converted_weights[our_name] = weight

        return LoadedWeights(
            weights=converted_weights,
            dtype=dtype,
            source_path=model_path,
            tensor_count=len(converted_weights),
        )

    @staticmethod
    def build_nested_weights(loaded: LoadedWeights) -> dict:
        """Convert flat weights to nested structure for model.update().

        Args:
            loaded: LoadedWeights from load_weights()

        Returns:
            Nested dictionary structure
        """
        flat_weights = loaded.weights

        # Find maximum layer index
        max_layer_idx = loaded.layer_count - 1

        # Build nested structure
        nested: dict = {}
        for name, weight in flat_weights.items():
            parts = name.split(".")
            current = nested

            i = 0
            while i < len(parts) - 1:
                part = parts[i]

                if part == "layers":
                    if part not in current:
                        current[part] = [{} for _ in range(max_layer_idx + 1)]
                    layer_idx = int(parts[i + 1])
                    current = current[part][layer_idx]
                    i += 2
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    i += 1

            current[parts[-1]] = weight

        return nested
