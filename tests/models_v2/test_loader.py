"""
Tests for model loader module.

Tests create_model, create_from_preset, and get_factory_by_architecture.
Async loading tests require mock file system.
"""

import pytest

from chuk_lazarus.models_v2.loader import (
    create_model,
    get_factory_by_architecture,
)
from chuk_lazarus.models_v2.models.base import Model


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_llama(self):
        """Test creating a Llama model."""
        model = create_model(
            "llama",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        assert isinstance(model, Model)

    def test_create_mamba(self):
        """Test creating a Mamba model directly (requires MambaConfig)."""
        # MambaForCausalLM requires MambaConfig, not ModelConfig
        # So we test directly with the family classes
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)
        assert isinstance(model, Model)

    def test_create_with_dict_config(self):
        """Test creating with dict config."""
        config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        }
        model = create_model("llama", config=config)
        assert isinstance(model, Model)

    def test_create_with_config_object(self):
        """Test creating with ModelConfig object."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = create_model("llama", config=config)
        assert isinstance(model, Model)

    def test_create_with_kwargs_override(self):
        """Test creating with kwargs overriding config."""
        config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        }
        model = create_model("llama", config=config, vocab_size=2000)
        assert isinstance(model, Model)

    def test_create_unknown_model_type(self):
        """Test error for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown_model_type")


class TestGetFactoryByArchitecture:
    """Tests for get_factory_by_architecture function."""

    def test_llama_architecture(self):
        """Test getting factory for LlamaForCausalLM."""
        factory = get_factory_by_architecture("LlamaForCausalLM")
        assert factory is not None

    def test_mamba_architecture(self):
        """Test getting factory for MambaForCausalLM or similar."""
        _ = get_factory_by_architecture("MambaForCausalLM")
        # May or may not be registered
        # Just verify it returns something or None without error

    def test_unknown_architecture(self):
        """Test getting factory for unknown architecture."""
        factory = get_factory_by_architecture("UnknownArchitecture")
        assert factory is None


class TestLoadWeightsAsync:
    """Tests for async weight loading."""

    @pytest.mark.asyncio
    async def test_load_npz_async(self):
        """Test loading NPZ weights."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_npz_async

        # Create temp NPZ file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, weight1=np.random.randn(10, 10).astype(np.float32))
            weights = await load_npz_async(Path(f.name))

        assert "weight1" in weights
        assert weights["weight1"].shape == (10, 10)


class TestLoadWeightsErrors:
    """Tests for weight loading error handling."""

    @pytest.mark.asyncio
    async def test_no_weights_found(self):
        """Test error when no weights found."""
        import tempfile
        from pathlib import Path

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Create empty temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No weights found"):
                await load_weights_async(Path(tmpdir))


class TestLoadWeightsAsyncDtypes:
    """Tests for weight loading with different dtypes."""

    @pytest.mark.asyncio
    async def test_load_npz_float32(self):
        """Test loading NPZ weights with float32 dtype."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Create temp directory with NPZ file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.npz"
            np.savez(str(path), weight1=np.random.randn(10, 10).astype(np.float32))
            weights = await load_weights_async(Path(tmpdir), dtype="float32")

        assert "weight1" in weights

    @pytest.mark.asyncio
    async def test_load_npz_float16(self):
        """Test loading NPZ weights with float16 dtype conversion."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Create temp directory with NPZ file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.npz"
            np.savez(str(path), weight1=np.random.randn(10, 10).astype(np.float32))
            weights = await load_weights_async(Path(tmpdir), dtype="float16")

        assert "weight1" in weights

    @pytest.mark.asyncio
    async def test_load_npz_bfloat16(self):
        """Test loading NPZ weights with bfloat16 dtype conversion."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Create temp directory with NPZ file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.npz"
            np.savez(str(path), weight1=np.random.randn(10, 10).astype(np.float32))
            weights = await load_weights_async(Path(tmpdir), dtype="bfloat16")

        assert "weight1" in weights


class TestCreateFromPreset:
    """Tests for create_from_preset function."""

    def test_llama_preset(self):
        """Test creating model from llama preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        model = create_from_preset("llama2_7b", model_type="llama")
        assert model is not None

    def test_mistral_preset(self):
        """Test creating model from mistral preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        model = create_from_preset("mistral_7b", model_type="llama")
        assert model is not None

    def test_mamba_preset(self):
        """Test creating model from mamba preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        model = create_from_preset("mamba_130m", model_type="mamba")
        assert model is not None

    def test_llama3_preset(self):
        """Test creating model from llama3 preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        model = create_from_preset("llama3_8b", model_type="llama")
        assert model is not None

    def test_unknown_preset_error(self):
        """Test error for unknown preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            create_from_preset("nonexistent_preset")


class TestLoadModelSync:
    """Tests for synchronous model loading."""

    def test_load_model_sync_with_create_model(self):
        """Test creating model with create_model directly."""
        from chuk_lazarus.models_v2.loader import create_model

        # Create model directly without loading weights
        model = create_model(
            "llama",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

        assert model is not None


class TestLoadModelAsyncFunctions:
    """Tests for async model loading functions."""

    @pytest.mark.asyncio
    async def test_load_safetensors_async(self):
        """Test loading safetensors format weights."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_safetensors_async

        # Check if safetensors is available
        try:
            import safetensors  # noqa: F401
        except ImportError:
            pytest.skip("safetensors not installed")

        from safetensors.numpy import save_file

        # Create temp safetensors file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            weights = {"weight1": np.random.randn(10, 10).astype(np.float32)}
            save_file(weights, str(path))

            loaded = await load_safetensors_async(path)

        assert "weight1" in loaded

    @pytest.mark.asyncio
    async def test_download_from_hub_skipped(self):
        """Test that download_from_hub requires network and is skipped."""
        # This just tests the function exists
        from chuk_lazarus.models_v2.loader import download_from_hub_async

        assert download_from_hub_async is not None

    @pytest.mark.asyncio
    async def test_load_weights_async_safetensors(self):
        """Test load_weights_async with safetensors format."""
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Check if safetensors is available
        try:
            from safetensors.numpy import save_file
        except ImportError:
            pytest.skip("safetensors not installed")

        # Create temp directory with safetensors file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.safetensors"
            weights = {"weight1": np.random.randn(10, 10).astype(np.float32)}
            save_file(weights, str(path))

            loaded = await load_weights_async(Path(tmpdir), dtype="float16")

        assert "weight1" in loaded

    @pytest.mark.asyncio
    async def test_load_sharded_safetensors_async(self):
        """Test loading sharded safetensors files."""
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Check if safetensors is available
        try:
            from safetensors.numpy import save_file
        except ImportError:
            pytest.skip("safetensors not installed")

        # Create temp directory with sharded safetensors
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create index file
            index = {
                "weight_map": {
                    "layer.0.weight": "model-00001-of-00002.safetensors",
                    "layer.1.weight": "model-00002-of-00002.safetensors",
                }
            }
            index_path = tmppath / "model.safetensors.index.json"
            with open(index_path, "w") as f:
                json.dump(index, f)

            # Create shard files
            save_file(
                {"layer.0.weight": np.random.randn(10, 10).astype(np.float32)},
                str(tmppath / "model-00001-of-00002.safetensors"),
            )
            save_file(
                {"layer.1.weight": np.random.randn(10, 10).astype(np.float32)},
                str(tmppath / "model-00002-of-00002.safetensors"),
            )

            loaded = await load_weights_async(tmppath, dtype="float32")

        assert "layer.0.weight" in loaded
        assert "layer.1.weight" in loaded


class TestLoaderWithMockedHub:
    """Tests for hub download functionality."""

    @pytest.mark.asyncio
    async def test_download_from_hub_import_error(self):
        """Test proper error when huggingface_hub not available."""
        import sys

        # Temporarily mock huggingface_hub as unavailable
        original = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None

        try:
            # Import fresh
            import importlib

            from chuk_lazarus.models_v2 import loader

            importlib.reload(loader)

            # This should raise ImportError
            with pytest.raises(ImportError, match="huggingface_hub"):
                await loader.download_from_hub_async("test/model")
        finally:
            # Restore
            if original:
                sys.modules["huggingface_hub"] = original
            else:
                sys.modules.pop("huggingface_hub", None)

    def test_codellama_preset(self):
        """Test creating model from codellama preset."""
        from chuk_lazarus.models_v2.loader import create_from_preset

        model = create_from_preset("code_llama_7b", model_type="llama")
        assert model is not None


class TestLoadWeightsAsyncIntegerWeights:
    """Tests for loading weights that include integer types."""

    @pytest.mark.asyncio
    async def test_load_npz_with_int_weights(self):
        """Test loading NPZ with integer weights (should not be converted)."""
        import tempfile
        from pathlib import Path

        import mlx.core as mx
        import numpy as np

        from chuk_lazarus.models_v2.loader import load_weights_async

        # Create temp directory with NPZ file containing int weights
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.npz"
            np.savez(
                str(path),
                float_weight=np.random.randn(10, 10).astype(np.float32),
                int_weight=np.array([1, 2, 3, 4, 5], dtype=np.int64),
            )
            weights = await load_weights_async(Path(tmpdir), dtype="float16")

        # Float weights should be converted
        assert weights["float_weight"].dtype == mx.float16
        # Int weights should remain int
        assert weights["int_weight"].dtype == mx.int64
