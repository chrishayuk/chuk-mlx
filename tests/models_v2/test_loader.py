"""
Tests for model loader module.

Tests create_model, create_from_preset, and model loading functions.
"""

import pytest

from chuk_lazarus.models_v2.loader import create_model
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

    def test_code_llama_via_config(self):
        """Test creating Code Llama model via direct config."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.code_llama_7b()
        model = LlamaForCausalLM(config)
        assert model is not None


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


class TestLoadModelAsync:
    """Tests for async model loading functions."""

    @pytest.mark.asyncio
    async def test_load_model_async_import(self):
        """Test that load_model_async function exists and is importable."""
        from chuk_lazarus.models_v2.loader import load_model_async

        assert load_model_async is not None
        assert callable(load_model_async)

    @pytest.mark.asyncio
    async def test_load_model_with_lora_async_import(self):
        """Test that load_model_with_lora_async function exists and is importable."""
        from chuk_lazarus.models_v2.loader import load_model_with_lora_async

        assert load_model_with_lora_async is not None
        assert callable(load_model_with_lora_async)


class TestLoadModelTuple:
    """Tests for load_model_tuple function."""

    def test_load_model_tuple_import(self):
        """Test that load_model_tuple function exists and is importable."""
        from chuk_lazarus.models_v2.loader import load_model_tuple

        assert load_model_tuple is not None
        assert callable(load_model_tuple)


class TestSaveAdapter:
    """Tests for save_adapter function."""

    def test_save_adapter_import(self):
        """Test that save_adapter function exists and is importable."""
        from chuk_lazarus.models_v2.loader import save_adapter

        assert save_adapter is not None
        assert callable(save_adapter)
