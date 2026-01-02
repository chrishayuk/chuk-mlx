"""Tests for GPT-2 weight conversion utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np

from chuk_lazarus.models_v2.families.gpt2.convert import (
    _map_weight_name,
    _reverse_map_weight_name,
    convert_hf_weights,
    get_num_params,
    load_hf_config,
    print_weight_shapes,
)


class TestLoadHfConfig:
    """Tests for load_hf_config function."""

    def test_load_valid_config(self):
        """Test loading a valid config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "model_type": "gpt2",
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(tmpdir)

            assert result["model_type"] == "gpt2"
            assert result["n_embd"] == 768


class TestWeightNameMapping:
    """Tests for weight name mapping functions."""

    def test_direct_weight_map_wte(self):
        """Test direct mapping for token embeddings."""
        result = _map_weight_name("wte.weight")
        assert result == "model.embed_tokens.weight.weight"

    def test_direct_weight_map_wpe(self):
        """Test direct mapping for position embeddings."""
        result = _map_weight_name("wpe.weight")
        assert result == "model.position_embeddings.weight"

    def test_direct_weight_map_ln_f(self):
        """Test direct mapping for final layer norm."""
        result = _map_weight_name("ln_f.weight")
        assert result == "model.norm.weight"
        result = _map_weight_name("ln_f.bias")
        assert result == "model.norm.bias"

    def test_layer_weight_map_ln_1(self):
        """Test layer-level mapping for pre-attention layer norm."""
        result = _map_weight_name("h.0.ln_1.weight")
        assert result == "model.layers.0.input_layernorm.weight"
        result = _map_weight_name("h.5.ln_1.bias")
        assert result == "model.layers.5.input_layernorm.bias"

    def test_layer_weight_map_ln_2(self):
        """Test layer-level mapping for post-attention layer norm."""
        result = _map_weight_name("h.0.ln_2.weight")
        assert result == "model.layers.0.post_attention_layernorm.weight"
        result = _map_weight_name("h.3.ln_2.bias")
        assert result == "model.layers.3.post_attention_layernorm.bias"

    def test_layer_weight_map_attention(self):
        """Test layer-level mapping for attention weights."""
        # Combined QKV projection
        result = _map_weight_name("h.0.attn.c_attn.weight")
        assert result == "model.layers.0.self_attn.c_attn.weight"
        result = _map_weight_name("h.2.attn.c_attn.bias")
        assert result == "model.layers.2.self_attn.c_attn.bias"

        # Output projection
        result = _map_weight_name("h.0.attn.c_proj.weight")
        assert result == "model.layers.0.self_attn.o_proj.weight"

    def test_layer_weight_map_mlp(self):
        """Test layer-level mapping for MLP weights."""
        result = _map_weight_name("h.0.mlp.c_fc.weight")
        assert result == "model.layers.0.mlp.c_fc.weight"
        result = _map_weight_name("h.1.mlp.c_fc.bias")
        assert result == "model.layers.1.mlp.c_fc.bias"
        result = _map_weight_name("h.0.mlp.c_proj.weight")
        assert result == "model.layers.0.mlp.c_proj.weight"

    def test_unrecognized_weight_returns_none(self):
        """Test that unrecognized weights return None."""
        result = _map_weight_name("some.random.weight")
        assert result is None

        result = _map_weight_name("attn.bias")  # GPT-2 attention bias buffer
        assert result is None


class TestReverseWeightNameMapping:
    """Tests for reverse weight name mapping."""

    def test_reverse_map_direct(self):
        """Test reverse mapping for direct weights."""
        result = _reverse_map_weight_name("model.embed_tokens.weight.weight")
        assert result == "wte.weight"

        result = _reverse_map_weight_name("model.position_embeddings.weight")
        assert result == "wpe.weight"

        result = _reverse_map_weight_name("model.norm.weight")
        assert result == "ln_f.weight"

    def test_reverse_map_layer_norms(self):
        """Test reverse mapping for layer norms."""
        result = _reverse_map_weight_name("model.layers.0.input_layernorm.weight")
        assert result == "h.0.ln_1.weight"

        result = _reverse_map_weight_name("model.layers.2.post_attention_layernorm.bias")
        assert result == "h.2.ln_2.bias"

    def test_reverse_map_attention(self):
        """Test reverse mapping for attention weights."""
        result = _reverse_map_weight_name("model.layers.0.self_attn.c_attn.weight")
        assert result == "h.0.attn.c_attn.weight"

        result = _reverse_map_weight_name("model.layers.1.self_attn.o_proj.weight")
        assert result == "h.1.attn.c_proj.weight"

    def test_reverse_map_mlp(self):
        """Test reverse mapping for MLP weights."""
        result = _reverse_map_weight_name("model.layers.0.mlp.c_fc.weight")
        assert result == "h.0.mlp.c_fc.weight"

        result = _reverse_map_weight_name("model.layers.0.mlp.c_proj.bias")
        assert result == "h.0.mlp.c_proj.bias"


class TestConvertHfWeights:
    """Tests for HuggingFace weight conversion."""

    def test_convert_basic_weights(self):
        """Test basic weight conversion."""
        hf_weights = {
            "wte.weight": np.random.randn(1000, 64).astype(np.float32),
            "wpe.weight": np.random.randn(256, 64).astype(np.float32),
            "ln_f.weight": np.random.randn(64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight.weight" in converted
        assert "model.position_embeddings.weight" in converted
        assert "model.norm.weight" in converted

    def test_convert_layer_weights(self):
        """Test layer weight conversion."""
        hf_weights = {
            "h.0.ln_1.weight": np.random.randn(64).astype(np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(192, 64).astype(np.float32),
            "h.0.mlp.c_fc.weight": np.random.randn(256, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.layers.0.input_layernorm.weight" in converted
        assert "model.layers.0.self_attn.c_attn.weight" in converted
        assert "model.layers.0.mlp.c_fc.weight" in converted

    def test_convert_skips_unmapped_weights(self):
        """Test that unmapped weights are skipped."""
        hf_weights = {
            "wte.weight": np.random.randn(1000, 64).astype(np.float32),
            "attn.bias": np.random.randn(100, 100).astype(np.float32),  # Attention bias buffer
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight.weight" in converted
        assert len(converted) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_num_params(self):
        """Test parameter counting."""
        weights = {
            "a": np.random.randn(10, 10).astype(np.float32),
            "b": np.random.randn(5).astype(np.float32),
        }

        count = get_num_params(weights)

        assert count == 100 + 5

    def test_get_num_params_empty(self):
        """Test parameter counting with empty dict."""
        count = get_num_params({})
        assert count == 0

    def test_print_weight_shapes(self, capsys):
        """Test weight shape printing."""
        weights = {
            "b": np.random.randn(5, 5).astype(np.float32),
            "a": np.random.randn(10).astype(np.float32),
        }

        print_weight_shapes(weights)

        captured = capsys.readouterr()
        assert "a:" in captured.out
        assert "b:" in captured.out
        assert "(10,)" in captured.out
        assert "(5, 5)" in captured.out
