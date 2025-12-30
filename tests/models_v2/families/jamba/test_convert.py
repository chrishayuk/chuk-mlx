"""
Tests for Jamba weight conversion utilities.
"""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from chuk_lazarus.models_v2.families.jamba.convert import (
    JAMBA_WEIGHT_MAP,
    _map_weight_name,
    _reverse_map_weight_name,
    convert_hf_weights,
    convert_mlx_to_hf,
    get_num_params,
    load_hf_config,
    load_weights,
    print_weight_shapes,
)


class TestLoadHfConfig:
    """Tests for load_hf_config function."""

    def test_load_valid_config(self):
        """Test loading a valid config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "vocab_size": 65536,
                "model_type": "jamba",
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(tmpdir)

            assert result["hidden_size"] == 4096
            assert result["model_type"] == "jamba"

    def test_load_config_with_path_object(self):
        """Test loading config with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_data = {"model_type": "jamba", "num_experts": 16}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            result = load_hf_config(Path(tmpdir))

            assert result["model_type"] == "jamba"
            assert result["num_experts"] == 16


class TestLoadWeights:
    """Tests for load_weights function."""

    def test_load_empty_directory(self):
        """Test loading from directory with no safetensor files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_weights(tmpdir)
            assert result == {}

    def test_load_single_safetensor(self):
        """Test loading from directory with single safetensor file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.embed_tokens.weight": mx.random.normal((100, 64)),
            }
            sf_path = Path(tmpdir) / "model.safetensors"
            mx.save_safetensors(str(sf_path), weights)

            result = load_weights(tmpdir)

            # Maps to weight.weight because TokenEmbedding wraps nn.Embedding
            assert "model.embed_tokens.weight.weight" in result

    def test_load_multiple_safetensors(self):
        """Test loading from directory with multiple safetensor files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights1 = {"model.embed_tokens.weight": mx.random.normal((100, 64))}
            weights2 = {"model.final_layernorm.weight": mx.random.normal((64,))}

            mx.save_safetensors(str(Path(tmpdir) / "model-00001-of-00002.safetensors"), weights1)
            mx.save_safetensors(str(Path(tmpdir) / "model-00002-of-00002.safetensors"), weights2)

            result = load_weights(tmpdir)

            # embed_tokens maps to weight.weight because TokenEmbedding wraps nn.Embedding
            assert "model.embed_tokens.weight.weight" in result
            assert "model.norm.weight" in result  # Mapped from final_layernorm

    def test_load_with_layer_weights(self):
        """Test loading with layer-specific weights that need mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.layers.0.mamba.in_proj.weight": mx.random.normal((128, 64)),
                "model.layers.0.feed_forward.router.weight": mx.random.normal((16, 64)),
            }
            sf_path = Path(tmpdir) / "model.safetensors"
            mx.save_safetensors(str(sf_path), weights)

            result = load_weights(tmpdir)

            # Verify Mamba naming conversion
            assert "model.layers.0.mamba.ssm.in_proj.weight" in result
            # Verify MoE router naming conversion
            assert "model.layers.0.feed_forward.router.gate.weight" in result

    def test_load_skips_unmapped_weights(self):
        """Test that unmapped weights are skipped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = {
                "model.embed_tokens.weight": mx.random.normal((100, 64)),
                "rotary_emb.inv_freq": mx.random.normal((32,)),  # Should be skipped
            }
            sf_path = Path(tmpdir) / "model.safetensors"
            mx.save_safetensors(str(sf_path), weights)

            result = load_weights(tmpdir)

            # embed_tokens maps to weight.weight
            assert "model.embed_tokens.weight.weight" in result
            assert "rotary_emb.inv_freq" not in result
            assert len(result) == 1


class TestWeightNameMapping:
    """Tests for weight name mapping functions."""

    def test_direct_weight_map_embed_tokens(self):
        """Test direct mapping for embed_tokens (wraps nn.Embedding)."""
        result = _map_weight_name("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight.weight"

    def test_direct_weight_map_norm(self):
        """Test direct mapping for final_layernorm -> norm."""
        result = _map_weight_name("model.final_layernorm.weight")
        assert result == "model.norm.weight"

    def test_direct_weight_map_lm_head(self):
        """Test direct mapping for lm_head."""
        result = _map_weight_name("lm_head.weight")
        assert result == "lm_head.lm_head.weight"

    def test_layer_weight_map_attention(self):
        """Test layer-level mapping for attention weights."""
        result = _map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

        result = _map_weight_name("model.layers.5.self_attn.k_proj.weight")
        assert result == "model.layers.5.self_attn.k_proj.weight"

    def test_layer_weight_map_mamba(self):
        """Test layer-level mapping for Mamba weights."""
        result = _map_weight_name("model.layers.0.mamba.in_proj.weight")
        assert result == "model.layers.0.mamba.ssm.in_proj.weight"

        result = _map_weight_name("model.layers.3.mamba.out_proj.weight")
        assert result == "model.layers.3.mamba.ssm.out_proj.weight"

    def test_layer_weight_map_moe_router(self):
        """Test layer-level mapping for MoE router."""
        result = _map_weight_name("model.layers.0.feed_forward.router.weight")
        assert result == "model.layers.0.feed_forward.router.gate.weight"

    def test_layer_weight_map_moe_experts(self):
        """Test layer-level mapping for MoE experts passes through."""
        result = _map_weight_name("model.layers.0.feed_forward.experts.0.gate_proj.weight")
        assert result == "model.layers.0.feed_forward.experts.0.gate_proj.weight"

    def test_layer_weight_map_layernorms(self):
        """Test layer-level mapping for normalization layers."""
        result = _map_weight_name("model.layers.0.input_layernorm.weight")
        assert result == "model.layers.0.input_layernorm.weight"

        result = _map_weight_name("model.layers.0.pre_ff_layernorm.weight")
        assert result == "model.layers.0.pre_ff_layernorm.weight"

    def test_unrecognized_weight_returns_none(self):
        """Test that unrecognized top-level weights return None."""
        result = _map_weight_name("some.random.weight")
        assert result is None

        result = _map_weight_name("rotary_emb.inv_freq")
        assert result is None

    def test_maps_jamba_mamba2_layernorms(self):
        """Test that Jamba's Mamba2-style layernorms are mapped correctly.

        These layernorms are used by Jamba Reasoning 3B (Mamba2-based models).
        They enable additional normalization within the SSM computation.
        """
        # dt_layernorm, b_layernorm, c_layernorm are mapped to mamba.ssm.*
        result = _map_weight_name("model.layers.0.mamba.dt_layernorm.weight")
        assert result == "model.layers.0.mamba.ssm.dt_layernorm.weight"

        result = _map_weight_name("model.layers.0.mamba.b_layernorm.weight")
        assert result == "model.layers.0.mamba.ssm.b_layernorm.weight"

        result = _map_weight_name("model.layers.0.mamba.c_layernorm.weight")
        assert result == "model.layers.0.mamba.ssm.c_layernorm.weight"


class TestReverseWeightNameMapping:
    """Tests for reverse weight name mapping."""

    def test_reverse_map_direct(self):
        """Test reverse mapping for direct weights."""
        # embed_tokens.weight.weight maps back to embed_tokens.weight
        result = _reverse_map_weight_name("model.embed_tokens.weight.weight")
        assert result == "model.embed_tokens.weight"

        result = _reverse_map_weight_name("model.norm.weight")
        assert result == "model.final_layernorm.weight"

        result = _reverse_map_weight_name("lm_head.lm_head.weight")
        assert result == "lm_head.weight"

    def test_reverse_map_mamba(self):
        """Test reverse mapping for Mamba weights."""
        result = _reverse_map_weight_name("model.layers.0.mamba.ssm.in_proj.weight")
        assert result == "model.layers.0.mamba.in_proj.weight"

    def test_reverse_map_moe_router(self):
        """Test reverse mapping for MoE router."""
        result = _reverse_map_weight_name("model.layers.0.feed_forward.router.gate.weight")
        assert result == "model.layers.0.feed_forward.router.weight"

    def test_reverse_map_layer_patterns(self):
        """Test reverse mapping for layer patterns."""
        result = _reverse_map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

    def test_reverse_map_unrecognized_returns_none(self):
        """Test unrecognized patterns return None."""
        result = _reverse_map_weight_name("some.random.weight")
        assert result is None


class TestConvertHfWeights:
    """Tests for HuggingFace weight conversion."""

    def test_convert_basic_weights(self):
        """Test basic weight conversion."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.final_layernorm.weight": np.random.randn(64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        # embed_tokens maps to weight.weight because TokenEmbedding wraps nn.Embedding
        assert "model.embed_tokens.weight.weight" in converted
        assert "model.norm.weight" in converted
        assert "lm_head.lm_head.weight" in converted

    def test_convert_layer_weights(self):
        """Test layer weight conversion."""
        hf_weights = {
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "model.layers.0.mamba.in_proj.weight": np.random.randn(128, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.layers.0.self_attn.q_proj.weight" in converted
        assert "model.layers.0.mamba.ssm.in_proj.weight" in converted

    def test_convert_skips_unmapped_weights(self):
        """Test that unmapped weights are skipped."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "rotary_emb.inv_freq": np.random.randn(32).astype(np.float32),  # Should be skipped
        }

        converted = convert_hf_weights(hf_weights)

        # embed_tokens maps to weight.weight
        assert "model.embed_tokens.weight.weight" in converted
        assert "rotary_emb.inv_freq" not in converted
        assert len(converted) == 1

    def test_convert_with_tied_embeddings(self):
        """Test conversion with tied word embeddings."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights, tie_word_embeddings=True)

        # embed_tokens maps to weight.weight
        assert "model.embed_tokens.weight.weight" in converted
        assert "lm_head.lm_head.weight" not in converted  # Should be skipped when tied
        assert len(converted) == 1


class TestConvertMlxToHf:
    """Tests for MLX to HuggingFace conversion."""

    def test_convert_numpy_weights(self):
        """Test conversion of numpy arrays."""
        mlx_weights = {
            # Our model uses weight.weight for embeddings
            "model.embed_tokens.weight.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.final_layernorm.weight" in converted

    def test_convert_layer_weights_to_hf(self):
        """Test layer weight conversion to HF format."""
        mlx_weights = {
            "model.layers.0.mamba.ssm.in_proj.weight": np.random.randn(128, 64).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.layers.0.mamba.in_proj.weight" in converted

    def test_convert_object_with_array_interface(self):
        """Test conversion of objects with __array__ method."""

        class ArrayLike:
            def __init__(self, data):
                self._data = data

            def __array__(self):
                return self._data

        mlx_weights = {
            # Our model uses weight.weight for embeddings
            "model.embed_tokens.weight.weight": ArrayLike(
                np.random.randn(100, 64).astype(np.float32)
            ),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert isinstance(converted["model.embed_tokens.weight"], np.ndarray)

    def test_convert_object_with_numpy_method(self):
        """Test conversion of objects with .numpy() method (like MLX arrays)."""

        class MlxLike:
            def __init__(self, data):
                self._data = data

            def numpy(self):
                return self._data

        mlx_weights = {
            # Our model uses weight.weight for embeddings
            "model.embed_tokens.weight.weight": MlxLike(
                np.random.randn(100, 64).astype(np.float32)
            ),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert isinstance(converted["model.embed_tokens.weight"], np.ndarray)

    def test_convert_skips_unmapped_weights(self):
        """Test that unmapped MLX weights are skipped."""
        mlx_weights = {
            # Our model uses weight.weight for embeddings
            "model.embed_tokens.weight.weight": np.random.randn(100, 64).astype(np.float32),
            "some.random.weight": np.random.randn(64).astype(np.float32),  # Should be skipped
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert "some.random.weight" not in converted
        assert len(converted) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_num_params(self):
        """Test parameter counting."""
        weights = {
            "weight1": np.random.randn(100, 64).astype(np.float32),  # 6400
            "weight2": np.random.randn(64).astype(np.float32),  # 64
        }

        total = get_num_params(weights)
        assert total == 6400 + 64

    def test_get_num_params_empty(self):
        """Test parameter counting with empty dict."""
        total = get_num_params({})
        assert total == 0

    def test_print_weight_shapes(self, capsys):
        """Test weight shape printing."""
        weights = {
            "a.weight": np.random.randn(100, 64).astype(np.float32),
            "b.weight": np.random.randn(64).astype(np.float32),
        }

        print_weight_shapes(weights)

        captured = capsys.readouterr()
        assert "a.weight: (100, 64)" in captured.out
        assert "b.weight: (64,)" in captured.out


class TestWeightMaps:
    """Tests for weight map constants."""

    def test_jamba_weight_map_completeness(self):
        """Test that JAMBA_WEIGHT_MAP has expected keys."""
        assert "model.embed_tokens.weight" in JAMBA_WEIGHT_MAP
        assert "model.final_layernorm.weight" in JAMBA_WEIGHT_MAP
        assert "lm_head.weight" in JAMBA_WEIGHT_MAP
