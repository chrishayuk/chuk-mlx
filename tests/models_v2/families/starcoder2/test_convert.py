"""
Tests for StarCoder2 weight conversion utilities.
"""

import numpy as np

from chuk_lazarus.models_v2.families.starcoder2.convert import (
    STARCODER2_WEIGHT_MAP,
    _map_weight_name,
    _reverse_map_weight_name,
    convert_hf_weights,
    convert_mlx_to_hf,
    get_num_params,
    print_weight_shapes,
)


class TestWeightMapping:
    """Tests for weight name mapping."""

    def test_direct_mapping_embed_tokens(self):
        """Test direct mapping for embed_tokens."""
        result = _map_weight_name("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight"

    def test_direct_mapping_norm(self):
        """Test direct mapping for final norm."""
        result = _map_weight_name("model.norm.weight")
        assert result == "model.norm.weight"

        result = _map_weight_name("model.norm.bias")
        assert result == "model.norm.bias"

    def test_direct_mapping_lm_head(self):
        """Test direct mapping for lm_head."""
        result = _map_weight_name("lm_head.weight")
        assert result == "lm_head.lm_head.weight"

    def test_layer_mapping_attention(self):
        """Test layer mapping for attention weights."""
        result = _map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

        result = _map_weight_name("model.layers.5.self_attn.k_proj.bias")
        assert result == "model.layers.5.self_attn.k_proj.bias"

        result = _map_weight_name("model.layers.10.self_attn.v_proj.weight")
        assert result == "model.layers.10.self_attn.v_proj.weight"

        result = _map_weight_name("model.layers.0.self_attn.o_proj.weight")
        assert result == "model.layers.0.self_attn.o_proj.weight"

    def test_layer_mapping_mlp_c_fc(self):
        """Test layer mapping for MLP c_fc -> up_proj."""
        result = _map_weight_name("model.layers.0.mlp.c_fc.weight")
        assert result == "model.layers.0.mlp.up_proj.weight"

        result = _map_weight_name("model.layers.5.mlp.c_fc.bias")
        assert result == "model.layers.5.mlp.up_proj.bias"

    def test_layer_mapping_mlp_c_proj(self):
        """Test layer mapping for MLP c_proj -> down_proj."""
        result = _map_weight_name("model.layers.0.mlp.c_proj.weight")
        assert result == "model.layers.0.mlp.down_proj.weight"

        result = _map_weight_name("model.layers.5.mlp.c_proj.bias")
        assert result == "model.layers.5.mlp.down_proj.bias"

    def test_layer_mapping_layernorm(self):
        """Test layer mapping for LayerNorm."""
        result = _map_weight_name("model.layers.0.input_layernorm.weight")
        assert result == "model.layers.0.input_layernorm.weight"

        result = _map_weight_name("model.layers.0.input_layernorm.bias")
        assert result == "model.layers.0.input_layernorm.bias"

        result = _map_weight_name("model.layers.0.post_attention_layernorm.weight")
        assert result == "model.layers.0.post_attention_layernorm.weight"

    def test_unrecognized_weight(self):
        """Test unrecognized weight name returns None."""
        result = _map_weight_name("unknown.weight")
        assert result is None


class TestReverseWeightMapping:
    """Tests for reverse weight name mapping."""

    def test_reverse_direct_mapping(self):
        """Test reverse direct mapping."""
        result = _reverse_map_weight_name("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight"

        result = _reverse_map_weight_name("model.norm.weight")
        assert result == "model.norm.weight"

        result = _reverse_map_weight_name("lm_head.lm_head.weight")
        assert result == "lm_head.weight"

    def test_reverse_layer_mapping_attention(self):
        """Test reverse layer mapping for attention."""
        result = _reverse_map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

    def test_reverse_layer_mapping_mlp_up_proj(self):
        """Test reverse layer mapping for up_proj -> c_fc."""
        result = _reverse_map_weight_name("model.layers.0.mlp.up_proj.weight")
        assert result == "model.layers.0.mlp.c_fc.weight"

    def test_reverse_layer_mapping_mlp_down_proj(self):
        """Test reverse layer mapping for down_proj -> c_proj."""
        result = _reverse_map_weight_name("model.layers.0.mlp.down_proj.weight")
        assert result == "model.layers.0.mlp.c_proj.weight"

    def test_reverse_unrecognized_weight(self):
        """Test unrecognized weight name returns None."""
        result = _reverse_map_weight_name("unknown.weight")
        assert result is None


class TestConvertHFWeights:
    """Tests for convert_hf_weights function."""

    def test_basic_conversion(self):
        """Test basic weight conversion."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
            "model.norm.bias": np.random.randn(64).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "model.layers.0.mlp.c_fc.weight": np.random.randn(128, 64).astype(np.float32),
            "model.layers.0.mlp.c_proj.weight": np.random.randn(64, 128).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted
        assert "model.layers.0.self_attn.q_proj.weight" in converted
        assert "model.layers.0.mlp.up_proj.weight" in converted
        assert "model.layers.0.mlp.down_proj.weight" in converted

    def test_conversion_with_lm_head(self):
        """Test conversion with lm_head weight."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights, tie_word_embeddings=False)

        assert "lm_head.lm_head.weight" in converted

    def test_conversion_tied_embeddings(self):
        """Test conversion with tied embeddings skips lm_head."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights, tie_word_embeddings=True)

        # lm_head should be skipped when tied
        assert "lm_head.lm_head.weight" not in converted
        assert "model.embed_tokens.weight" in converted

    def test_skips_unmapped_weights(self):
        """Test unmapped weights are skipped."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "unknown.weight": np.random.randn(100).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight" in converted
        assert "unknown.weight" not in converted


class TestConvertMLXToHF:
    """Tests for convert_mlx_to_hf function."""

    def test_basic_conversion(self):
        """Test basic conversion from MLX to HF format."""
        import mlx.core as mx

        mlx_weights = {
            "model.embed_tokens.weight": mx.random.normal((1000, 64)),
            "model.norm.weight": mx.random.normal((64,)),
            "model.layers.0.mlp.up_proj.weight": mx.random.normal((128, 64)),
            "model.layers.0.mlp.down_proj.weight": mx.random.normal((64, 128)),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted
        assert "model.layers.0.mlp.c_fc.weight" in converted
        assert "model.layers.0.mlp.c_proj.weight" in converted

    def test_numpy_array_handling(self):
        """Test handling of numpy arrays."""
        mlx_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert isinstance(converted["model.embed_tokens.weight"], np.ndarray)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_num_params(self):
        """Test get_num_params function."""
        weights = {
            "weight1": np.random.randn(100, 64).astype(np.float32),
            "weight2": np.random.randn(64).astype(np.float32),
            "weight3": np.random.randn(50, 50).astype(np.float32),
        }

        num_params = get_num_params(weights)

        expected = 100 * 64 + 64 + 50 * 50
        assert num_params == expected

    def test_print_weight_shapes(self, capsys):
        """Test print_weight_shapes function."""
        weights = {
            "a.weight": np.random.randn(100, 64).astype(np.float32),
            "b.weight": np.random.randn(64).astype(np.float32),
        }

        print_weight_shapes(weights)

        captured = capsys.readouterr()
        assert "a.weight: (100, 64)" in captured.out
        assert "b.weight: (64,)" in captured.out


class TestWeightMapConsistency:
    """Tests for weight map consistency."""

    def test_weight_map_has_required_keys(self):
        """Test STARCODER2_WEIGHT_MAP has required keys."""
        assert "model.embed_tokens.weight" in STARCODER2_WEIGHT_MAP
        assert "model.norm.weight" in STARCODER2_WEIGHT_MAP
        assert "model.norm.bias" in STARCODER2_WEIGHT_MAP
        assert "lm_head.weight" in STARCODER2_WEIGHT_MAP

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves weights."""
        import mlx.core as mx

        # Create sample weights in HF format
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
            "model.norm.bias": np.random.randn(64).astype(np.float32),
            "model.layers.0.mlp.c_fc.weight": np.random.randn(128, 64).astype(np.float32),
            "model.layers.0.mlp.c_proj.weight": np.random.randn(64, 128).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        }

        # Convert HF -> MLX
        mlx_weights = convert_hf_weights(hf_weights)
        mlx_weights = {k: mx.array(v) for k, v in mlx_weights.items()}

        # Convert MLX -> HF
        hf_weights_back = convert_mlx_to_hf(mlx_weights)

        # Check that keys are preserved (accounting for c_fc/up_proj naming)
        assert "model.embed_tokens.weight" in hf_weights_back
        assert "model.norm.weight" in hf_weights_back
        assert "model.layers.0.mlp.c_fc.weight" in hf_weights_back
        assert "model.layers.0.mlp.c_proj.weight" in hf_weights_back

        # Check shapes match
        assert (
            hf_weights_back["model.embed_tokens.weight"].shape
            == hf_weights["model.embed_tokens.weight"].shape
        )
        assert hf_weights_back["model.norm.weight"].shape == hf_weights["model.norm.weight"].shape
