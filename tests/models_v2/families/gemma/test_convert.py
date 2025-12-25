"""
Tests for Gemma weight conversion utilities.
"""

import numpy as np

from chuk_lazarus.models_v2.families.gemma.convert import (
    GEMMA_LAYER_PATTERNS,
    GEMMA_WEIGHT_MAP,
    _map_weight_name,
    _reverse_map_weight_name,
    convert_hf_weights,
    convert_mlx_community_weights,
    convert_mlx_to_hf,
    get_num_params,
    print_weight_shapes,
)


class TestWeightNameMapping:
    """Tests for weight name mapping functions."""

    def test_direct_weight_map_embed_tokens(self):
        """Test direct mapping for embed_tokens."""
        result = _map_weight_name("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight"

    def test_direct_weight_map_norm(self):
        """Test direct mapping for norm."""
        result = _map_weight_name("model.norm.weight")
        assert result == "model.norm.weight"

    def test_direct_weight_map_lm_head(self):
        """Test direct mapping for lm_head."""
        result = _map_weight_name("lm_head.weight")
        assert result == "lm_head.weight"

    def test_layer_weight_map_attention(self):
        """Test layer-level mapping for attention weights."""
        result = _map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

        result = _map_weight_name("model.layers.5.self_attn.k_proj.weight")
        assert result == "model.layers.5.self_attn.k_proj.weight"

        result = _map_weight_name("model.layers.10.self_attn.v_proj.weight")
        assert result == "model.layers.10.self_attn.v_proj.weight"

        result = _map_weight_name("model.layers.3.self_attn.o_proj.weight")
        assert result == "model.layers.3.self_attn.o_proj.weight"

    def test_layer_weight_map_qk_norms(self):
        """Test layer-level mapping for Gemma-specific QK norms."""
        result = _map_weight_name("model.layers.0.self_attn.q_norm.weight")
        assert result == "model.layers.0.self_attn.q_norm.weight"

        result = _map_weight_name("model.layers.2.self_attn.k_norm.weight")
        assert result == "model.layers.2.self_attn.k_norm.weight"

    def test_layer_weight_map_mlp(self):
        """Test layer-level mapping for MLP weights."""
        result = _map_weight_name("model.layers.0.mlp.gate_proj.weight")
        assert result == "model.layers.0.mlp.gate_proj.weight"

        result = _map_weight_name("model.layers.1.mlp.up_proj.weight")
        assert result == "model.layers.1.mlp.up_proj.weight"

        result = _map_weight_name("model.layers.2.mlp.down_proj.weight")
        assert result == "model.layers.2.mlp.down_proj.weight"

    def test_layer_weight_map_layernorms(self):
        """Test layer-level mapping for Gemma's 4 normalization layers."""
        result = _map_weight_name("model.layers.0.input_layernorm.weight")
        assert result == "model.layers.0.input_layernorm.weight"

        result = _map_weight_name("model.layers.0.post_attention_layernorm.weight")
        assert result == "model.layers.0.post_attention_layernorm.weight"

        result = _map_weight_name("model.layers.0.pre_feedforward_layernorm.weight")
        assert result == "model.layers.0.pre_feedforward_layernorm.weight"

        result = _map_weight_name("model.layers.0.post_feedforward_layernorm.weight")
        assert result == "model.layers.0.post_feedforward_layernorm.weight"

    def test_layer_weight_map_unknown_pattern(self):
        """Test passthrough for unknown layer patterns."""
        result = _map_weight_name("model.layers.0.unknown_layer.weight")
        assert result == "model.layers.0.unknown_layer.weight"

    def test_unrecognized_weight_returns_none(self):
        """Test that unrecognized top-level weights return None."""
        result = _map_weight_name("some.random.weight")
        assert result is None

        result = _map_weight_name("rotary_emb.inv_freq")
        assert result is None


class TestReverseWeightNameMapping:
    """Tests for reverse weight name mapping."""

    def test_reverse_map_direct(self):
        """Test reverse mapping for direct weights."""
        result = _reverse_map_weight_name("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight"

        result = _reverse_map_weight_name("model.norm.weight")
        assert result == "model.norm.weight"

        result = _reverse_map_weight_name("lm_head.weight")
        assert result == "lm_head.weight"

    def test_reverse_map_layer_patterns(self):
        """Test reverse mapping for layer patterns."""
        result = _reverse_map_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

        result = _reverse_map_weight_name("model.layers.5.mlp.gate_proj.weight")
        assert result == "model.layers.5.mlp.gate_proj.weight"

    def test_reverse_map_unknown_pattern(self):
        """Test passthrough for unknown layer patterns."""
        result = _reverse_map_weight_name("model.layers.0.unknown.weight")
        assert result == "model.layers.0.unknown.weight"

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
            "model.norm.weight": np.random.randn(64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted
        assert "lm_head.weight" in converted
        assert len(converted) == 3

    def test_convert_layer_weights(self):
        """Test layer weight conversion."""
        hf_weights = {
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(128, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.layers.0.self_attn.q_proj.weight" in converted
        assert "model.layers.0.self_attn.k_proj.weight" in converted
        assert "model.layers.0.mlp.gate_proj.weight" in converted

    def test_convert_skips_unmapped_weights(self):
        """Test that unmapped weights are skipped."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "rotary_emb.inv_freq": np.random.randn(32).astype(np.float32),  # Should be skipped
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight" in converted
        assert "rotary_emb.inv_freq" not in converted
        assert len(converted) == 1

    def test_convert_with_tied_embeddings(self):
        """Test conversion with tied word embeddings."""
        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 64).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights, tie_word_embeddings=True)

        assert "model.embed_tokens.weight" in converted
        assert "lm_head.weight" not in converted  # Should be skipped when tied
        assert len(converted) == 1


class TestConvertMlxCommunityWeights:
    """Tests for MLX community weight conversion."""

    def test_convert_mlx_community_passthrough(self):
        """Test that MLX community weights pass through."""
        weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
        }

        converted = convert_mlx_community_weights(weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.layers.0.self_attn.q_proj.weight" in converted


class TestConvertMlxToHf:
    """Tests for MLX to HuggingFace conversion."""

    def test_convert_numpy_weights(self):
        """Test conversion of numpy arrays."""
        mlx_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 64).astype(np.float32),
            "model.norm.weight": np.random.randn(64).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted

    def test_convert_layer_weights_to_hf(self):
        """Test layer weight conversion to HF format."""
        mlx_weights = {
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(128, 64).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.layers.0.self_attn.q_proj.weight" in converted
        assert "model.layers.0.mlp.gate_proj.weight" in converted


class TestMlxArrayConversion:
    """Tests for MLX array conversion in convert_mlx_to_hf."""

    def test_convert_mlx_array_passthrough(self):
        """Test that MLX arrays are passed through (can be converted via np.asarray)."""
        import mlx.core as mx

        mlx_weights = {
            "model.embed_tokens.weight": mx.random.normal((100, 64)),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        # MLX arrays are passed through since they lack .numpy() and __array__
        # The current implementation passes them through unchanged
        # To get numpy, caller should use np.asarray on the result
        assert "model.embed_tokens.weight" in converted
        # Result can be converted to numpy via np.asarray
        result = np.asarray(converted["model.embed_tokens.weight"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 64)

    def test_convert_object_with_array_interface(self):
        """Test conversion of objects with __array__ method."""

        class ArrayLike:
            def __init__(self, data):
                self._data = data

            def __array__(self):
                return self._data

        mlx_weights = {
            "model.embed_tokens.weight": ArrayLike(np.random.randn(100, 64).astype(np.float32)),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert isinstance(converted["model.embed_tokens.weight"], np.ndarray)


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

    def test_gemma_weight_map_completeness(self):
        """Test that GEMMA_WEIGHT_MAP has expected keys."""
        assert "model.embed_tokens.weight" in GEMMA_WEIGHT_MAP
        assert "model.norm.weight" in GEMMA_WEIGHT_MAP
        assert "lm_head.weight" in GEMMA_WEIGHT_MAP

    def test_gemma_layer_patterns_completeness(self):
        """Test that GEMMA_LAYER_PATTERNS has expected keys."""
        # Attention weights
        assert "self_attn.q_proj.weight" in GEMMA_LAYER_PATTERNS
        assert "self_attn.k_proj.weight" in GEMMA_LAYER_PATTERNS
        assert "self_attn.v_proj.weight" in GEMMA_LAYER_PATTERNS
        assert "self_attn.o_proj.weight" in GEMMA_LAYER_PATTERNS

        # QK norms
        assert "self_attn.q_norm.weight" in GEMMA_LAYER_PATTERNS
        assert "self_attn.k_norm.weight" in GEMMA_LAYER_PATTERNS

        # MLP weights
        assert "mlp.gate_proj.weight" in GEMMA_LAYER_PATTERNS
        assert "mlp.up_proj.weight" in GEMMA_LAYER_PATTERNS
        assert "mlp.down_proj.weight" in GEMMA_LAYER_PATTERNS

        # Layer norms
        assert "input_layernorm.weight" in GEMMA_LAYER_PATTERNS
        assert "post_attention_layernorm.weight" in GEMMA_LAYER_PATTERNS
        assert "pre_feedforward_layernorm.weight" in GEMMA_LAYER_PATTERNS
        assert "post_feedforward_layernorm.weight" in GEMMA_LAYER_PATTERNS
