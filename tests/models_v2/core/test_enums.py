"""
Tests for models_v2.core.enums.

Ensures all enums are properly defined with correct values and string conversion.
"""

import pytest

from chuk_lazarus.models_v2.core.enums import (
    ActivationType,
    AttentionType,
    BackboneType,
    BlockType,
    FFNType,
    HeadType,
    InitType,
    ModelMode,
    NormType,
    PoolingType,
    PositionEmbeddingType,
    RecurrentType,
    SSMType,
)


class TestModelMode:
    """Tests for ModelMode enum."""

    def test_values(self):
        """Test all ModelMode values exist."""
        assert ModelMode.TRAIN.value == "train"
        assert ModelMode.INFERENCE.value == "inference"
        assert ModelMode.EVAL.value == "eval"

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(ModelMode.TRAIN) == "train"
        assert str(ModelMode.INFERENCE) == "inference"
        assert str(ModelMode.EVAL) == "eval"

    def test_from_string(self):
        """Test creation from string."""
        assert ModelMode("train") == ModelMode.TRAIN
        assert ModelMode("inference") == ModelMode.INFERENCE

    def test_invalid_value(self):
        """Test invalid value raises error."""
        with pytest.raises(ValueError):
            ModelMode("invalid")


class TestBlockType:
    """Tests for BlockType enum."""

    def test_all_values(self):
        """Test all BlockType values exist."""
        expected = {
            "transformer",
            "mamba",
            "mamba2",
            "lstm",
            "gru",
            "mingru",
            "conv",
            "hybrid",
        }
        actual = {bt.value for bt in BlockType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(BlockType.TRANSFORMER) == "transformer"
        assert str(BlockType.MAMBA) == "mamba"
        assert str(BlockType.LSTM) == "lstm"


class TestBackboneType:
    """Tests for BackboneType enum."""

    def test_all_values(self):
        """Test all BackboneType values exist."""
        expected = {
            "transformer",
            "mamba",
            "recurrent",
            "hybrid",
            "encoder_decoder",
        }
        actual = {bt.value for bt in BackboneType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(BackboneType.TRANSFORMER) == "transformer"
        assert str(BackboneType.HYBRID) == "hybrid"


class TestHeadType:
    """Tests for HeadType enum."""

    def test_all_values(self):
        """Test all HeadType values exist."""
        expected = {
            "lm",
            "classifier",
            "regression",
            "sequence_labeling",
            "contrastive",
        }
        actual = {ht.value for ht in HeadType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(HeadType.LM) == "lm"
        assert str(HeadType.CLASSIFIER) == "classifier"


class TestAttentionType:
    """Tests for AttentionType enum."""

    def test_all_values(self):
        """Test all AttentionType values exist."""
        expected = {
            "multi_head",
            "grouped_query",
            "multi_query",
            "multi_latent",
            "sliding_window",
            "linear",
            "flash",
        }
        actual = {at.value for at in AttentionType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(AttentionType.MULTI_HEAD) == "multi_head"
        assert str(AttentionType.GROUPED_QUERY) == "grouped_query"


class TestNormType:
    """Tests for NormType enum."""

    def test_all_values(self):
        """Test all NormType values exist."""
        expected = {
            "rms_norm",
            "layer_norm",
            "gemma_norm",
            "batch_norm",
            "group_norm",
            "none",
        }
        actual = {nt.value for nt in NormType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(NormType.RMS_NORM) == "rms_norm"
        assert str(NormType.LAYER_NORM) == "layer_norm"


class TestActivationType:
    """Tests for ActivationType enum."""

    def test_all_values(self):
        """Test all ActivationType values exist."""
        expected = {
            "silu",
            "gelu",
            "gelu_approx",
            "relu",
            "relu2",
            "tanh",
            "sigmoid",
            "none",
        }
        actual = {at.value for at in ActivationType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(ActivationType.SILU) == "silu"
        assert str(ActivationType.GELU) == "gelu"


class TestPositionEmbeddingType:
    """Tests for PositionEmbeddingType enum."""

    def test_all_values(self):
        """Test all PositionEmbeddingType values exist."""
        expected = {
            "rope",
            "alibi",
            "learned",
            "sinusoidal",
            "relative",
            "none",
        }
        actual = {pet.value for pet in PositionEmbeddingType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(PositionEmbeddingType.ROPE) == "rope"
        assert str(PositionEmbeddingType.ALIBI) == "alibi"


class TestPoolingType:
    """Tests for PoolingType enum."""

    def test_all_values(self):
        """Test all PoolingType values exist."""
        expected = {"cls", "mean", "max", "last", "first"}
        actual = {pt.value for pt in PoolingType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(PoolingType.MEAN) == "mean"
        assert str(PoolingType.CLS) == "cls"


class TestFFNType:
    """Tests for FFNType enum."""

    def test_all_values(self):
        """Test all FFNType values exist."""
        expected = {"mlp", "swiglu", "geglu", "gated", "moe"}
        actual = {ft.value for ft in FFNType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(FFNType.SWIGLU) == "swiglu"
        assert str(FFNType.MOE) == "moe"


class TestSSMType:
    """Tests for SSMType enum."""

    def test_all_values(self):
        """Test all SSMType values exist."""
        expected = {"mamba", "mamba2", "s4", "s4d", "h3"}
        actual = {st.value for st in SSMType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(SSMType.MAMBA) == "mamba"
        assert str(SSMType.MAMBA2) == "mamba2"


class TestRecurrentType:
    """Tests for RecurrentType enum."""

    def test_all_values(self):
        """Test all RecurrentType values exist."""
        expected = {"lstm", "gru", "mingru", "rnn"}
        actual = {rt.value for rt in RecurrentType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(RecurrentType.LSTM) == "lstm"
        assert str(RecurrentType.GRU) == "gru"


class TestInitType:
    """Tests for InitType enum."""

    def test_all_values(self):
        """Test all InitType values exist."""
        expected = {"normal", "xavier", "kaiming", "orthogonal", "zeros", "ones"}
        actual = {it.value for it in InitType}
        assert actual == expected

    def test_str_conversion(self):
        """Test string conversion."""
        assert str(InitType.XAVIER) == "xavier"
        assert str(InitType.KAIMING) == "kaiming"
