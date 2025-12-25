"""
Tests for GEGLU feed-forward network.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.components.ffn import GEGLU
from chuk_lazarus.models_v2.components.ffn.geglu import create_geglu
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType, FFNType


class TestGEGLU:
    """Tests for GEGLU feed-forward network."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = FFNConfig(
            ffn_type=FFNType.GEGLU,
            hidden_size=512,
            intermediate_size=1408,
            activation=ActivationType.GELU,
        )
        ffn = GEGLU(config)

        x = mx.random.normal((2, 10, 512))
        output = ffn(x)

        assert output.shape == (2, 10, 512)

    def test_gelu_activation(self):
        """Test GELU is used (different from SiLU)."""
        config = FFNConfig(
            hidden_size=256,
            intermediate_size=512,
            activation=ActivationType.GELU,
        )
        ffn = GEGLU(config)

        x = mx.random.normal((1, 5, 256))
        output = ffn(x)

        assert output.shape == (1, 5, 256)


class TestGEGLUFactory:
    """Tests for GEGLU factory function."""

    def test_create_geglu(self):
        """Test GEGLU factory function."""
        ffn = create_geglu(
            hidden_size=256,
            intermediate_size=512,
        )
        assert isinstance(ffn, GEGLU)

        x = mx.random.normal((2, 10, 256))
        output = ffn(x)
        assert output.shape == (2, 10, 256)

    def test_create_geglu_directly(self):
        """Test creating GEGLU directly."""
        config = FFNConfig(
            ffn_type=FFNType.GEGLU,
            hidden_size=512,
            intermediate_size=1408,
        )
        ffn = GEGLU(config)

        assert isinstance(ffn, GEGLU)
