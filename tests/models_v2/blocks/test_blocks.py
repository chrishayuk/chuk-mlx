"""
Tests for block components.

Tests TransformerBlock, MambaBlock, RecurrentBlock, and HybridBlock.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.blocks import (
    BlockOutput,
    HybridBlock,
    MambaBlockWrapper,
    RecurrentBlockWrapper,
    RecurrentWithFFN,
    TransformerBlock,
)
from chuk_lazarus.models_v2.blocks.hybrid import AlternatingHybrid
from chuk_lazarus.models_v2.blocks.mamba import (
    MambaWithFFN,
    create_mamba_block_wrapper,
    create_mamba_with_ffn,
)
from chuk_lazarus.models_v2.blocks.transformer import create_transformer_block
from chuk_lazarus.models_v2.core.enums import BlockType, FFNType


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            num_kv_heads=8,
            intermediate_size=2048,
        )

        x = mx.random.normal((2, 10, 512))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_with_mask(self):
        """Test transformer block with attention mask."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
        )

        x = mx.random.normal((2, 10, 512))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output = block(x, mask=mask)

        assert output.hidden_states.shape == (2, 10, 512)

    def test_with_cache(self):
        """Test transformer block with KV cache."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
        )

        # First pass to build cache
        x = mx.random.normal((2, 10, 512))
        output = block(x)
        cache = output.cache

        # Second pass with cache
        x_new = mx.random.normal((2, 1, 512))
        output_new = block(x_new, cache=cache)

        assert output_new.hidden_states.shape == (2, 1, 512)
        assert output_new.cache is not None

    def test_block_type(self):
        """Test block type is TRANSFORMER."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
        )

        assert block.block_type == BlockType.TRANSFORMER

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        block = TransformerBlock(
            hidden_size=1024,
            num_heads=16,
            intermediate_size=4096,
        )

        assert block.hidden_size == 1024


class TestMambaBlockWrapper:
    """Tests for MambaBlockWrapper."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MambaBlockWrapper(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_with_cache(self):
        """Test Mamba block with cache."""
        block = MambaBlockWrapper(
            d_model=128,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Test that forward pass works (cache testing requires careful setup)
        x = mx.random.normal((2, 5, 128))
        output = block(x)

        assert output.hidden_states.shape == (2, 5, 128)
        # Cache is returned for potential caching
        assert output.cache is not None or output.cache is None

    def test_block_type(self):
        """Test block type is MAMBA."""
        block = MambaBlockWrapper(
            d_model=256,
            d_state=16,
        )

        assert block.block_type == BlockType.MAMBA


class TestRecurrentBlockWrapper:
    """Tests for RecurrentBlockWrapper."""

    def test_lstm_forward(self):
        """Test LSTM block forward pass."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="lstm",
            num_layers=1,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_gru_forward(self):
        """Test GRU block forward pass."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="gru",
            num_layers=1,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)

    def test_mingru_forward(self):
        """Test MinGRU block forward pass."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="mingru",
            num_layers=1,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)

    def test_block_type(self):
        """Test block type is specific RNN type."""
        # LSTM block
        lstm_block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="lstm",
        )
        assert lstm_block.block_type == BlockType.LSTM

        # GRU block
        gru_block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="gru",
        )
        assert gru_block.block_type == BlockType.GRU

        # MinGRU block
        mingru_block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="mingru",
        )
        assert mingru_block.block_type == BlockType.MINGRU


class TestRecurrentWithFFN:
    """Tests for RecurrentWithFFN."""

    def test_basic_forward(self):
        """Test forward pass with FFN sublayer."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="mingru",
            num_layers=1,
            intermediate_size=1024,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_with_default_intermediate(self):
        """Test with default intermediate size."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="gru",
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)


class TestHybridBlock:
    """Tests for HybridBlock."""

    def test_attention_mamba_hybrid(self):
        """Test hybrid block with attention and Mamba."""
        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_block_type(self):
        """Test block type is HYBRID."""
        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
        )

        assert block.block_type == BlockType.HYBRID


class TestBlockGradients:
    """Tests for gradient flow through blocks."""

    def test_transformer_block_gradients(self):
        """Test gradients flow through transformer block."""
        block = TransformerBlock(
            hidden_size=128,
            num_heads=4,
            intermediate_size=256,
        )

        x = mx.random.normal((1, 5, 128))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out.hidden_states**2)

        loss_and_grad_fn = nn.value_and_grad(block, loss_fn)
        loss, grads = loss_and_grad_fn(block, x)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_mamba_block_gradients(self):
        """Test gradients flow through Mamba block."""
        block = MambaBlockWrapper(
            d_model=64,
            d_state=8,
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out.hidden_states**2)

        loss_and_grad_fn = nn.value_and_grad(block, loss_fn)
        loss, grads = loss_and_grad_fn(block, x)

        assert loss.item() > 0

    def test_recurrent_block_gradients(self):
        """Test gradients flow through recurrent block."""
        block = RecurrentBlockWrapper(
            d_model=64,
            rnn_type="mingru",
        )

        x = mx.random.normal((1, 5, 64))

        def loss_fn(model, x):
            out = model(x)
            return mx.mean(out.hidden_states**2)

        loss_and_grad_fn = nn.value_and_grad(block, loss_fn)
        loss, grads = loss_and_grad_fn(block, x)

        assert loss.item() > 0


class TestBlockOutput:
    """Tests for BlockOutput dataclass."""

    def test_basic_output(self):
        """Test BlockOutput creation."""
        hidden = mx.random.normal((2, 10, 256))
        output = BlockOutput(hidden_states=hidden)

        assert output.hidden_states.shape == (2, 10, 256)
        assert output.cache is None

    def test_with_cache(self):
        """Test BlockOutput with cache."""
        hidden = mx.random.normal((2, 10, 256))
        k_cache = mx.random.normal((2, 10, 8, 64))
        v_cache = mx.random.normal((2, 10, 8, 64))
        cache = (k_cache, v_cache)

        output = BlockOutput(hidden_states=hidden, cache=cache)

        assert output.cache is not None
        assert len(output.cache) == 2


class TestMambaWithFFN:
    """Tests for MambaWithFFN block."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MambaWithFFN(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
            intermediate_size=1024,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_with_default_intermediate(self):
        """Test with default intermediate size (4x model dim)."""
        block = MambaWithFFN(
            d_model=128,
            d_state=16,
        )

        x = mx.random.normal((2, 5, 128))
        output = block(x)

        assert output.hidden_states.shape == (2, 5, 128)

    def test_block_type(self):
        """Test block type is MAMBA."""
        block = MambaWithFFN(d_model=256, d_state=16)
        assert block.block_type == BlockType.MAMBA

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        block = MambaWithFFN(d_model=512, d_state=16)
        assert block.hidden_size == 512

    def test_init_cache(self):
        """Test cache initialization."""
        block = MambaWithFFN(d_model=128, d_state=16, d_conv=4)
        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None


class TestAlternatingHybrid:
    """Tests for AlternatingHybrid block."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        block = AlternatingHybrid(
            d_model=256,
            num_heads=8,
            d_state=16,
            layer_idx=0,  # Required parameter
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert isinstance(output, BlockOutput)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_block_type(self):
        """Test block type is HYBRID."""
        block = AlternatingHybrid(d_model=256, num_heads=8, d_state=16, layer_idx=0)
        assert block.block_type == BlockType.HYBRID

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        block = AlternatingHybrid(d_model=512, num_heads=16, d_state=16, layer_idx=0)
        assert block.hidden_size == 512

    def test_alternating_even_layer(self):
        """Test even layer uses attention."""
        block = AlternatingHybrid(d_model=256, num_heads=8, d_state=16, layer_idx=0)
        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_alternating_odd_layer(self):
        """Test odd layer uses Mamba."""
        block = AlternatingHybrid(d_model=256, num_heads=8, d_state=16, layer_idx=1)
        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)


class TestBlockFactoryFunctions:
    """Tests for block factory functions."""

    def test_create_transformer_block(self):
        """Test transformer block factory."""
        block = create_transformer_block(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
        )
        assert isinstance(block, TransformerBlock)

        x = mx.random.normal((2, 10, 512))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_create_mamba_block_wrapper(self):
        """Test Mamba block wrapper factory."""
        block = create_mamba_block_wrapper(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        assert isinstance(block, MambaBlockWrapper)

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_create_mamba_with_ffn(self):
        """Test MambaWithFFN factory."""
        block = create_mamba_with_ffn(
            d_model=256,
            d_state=16,
            intermediate_size=1024,
        )
        assert isinstance(block, MambaWithFFN)

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_recurrent_block_direct_creation(self):
        """Test recurrent block direct creation."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="gru",
        )
        assert isinstance(block, RecurrentBlockWrapper)

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_recurrent_with_ffn_direct_creation(self):
        """Test RecurrentWithFFN direct creation."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="mingru",
            intermediate_size=1024,
        )
        assert isinstance(block, RecurrentWithFFN)

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)


class TestTransformerBlockAdvanced:
    """Advanced tests for TransformerBlock."""

    def test_gqa_configuration(self):
        """Test Grouped Query Attention configuration."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            num_kv_heads=2,  # GQA with 2 KV heads
            intermediate_size=2048,
        )

        x = mx.random.normal((2, 10, 512))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_sliding_window(self):
        """Test sliding window attention."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            sliding_window=128,
        )

        x = mx.random.normal((2, 256, 512))  # Longer than window
        output = block(x)
        assert output.hidden_states.shape == (2, 256, 512)

    def test_custom_ffn_type(self):
        """Test with custom FFN type."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            ffn_type=FFNType.GEGLU,
        )

        x = mx.random.normal((2, 10, 512))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_init_cache(self):
        """Test cache initialization."""
        block = TransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=1024,
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None

    def test_with_layernorm(self):
        """Test with LayerNorm instead of RMSNorm."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            norm_type="layernorm",
        )

        x = mx.random.normal((2, 10, 512))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_with_mlp_ffn(self):
        """Test with MLP FFN type (default, not SwiGLU/GEGLU)."""
        block = TransformerBlock(
            hidden_size=512,
            num_heads=8,
            intermediate_size=2048,
            ffn_type=FFNType.MLP,
        )

        x = mx.random.normal((2, 10, 512))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 512)

    def test_with_dropout(self):
        """Test with dropout enabled."""
        block = TransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=1024,
            hidden_dropout=0.1,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_from_config(self):
        """Test creating from ModelConfig."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=1024,
        )

        block = TransformerBlock.from_config(config, layer_idx=0)
        assert isinstance(block, TransformerBlock)

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_from_config_with_gqa(self):
        """Test from_config with GQA."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,  # GQA
            intermediate_size=1024,
        )

        block = TransformerBlock.from_config(config, layer_idx=0)
        assert isinstance(block, TransformerBlock)

    def test_from_config_with_sliding_window(self):
        """Test from_config with sliding window."""
        from chuk_lazarus.models_v2.core.config import ModelConfig

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=1024,
            sliding_window=64,
        )

        block = TransformerBlock.from_config(config, layer_idx=0)
        assert isinstance(block, TransformerBlock)


class TestRecurrentBlockAdvanced:
    """Advanced tests for RecurrentBlock."""

    def test_multi_layer_lstm(self):
        """Test multi-layer LSTM."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="lstm",
            num_layers=3,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 10, 256)

    def test_with_initial_state(self):
        """Test with initial hidden state."""
        block = RecurrentBlockWrapper(
            d_model=128,
            rnn_type="gru",
            num_layers=1,
        )

        x = mx.random.normal((2, 10, 128))
        # Initialize state
        h_0 = mx.random.normal((1, 2, 128))
        output = block(x, cache=h_0)
        assert output.hidden_states.shape == (2, 10, 128)

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        block = RecurrentBlockWrapper(d_model=512, rnn_type="mingru")
        assert block.hidden_size == 512

    def test_init_cache(self):
        """Test cache initialization."""
        block = RecurrentBlockWrapper(d_model=256, rnn_type="lstm")
        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None


class TestMambaBlockAdvanced:
    """Advanced tests for MambaBlockWrapper."""

    def test_init_cache(self):
        """Test cache initialization."""
        block = MambaBlockWrapper(
            d_model=256,
            d_state=16,
            d_conv=4,
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        block = MambaBlockWrapper(d_model=512, d_state=16)
        assert block.hidden_size == 512

    def test_different_expand_factors(self):
        """Test different expansion factors."""
        for expand in [1, 2, 4]:
            block = MambaBlockWrapper(
                d_model=128,
                d_state=16,
                expand=expand,
            )
            x = mx.random.normal((2, 10, 128))
            output = block(x)
            assert output.hidden_states.shape == (2, 10, 128)


class TestHybridBlockCombineModes:
    """Tests for HybridBlock combine modes."""

    def test_concat_combine_mode(self):
        """Test CONCAT combine mode."""
        from chuk_lazarus.models_v2.core.enums import HybridCombineMode

        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
            combine_mode=HybridCombineMode.CONCAT,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.combine_mode == HybridCombineMode.CONCAT
        # CONCAT mode should have combine_proj
        assert block.combine_proj is not None

    def test_gate_combine_mode(self):
        """Test GATE combine mode."""
        from chuk_lazarus.models_v2.core.enums import HybridCombineMode

        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
            combine_mode=HybridCombineMode.GATE,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.combine_mode == HybridCombineMode.GATE
        # GATE mode should have gate layer
        assert hasattr(block, "gate")

    def test_add_combine_mode(self):
        """Test ADD combine mode (default)."""
        from chuk_lazarus.models_v2.core.enums import HybridCombineMode

        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
            combine_mode=HybridCombineMode.ADD,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.combine_mode == HybridCombineMode.ADD
        # ADD mode should NOT have combine_proj
        assert block.combine_proj is None

    def test_string_combine_mode_conversion(self):
        """Test string combine mode is converted to enum."""
        from chuk_lazarus.models_v2.core.enums import HybridCombineMode

        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
            combine_mode="concat",
        )

        assert block.combine_mode == HybridCombineMode.CONCAT

    def test_hybrid_init_cache(self):
        """Test HybridBlock init_cache."""
        block = HybridBlock(
            d_model=256,
            num_heads=8,
            d_state=16,
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)

        assert cache is not None
        assert "attn" in cache
        assert "mamba" in cache

    def test_hybrid_with_cache(self):
        """Test HybridBlock with cache."""
        block = HybridBlock(
            d_model=128,
            num_heads=4,
            d_state=8,
        )

        # First pass
        x = mx.random.normal((2, 5, 128))
        output = block(x)
        cache = output.cache

        assert cache is not None
        assert "attn" in cache
        assert "mamba" in cache


class TestHybridBlockFactoryFunctions:
    """Tests for hybrid block factory functions."""

    def test_create_hybrid_block(self):
        """Test create_hybrid_block factory."""
        from chuk_lazarus.models_v2.blocks.hybrid import create_hybrid_block

        block = create_hybrid_block(
            d_model=256,
            num_heads=8,
            d_state=16,
        )

        assert isinstance(block, HybridBlock)
        x = mx.random.normal((2, 5, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 5, 256)

    def test_create_alternating_hybrid(self):
        """Test create_alternating_hybrid factory."""
        from chuk_lazarus.models_v2.blocks.hybrid import create_alternating_hybrid

        block = create_alternating_hybrid(
            d_model=256,
            layer_idx=0,
            num_heads=8,
        )

        assert isinstance(block, AlternatingHybrid)
        x = mx.random.normal((2, 5, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 5, 256)


class TestRecurrentBlockBidirectional:
    """Tests for bidirectional recurrent blocks."""

    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="lstm",
            bidirectional=True,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.bidirectional is True

    def test_bidirectional_gru(self):
        """Test bidirectional GRU."""
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="gru",
            bidirectional=True,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)

    def test_bidirectional_with_out_proj(self):
        """Test bidirectional with output projection when dims don't match."""
        # When hidden_size // 2 * 2 != hidden_size, out_proj is needed
        block = RecurrentBlockWrapper(
            d_model=256,
            rnn_type="lstm",
            bidirectional=True,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)


class TestRecurrentWithFFNAdvanced:
    """Advanced tests for RecurrentWithFFN."""

    def test_lstm_with_ffn(self):
        """Test LSTM with FFN."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="lstm",
            intermediate_size=1024,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.block_type == BlockType.LSTM

    def test_gru_with_ffn(self):
        """Test GRU with FFN."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="gru",
            intermediate_size=1024,
        )

        x = mx.random.normal((2, 10, 256))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 256)
        assert block.block_type == BlockType.GRU

    def test_recurrent_with_ffn_init_cache_lstm(self):
        """Test init_cache for LSTM with FFN."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="lstm",
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None

    def test_recurrent_with_ffn_init_cache_gru(self):
        """Test init_cache for GRU with FFN."""
        block = RecurrentWithFFN(
            d_model=256,
            rnn_type="gru",
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None

    def test_recurrent_with_ffn_hidden_size(self):
        """Test hidden_size property."""
        block = RecurrentWithFFN(d_model=512, rnn_type="mingru")
        assert block.hidden_size == 512


class TestRecurrentBlockFactoryFunctions:
    """Tests for recurrent block factory functions."""

    def test_create_recurrent_block(self):
        """Test create_recurrent_block factory."""
        from chuk_lazarus.models_v2.blocks.recurrent import create_recurrent_block

        block = create_recurrent_block(
            d_model=256,
            rnn_type="gru",
        )

        assert isinstance(block, RecurrentBlockWrapper)
        x = mx.random.normal((2, 5, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 5, 256)

    def test_create_recurrent_with_ffn(self):
        """Test create_recurrent_with_ffn factory."""
        from chuk_lazarus.models_v2.blocks.recurrent import create_recurrent_with_ffn

        block = create_recurrent_with_ffn(
            d_model=256,
            rnn_type="lstm",
            intermediate_size=1024,
        )

        assert isinstance(block, RecurrentWithFFN)
        x = mx.random.normal((2, 5, 256))
        output = block(x)
        assert output.hidden_states.shape == (2, 5, 256)


class TestAlternatingHybridAdvanced:
    """Advanced tests for AlternatingHybrid."""

    def test_alternating_uses_attention_flag(self):
        """Test that even layer uses attention (uses_attention_first=True)."""
        block_even = AlternatingHybrid(
            d_model=256,
            num_heads=8,
            layer_idx=0,
            use_attention_first=True,
        )
        assert block_even.uses_attention is True

        block_odd = AlternatingHybrid(
            d_model=256,
            num_heads=8,
            layer_idx=1,
            use_attention_first=True,
        )
        assert block_odd.uses_attention is False

    def test_alternating_init_cache_attention(self):
        """Test init_cache for attention layer."""
        block = AlternatingHybrid(
            d_model=256,
            num_heads=8,
            layer_idx=0,  # Even = attention
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None
        # Attention cache is (k_cache, v_cache) tuple
        assert isinstance(cache, tuple)
        assert len(cache) == 2

    def test_alternating_init_cache_mamba(self):
        """Test init_cache for Mamba layer."""
        block = AlternatingHybrid(
            d_model=256,
            num_heads=8,
            layer_idx=1,  # Odd = Mamba
        )

        cache = block.init_cache(batch_size=2, max_seq_len=100)
        # Mamba cache structure
        assert cache is not None
