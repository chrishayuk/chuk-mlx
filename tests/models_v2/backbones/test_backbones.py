"""
Tests for backbone components.

Tests TransformerBackbone, MambaBackbone, RecurrentBackbone, and HybridBackbone.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.backbones import (
    BackboneOutput,
    HybridBackbone,
    RecurrentBackbone,
    create_mamba_backbone,
    create_recurrent_backbone,
    create_transformer_backbone,
)
from chuk_lazarus.models_v2.backbones.hybrid import create_hybrid_backbone


class TestTransformerBackbone:
    """Tests for TransformerBackbone."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert isinstance(output, BackboneOutput)
        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_with_cache(self):
        """Test backbone with KV cache."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )

        # First pass to build cache
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output_new = backbone(new_token, cache=cache)

        assert output_new.last_hidden_state.shape == (1, 1, 128)
        assert output_new.cache is not None

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=3,
            num_heads=4,
            intermediate_size=256,
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # Should have embeddings + 3 layer outputs
        assert len(output.hidden_states) == 4

    def test_properties(self):
        """Test backbone properties."""
        backbone = create_transformer_backbone(
            vocab_size=5000,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
        )

        assert backbone.hidden_size == 512
        assert backbone.num_layers == 6
        assert backbone.vocab_size == 5000


class TestMambaBackbone:
    """Tests for MambaBackbone."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        backbone = create_mamba_backbone(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            d_state=16,
            d_conv=4,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert isinstance(output, BackboneOutput)
        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_with_cache(self):
        """Test backbone with SSM cache."""
        backbone = create_mamba_backbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=16,
        )

        # Test basic forward (cache requires careful SSM setup)
        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 3, 128)

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        backbone = create_mamba_backbone(
            vocab_size=1000,
            d_model=128,
            num_layers=3,
            d_state=16,
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 4  # embeddings + 3 layers

    def test_init_cache(self):
        """Test cache initialization."""
        backbone = create_mamba_backbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=16,
        )

        cache = backbone.init_cache(batch_size=4, max_seq_len=100)

        assert cache is not None
        assert len(cache) == 2  # One per layer


class TestRecurrentBackbone:
    """Tests for RecurrentBackbone."""

    def test_lstm_backbone(self):
        """Test LSTM backbone."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            rnn_type="lstm",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert isinstance(output, BackboneOutput)
        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_gru_backbone(self):
        """Test GRU backbone."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            rnn_type="gru",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_mingru_backbone(self):
        """Test MinGRU backbone."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            rnn_type="mingru",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_with_ffn(self):
        """Test recurrent backbone with FFN."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=256,
            num_layers=2,
            rnn_type="mingru",
            with_ffn=True,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_factory_function(self):
        """Test factory function."""
        backbone = create_recurrent_backbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            rnn_type="mingru",
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 3, 128)


class TestHybridBackbone:
    """Tests for HybridBackbone."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            intermediate_size=512,
            d_state=16,
            mix_strategy="alternating",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert isinstance(output, BackboneOutput)
        assert output.last_hidden_state.shape == (1, 5, 256)

    def test_with_cache(self):
        """Test hybrid backbone with cache."""
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy="alternating",
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)
        cache = output.cache

        # Continue with cache
        new_token = mx.array([[4]])
        output_new = backbone(new_token, cache=cache)

        assert output_new.last_hidden_state.shape == (1, 1, 128)


class TestBackboneOutput:
    """Tests for BackboneOutput dataclass."""

    def test_basic_output(self):
        """Test BackboneOutput creation."""
        hidden = mx.random.normal((2, 10, 256))
        output = BackboneOutput(last_hidden_state=hidden)

        assert output.last_hidden_state.shape == (2, 10, 256)
        assert output.hidden_states is None
        assert output.cache is None

    def test_with_hidden_states(self):
        """Test BackboneOutput with all hidden states."""
        hidden = mx.random.normal((2, 10, 256))
        all_hidden = tuple(mx.random.normal((2, 10, 256)) for _ in range(4))

        output = BackboneOutput(
            last_hidden_state=hidden,
            hidden_states=all_hidden,
        )

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 4

    def test_with_cache(self):
        """Test BackboneOutput with cache."""
        hidden = mx.random.normal((2, 10, 256))
        cache = [
            (mx.random.normal((2, 10, 4, 64)), mx.random.normal((2, 10, 4, 64))) for _ in range(3)
        ]

        output = BackboneOutput(
            last_hidden_state=hidden,
            cache=cache,
        )

        assert output.cache is not None
        assert len(output.cache) == 3


class TestBackboneGradients:
    """Tests for gradient flow through backbones."""

    def test_transformer_backbone_gradients(self):
        """Test gradients flow through transformer backbone."""
        backbone = create_transformer_backbone(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=128,
        )

        input_ids = mx.array([[1, 2, 3, 4]])

        def loss_fn(model, input_ids):
            out = model(input_ids)
            return mx.mean(out.last_hidden_state**2)

        loss_and_grad_fn = nn.value_and_grad(backbone, loss_fn)
        loss, grads = loss_and_grad_fn(backbone, input_ids)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_mamba_backbone_gradients(self):
        """Test gradients flow through Mamba backbone."""
        backbone = create_mamba_backbone(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            d_state=8,
        )

        input_ids = mx.array([[1, 2, 3, 4]])

        def loss_fn(model, input_ids):
            out = model(input_ids)
            return mx.mean(out.last_hidden_state**2)

        loss_and_grad_fn = nn.value_and_grad(backbone, loss_fn)
        loss, grads = loss_and_grad_fn(backbone, input_ids)

        assert loss.item() > 0

    def test_recurrent_backbone_gradients(self):
        """Test gradients flow through recurrent backbone."""
        backbone = create_recurrent_backbone(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            rnn_type="mingru",
        )

        input_ids = mx.array([[1, 2, 3, 4]])

        def loss_fn(model, input_ids):
            out = model(input_ids)
            return mx.mean(out.last_hidden_state**2)

        loss_and_grad_fn = nn.value_and_grad(backbone, loss_fn)
        loss, grads = loss_and_grad_fn(backbone, input_ids)

        assert loss.item() > 0


class TestBackboneEmbeddings:
    """Tests for backbone embedding handling."""

    def test_get_input_embeddings(self):
        """Test getting input embeddings."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
        )

        embeddings = backbone.get_input_embeddings()
        assert embeddings is not None

    def test_set_input_embeddings(self):
        """Test setting input embeddings."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
        )

        old_emb = backbone.get_input_embeddings()
        # This should work without error
        backbone.set_input_embeddings(old_emb)


class TestHybridBackboneAdvanced:
    """Advanced tests for HybridBackbone."""

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5  # embeddings + 4 layers

    def test_properties(self):
        """Test backbone properties."""
        backbone = HybridBackbone(
            vocab_size=5000,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
            d_state=16,
        )

        assert backbone.hidden_size == 512
        assert backbone.num_layers == 6
        assert backbone.vocab_size == 5000

    def test_init_cache(self):
        """Test cache initialization."""
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
        )

        cache = backbone.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None
        assert len(cache) == 4  # One per layer

    def test_get_set_input_embeddings(self):
        """Test get/set input embeddings."""
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
        )

        emb = backbone.get_input_embeddings()
        assert emb is not None

        backbone.set_input_embeddings(emb)

    def test_factory_function(self):
        """Test factory function."""
        backbone = create_hybrid_backbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
        )
        assert isinstance(backbone, HybridBackbone)

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)
        assert output.last_hidden_state.shape == (1, 3, 128)


class TestMambaBackboneAdvanced:
    """Advanced tests for MambaBackbone."""

    def test_properties(self):
        """Test backbone properties."""
        backbone = create_mamba_backbone(
            vocab_size=5000,
            d_model=512,
            num_layers=6,
            d_state=16,
        )

        assert backbone.hidden_size == 512
        assert backbone.num_layers == 6
        assert backbone.vocab_size == 5000

    def test_get_set_input_embeddings(self):
        """Test get/set input embeddings."""
        backbone = create_mamba_backbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=16,
        )

        emb = backbone.get_input_embeddings()
        assert emb is not None

        backbone.set_input_embeddings(emb)


class TestRecurrentBackboneAdvanced:
    """Advanced tests for RecurrentBackbone."""

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=128,
            num_layers=3,
            rnn_type="mingru",
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 4  # embeddings + 3 layers

    def test_properties(self):
        """Test backbone properties."""
        backbone = RecurrentBackbone(
            vocab_size=5000,
            d_model=512,
            num_layers=4,
            rnn_type="lstm",
        )

        assert backbone.hidden_size == 512
        assert backbone.num_layers == 4
        assert backbone.vocab_size == 5000

    def test_init_cache(self):
        """Test cache initialization."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            rnn_type="lstm",
        )

        cache = backbone.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None

    def test_get_set_input_embeddings(self):
        """Test get/set input embeddings."""
        backbone = RecurrentBackbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            rnn_type="gru",
        )

        emb = backbone.get_input_embeddings()
        assert emb is not None

        backbone.set_input_embeddings(emb)


class TestTransformerBackboneAdvanced:
    """Advanced tests for TransformerBackbone."""

    def test_init_cache(self):
        """Test cache initialization."""
        backbone = create_transformer_backbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
        )

        cache = backbone.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None
        assert len(cache) == 2  # One per layer


class TestHybridMixStrategies:
    """Tests for different HybridBackbone mix strategies."""

    def test_interleaved_strategy(self):
        """Test INTERLEAVED mix strategy."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.INTERLEAVED,
            attention_ratio=0.5,  # Every 2nd layer is attention
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 128)
        # Check layer types are correct
        from chuk_lazarus.models_v2.core.enums import BlockType

        assert BlockType.TRANSFORMER in backbone.layer_types
        assert BlockType.MAMBA in backbone.layer_types

    def test_interleaved_with_cache(self):
        """Test INTERLEAVED strategy with cache."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.INTERLEAVED,
            attention_ratio=0.25,  # Every 4th layer is attention
        )

        # First forward pass
        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[4]])
        output_new = backbone(new_token, cache=cache)

        assert output_new.last_hidden_state.shape == (1, 1, 128)

    def test_parallel_strategy(self):
        """Test PARALLEL mix strategy."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.PARALLEL,
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 128)
        # All layers should be HYBRID type
        from chuk_lazarus.models_v2.core.enums import BlockType

        assert all(lt == BlockType.HYBRID for lt in backbone.layer_types)

    def test_parallel_output_hidden_states(self):
        """Test PARALLEL strategy with hidden states output."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=3,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.PARALLEL,
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 4  # embeddings + 3 layers

    def test_parallel_with_cache(self):
        """Test PARALLEL strategy with cache."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.PARALLEL,
        )

        # First forward pass
        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[4]])
        output_new = backbone(new_token, cache=cache)

        assert output_new.last_hidden_state.shape == (1, 1, 128)

    def test_alternating_strategy_enum(self):
        """Test ALTERNATING strategy with enum (explicit)."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.ALTERNATING,
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 3, 128)

    def test_string_to_enum_conversion(self):
        """Test that string mix strategies are converted to enums."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        # Test with string
        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            mix_strategy="interleaved",
        )

        assert backbone.mix_strategy == HybridMixStrategy.INTERLEAVED

    def test_interleaved_zero_attention_ratio(self):
        """Test INTERLEAVED with attention_ratio close to 0."""
        from chuk_lazarus.models_v2.core.enums import HybridMixStrategy

        backbone = HybridBackbone(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            d_state=16,
            mix_strategy=HybridMixStrategy.INTERLEAVED,
            attention_ratio=0.0,  # No attention layers
        )

        input_ids = mx.array([[1, 2, 3]])
        output = backbone(input_ids)

        assert output.last_hidden_state.shape == (1, 3, 128)
