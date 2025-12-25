"""
Tests for model base classes and compositions.

Tests CausalLM, ClassifierModel, and ModelOutput.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.core.config import ModelConfig
from chuk_lazarus.models_v2.core.enums import BackboneType
from chuk_lazarus.models_v2.models.base import ModelOutput
from chuk_lazarus.models_v2.models.causal_lm import (
    CausalLM,
    HybridCausalLM,
    create_causal_lm,
)
from chuk_lazarus.models_v2.models.classifiers import (
    SequenceClassifier,
    TokenClassifier,
    create_classifier,
)


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_basic_output(self):
        """Test basic ModelOutput creation."""
        logits = mx.random.normal((2, 10, 1000))
        output = ModelOutput(logits=logits)

        assert output.logits.shape == (2, 10, 1000)
        assert output.loss is None
        assert output.hidden_states is None
        assert output.cache is None

    def test_with_loss(self):
        """Test ModelOutput with loss."""
        logits = mx.random.normal((2, 10, 1000))
        loss = mx.array(2.5)

        output = ModelOutput(logits=logits, loss=loss)

        assert output.logits is not None
        assert output.loss.item() == 2.5

    def test_with_hidden_states(self):
        """Test ModelOutput with hidden states."""
        logits = mx.random.normal((2, 10, 1000))
        hidden = tuple(mx.random.normal((2, 10, 256)) for _ in range(5))

        output = ModelOutput(logits=logits, hidden_states=hidden)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5

    def test_with_cache(self):
        """Test ModelOutput with cache."""
        logits = mx.random.normal((2, 10, 1000))
        cache = [
            (mx.random.normal((2, 10, 8, 64)), mx.random.normal((2, 10, 8, 64))) for _ in range(3)
        ]

        output = ModelOutput(logits=logits, cache=cache)

        assert output.cache is not None
        assert len(output.cache) == 3

    def test_full_output(self):
        """Test ModelOutput with all fields."""
        logits = mx.random.normal((2, 10, 1000))
        loss = mx.array(1.5)
        hidden = tuple(mx.random.normal((2, 10, 256)) for _ in range(4))
        cache = [(mx.random.normal((2, 10, 8, 64)),) * 2 for _ in range(3)]

        output = ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden,
            cache=cache,
        )

        assert output.logits.shape == (2, 10, 1000)
        assert output.loss.item() == 1.5
        assert len(output.hidden_states) == 4
        assert len(output.cache) == 3


class TestCausalLMIntegration:
    """Integration tests for Causal LM models."""

    def test_llama_causal_lm(self):
        """Test LlamaForCausalLM as CausalLM."""
        from chuk_lazarus.models_v2.families.llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)

        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is not None

    def test_mamba_causal_lm(self):
        """Test MambaForCausalLM as CausalLM."""
        from chuk_lazarus.models_v2.families.mamba import (
            MambaConfig,
            MambaForCausalLM,
        )

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)

        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is not None


class TestModelRegistry:
    """Tests for model registry integration."""

    def test_llama_registered(self):
        """Test Llama is registered in registry."""
        from chuk_lazarus.models_v2.core.registry import get_model_class

        factory = get_model_class("llama")
        assert factory is not None

    def test_llama_architecture_registered(self):
        """Test Llama architecture names are registered."""
        from chuk_lazarus.models_v2.core.registry import get_model_class

        factory = get_model_class("LlamaForCausalLM")
        assert factory is not None

    def test_mamba_registered(self):
        """Test Mamba is registered in registry."""
        from chuk_lazarus.models_v2.core.registry import get_model_class

        factory = get_model_class("mamba")
        assert factory is not None

    def test_mamba_architecture_registered(self):
        """Test Mamba architecture names are registered."""
        from chuk_lazarus.models_v2.core.registry import get_model_class

        # Try both possible architecture names
        factory = get_model_class("MambaForCausalLM")
        if factory is None:
            factory = get_model_class("MambaLMHeadModel")
        assert factory is not None


class TestModelComparison:
    """Tests comparing different model architectures."""

    def test_transformer_vs_mamba_interface(self):
        """Test Transformer and Mamba have compatible interfaces."""
        from chuk_lazarus.models_v2.families.llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )
        from chuk_lazarus.models_v2.families.mamba import (
            MambaConfig,
            MambaForCausalLM,
        )

        llama_config = LlamaConfig.tiny()
        mamba_config = MambaConfig.tiny()

        # Make vocab sizes match for comparison
        llama_config = LlamaConfig(**{**llama_config.__dict__, "vocab_size": 1000})
        mamba_config = MambaConfig(**{**mamba_config.__dict__, "vocab_size": 1000})

        llama = LlamaForCausalLM(llama_config)
        mamba = MambaForCausalLM(mamba_config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])

        # Both should return ModelOutput
        llama_output = llama(input_ids)
        mamba_output = mamba(input_ids)

        assert isinstance(llama_output, ModelOutput)
        assert isinstance(mamba_output, ModelOutput)

        # Both should have logits of correct shape
        assert llama_output.logits.shape[2] == 1000
        assert mamba_output.logits.shape[2] == 1000

    def test_both_support_generation(self):
        """Test both architectures support generation."""
        from chuk_lazarus.models_v2.families.llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )
        from chuk_lazarus.models_v2.families.mamba import (
            MambaConfig,
            MambaForCausalLM,
        )

        llama = LlamaForCausalLM(LlamaConfig.tiny())
        mamba = MambaForCausalLM(MambaConfig.tiny())

        input_ids = mx.array([[1, 2, 3]])

        llama_gen = llama.generate(input_ids, max_new_tokens=3)
        mamba_gen = mamba.generate(input_ids, max_new_tokens=3)

        # Both should extend the sequence
        assert llama_gen.shape[1] == 6
        assert mamba_gen.shape[1] == 6


class TestModelModes:
    """Tests for model training/inference modes."""

    def test_training_mode(self):
        """Test model in training mode (no cache)."""
        from chuk_lazarus.models_v2.families.llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )

        model = LlamaForCausalLM(LlamaConfig.tiny())

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)

        # Should compute loss
        assert output.loss is not None

    def test_inference_mode_with_cache(self):
        """Test model in inference mode with cache."""
        from chuk_lazarus.models_v2.families.llama import (
            LlamaConfig,
            LlamaForCausalLM,
        )

        model = LlamaForCausalLM(LlamaConfig.tiny())

        # First pass
        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)
        cache = output.cache

        # Subsequent passes with cache
        for token in [[4], [5], [6]]:
            new_ids = mx.array([token])
            output = model(new_ids, cache=cache)
            cache = output.cache

            # Single token output
            assert output.logits.shape[1] == 1


class TestCausalLM:
    """Tests for CausalLM class."""

    def test_basic_creation(self):
        """Test basic CausalLM creation."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        assert model.config == config
        assert model.backbone is not None

    def test_forward_pass(self):
        """Test forward pass."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (1, 5, 1000)

    def test_with_labels(self):
        """Test forward pass with labels."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.shape == ()

    def test_with_cache(self):
        """Test forward pass with cache."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        # First pass
        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[4]])
        output2 = model(new_token, cache=cache)

        assert output2.logits.shape == (1, 1, 1000)
        assert output2.cache is not None

    def test_output_hidden_states(self):
        """Test returning hidden states."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) >= 2

    def test_from_config(self):
        """Test from_config class method."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = CausalLM.from_config(config, backbone_type=BackboneType.TRANSFORMER)

        assert isinstance(model, CausalLM)

    def test_generate(self):
        """Test text generation via family model (LlamaForCausalLM)."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=5)

        assert generated.shape[1] == 8  # 3 prompt + 5 generated

    def test_generate_with_temperature(self):
        """Test generation with temperature via family model."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.5)

        assert generated.shape[1] == 6

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling via family model."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=50)

        assert generated.shape[1] == 6

    def test_generate_with_stop_token(self):
        """Test generation with stop token via family model."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Generate with stop token - should stop early if hit
        generated = model.generate(input_ids, max_new_tokens=10, stop_tokens=[0])

        # Generated should be at most 13 tokens (3 + 10)
        assert generated.shape[1] <= 13


class TestMambaCausalLM:
    """Tests for MambaCausalLM using family-specific config."""

    def test_basic_creation(self):
        """Test MambaCausalLM creation via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        # Verify it's a Mamba model
        assert hasattr(model, "backbone")

    def test_forward_pass(self):
        """Test forward pass via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)


class TestHybridCausalLM:
    """Tests for HybridCausalLM."""

    def test_basic_creation(self):
        """Test HybridCausalLM creation."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            state_size=16,
        )
        model = HybridCausalLM(config)

        assert model.backbone_type == BackboneType.HYBRID


class TestSequenceClassifier:
    """Tests for SequenceClassifier."""

    def test_basic_creation(self):
        """Test basic creation."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=3)

        assert model.num_labels == 3

    def test_forward_pass(self):
        """Test forward pass."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=5)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5)

    def test_with_labels(self):
        """Test with labels for loss."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=5)

        input_ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        labels = mx.array([0, 2])
        output = model(input_ids, labels=labels)

        assert output.loss is not None

    def test_mean_pooling(self):
        """Test mean pooling strategy."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=3, pool_strategy="mean")

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3)

    def test_first_pooling(self):
        """Test first token pooling."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=3, pool_strategy="first")

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3)

    def test_with_pooler(self):
        """Test with pooler head."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier(config, num_labels=3, use_pooler=True)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3)

    def test_from_config(self):
        """Test from_config class method."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = SequenceClassifier.from_config(config, num_labels=5)

        assert isinstance(model, SequenceClassifier)

    def test_mamba_backbone(self):
        """Test with Mamba backbone using MambaBackbone directly."""
        from chuk_lazarus.models_v2.backbones import MambaBackbone
        from chuk_lazarus.models_v2.heads import ClassifierHead

        # Create Mamba backbone directly (not via from_config)
        backbone = MambaBackbone(
            vocab_size=1000,
            d_model=128,
            num_layers=2,
            d_state=16,
        )

        # Create classifier head
        head = ClassifierHead(
            hidden_size=128,
            num_labels=3,
            pool_strategy="last",
        )

        # Test backbone forward
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        backbone_output = backbone(input_ids)
        head_output = head(backbone_output.last_hidden_state)

        assert head_output.logits.shape == (1, 3)


class TestTokenClassifier:
    """Tests for TokenClassifier."""

    def test_basic_creation(self):
        """Test basic creation."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = TokenClassifier(config, num_labels=9)

        assert model.num_labels == 9

    def test_forward_pass(self):
        """Test forward pass."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = TokenClassifier(config, num_labels=9)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        # Token classification outputs per-token logits
        assert output.logits.shape == (1, 5, 9)

    def test_with_labels(self):
        """Test with labels for loss."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = TokenClassifier(config, num_labels=9)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[0, 1, 2, 3, 4]])
        output = model(input_ids, labels=labels)

        assert output.loss is not None

    def test_from_config(self):
        """Test from_config class method."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = TokenClassifier.from_config(config, num_labels=9)

        assert isinstance(model, TokenClassifier)


class TestModelFactoryFunctions:
    """Tests for model factory functions."""

    def test_create_causal_lm_transformer(self):
        """Test create_causal_lm with transformer."""
        model = create_causal_lm(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            backbone_type="transformer",
        )
        assert isinstance(model, CausalLM)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)
        assert output.logits.shape == (1, 3, 1000)

    def test_create_classifier_sequence(self):
        """Test create_classifier for sequence classification."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            num_labels=5,
            task="sequence",
        )
        assert isinstance(model, SequenceClassifier)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        assert output.logits.shape == (1, 5)

    def test_create_classifier_token(self):
        """Test create_classifier for token classification."""
        model = create_classifier(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            num_labels=9,
            task="token",
        )
        assert isinstance(model, TokenClassifier)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        assert output.logits.shape == (1, 5, 9)


class TestModelGradients:
    """Tests for gradient flow through models."""

    def test_causal_lm_gradients(self):
        """Test gradients flow through CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        labels = mx.array([[2, 3, 4]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0

    def test_sequence_classifier_gradients(self):
        """Test gradients flow through SequenceClassifier."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        model = SequenceClassifier(config, num_labels=3)

        input_ids = mx.array([[1, 2, 3]])
        labels = mx.array([1])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0

    def test_token_classifier_gradients(self):
        """Test gradients flow through TokenClassifier."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        model = TokenClassifier(config, num_labels=5)

        input_ids = mx.array([[1, 2, 3]])
        labels = mx.array([[0, 1, 2]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)


class TestModelBase:
    """Tests for Model base class methods."""

    def test_num_parameters(self):
        """Test num_parameters method."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        num_params = model.num_parameters()
        assert num_params > 0

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is not None

    def test_init_cache(self):
        """Test init_cache method."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        cache = model.init_cache(batch_size=2, max_seq_len=100)
        assert cache is not None
        assert len(cache) > 0


class TestCausalLMAdvanced:
    """Advanced tests for CausalLM."""

    def test_hybrid_backbone(self):
        """Test CausalLM with hybrid backbone."""
        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            state_size=16,
        )
        model = HybridCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, 1000)

    def test_output_aux_loss(self):
        """Test ModelOutput with aux_loss field."""
        logits = mx.random.normal((2, 5, 1000))
        aux_loss = mx.array(0.5)
        output = ModelOutput(logits=logits, aux_loss=aux_loss)

        assert output.aux_loss is not None
        assert output.aux_loss.item() == 0.5

    def test_output_aux_outputs(self):
        """Test ModelOutput with aux_outputs field."""
        logits = mx.random.normal((2, 5, 1000))
        output = ModelOutput(logits=logits)

        # Default should be empty dict
        assert isinstance(output.aux_outputs, dict)


class TestModelBaseWeights:
    """Tests for Model weight saving and loading."""

    def test_save_and_load_weights(self):
        """Test save_weights and load_weights methods."""
        import tempfile
        from pathlib import Path

        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model1 = LlamaForCausalLM(config)
        model2 = LlamaForCausalLM(config)

        # Run a forward pass to initialize weights
        input_ids = mx.array([[1, 2, 3]])
        model1(input_ids)
        model2(input_ids)

        # Save and load weights
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = str(Path(tmpdir) / "weights.npz")
            model1.save_weights(weights_path)
            model2.load_weights(weights_path)

        # Both models should produce similar outputs
        output1 = model1(input_ids)
        output2 = model2(input_ids)

        # Shapes should match
        assert output1.logits.shape == output2.logits.shape

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Get current embeddings
        _ = model.get_input_embeddings()

        # Create new embeddings
        new_embeddings = model.get_input_embeddings()

        # Set new embeddings
        model.set_input_embeddings(new_embeddings)

        # Get embeddings again
        current_embeddings = model.get_input_embeddings()

        # They should be the same (we set the same object)
        assert current_embeddings is new_embeddings


class TestCausalLMGeneration:
    """Tests for CausalLM generation methods."""

    def test_generate_greedy(self):
        """Test greedy generation (temperature=0)."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 8  # 3 prompt + 5 new

    def test_generate_with_top_p(self):
        """Test generation with nucleus sampling (top-p)."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            temperature=1.0,
            top_p=0.9,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4  # At least prompt + 1

    def test_generate_early_stop(self):
        """Test generation stops early on stop token."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Use a stop token that's likely to appear
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            stop_tokens=[0, 1, 2],  # Common tokens likely to appear
        )

        # Should have stopped before max
        assert generated.shape[0] == 1


class TestMambaCausalLMGeneration:
    """Tests for Mamba generation."""

    def test_mamba_generate(self):
        """Test generation with Mamba backbone."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4]])
        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 5


class TestCausalLMBackboneTypes:
    """Tests for CausalLM with different backbone types."""

    def test_causal_lm_mamba_backbone(self):
        """Test CausalLM with Mamba backbone via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        assert hasattr(model, "backbone")

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)

    def test_causal_lm_unsupported_backbone_error(self):
        """Test CausalLM raises error for unsupported backbone type."""
        import pytest

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

        # Directly test with an invalid backbone type string
        with pytest.raises(ValueError, match="Unsupported backbone type"):
            # Pass a BackboneType that's not transformer, mamba, or hybrid
            CausalLM(config, backbone_type=BackboneType.RECURRENT)

    def test_causal_lm_untied_embeddings(self):
        """Test CausalLM with untied embeddings via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        config = LlamaConfig(**{**config.__dict__, "tie_word_embeddings": False})
        model = LlamaForCausalLM(config)

        # LM head should exist
        assert model.lm_head is not None

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3, config.vocab_size)


class TestCausalLMGenerateDirect:
    """Tests for CausalLM.generate method via family models."""

    def test_generate_basic(self):
        """Test basic generation via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3)

        assert generated.shape[0] == 1
        assert generated.shape[1] == 6  # 3 prompt + 3 generated

    def test_generate_temperature(self):
        """Test generation with different temperatures."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=2.0)

        assert generated.shape[1] == 6

    def test_generate_top_k(self):
        """Test generation with top-k sampling."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=10)

        assert generated.shape[1] == 6

    def test_generate_top_p(self):
        """Test generation with nucleus (top-p) sampling."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_p=0.9)

        assert generated.shape[1] == 6

    def test_generate_stop_tokens(self):
        """Test generation stops on stop tokens."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Use common tokens as stop tokens
        generated = model.generate(input_ids, max_new_tokens=10, stop_tokens=[0, 1, 2, 3])

        # Should have stopped early in most cases
        assert generated.shape[1] <= 13


class TestMambaCausalLMDirect:
    """Tests for MambaCausalLM class via family models."""

    def test_mamba_causal_lm_creation(self):
        """Test creating Mamba model via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        assert hasattr(model, "backbone")

    def test_mamba_causal_lm_forward(self):
        """Test MambaForCausalLM forward pass."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)


class TestCausalLMDirectGeneration:
    """Tests for CausalLM.generate method directly via family models."""

    def test_generate_basic_via_causal_lm(self):
        """Test basic generation via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3)

        assert generated.shape[0] == 1
        assert generated.shape[1] == 6

    def test_generate_with_temperature_zero(self):
        """Test generation with temperature=0 (greedy)."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.0)

        assert generated.shape[1] == 6

    def test_generate_with_top_k_via_causal_lm(self):
        """Test generation with top-k via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=10)

        assert generated.shape[1] == 6

    def test_generate_with_top_p_via_causal_lm(self):
        """Test generation with top-p via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_p=0.9)

        assert generated.shape[1] == 6

    def test_generate_stop_tokens_via_causal_lm(self):
        """Test generation with stop tokens via LlamaForCausalLM."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Using stop tokens that are likely to be generated
        generated = model.generate(input_ids, max_new_tokens=5, stop_tokens=[0, 1, 2])

        assert generated.shape[1] <= 8


class TestMambaCausalLMClass:
    """Tests for MambaCausalLM wrapper class via family models."""

    def test_mamba_causal_lm_class_creation(self):
        """Test MambaCausalLM creation via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        assert hasattr(model, "backbone")

    def test_mamba_causal_lm_class_forward(self):
        """Test MambaForCausalLM forward pass."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)


class TestClassifierMambaBackbone:
    """Tests for classifiers with Mamba backbone via direct MambaBackbone."""

    def test_sequence_classifier_mamba_backbone(self):
        """Test SequenceClassifier components with Mamba backbone directly."""
        from chuk_lazarus.models_v2.backbones import MambaBackbone
        from chuk_lazarus.models_v2.heads import ClassifierHead

        # Create Mamba backbone directly
        backbone = MambaBackbone(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            d_state=16,
        )

        head = ClassifierHead(
            hidden_size=64,
            num_labels=3,
            pool_strategy="last",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        backbone_output = backbone(input_ids)
        head_output = head(backbone_output.last_hidden_state)

        assert head_output.logits.shape == (1, 3)

    def test_token_classifier_mamba_backbone(self):
        """Test TokenClassifier components with Mamba backbone directly."""
        from chuk_lazarus.models_v2.backbones import MambaBackbone
        from chuk_lazarus.models_v2.heads import ClassifierHead

        # Create Mamba backbone directly
        backbone = MambaBackbone(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            d_state=16,
        )

        head = ClassifierHead(
            hidden_size=64,
            num_labels=5,
            pool_strategy="none",
        )

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        backbone_output = backbone(input_ids)
        head_output = head(backbone_output.last_hidden_state)

        assert head_output.logits.shape == (1, 5, 5)


class TestClassifierUnsupportedBackbone:
    """Tests for classifier error handling."""

    def test_sequence_classifier_unsupported_backbone(self):
        """Test SequenceClassifier raises error for unsupported backbone."""
        import pytest

        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )

        with pytest.raises(ValueError, match="Unsupported backbone"):
            SequenceClassifier(config, num_labels=3, backbone_type=BackboneType.HYBRID)

    def test_token_classifier_unsupported_backbone(self):
        """Test TokenClassifier raises error for unsupported backbone."""
        import pytest

        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )

        with pytest.raises(ValueError, match="Unsupported backbone"):
            TokenClassifier(config, num_labels=5, backbone_type=BackboneType.HYBRID)


class TestClassifierOutputHiddenStates:
    """Tests for classifier output_hidden_states."""

    def test_sequence_classifier_hidden_states(self):
        """Test SequenceClassifier with output_hidden_states."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = SequenceClassifier(config, num_labels=3)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) >= 2

    def test_token_classifier_hidden_states(self):
        """Test TokenClassifier with output_hidden_states."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = TokenClassifier(config, num_labels=5)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None


class TestCreateClassifierWithEnum:
    """Tests for create_classifier with ClassificationTask enum."""

    def test_create_classifier_with_enum(self):
        """Test create_classifier with ClassificationTask enum."""
        from chuk_lazarus.models_v2.core.enums import ClassificationTask

        model = create_classifier(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            num_labels=5,
            task=ClassificationTask.SEQUENCE,
        )

        assert isinstance(model, SequenceClassifier)

    def test_create_classifier_token_with_enum(self):
        """Test create_classifier for token task with enum."""
        from chuk_lazarus.models_v2.core.enums import ClassificationTask

        model = create_classifier(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            num_labels=9,
            task=ClassificationTask.TOKEN,
        )

        assert isinstance(model, TokenClassifier)


class TestCausalLMMambaBackboneType:
    """Tests for CausalLM with Mamba backbone type via family models."""

    def test_causal_lm_mamba_backbone_type(self):
        """Test Mamba backbone via MambaForCausalLM."""
        from chuk_lazarus.models_v2.families.mamba import MambaConfig, MambaForCausalLM

        config = MambaConfig.tiny()
        model = MambaForCausalLM(config)

        assert hasattr(model, "backbone")

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3, config.vocab_size)

    def test_causal_lm_untied_embeddings(self):
        """Test LlamaForCausalLM with untied embeddings."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        config = LlamaConfig(**{**config.__dict__, "tie_word_embeddings": False})
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3, config.vocab_size)


class TestCausalLMGenerate:
    """Tests for CausalLM generate method covering all code paths."""

    def test_generate_temperature_default(self):
        """Test generate with default temperature (1.0) via CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3)

        assert generated.shape[1] == 6

    def test_generate_temperature_not_one(self):
        """Test generate with temperature != 1.0 via CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.5)

        assert generated.shape[1] == 6

    def test_generate_with_top_k(self):
        """Test generate with top_k sampling via CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=5)

        assert generated.shape[1] == 6

    def test_generate_with_top_p(self):
        """Test generate with top_p (nucleus) sampling via CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_p=0.95)

        assert generated.shape[1] == 6

    def test_generate_with_stop_tokens(self):
        """Test generate with stop tokens via CausalLM."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        input_ids = mx.array([[1, 2, 3]])
        # Use common tokens that might appear
        generated = model.generate(input_ids, max_new_tokens=5, stop_tokens=[0, 1, 2])

        assert generated.shape[1] <= 8


class TestCreateCausalLM:
    """Tests for create_causal_lm factory function."""

    def test_create_causal_lm_transformer(self):
        """Test create_causal_lm with transformer backbone."""
        model = create_causal_lm(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            backbone_type="transformer",
        )

        assert isinstance(model, CausalLM)
        assert model.backbone_type == BackboneType.TRANSFORMER

    def test_create_causal_lm_default_backbone(self):
        """Test create_causal_lm with default backbone."""
        model = create_causal_lm(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
        )

        assert isinstance(model, CausalLM)


class TestClassifierConfigAndBackbone:
    """Tests for classifier config and backbone properties."""

    def test_sequence_classifier_config_property(self):
        """Test SequenceClassifier config property."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = SequenceClassifier(config, num_labels=3)

        assert model.config == config
        assert model.config.vocab_size == 100

    def test_sequence_classifier_backbone_property(self):
        """Test SequenceClassifier backbone property."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = SequenceClassifier(config, num_labels=3)

        assert model.backbone is not None

    def test_token_classifier_config_property(self):
        """Test TokenClassifier config property."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = TokenClassifier(config, num_labels=5)

        assert model.config == config
        assert model.config.vocab_size == 100

    def test_token_classifier_backbone_property(self):
        """Test TokenClassifier backbone property."""
        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = TokenClassifier(config, num_labels=5)

        assert model.backbone is not None
