"""
Tests for StarCoder and StarCoder2 model implementation.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.models_v2.families.starcoder2 import (
    StarCoder2Block,
    StarCoder2Config,
    StarCoder2ForCausalLM,
    StarCoder2Model,
    StarCoderBlock,
    StarCoderConfig,
    StarCoderForCausalLM,
    StarCoderModel,
)


class TestStarCoder2Block:
    """Tests for StarCoder2Block."""

    def test_basic_forward(self):
        """Test basic forward pass through block."""
        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=0)

        x = mx.random.normal((2, 10, config.hidden_size))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, config.hidden_size)

    def test_with_mask(self):
        """Test forward with attention mask."""
        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=0)

        x = mx.random.normal((1, 5, config.hidden_size))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = block(x, mask=mask)

        assert output.hidden_states.shape == (1, 5, config.hidden_size)

    def test_with_cache(self):
        """Test forward with KV cache."""
        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=0)

        # First pass
        x = mx.random.normal((1, 5, config.hidden_size))
        output1 = block(x)

        # Second pass with cache
        x2 = mx.random.normal((1, 1, config.hidden_size))
        output2 = block(x2, cache=output1.cache)

        assert output2.hidden_states.shape == (1, 1, config.hidden_size)
        assert output2.cache is not None

    def test_block_type_property(self):
        """Test block_type property returns TRANSFORMER."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=0)

        assert block.block_type == BlockType.TRANSFORMER

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=0)

        assert block.hidden_size == config.hidden_size

    def test_layer_idx(self):
        """Test layer_idx is stored correctly."""
        config = StarCoder2Config.tiny()
        block = StarCoder2Block(config, layer_idx=3)

        assert block.layer_idx == 3

    def test_without_sliding_window(self):
        """Test block without sliding window uses GQA."""
        config = StarCoder2Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            sliding_window=None,
        )
        block = StarCoder2Block(config, layer_idx=0)

        x = mx.random.normal((1, 5, config.hidden_size))
        output = block(x)

        assert output.hidden_states.shape == (1, 5, config.hidden_size)


class TestStarCoder2Model:
    """Tests for StarCoder2Model (backbone)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_with_attention_mask(self):
        """Test forward with attention mask."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_with_cache(self):
        """Test forward with KV cache."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output1 = model(input_ids)
        cache = output1.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output2 = model(new_token, cache=cache)

        assert output2.last_hidden_state.shape == (1, 1, config.hidden_size)
        assert output2.cache is not None
        assert len(output2.cache) == config.num_hidden_layers

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # embeddings + num_layers
        assert len(output.hidden_states) == config.num_hidden_layers + 1

    def test_properties(self):
        """Test model properties."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        assert model.hidden_size == config.hidden_size
        assert model.num_layers == config.num_hidden_layers
        assert model.vocab_size == config.vocab_size

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is not None
        assert embeddings is model.embed_tokens

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        config = StarCoder2Config.tiny()
        model = StarCoder2Model(config)

        from chuk_lazarus.models_v2.components.embeddings import create_token_embedding

        new_embeddings = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        model.set_input_embeddings(new_embeddings)
        assert model.embed_tokens is new_embeddings


class TestStarCoder2ForCausalLM:
    """Tests for StarCoder2ForCausalLM (full model)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is None

    def test_with_labels(self):
        """Test forward pass with labels for loss computation."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is not None
        assert output.loss.shape == ()

    def test_with_cache(self):
        """Test forward pass with cache."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        # First pass to build cache
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output_new = model(new_token, cache=cache)

        assert output_new.logits.shape == (1, 1, config.vocab_size)

    def test_with_attention_mask(self):
        """Test forward with attention mask."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = model(input_ids, attention_mask=mask)

        assert output.logits.shape == (1, 5, config.vocab_size)

    def test_output_hidden_states(self):
        """Test output_hidden_states option."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None

    def test_generate(self):
        """Test text generation."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5  # prompt + generated

    def test_generate_with_temperature(self):
        """Test generation with different temperature."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.5,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            top_k=10,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            repetition_penalty=1.2,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            stop_tokens=[0],
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 3 + 10

    def test_tied_embeddings(self):
        """Test tied word embeddings (default)."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        # Default should have tied embeddings
        assert model.lm_head.tied_embeddings is not None

    def test_untied_embeddings(self):
        """Test untied word embeddings."""
        config = StarCoder2Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            tie_word_embeddings=False,
        )
        model = StarCoder2ForCausalLM(config)

        # LM head should have its own projection
        assert model.lm_head.lm_head is not None

    def test_config_property(self):
        """Test config property."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        assert model.config is config

    def test_backbone_property(self):
        """Test backbone property."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        assert model.backbone is model.model

    def test_from_config(self):
        """Test from_config factory method."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM.from_config(config)

        assert model.config == config


class TestStarCoder2Gradients:
    """Tests for gradient flow through StarCoder2."""

    def test_forward_backward(self):
        """Test full forward-backward pass."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0
        # Check some gradients exist
        assert any(g is not None for g in grads.values())

    def test_gradients_through_attention(self):
        """Test gradients flow through attention."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])

        def loss_fn(model, input_ids):
            output = model(input_ids)
            return mx.mean(output.logits**2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids)

        assert loss.item() > 0


class TestStarCoder2BatchHandling:
    """Tests for batch handling."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        for batch_size in [1, 2, 4]:
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = StarCoder2Config.tiny()
        model = StarCoder2ForCausalLM(config)

        for seq_len in [1, 5, 10, 20]:
            input_ids = mx.random.randint(0, config.vocab_size, (1, seq_len))
            output = model(input_ids)
            assert output.logits.shape == (1, seq_len, config.vocab_size)


class TestStarCoder2FromPretrained:
    """Tests for from_pretrained_async method."""

    @pytest.mark.asyncio
    async def test_from_pretrained_with_config(self, tmp_path):
        """Test from_pretrained_async with provided config."""
        import json

        # Create a config file
        config_data = {
            "model_type": "starcoder2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 256,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load with provided config
        config = StarCoder2Config.tiny()
        model = await StarCoder2ForCausalLM.from_pretrained_async(str(tmp_path), config=config)

        assert model is not None
        assert model.config == config

    @pytest.mark.asyncio
    async def test_from_pretrained_loads_config(self, tmp_path):
        """Test from_pretrained_async loads config from file."""
        import json

        # Create a config file
        config_data = {
            "model_type": "starcoder2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 256,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load without provided config - should load from file
        model = await StarCoder2ForCausalLM.from_pretrained_async(str(tmp_path))

        assert model is not None
        assert model.config.vocab_size == 1000
        assert model.config.hidden_size == 64

    @pytest.mark.asyncio
    async def test_from_pretrained_with_weights(self, tmp_path):
        """Test from_pretrained_async with safetensors weights."""
        import json

        # Create a config file
        config_data = {
            "model_type": "starcoder2",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 256,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Try loading - it will skip weights if safetensors not available or file missing
        config = StarCoder2Config.tiny()
        model = await StarCoder2ForCausalLM.from_pretrained_async(str(tmp_path), config=config)

        assert model is not None


# =============================================================================
# StarCoder (Original) Tests
# =============================================================================


class TestStarCoderBlock:
    """Tests for StarCoderBlock (original GPT-BigCode architecture)."""

    def test_basic_forward(self):
        """Test basic forward pass through block."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        x = mx.random.normal((2, 10, config.hidden_size))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, config.hidden_size)

    def test_with_mask(self):
        """Test forward with attention mask."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        x = mx.random.normal((1, 5, config.hidden_size))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = block(x, mask=mask)

        assert output.hidden_states.shape == (1, 5, config.hidden_size)

    def test_with_cache(self):
        """Test forward with KV cache."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        # First pass
        x = mx.random.normal((1, 5, config.hidden_size))
        output1 = block(x)

        # Second pass with cache
        x2 = mx.random.normal((1, 1, config.hidden_size))
        output2 = block(x2, cache=output1.cache)

        assert output2.hidden_states.shape == (1, 1, config.hidden_size)
        assert output2.cache is not None

    def test_block_type_property(self):
        """Test block_type property returns TRANSFORMER."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        assert block.block_type == BlockType.TRANSFORMER

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        assert block.hidden_size == config.hidden_size

    def test_layer_idx(self):
        """Test layer_idx is stored correctly."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=3)

        assert block.layer_idx == 3

    def test_mqa_kv_heads(self):
        """Test block uses MQA (single KV head)."""
        config = StarCoderConfig.tiny()
        block = StarCoderBlock(config, layer_idx=0)

        # MQA should have 1 KV head
        assert config.num_key_value_heads == 1

        x = mx.random.normal((1, 5, config.hidden_size))
        output = block(x)

        assert output.hidden_states.shape == (1, 5, config.hidden_size)


class TestStarCoderModel:
    """Tests for StarCoderModel (backbone)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_learned_position_embeddings(self):
        """Test that learned position embeddings are used."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        # Model should have embed_positions
        assert hasattr(model, "embed_positions")
        assert model.embed_positions is not None

    def test_with_attention_mask(self):
        """Test forward with attention mask."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_with_cache(self):
        """Test forward with KV cache."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output1 = model(input_ids)
        cache = output1.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output2 = model(new_token, cache=cache)

        assert output2.last_hidden_state.shape == (1, 1, config.hidden_size)
        assert output2.cache is not None
        assert len(output2.cache) == config.num_hidden_layers

    def test_with_position_ids(self):
        """Test forward with explicit position_ids."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        position_ids = mx.array([[0, 1, 2, 3, 4]])

        output = model(input_ids, position_ids=position_ids)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # embeddings + num_layers
        assert len(output.hidden_states) == config.num_hidden_layers + 1

    def test_properties(self):
        """Test model properties."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        assert model.hidden_size == config.hidden_size
        assert model.num_layers == config.num_hidden_layers
        assert model.vocab_size == config.vocab_size

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is not None
        assert embeddings is model.embed_tokens

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        config = StarCoderConfig.tiny()
        model = StarCoderModel(config)

        from chuk_lazarus.models_v2.components.embeddings import create_token_embedding

        new_embeddings = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        model.set_input_embeddings(new_embeddings)
        assert model.embed_tokens is new_embeddings


class TestStarCoderForCausalLM:
    """Tests for StarCoderForCausalLM (full model)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is None

    def test_with_labels(self):
        """Test forward pass with labels for loss computation."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is not None
        assert output.loss.shape == ()

    def test_with_cache(self):
        """Test forward pass with cache."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        # First pass to build cache
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output_new = model(new_token, cache=cache)

        assert output_new.logits.shape == (1, 1, config.vocab_size)

    def test_with_attention_mask(self):
        """Test forward with attention mask."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = model(input_ids, attention_mask=mask)

        assert output.logits.shape == (1, 5, config.vocab_size)

    def test_with_position_ids(self):
        """Test forward with explicit position_ids."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        position_ids = mx.array([[0, 1, 2, 3, 4]])

        output = model(input_ids, position_ids=position_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)

    def test_output_hidden_states(self):
        """Test output_hidden_states option."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None

    def test_generate(self):
        """Test text generation."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5  # prompt + generated

    def test_generate_with_temperature(self):
        """Test generation with different temperature."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.5,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            top_k=10,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            repetition_penalty=1.2,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            stop_tokens=[0],
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 3 + 10

    def test_tied_embeddings(self):
        """Test tied word embeddings (default)."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        # Default should have tied embeddings
        assert model.lm_head.tied_embeddings is not None

    def test_untied_embeddings(self):
        """Test untied word embeddings."""
        config = StarCoderConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=128,
            tie_word_embeddings=False,
        )
        model = StarCoderForCausalLM(config)

        # LM head should have its own projection
        assert model.lm_head.lm_head is not None

    def test_config_property(self):
        """Test config property."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        assert model.config is config

    def test_backbone_property(self):
        """Test backbone property."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        assert model.backbone is model.model

    def test_from_config(self):
        """Test from_config factory method."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM.from_config(config)

        assert model.config == config


class TestStarCoderGradients:
    """Tests for gradient flow through StarCoder (original)."""

    def test_forward_backward(self):
        """Test full forward-backward pass."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0
        # Check some gradients exist
        assert any(g is not None for g in grads.values())

    def test_gradients_through_attention(self):
        """Test gradients flow through attention."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])

        def loss_fn(model, input_ids):
            output = model(input_ids)
            return mx.mean(output.logits**2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids)

        assert loss.item() > 0


class TestStarCoderBatchHandling:
    """Tests for batch handling in StarCoder (original)."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        for batch_size in [1, 2, 4]:
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = StarCoderConfig.tiny()
        model = StarCoderForCausalLM(config)

        for seq_len in [1, 5, 10, 20]:
            input_ids = mx.random.randint(0, config.vocab_size, (1, seq_len))
            output = model(input_ids)
            assert output.logits.shape == (1, seq_len, config.vocab_size)
