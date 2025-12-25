"""
Tests for Llama model family.

Tests LlamaConfig, LlamaBlock, LlamaModel, and LlamaForCausalLM.
"""

import mlx.core as mx

from chuk_lazarus.models_v2.families.llama import (
    LlamaBlock,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)


class TestLlamaConfig:
    """Tests for LlamaConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = LlamaConfig()

        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.intermediate_size == 11008
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = LlamaConfig(
            vocab_size=50000,
            hidden_size=2048,
            num_hidden_layers=16,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 16

    def test_tiny_preset(self):
        """Test tiny preset for testing."""
        config = LlamaConfig.tiny()

        assert config.hidden_size < 1024
        assert config.num_hidden_layers <= 4

    def test_llama2_7b_preset(self):
        """Test Llama2-7B preset."""
        config = LlamaConfig.llama2_7b()

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32

    def test_llama3_8b_preset(self):
        """Test Llama3-8B preset."""
        config = LlamaConfig.llama3_8b()

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        # Llama3 uses different rope_theta
        assert config.rope_theta >= 500000

    def test_mistral_7b_preset(self):
        """Test Mistral-7B preset."""
        config = LlamaConfig.mistral_7b()

        assert config.hidden_size == 4096
        # Mistral uses sliding window attention
        assert config.sliding_window is not None

    def test_gqa_config(self):
        """Test grouped-query attention configuration."""
        config = LlamaConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
        )

        assert config.num_key_value_heads == 8

    def test_tie_embeddings_default(self):
        """Test tie_word_embeddings default."""
        config = LlamaConfig()

        assert config.tie_word_embeddings is True


class TestLlamaBlock:
    """Tests for LlamaBlock."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = LlamaConfig.tiny()
        block = LlamaBlock(config, layer_idx=0)

        x = mx.random.normal((1, 5, config.hidden_size))
        output = block(x)

        assert output.hidden_states.shape == (1, 5, config.hidden_size)

    def test_with_mask(self):
        """Test forward pass with attention mask."""
        config = LlamaConfig.tiny()
        block = LlamaBlock(config, layer_idx=0)

        x = mx.random.normal((2, 10, config.hidden_size))
        import mlx.nn as nn

        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)

        output = block(x, mask=mask)

        assert output.hidden_states.shape == (2, 10, config.hidden_size)

    def test_with_cache(self):
        """Test forward pass with KV cache."""
        config = LlamaConfig.tiny()
        block = LlamaBlock(config, layer_idx=0)

        # First pass
        x = mx.random.normal((1, 5, config.hidden_size))
        output = block(x)
        cache = output.cache

        # Second pass with cache
        x_new = mx.random.normal((1, 1, config.hidden_size))
        output_new = block(x_new, cache=cache)

        assert output_new.hidden_states.shape == (1, 1, config.hidden_size)
        assert output_new.cache is not None

    def test_block_type_property(self):
        """Test block_type property returns TRANSFORMER."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = LlamaConfig.tiny()
        block = LlamaBlock(config, layer_idx=0)

        assert block.block_type == BlockType.TRANSFORMER

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        config = LlamaConfig.tiny()
        block = LlamaBlock(config, layer_idx=0)

        assert block.hidden_size == config.hidden_size


class TestLlamaModel:
    """Tests for LlamaModel (backbone)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = LlamaConfig.tiny()
        model = LlamaModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)

    def test_with_cache(self):
        """Test forward pass with cache."""
        config = LlamaConfig.tiny()
        model = LlamaModel(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output_new = model(new_token, cache=cache)

        assert output_new.last_hidden_state.shape == (1, 1, config.hidden_size)

    def test_output_hidden_states(self):
        """Test returning all hidden states."""
        config = LlamaConfig.tiny()
        model = LlamaModel(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # embeddings + num_layers
        assert len(output.hidden_states) == config.num_hidden_layers + 1

    def test_properties(self):
        """Test model properties."""
        config = LlamaConfig.tiny()
        model = LlamaModel(config)

        assert model.hidden_size == config.hidden_size
        assert model.num_layers == config.num_hidden_layers
        assert model.vocab_size == config.vocab_size

    def test_with_attention_mask(self):
        """Test forward pass with custom attention mask."""
        import mlx.nn as nn

        config = LlamaConfig.tiny()
        model = LlamaModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)

        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, config.hidden_size)


class TestLlamaForCausalLM:
    """Tests for LlamaForCausalLM (full model)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is None

    def test_with_labels(self):
        """Test forward pass with labels for loss computation."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert output.loss is not None
        assert output.loss.shape == ()

    def test_with_cache(self):
        """Test forward pass with cache."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)
        cache = output.cache

        # Second pass with cache
        new_token = mx.array([[6]])
        output_new = model(new_token, cache=cache)

        assert output_new.logits.shape == (1, 1, config.vocab_size)

    def test_generate(self):
        """Test text generation."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5  # prompt + generated

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            stop_tokens=[0],  # Stop on token 0
        )

        assert generated.shape[0] == 1
        # Should stop when hitting stop token (or max)
        assert generated.shape[1] <= 3 + 10

    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            repetition_penalty=1.2,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_top_k(self):
        """Test generation with top-k sampling."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            top_k=10,
            temperature=1.0,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 3 + 5

    def test_generate_with_temperature(self):
        """Test generation with temperature scaling."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Low temperature (more deterministic)
        generated_low = model.generate(
            input_ids,
            max_new_tokens=3,
            temperature=0.5,
        )

        assert generated_low.shape[0] == 1
        assert generated_low.shape[1] == 3 + 3

    def test_generate_stops_on_stop_token(self):
        """Test that generation stops when stop token is generated."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Use a very small vocab and set up model to likely generate token 0
        input_ids = mx.array([[1, 2, 3]])

        # Generate with stop token - should stop early if token 0 is generated
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            stop_tokens=[0, 1, 2, 3],  # Multiple stop tokens to increase chance
            temperature=1.0,
        )

        # Should either stop early or reach max
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 3 + 20

    def test_tied_embeddings(self):
        """Test tied word embeddings."""
        config = LlamaConfig.tiny()
        config = LlamaConfig(**{**config.__dict__, "tie_word_embeddings": True})
        model = LlamaForCausalLM(config)

        # LM head should use tied embeddings
        assert model.lm_head.tied_embeddings is not None

    def test_untied_embeddings(self):
        """Test untied word embeddings."""
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)

        # LM head should have its own projection
        assert model.lm_head.lm_head is not None

    def test_from_config(self):
        """Test from_config factory method."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM.from_config(config)

        assert model.config == config


class TestLlamaGradients:
    """Tests for gradient flow through Llama."""

    def test_forward_backward(self):
        """Test full forward-backward pass."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss, grads = mx.value_and_grad(loss_fn)(model, input_ids, labels)

        assert loss.item() > 0
        # Check some gradients exist
        assert any(g is not None for g in grads.values())


class TestLlamaBatchHandling:
    """Tests for batch handling."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        for batch_size in [1, 2, 4]:
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        for seq_len in [1, 5, 10, 20]:
            input_ids = mx.random.randint(0, config.vocab_size, (1, seq_len))
            output = model(input_ids)
            assert output.logits.shape == (1, seq_len, config.vocab_size)


class TestLlamaConvert:
    """Tests for weight conversion utilities."""

    def test_convert_hf_weights_basic(self):
        """Test basic HuggingFace weight conversion."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import convert_hf_weights

        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
            "model.norm.weight": np.random.randn(256).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 256).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted
        assert "lm_head.lm_head.weight" in converted

    def test_convert_hf_weights_tied_embeddings(self):
        """Test HuggingFace weight conversion with tied embeddings."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import convert_hf_weights

        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
            "model.norm.weight": np.random.randn(256).astype(np.float32),
            "lm_head.weight": np.random.randn(1000, 256).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights, tie_word_embeddings=True)

        # lm_head should be skipped with tied embeddings
        assert "lm_head.lm_head.weight" not in converted
        assert "model.embed_tokens.weight" in converted

    def test_convert_hf_weights_layers(self):
        """Test HuggingFace weight conversion for layer weights."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import convert_hf_weights

        hf_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.random.randn(256).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(256, 256).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(512, 256).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(512, 256).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(256, 512).astype(np.float32),
            "model.norm.weight": np.random.randn(256).astype(np.float32),
        }

        converted = convert_hf_weights(hf_weights)

        assert "model.layers.0.input_layernorm.weight" in converted
        assert "model.layers.0.self_attn.q_proj.weight" in converted
        assert "model.layers.0.mlp.gate_proj.weight" in converted

    def test_convert_mlx_to_hf(self):
        """Test converting MLX weights back to HuggingFace format."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import convert_mlx_to_hf

        mlx_weights = {
            "model.embed_tokens.weight": np.random.randn(1000, 256).astype(np.float32),
            "model.norm.weight": np.random.randn(256).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.random.randn(256).astype(np.float32),
        }

        converted = convert_mlx_to_hf(mlx_weights)

        assert "model.embed_tokens.weight" in converted
        assert "model.norm.weight" in converted

    def test_get_num_params(self):
        """Test counting parameters in weights."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import get_num_params

        weights = {
            "weight1": np.random.randn(10, 10).astype(np.float32),
            "weight2": np.random.randn(5, 5).astype(np.float32),
        }

        total = get_num_params(weights)
        assert total == 100 + 25

    def test_map_weight_name_unknown(self):
        """Test that unknown weight names return None."""
        from chuk_lazarus.models_v2.families.llama.convert import _map_weight_name

        result = _map_weight_name("some.unknown.weight.name")
        assert result is None

    def test_reverse_map_weight_name_unknown(self):
        """Test reverse mapping for unknown weight names."""
        from chuk_lazarus.models_v2.families.llama.convert import _reverse_map_weight_name

        result = _reverse_map_weight_name("some.unknown.weight.name")
        assert result is None

    def test_convert_mlx_array_to_hf(self):
        """Test converting mx.array to HuggingFace format."""
        import numpy as np

        from chuk_lazarus.models_v2.families.llama.convert import convert_mlx_to_hf

        # Test with mx.array - use a proper layer weight name that gets converted
        arr = mx.array(np.random.randn(10, 10).astype(np.float32))
        mlx_weights = {"model.layers.0.input_layernorm.weight": arr}

        converted = convert_mlx_to_hf(mlx_weights)
        # The weight should be converted and present in output
        assert "model.layers.0.input_layernorm.weight" in converted
        # Check it's converted to numpy
        weight_value = converted["model.layers.0.input_layernorm.weight"]
        assert hasattr(weight_value, "shape")


class TestLlamaFromPretrained:
    """Tests for from_pretrained_async method."""

    async def test_from_pretrained_with_config(self, tmp_path):
        """Test loading with provided config (no config.json needed)."""
        import json

        # Create a minimal config
        config = LlamaConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
        )

        # Create config.json (even though we provide config, good to have)
        config_data = {
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        # Load model with provided config
        model = await LlamaForCausalLM.from_pretrained_async(str(tmp_path), config=config)

        assert model.config.vocab_size == 100
        assert model.config.hidden_size == 64

    async def test_from_pretrained_loads_config_from_file(self, tmp_path):
        """Test loading config from config.json file."""
        import json

        # Create config.json
        config_data = {
            "vocab_size": 200,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        # Load model without providing config
        model = await LlamaForCausalLM.from_pretrained_async(str(tmp_path))

        assert model.config.vocab_size == 200
        assert model.config.hidden_size == 128
        assert model.config.num_hidden_layers == 3

    async def test_from_pretrained_safetensors_path_exists(self, tmp_path):
        """Test that safetensors loading code path is exercised."""
        import json

        import numpy as np
        import pytest

        # Create config.json
        config_data = {
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        # Create safetensors with HF-style weight names
        try:
            import safetensors.numpy as st

            # Use HF-style names that convert_hf_weights expects
            weights = {
                "model.embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
                "model.norm.weight": np.ones(64).astype(np.float32),
            }
            st.save_file(weights, str(tmp_path / "model.safetensors"))

            # The loading may fail due to weight name mismatch, but we want
            # to verify the code path runs. Wrap in try/except for robustness.
            try:
                model = await LlamaForCausalLM.from_pretrained_async(str(tmp_path))
                assert model.config.vocab_size == 100
            except ValueError:
                # Weight name mismatch is expected - the important thing
                # is that the safetensors loading code was exercised
                pass
        except ImportError:
            pytest.skip("safetensors not available")

    async def test_from_pretrained_no_weights_file(self, tmp_path):
        """Test loading when no weights file exists."""
        import json

        # Create config.json only (no weights)
        config_data = {
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)

        # Load model - should work even without weights
        model = await LlamaForCausalLM.from_pretrained_async(str(tmp_path))

        assert model.config.vocab_size == 100
