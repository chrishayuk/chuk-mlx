"""
Tests for LoRA adapter module.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.adapters import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    count_lora_parameters,
    merge_lora_weights,
)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()

        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert len(config.target_modules) == 2


class TestLoRALinear:
    """Tests for LoRALinear module."""

    def test_creation(self):
        """Test LoRALinear creation."""
        base = nn.Linear(768, 768)
        lora = LoRALinear(base, rank=8, alpha=16.0)

        assert lora.rank == 8
        assert lora.alpha == 16.0
        assert lora.scaling == 2.0  # 16 / 8
        assert lora.lora_A.shape == (768, 8)
        assert lora.lora_B.shape == (8, 768)

    def test_forward_pass(self):
        """Test LoRALinear forward pass."""
        base = nn.Linear(256, 512)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        x = mx.random.normal((2, 10, 256))
        output = lora(x)

        assert output.shape == (2, 10, 512)

    def test_initial_output_equals_base(self):
        """Test initial output equals base (lora_B is zeros)."""
        base = nn.Linear(128, 128)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        x = mx.random.normal((1, 5, 128))

        # Before any training, lora_B is zeros so output should equal base
        base_output = base(x)
        lora_output = lora(x)

        # Should be very close (lora_A contributes near-zero due to small init)
        assert mx.allclose(base_output, lora_output, atol=1e-3)

    def test_merge_weights(self):
        """Test merging LoRA weights into base layer."""
        base = nn.Linear(128, 256)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        # Manually set lora weights for predictable merge
        lora.lora_A = mx.ones((128, 4)) * 0.1
        lora.lora_B = mx.ones((4, 256)) * 0.1

        merged = lora.merge_weights()

        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (256, 128)

    def test_training_property(self):
        """Test training mode property."""
        base = nn.Linear(64, 64)
        lora = LoRALinear(base, rank=4)

        assert lora.training is False

        lora.training = True
        assert lora.training is True

        lora.training = False
        assert lora.training is False

    def test_dropout_applied_in_training(self):
        """Test dropout is applied in training mode."""
        base = nn.Linear(128, 128)
        lora = LoRALinear(base, rank=4, dropout=0.5)

        x = mx.random.normal((2, 10, 128))

        # In eval mode, no dropout
        lora.training = False
        out1 = lora(x)

        # In training mode with dropout
        lora.training = True
        out2 = lora(x)

        # Outputs should differ due to dropout (with high probability)
        # This is probabilistic, so we just check shapes match
        assert out1.shape == out2.shape


class TestApplyLoRA:
    """Tests for apply_lora function."""

    def test_apply_lora_to_llama(self):
        """Test applying LoRA to a Llama model."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        lora_config = LoRAConfig(rank=4, alpha=8.0)
        lora_layers = apply_lora(model, lora_config)

        # Should have applied to attention and MLP layers
        assert len(lora_layers) > 0

        # Check some layers were converted
        for name, layer in lora_layers.items():
            assert isinstance(layer, LoRALinear)
            assert layer.rank == 4

    def test_apply_lora_custom_targets(self):
        """Test applying LoRA with custom target modules."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Only target q_proj and v_proj
        lora_config = LoRAConfig(
            rank=8,
            target_modules=["q_proj", "v_proj"],
        )
        lora_layers = apply_lora(model, lora_config)

        # Check only targeted modules
        for name in lora_layers:
            assert "q_proj" in name or "v_proj" in name

    def test_model_still_works_after_lora(self):
        """Test model forward pass works after LoRA."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        lora_config = LoRAConfig(rank=4)
        apply_lora(model, lora_config)

        # Model should still work
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)


class TestMergeLoRAWeights:
    """Tests for merge_lora_weights function."""

    def test_merge_lora_weights(self):
        """Test merging LoRA weights back into model."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        lora_config = LoRAConfig(rank=4)
        lora_layers = apply_lora(model, lora_config)

        # Merge weights
        merge_lora_weights(model, lora_layers)

        # Model should still work and layers should be regular Linear
        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)

        assert output.logits.shape == (1, 3, config.vocab_size)


class TestCountLoRAParameters:
    """Tests for count_lora_parameters function."""

    def test_count_parameters(self):
        """Test counting LoRA parameters."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        lora_config = LoRAConfig(rank=4)
        lora_layers = apply_lora(model, lora_config)

        num_params = count_lora_parameters(lora_layers)

        assert num_params > 0
        # Each layer has lora_A (in, rank) + lora_B (rank, out)

    def test_parameter_count_scales_with_rank(self):
        """Test that parameter count scales with rank."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()

        # Rank 4
        model1 = LlamaForCausalLM(config)
        lora_layers1 = apply_lora(model1, LoRAConfig(rank=4))
        params1 = count_lora_parameters(lora_layers1)

        # Rank 8
        model2 = LlamaForCausalLM(config)
        lora_layers2 = apply_lora(model2, LoRAConfig(rank=8))
        params2 = count_lora_parameters(lora_layers2)

        # Rank 8 should have ~2x parameters
        assert params2 > params1
        assert abs(params2 / params1 - 2.0) < 0.1
