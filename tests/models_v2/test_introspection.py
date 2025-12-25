"""
Tests for model introspection utilities.
"""

import mlx.nn as nn

from chuk_lazarus.models_v2.core.config import ModelConfig
from chuk_lazarus.models_v2.introspection import (
    FLOPsEstimate,
    MemoryEstimate,
    ModelCapabilities,
    ParameterStats,
    count_parameters,
    detect_model_capabilities,
    estimate_flops,
    estimate_memory,
    introspect,
    print_introspection,
)


class TestParameterStats:
    """Tests for ParameterStats dataclass."""

    def test_creation(self):
        """Test creating ParameterStats."""
        stats = ParameterStats(
            total=1_000_000,
            trainable=900_000,
            frozen=100_000,
        )
        assert stats.total == 1_000_000
        assert stats.trainable == 900_000
        assert stats.frozen == 100_000

    def test_trainable_ratio(self):
        """Test trainable_ratio property."""
        stats = ParameterStats(
            total=1_000_000,
            trainable=500_000,
            frozen=500_000,
        )
        assert stats.trainable_ratio == 0.5

    def test_trainable_ratio_zero_total(self):
        """Test trainable_ratio with zero total."""
        stats = ParameterStats(total=0, trainable=0, frozen=0)
        assert stats.trainable_ratio == 0.0

    def test_summary(self):
        """Test summary method."""
        stats = ParameterStats(
            total=1_000_000,
            trainable=1_000_000,
            frozen=0,
        )
        summary = stats.summary()
        assert "1,000,000" in summary
        assert "100.0%" in summary

    def test_component_breakdown(self):
        """Test component breakdown fields."""
        stats = ParameterStats(
            total=100,
            trainable=100,
            frozen=0,
            embedding=30,
            attention=25,
            ffn=20,
            norm=10,
            head=10,
            other=5,
        )
        assert stats.embedding == 30
        assert stats.attention == 25
        assert stats.ffn == 20
        assert stats.norm == 10
        assert stats.head == 10
        assert stats.other == 5

    def test_per_layer_stats(self):
        """Test per-layer stats."""
        stats = ParameterStats(
            total=100,
            trainable=100,
            frozen=0,
            per_layer={0: 25, 1: 25, 2: 25, 3: 25},
        )
        assert len(stats.per_layer) == 4
        assert stats.per_layer[0] == 25


class TestFLOPsEstimate:
    """Tests for FLOPsEstimate dataclass."""

    def test_creation(self):
        """Test creating FLOPsEstimate."""
        flops = FLOPsEstimate(
            attention=1_000_000,
            ffn=2_000_000,
            embedding=100_000,
            norm=50_000,
            total=3_150_000,
            per_token=1000,
        )
        assert flops.total == 3_150_000
        assert flops.per_token == 1000

    def test_estimate_transformer(self):
        """Test FLOPs estimation for transformer."""
        flops = FLOPsEstimate.estimate_transformer(
            hidden_size=768,
            num_layers=12,
            vocab_size=50257,
            intermediate_size=3072,
            seq_length=512,
            num_heads=12,
            batch_size=1,
        )
        assert flops.total > 0
        assert flops.attention > 0
        assert flops.ffn > 0
        assert flops.per_token > 0

    def test_estimate_transformer_with_batch(self):
        """Test FLOPs estimation with batch size."""
        flops_batch_1 = FLOPsEstimate.estimate_transformer(
            hidden_size=256,
            num_layers=4,
            vocab_size=1000,
            intermediate_size=1024,
            seq_length=128,
            num_heads=4,
            batch_size=1,
        )
        flops_batch_4 = FLOPsEstimate.estimate_transformer(
            hidden_size=256,
            num_layers=4,
            vocab_size=1000,
            intermediate_size=1024,
            seq_length=128,
            num_heads=4,
            batch_size=4,
        )
        # Batch size 4 should have ~4x total FLOPs
        assert flops_batch_4.total > flops_batch_1.total

    def test_estimate_transformer_zero_seq_length(self):
        """Test FLOPs estimation with zero sequence length."""
        flops = FLOPsEstimate.estimate_transformer(
            hidden_size=256,
            num_layers=4,
            vocab_size=1000,
            intermediate_size=1024,
            seq_length=0,
            num_heads=4,
            batch_size=1,
        )
        assert flops.per_token == 0

    def test_summary(self):
        """Test summary method."""
        flops = FLOPsEstimate(
            attention=1_000_000_000_000,  # 1 TFLOPs
            ffn=500_000_000_000,
            total=1_500_000_000_000,
        )
        summary = flops.summary()
        assert "1.50T" in summary or "1.5T" in summary


class TestMemoryEstimate:
    """Tests for MemoryEstimate dataclass."""

    def test_creation(self):
        """Test creating MemoryEstimate."""
        mem = MemoryEstimate(
            parameters_mb=100.0,
            activations_mb=50.0,
            gradients_mb=100.0,
            optimizer_state_mb=200.0,
            total_training_mb=450.0,
            total_inference_mb=120.0,
        )
        assert mem.parameters_mb == 100.0
        assert mem.total_training_mb == 450.0

    def test_estimate(self):
        """Test memory estimation."""
        mem = MemoryEstimate.estimate(
            num_parameters=100_000_000,  # 100M params
            hidden_size=768,
            num_layers=12,
            seq_length=512,
            batch_size=1,
            dtype_bytes=4,
            optimizer="adam",
        )
        assert mem.parameters_mb > 0
        assert mem.total_inference_mb > 0
        assert mem.total_training_mb > mem.total_inference_mb

    def test_estimate_sgd_optimizer(self):
        """Test memory estimation with SGD optimizer."""
        mem_adam = MemoryEstimate.estimate(
            num_parameters=1_000_000,
            hidden_size=128,
            num_layers=4,
            seq_length=64,
            batch_size=1,
            dtype_bytes=4,
            optimizer="adam",
        )
        mem_sgd = MemoryEstimate.estimate(
            num_parameters=1_000_000,
            hidden_size=128,
            num_layers=4,
            seq_length=64,
            batch_size=1,
            dtype_bytes=4,
            optimizer="sgd",
        )
        # Adam uses 2x state, SGD uses 1x
        assert mem_adam.optimizer_state_mb > mem_sgd.optimizer_state_mb

    def test_summary(self):
        """Test summary method."""
        mem = MemoryEstimate(
            parameters_mb=100.0,
            total_inference_mb=120.0,
            total_training_mb=450.0,
        )
        summary = mem.summary()
        assert "120.0MB" in summary
        assert "450.0MB" in summary


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_creation(self):
        """Test creating ModelCapabilities."""
        caps = ModelCapabilities(
            is_causal_lm=True,
            supports_kv_cache=True,
        )
        assert caps.is_causal_lm is True
        assert caps.supports_kv_cache is True
        assert caps.is_classifier is False

    def test_all_capabilities(self):
        """Test all capability flags."""
        caps = ModelCapabilities(
            is_causal_lm=True,
            is_classifier=True,
            is_encoder_decoder=True,
            supports_kv_cache=True,
            supports_lora=True,
            supports_moe=True,
            has_memory=True,
            has_planning=True,
            has_tool_use=True,
            domains=["code", "math"],
        )
        assert caps.is_causal_lm is True
        assert caps.is_classifier is True
        assert caps.is_encoder_decoder is True
        assert caps.supports_kv_cache is True
        assert caps.supports_lora is True
        assert caps.supports_moe is True
        assert caps.has_memory is True
        assert caps.has_planning is True
        assert caps.has_tool_use is True
        assert "code" in caps.domains

    def test_summary(self):
        """Test summary method."""
        caps = ModelCapabilities(
            is_causal_lm=True,
            supports_moe=True,
            has_tool_use=True,
        )
        summary = caps.summary()
        assert "causal_lm" in summary
        assert "moe" in summary
        assert "tool_use" in summary

    def test_summary_classifier(self):
        """Test summary with classifier capability."""
        caps = ModelCapabilities(is_classifier=True)
        summary = caps.summary()
        assert "classifier" in summary

    def test_summary_memory(self):
        """Test summary with memory capability."""
        caps = ModelCapabilities(has_memory=True)
        summary = caps.summary()
        assert "memory" in summary

    def test_summary_planning(self):
        """Test summary with planning capability."""
        caps = ModelCapabilities(has_planning=True)
        summary = caps.summary()
        assert "planning" in summary

    def test_summary_empty(self):
        """Test summary with no capabilities."""
        caps = ModelCapabilities()
        summary = caps.summary()
        assert "Capabilities: []" in summary


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_simple_model(self):
        """Test parameter counting on simple model."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        stats = count_parameters(model)
        # Linear(10, 20): 10*20 + 20 = 220 params
        # Linear(20, 5): 20*5 + 5 = 105 params
        # Total: 325 params
        assert stats.total == 325
        assert stats.trainable == 325

    def test_count_model_with_embedding(self):
        """Test parameter counting with embedding layers."""

        class EmbedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 10)

            def __call__(self, x):
                return self.linear(self.embed_tokens(x))

        model = EmbedModel()
        stats = count_parameters(model)

        # Embedding: 100 * 32 = 3200 params
        # Linear: 32 * 10 + 10 = 330 params
        assert stats.total == 3530
        assert stats.embedding == 3200

    def test_count_model_with_attention(self):
        """Test parameter counting with attention patterns in names."""

        class AttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.attn_output = nn.Linear(64, 64)

            def __call__(self, x):
                return x

        model = AttentionModel()
        stats = count_parameters(model)

        # 4 Linear layers: 4 * (64*64 + 64) = 4 * 4160 = 16640
        assert stats.total == 16640
        assert stats.attention > 0

    def test_count_model_with_ffn(self):
        """Test parameter counting with FFN patterns."""

        class FFNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(64, 256)
                self.down_proj = nn.Linear(256, 64)
                self.gate_proj = nn.Linear(64, 256)

            def __call__(self, x):
                return x

        model = FFNModel()
        stats = count_parameters(model)

        assert stats.ffn > 0

    def test_count_model_with_norm(self):
        """Test parameter counting with normalization layers."""

        class NormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)
                self.ln_final = nn.LayerNorm(64)

            def __call__(self, x):
                return self.ln_final(self.norm(x))

        model = NormModel()
        stats = count_parameters(model)

        # Each LayerNorm: 64 + 64 = 128 (weight + bias)
        assert stats.total == 256
        assert stats.norm == 256

    def test_count_model_with_head(self):
        """Test parameter counting with head patterns."""

        class HeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(64, 1000)

            def __call__(self, x):
                return self.lm_head(x)

        model = HeadModel()
        stats = count_parameters(model)

        # Linear: 64*1000 + 1000 = 65000
        assert stats.total == 65000
        assert stats.head == 65000

    def test_count_model_with_layers(self):
        """Test parameter counting with layer numbering."""

        class LayeredModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers_0 = nn.Linear(32, 32)
                self.layer_1 = nn.Linear(32, 32)
                self.layers_2 = nn.Linear(32, 32)

            def __call__(self, x):
                return x

        model = LayeredModel()
        stats = count_parameters(model)

        assert len(stats.per_layer) == 3
        assert 0 in stats.per_layer
        assert 1 in stats.per_layer
        assert 2 in stats.per_layer


class TestEstimateFLOPs:
    """Tests for estimate_flops function."""

    def test_estimate_from_config(self):
        """Test FLOPs estimation from config."""
        config = ModelConfig(
            hidden_size=768,
            num_hidden_layers=12,
            vocab_size=50257,
            intermediate_size=3072,
            num_attention_heads=12,
        )

        flops = estimate_flops(config, seq_length=512, batch_size=1)
        assert flops.total > 0
        assert flops.attention > 0
        assert flops.ffn > 0

    def test_estimate_with_different_seq_lengths(self):
        """Test FLOPs estimation with different sequence lengths."""
        config = ModelConfig(
            hidden_size=256,
            num_hidden_layers=4,
            vocab_size=1000,
            intermediate_size=1024,
            num_attention_heads=4,
        )

        flops_short = estimate_flops(config, seq_length=64, batch_size=1)
        flops_long = estimate_flops(config, seq_length=256, batch_size=1)

        assert flops_long.total > flops_short.total


class TestEstimateMemory:
    """Tests for estimate_memory function."""

    def test_estimate_memory_from_model(self):
        """Test memory estimation from model and config."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Create a minimal ModelConfig with required fields
        model_config = ModelConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
        )

        mem = estimate_memory(model, model_config, seq_length=128, batch_size=1)

        assert mem.parameters_mb > 0
        assert mem.total_inference_mb > 0
        assert mem.total_training_mb > 0


class TestDetectModelCapabilities:
    """Tests for detect_model_capabilities function."""

    def test_detect_causal_lm(self):
        """Test detecting CausalLM capabilities."""
        from chuk_lazarus.models_v2.core.enums import BackboneType
        from chuk_lazarus.models_v2.models.causal_lm import CausalLM

        config = ModelConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = CausalLM(config, backbone_type=BackboneType.TRANSFORMER)

        caps = detect_model_capabilities(model)

        assert caps.is_causal_lm is True
        assert caps.supports_kv_cache is True
        assert caps.is_classifier is False

    def test_detect_sequence_classifier(self):
        """Test detecting SequenceClassifier capabilities."""
        from chuk_lazarus.models_v2.models.classifiers import SequenceClassifier

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = SequenceClassifier(config, num_labels=3)

        caps = detect_model_capabilities(model)

        assert caps.is_classifier is True

    def test_detect_token_classifier(self):
        """Test detecting TokenClassifier capabilities."""
        from chuk_lazarus.models_v2.models.classifiers import TokenClassifier

        config = ModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        model = TokenClassifier(config, num_labels=5)

        caps = detect_model_capabilities(model)

        assert caps.is_classifier is True

    def test_detect_lora_support(self):
        """Test detecting LoRA support."""
        from chuk_lazarus.models_v2.adapters import LoRAConfig, apply_lora
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Apply LoRA
        lora_config = LoRAConfig(rank=4, alpha=8.0)
        apply_lora(model, lora_config)

        caps = detect_model_capabilities(model)

        assert caps.supports_lora is True

    def test_detect_simple_model(self):
        """Test detecting capabilities of a simple nn.Module."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        caps = detect_model_capabilities(model)

        # Simple model shouldn't have any special capabilities
        assert caps.is_causal_lm is False
        assert caps.is_classifier is False
        assert caps.supports_lora is False


class TestIntrospect:
    """Tests for introspect function."""

    def test_introspect_basic(self):
        """Test basic introspection without config."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        info = introspect(model)

        assert "parameters" in info
        assert "capabilities" in info
        assert "flops" not in info  # No config provided
        assert "memory" not in info  # No config provided

    def test_introspect_with_config(self):
        """Test introspection with config."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Create a minimal ModelConfig
        model_config = ModelConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
        )

        info = introspect(model, config=model_config, seq_length=64, batch_size=1)

        assert "parameters" in info
        assert "capabilities" in info
        assert "flops" in info
        assert "memory" in info

    def test_introspect_parameter_stats(self):
        """Test that introspect returns correct ParameterStats."""
        model = nn.Linear(10, 20)

        info = introspect(model)

        params = info["parameters"]
        assert isinstance(params, ParameterStats)
        assert params.total == 220  # 10*20 + 20


class TestPrintIntrospection:
    """Tests for print_introspection function."""

    def test_print_basic(self, capsys):
        """Test print_introspection with basic model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        print_introspection(model)

        captured = capsys.readouterr()
        assert "Model Introspection Report" in captured.out
        assert "Parameters:" in captured.out
        assert "Capabilities:" in captured.out

    def test_print_with_config(self, capsys):
        """Test print_introspection with config."""
        from chuk_lazarus.models_v2.families.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig.tiny()
        model = LlamaForCausalLM(config)

        # Create a minimal ModelConfig
        model_config = ModelConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
        )

        print_introspection(model, config=model_config)

        captured = capsys.readouterr()
        assert "Model Introspection Report" in captured.out
        assert "Parameters:" in captured.out
        assert "Estimated FLOPs:" in captured.out
        assert "Memory:" in captured.out

    def test_print_component_breakdown(self, capsys):
        """Test print_introspection shows component breakdown."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 32)
                self.q_proj = nn.Linear(32, 32)
                self.up_proj = nn.Linear(32, 64)
                self.norm = nn.LayerNorm(32)
                self.lm_head = nn.Linear(32, 100)

            def __call__(self, x):
                return x

        model = TestModel()

        print_introspection(model)

        captured = capsys.readouterr()
        assert "Embedding:" in captured.out
        assert "Attention:" in captured.out
        assert "FFN:" in captured.out
        assert "Norm:" in captured.out
        assert "Head:" in captured.out
