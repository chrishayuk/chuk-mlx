"""Tests for MoE expert CLI formatters."""

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.formatters import (
    format_ablation_result,
    format_chat_result,
    format_coactivation,
    format_comparison_result,
    format_entropy_analysis,
    format_header,
    format_model_info,
    format_router_weights,
    format_subheader,
    format_taxonomy,
    format_topk_result,
)
from chuk_lazarus.introspection.moe.enums import (
    ExpertCategory,
    ExpertRole,
    MoEArchitecture,
)
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertIdentity,
    ExpertPair,
    ExpertPattern,
    ExpertTaxonomy,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
    TopKVariationResult,
)


class TestFormatHeader:
    """Tests for format_header function."""

    def test_basic_header(self):
        """Test basic header formatting."""
        result = format_header("TEST HEADER")
        assert "TEST HEADER" in result
        assert "=" * 70 in result

    def test_custom_width(self):
        """Test header with custom width."""
        result = format_header("TEST", width=40)
        assert "=" * 40 in result
        assert "=" * 70 not in result


class TestFormatSubheader:
    """Tests for format_subheader function."""

    def test_basic_subheader(self):
        """Test basic subheader formatting."""
        result = format_subheader("TEST SUBHEADER")
        assert "TEST SUBHEADER" in result
        assert "-" * 70 in result

    def test_custom_width(self):
        """Test subheader with custom width."""
        result = format_subheader("TEST", width=40)
        assert "-" * 40 in result


class TestFormatModelInfo:
    """Tests for format_model_info function."""

    def test_basic_model_info(self):
        """Test formatting basic model info."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )
        result = format_model_info(info, "test/model")

        assert "test/model" in result
        assert "gpt_oss" in result
        assert "32" in result
        assert "4" in result
        assert "8" in result

    def test_model_with_shared_expert(self):
        """Test formatting model with shared expert."""
        info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=16,
            num_experts_per_tok=2,
            total_layers=4,
            architecture=MoEArchitecture.LLAMA4,
            has_shared_expert=True,
        )
        result = format_model_info(info, "test/model")

        assert "Has shared expert: Yes" in result


class TestFormatChatResult:
    """Tests for format_chat_result function."""

    @pytest.fixture
    def sample_chat_result(self):
        """Create sample chat result."""
        stats = GenerationStats(
            expert_idx=6,
            tokens_generated=20,
            layers_modified=8,
            moe_type="gpt_oss_batched",
            prompt_tokens=10,
        )
        return ExpertChatResult(
            prompt="127 * 89 = ",
            response="11303",
            expert_idx=6,
            stats=stats,
        )

    def test_basic_formatting(self, sample_chat_result):
        """Test basic chat result formatting."""
        result = format_chat_result(sample_chat_result, "test/model", "gpt_oss_batched")

        assert "CHAT WITH EXPERT 6" in result
        assert "test/model" in result
        assert "127 * 89 = " in result
        assert "11303" in result

    def test_verbose_formatting(self, sample_chat_result):
        """Test verbose chat result formatting."""
        result = format_chat_result(
            sample_chat_result, "test/model", "gpt_oss_batched", verbose=True
        )

        assert "Statistics:" in result
        assert "Tokens generated: 20" in result
        assert "Layers modified: 8" in result
        assert "Prompt tokens: 10" in result


class TestFormatComparisonResult:
    """Tests for format_comparison_result function."""

    @pytest.fixture
    def sample_comparison_result(self):
        """Create sample comparison result."""
        results = []
        for expert_idx in [6, 7, 20]:
            stats = GenerationStats(
                expert_idx=expert_idx,
                tokens_generated=15,
                layers_modified=8,
                moe_type="gpt_oss_batched",
            )
            results.append(
                ExpertChatResult(
                    prompt="Test",
                    response=f"Response from expert {expert_idx}",
                    expert_idx=expert_idx,
                    stats=stats,
                )
            )
        return ExpertComparisonResult(
            prompt="Test",
            expert_results=tuple(results),
        )

    def test_basic_formatting(self, sample_comparison_result):
        """Test basic comparison result formatting."""
        result = format_comparison_result(sample_comparison_result, "test/model")

        assert "EXPERT COMPARISON" in result
        assert "Expert 6" in result
        assert "Expert 7" in result
        assert "Expert 20" in result
        assert "Response from expert 6" in result

    def test_verbose_formatting(self, sample_comparison_result):
        """Test verbose comparison result formatting."""
        result = format_comparison_result(sample_comparison_result, "test/model", verbose=True)

        assert "(tokens: 15)" in result


class TestFormatTopkResult:
    """Tests for format_topk_result function."""

    def test_different_outputs(self):
        """Test formatting when outputs differ."""
        result_data = TopKVariationResult(
            prompt="Test prompt",
            k_value=2,
            default_k=4,
            response="Modified response",
            normal_response="Normal response",
        )
        result = format_topk_result(result_data, "test/model")

        assert "TOP-K EXPERIMENT" in result
        assert "k=2" in result
        assert "default: 4" in result
        assert "Modified response" in result
        assert "Normal response" in result
        assert "** OUTPUTS DIFFER **" in result

    def test_identical_outputs(self):
        """Test formatting when outputs are identical."""
        result_data = TopKVariationResult(
            prompt="Test prompt",
            k_value=2,
            default_k=4,
            response="Same response",
            normal_response="Same response",
        )
        result = format_topk_result(result_data, "test/model")

        assert "Outputs are identical" in result


class TestFormatRouterWeights:
    """Tests for format_router_weights function."""

    def test_basic_formatting(self):
        """Test basic router weights formatting."""
        weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="Hello",
                        expert_indices=(6, 7, 20, 1),
                        weights=(0.4, 0.3, 0.2, 0.1),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token=" world",
                        expert_indices=(7, 6, 15, 3),
                        weights=(0.35, 0.3, 0.2, 0.15),
                    ),
                ),
            )
        ]
        result = format_router_weights(weights, "test/model", "Hello world")

        assert "ROUTER WEIGHTS" in result
        assert "Layer 0" in result
        assert "Hello" in result
        assert "world" in result
        assert "E6" in result
        assert "E7" in result


class TestFormatCoactivation:
    """Tests for format_coactivation function."""

    def test_basic_formatting(self):
        """Test basic co-activation formatting."""
        analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            top_pairs=(
                ExpertPair(expert_a=6, expert_b=7, coactivation_count=25, coactivation_rate=0.25),
                ExpertPair(expert_a=6, expert_b=20, coactivation_count=15, coactivation_rate=0.15),
            ),
            generalist_experts=(6, 7),
        )
        result = format_coactivation(analysis, "test/model", 0)

        assert "CO-ACTIVATION ANALYSIS" in result
        assert "Layer 0" in result
        assert "Total activations: 100" in result
        assert "E6 + E7" in result
        assert "25 times" in result
        assert "Generalist experts: [6, 7]" in result


class TestFormatTaxonomy:
    """Tests for format_taxonomy function."""

    @pytest.fixture
    def sample_taxonomy(self):
        """Create sample taxonomy."""
        identity = ExpertIdentity(
            expert_idx=6,
            layer_idx=0,
            primary_category=ExpertCategory.MATH,
            role=ExpertRole.SPECIALIST,
            confidence=0.9,
            activation_rate=0.15,
            top_tokens=("127", "89", "*", "=", "+"),
        )
        pattern = ExpertPattern(
            expert_idx=6,
            layer_idx=0,
            pattern_type="numeric",
            trigger_tokens=("1", "2", "3"),
            confidence=0.85,
            sample_activations=100,
        )
        return ExpertTaxonomy(
            model_id="test/model",
            num_layers=8,
            num_experts=32,
            expert_identities=(identity,),
            patterns=(pattern,),
        )

    def test_basic_formatting(self, sample_taxonomy):
        """Test basic taxonomy formatting."""
        result = format_taxonomy(sample_taxonomy)

        assert "EXPERT TAXONOMY" in result
        assert "test/model" in result
        assert "Layers: 8" in result
        assert "Experts per layer: 32" in result
        assert "specialist" in result
        assert "math" in result

    def test_verbose_formatting(self, sample_taxonomy):
        """Test verbose taxonomy formatting."""
        result = format_taxonomy(sample_taxonomy, verbose=True)

        assert "Top tokens:" in result
        assert "'127'" in result


class TestFormatAblationResult:
    """Tests for format_ablation_result function."""

    def test_different_outputs(self):
        """Test formatting when outputs differ."""
        result = format_ablation_result(
            normal_output="Normal output",
            ablated_output="Different output",
            expert_indices=[6, 7],
            prompt="Test prompt",
            model_id="test/model",
        )

        assert "ABLATION" in result
        assert "Expert(s) 6, 7" in result
        assert "Normal:  Normal output" in result
        assert "Ablated: Different output" in result
        assert "** OUTPUTS DIFFER" in result

    def test_identical_outputs(self):
        """Test formatting when outputs are identical."""
        result = format_ablation_result(
            normal_output="Same output",
            ablated_output="Same output",
            expert_indices=[6],
            prompt="Test prompt",
            model_id="test/model",
        )

        assert "Outputs are identical" in result
        assert "Expert(s) had no effect" in result


class TestFormatEntropyAnalysis:
    """Tests for format_entropy_analysis function."""

    def test_basic_formatting(self):
        """Test basic entropy analysis formatting."""
        entropies = [
            (0, 1.5, 0.75),
            (1, 1.2, 0.60),
            (2, 1.8, 0.90),
        ]
        result = format_entropy_analysis(entropies, "test/model", "Test prompt")

        assert "ROUTING ENTROPY ANALYSIS" in result
        assert "Layer  Mean Entropy  Normalized" in result
        assert "1.500" in result
        assert "0.750" in result
        assert "#" in result  # Histogram bar
