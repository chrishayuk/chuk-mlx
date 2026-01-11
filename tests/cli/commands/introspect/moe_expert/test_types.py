"""Tests for moe_expert _types module."""

from argparse import Namespace

from chuk_lazarus.cli.commands._constants import (
    ContextVerdict,
    LayerPhase,
    TokenType,
)
from chuk_lazarus.cli.commands.introspect.moe_expert._types import (
    AttentionPatternConfig,
    AttentionPatternResult,
    ContextWindowAnalysisResult,
    DomainTestResult,
    ExpertWeight,
    ExploreAnalysisResult,
    ExploreConfig,
    FullTaxonomyConfig,
    LayerExpertTransition,
    MoEExpertConfig,
    PositionRouting,
    TaxonomyResult,
    Trigram,
    get_layer_phase,
)


class TestTrigram:
    """Tests for Trigram model."""

    def test_pattern_property(self):
        """Test trigram pattern string generation."""
        trigram = Trigram(
            prev_type="^",
            curr_type=TokenType.OP,
            next_type="num",
        )
        assert trigram.pattern == "^→OP→num"


class TestPositionRouting:
    """Tests for PositionRouting model."""

    def test_top_expert_with_experts(self):
        """Test top_expert property with experts list."""
        routing = PositionRouting(
            position=0,
            token="test",
            token_type=TokenType.CW,
            trigram="^→CW→$",
            experts=[
                ExpertWeight(expert_idx=5, weight=0.8),
                ExpertWeight(expert_idx=3, weight=0.2),
            ],
        )
        assert routing.top_expert == 5

    def test_top_expert_empty(self):
        """Test top_expert property with empty experts list."""
        routing = PositionRouting(
            position=0,
            token="test",
            token_type=TokenType.CW,
            trigram="^→CW→$",
            experts=[],
        )
        assert routing.top_expert is None


class TestTaxonomyResult:
    """Tests for TaxonomyResult model."""

    def test_to_display(self):
        """Test to_display method."""
        result = TaxonomyResult(
            model_id="test-model",
            num_experts=32,
            num_moe_layers=8,
            prompts_analyzed=100,
            pattern_experts=[],
            category_stats=[],
        )
        display = result.to_display()
        assert "TAXONOMY ANALYSIS" in display
        assert "test-model" in display
        assert "32" in display
        assert "8" in display
        assert "100" in display


class TestAttentionPatternResult:
    """Tests for AttentionPatternResult model."""

    def test_to_display(self):
        """Test to_display method."""
        result = AttentionPatternResult(
            model_id="test-model",
            prompt="test prompt",
            layer=6,
            query_position=3,
            query_token="test",
            attention_weights=[(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)],
            expert_routing=[
                ExpertWeight(expert_idx=5, weight=0.8),
                ExpertWeight(expert_idx=3, weight=0.2),
            ],
        )
        display = result.to_display()
        assert "ATTENTION PATTERN" in display
        assert "Layer 6" in display
        assert "Position 3" in display
        assert "test" in display
        assert "0.500" in display or "0.5" in display


class TestDomainTestResult:
    """Tests for DomainTestResult model."""

    def test_to_display(self):
        """Test to_display method."""
        result = DomainTestResult(
            model_id="test-model",
            domains_tested=["math", "code", "text"],
            expert_stats=[],
            generalist_count=5,
        )
        display = result.to_display()
        assert "DOMAIN TEST RESULTS" in display
        assert "test-model" in display
        assert "math" in display
        assert "5" in display


class TestContextWindowAnalysisResult:
    """Tests for ContextWindowAnalysisResult model."""

    def test_to_display(self):
        """Test to_display method."""
        result = ContextWindowAnalysisResult(
            model_id="test-model",
            num_layers=8,
            results=[],
            verdict=ContextVerdict.EXTENDED_CONTEXT_MATTERS,
        )
        display = result.to_display()
        assert "CONTEXT WINDOW ANALYSIS" in display
        assert "test-model" in display
        assert "EXTENDED CONTEXT MATTERS" in display


class TestLayerExpertTransition:
    """Tests for LayerExpertTransition model."""

    def test_transition_str_stable(self):
        """Test transition_str for stable routing."""
        transition = LayerExpertTransition(
            position=0,
            token="test",
            early_expert=5,
            middle_expert=5,
            late_expert=5,
            has_transition=False,
        )
        assert "stable" in transition.transition_str.lower()
        assert "E5" in transition.transition_str

    def test_transition_str_with_transitions(self):
        """Test transition_str with actual transitions."""
        transition = LayerExpertTransition(
            position=0,
            token="test",
            early_expert=5,
            middle_expert=10,
            late_expert=15,
            has_transition=True,
        )
        transition_str = transition.transition_str
        assert "E5→E10" in transition_str
        assert "E10→E15" in transition_str

    def test_transition_str_partial_transition(self):
        """Test transition_str with partial transition."""
        transition = LayerExpertTransition(
            position=0,
            token="test",
            early_expert=5,
            middle_expert=5,
            late_expert=10,
            has_transition=True,
        )
        transition_str = transition.transition_str
        assert "E5→E10" in transition_str

    def test_transition_str_unknown(self):
        """Test transition_str with no experts set."""
        transition = LayerExpertTransition(
            position=0,
            token="test",
            has_transition=False,
        )
        assert "unknown" in transition.transition_str.lower()

    def test_transition_str_early_to_middle_only(self):
        """Test transition with only early to middle change."""
        transition = LayerExpertTransition(
            position=0,
            token="test",
            early_expert=5,
            middle_expert=10,
            late_expert=10,
            has_transition=True,
        )
        assert "E5→E10" in transition.transition_str


class TestExploreAnalysisResult:
    """Tests for ExploreAnalysisResult model."""

    def test_to_display(self):
        """Test to_display method."""
        result = ExploreAnalysisResult(
            prompt="2+2=",
            layer=6,
            positions=[],
            transitions=[],
        )
        display = result.to_display()
        assert "EXPLORE ANALYSIS" in display
        assert "2+2=" in display
        assert "Layer: 6" in display


class TestMoEExpertConfig:
    """Tests for MoEExpertConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            model="test-model",
            prompt="test prompt",
            layer=6,
            position=3,
            action="trace",
            verbose=True,
            output="/tmp/output.json",
        )
        config = MoEExpertConfig.from_args(args)
        assert config.model == "test-model"
        assert config.prompt == "test prompt"
        assert config.layer == 6
        assert config.position == 3
        assert config.action == "trace"
        assert config.verbose is True
        assert config.output == "/tmp/output.json"

    def test_from_args_defaults(self):
        """Test creating config with defaults."""
        args = Namespace(model="test-model")
        config = MoEExpertConfig.from_args(args)
        assert config.model == "test-model"
        assert config.prompt is None
        assert config.layer is None
        assert config.action == "trace"
        assert config.verbose is False


class TestFullTaxonomyConfig:
    """Tests for FullTaxonomyConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            model="test-model",
            categories="math,code",
            verbose=True,
        )
        config = FullTaxonomyConfig.from_args(args)
        assert config.model == "test-model"
        assert config.categories == "math,code"
        assert config.verbose is True


class TestExploreConfig:
    """Tests for ExploreConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            model="test-model",
            layer=10,
        )
        config = ExploreConfig.from_args(args)
        assert config.model == "test-model"
        assert config.layer == 10

    def test_from_args_default_layer(self):
        """Test creating config with default layer."""
        args = Namespace(model="test-model")
        config = ExploreConfig.from_args(args)
        assert config.model == "test-model"


class TestAttentionPatternConfig:
    """Tests for AttentionPatternConfig model."""

    def test_from_args(self):
        """Test creating config from args."""
        args = Namespace(
            model="test-model",
            prompt="test prompt",
            position=3,
            layer=6,
            head=4,
            top_k=10,
        )
        config = AttentionPatternConfig.from_args(args)
        assert config.model == "test-model"
        assert config.prompt == "test prompt"
        assert config.position == 3
        assert config.layer == 6
        assert config.head == 4
        assert config.top_k == 10

    def test_from_args_defaults(self):
        """Test creating config with defaults."""
        args = Namespace(model="test-model")
        config = AttentionPatternConfig.from_args(args)
        assert config.model == "test-model"
        assert "King is to queen" in config.prompt
        assert config.position is None
        assert config.top_k == 5


class TestGetLayerPhase:
    """Tests for get_layer_phase function."""

    def test_early_phase(self):
        """Test early layer phase."""
        assert get_layer_phase(0) == LayerPhase.EARLY
        assert get_layer_phase(3) == LayerPhase.EARLY

    def test_middle_phase(self):
        """Test middle layer phase."""
        assert get_layer_phase(8) == LayerPhase.MIDDLE
        assert get_layer_phase(15) == LayerPhase.MIDDLE

    def test_late_phase(self):
        """Test late layer phase."""
        assert get_layer_phase(20) == LayerPhase.LATE
        assert get_layer_phase(30) == LayerPhase.LATE
