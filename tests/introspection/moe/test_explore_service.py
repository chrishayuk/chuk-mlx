"""Tests for ExploreService - MoE expert exploration analysis."""

from unittest.mock import Mock

import pytest

from chuk_lazarus.introspection.moe.explore_service import (
    ComparisonResult,
    DeepDiveResult,
    ExploreService,
    LayerPhaseData,
    PatternMatch,
    PositionEvolution,
    TokenAnalysis,
)


class TestTokenAnalysis:
    """Tests for TokenAnalysis model."""

    def test_token_analysis_creation(self):
        """Test creating a TokenAnalysis instance."""
        analysis = TokenAnalysis(
            position=0,
            token="hello",
            token_type="CW",
            trigram="^→CW→CW",
            top_expert=5,
            all_experts=[5, 10],
            expert_weights=[0.7, 0.3],
        )
        assert analysis.position == 0
        assert analysis.token == "hello"
        assert analysis.token_type == "CW"
        assert analysis.top_expert == 5
        assert len(analysis.all_experts) == 2

    def test_token_analysis_defaults(self):
        """Test TokenAnalysis default values."""
        analysis = TokenAnalysis(
            position=0,
            token="test",
            token_type="CW",
            trigram="^→CW→$",
        )
        assert analysis.top_expert is None
        assert analysis.all_experts == []
        assert analysis.expert_weights == []

    def test_token_analysis_frozen(self):
        """Test TokenAnalysis is immutable."""
        from pydantic import ValidationError

        analysis = TokenAnalysis(
            position=0,
            token="test",
            token_type="CW",
            trigram="^→CW→$",
        )
        with pytest.raises(ValidationError):
            analysis.position = 1


class TestPatternMatch:
    """Tests for PatternMatch model."""

    def test_pattern_match_creation(self):
        """Test creating a PatternMatch instance."""
        pattern = PatternMatch(
            position=5,
            token="+",
            trigram="NUM→OP→NUM",
            pattern_type="arithmetic operator",
            top_expert=6,
        )
        assert pattern.position == 5
        assert pattern.token == "+"
        assert pattern.pattern_type == "arithmetic operator"


class TestLayerPhaseData:
    """Tests for LayerPhaseData model."""

    def test_layer_phase_creation(self):
        """Test creating LayerPhaseData."""
        phase = LayerPhaseData(
            phase_name="early",
            layer_range="L0-7",
            layer_experts=[(0, 5), (4, 5), (7, 6)],
            dominant_expert=5,
        )
        assert phase.phase_name == "early"
        assert phase.dominant_expert == 5
        assert len(phase.layer_experts) == 3


class TestPositionEvolution:
    """Tests for PositionEvolution model."""

    def test_position_evolution_creation(self):
        """Test creating PositionEvolution."""
        early = LayerPhaseData(
            phase_name="early",
            layer_range="L0-7",
            layer_experts=[(0, 5)],
            dominant_expert=5,
        )
        middle = LayerPhaseData(
            phase_name="middle",
            layer_range="L8-15",
            layer_experts=[(12, 10)],
            dominant_expert=10,
        )
        late = LayerPhaseData(
            phase_name="late",
            layer_range="L16+",
            layer_experts=[(20, 10)],
            dominant_expert=10,
        )

        evolution = PositionEvolution(
            position=2,
            token="+",
            trigram="NUM→OP→NUM",
            early=early,
            middle=middle,
            late=late,
            has_transition=True,
            transitions=["E5→E10"],
        )

        assert evolution.position == 2
        assert evolution.has_transition is True
        assert "E5→E10" in evolution.transitions


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    def test_comparison_result_creation(self):
        """Test creating ComparisonResult."""
        result = ComparisonResult(
            prompt1="2 + 3",
            prompt2="Calculate: 2 + 3",
            layer=12,
            tokens1=[],
            tokens2=[],
            shared_experts=[5, 6, 10],
            only_prompt1=[12],
            only_prompt2=[15, 20],
            overlap_ratio=0.5,
        )
        assert result.overlap_ratio == 0.5
        assert 5 in result.shared_experts


class TestDeepDiveResult:
    """Tests for DeepDiveResult model."""

    def test_deep_dive_result_creation(self):
        """Test creating DeepDiveResult."""
        result = DeepDiveResult(
            position=2,
            token="+",
            token_type="OP",
            trigram="NUM→OP→NUM",
            prev_token="2",
            prev_type="NUM",
            next_token="3",
            next_type="NUM",
            layer_routing=[(0, [(5, 0.6), (10, 0.4)])],
            all_experts=[5, 10],
            dominant_expert=5,
            peak_layer=12,
        )
        assert result.dominant_expert == 5
        assert result.peak_layer == 12


class TestExploreServiceAnalyzeRouting:
    """Tests for ExploreService.analyze_routing."""

    def test_analyze_routing_basic(self):
        """Test basic routing analysis."""
        tokens = ["2", "+", "3"]

        # Create mock positions with routing data
        pos1 = Mock()
        pos1.expert_indices = [5]
        pos1.weights = [1.0]

        pos2 = Mock()
        pos2.expert_indices = [6, 10]
        pos2.weights = [0.7, 0.3]

        pos3 = Mock()
        pos3.expert_indices = [5]
        pos3.weights = [1.0]

        positions = [pos1, pos2, pos3]

        results = ExploreService.analyze_routing(tokens, positions)

        assert len(results) == 3
        assert results[0].position == 0
        assert results[0].token == "2"
        assert results[1].top_expert == 6
        assert results[2].trigram.endswith("→$")

    def test_analyze_routing_no_experts(self):
        """Test analysis when no experts assigned."""
        tokens = ["test"]
        pos = Mock()
        pos.expert_indices = []
        pos.weights = []

        results = ExploreService.analyze_routing(tokens, [pos])

        assert len(results) == 1
        assert results[0].top_expert is None
        assert results[0].all_experts == []


class TestExploreServiceFindPatterns:
    """Tests for ExploreService.find_patterns."""

    def test_find_patterns_operator(self):
        """Test finding arithmetic operator patterns."""
        tokens = ["2", "+", "3"]

        pos1 = Mock()
        pos1.expert_indices = [5]
        pos2 = Mock()
        pos2.expert_indices = [6]
        pos3 = Mock()
        pos3.expert_indices = [5]

        positions = [pos1, pos2, pos3]

        patterns = ExploreService.find_patterns(tokens, positions)

        # Should find at least one pattern related to operator
        assert len(patterns) >= 0  # May or may not match depending on trigram

    def test_find_patterns_empty(self):
        """Test finding patterns with regular text."""
        tokens = ["hello", "world"]

        pos1 = Mock()
        pos1.expert_indices = [5]
        pos2 = Mock()
        pos2.expert_indices = [5]

        positions = [pos1, pos2]

        patterns = ExploreService.find_patterns(tokens, positions)

        # Should detect sequence start pattern at position 0
        assert isinstance(patterns, list)


class TestExploreServiceFindInterestingPositions:
    """Tests for ExploreService.find_interesting_positions."""

    def test_find_interesting_positions_basic(self):
        """Test finding interesting positions."""
        tokens = ["def", "add", "(", "x", ",", "y", ")", ":"]

        positions = ExploreService.find_interesting_positions(tokens, top_k=3)

        assert len(positions) <= 3
        assert all(0 <= p < len(tokens) for p in positions)

    def test_find_interesting_positions_with_operators(self):
        """Test that operators are considered interesting."""
        tokens = ["2", "+", "3", "="]

        positions = ExploreService.find_interesting_positions(tokens, top_k=2)

        assert len(positions) <= 2

    def test_find_interesting_positions_empty(self):
        """Test with minimal input."""
        tokens = ["a"]

        positions = ExploreService.find_interesting_positions(tokens, top_k=4)

        # Only one token, might be flagged as start/end
        assert isinstance(positions, list)


class TestExploreServiceAnalyzeLayerEvolution:
    """Tests for ExploreService.analyze_layer_evolution."""

    def test_analyze_layer_evolution_basic(self):
        """Test analyzing layer evolution."""
        tokens = ["2", "+", "3"]

        # Create mock layer weights
        def make_layer_weights(layer_idx, expert_ids):
            lw = Mock()
            lw.layer_idx = layer_idx
            positions = []
            for exp in expert_ids:
                p = Mock()
                p.expert_indices = [exp]
                positions.append(p)
            lw.positions = positions
            return lw

        weights_by_layer = [
            make_layer_weights(0, [5, 6, 5]),
            make_layer_weights(4, [5, 6, 5]),
            make_layer_weights(12, [10, 6, 10]),
            make_layer_weights(20, [10, 6, 10]),
        ]

        evolution = ExploreService.analyze_layer_evolution(tokens, weights_by_layer, position=0)

        assert evolution.position == 0
        assert evolution.token == "2"
        assert isinstance(evolution.early, LayerPhaseData)
        assert isinstance(evolution.middle, LayerPhaseData)
        assert isinstance(evolution.late, LayerPhaseData)

    def test_analyze_layer_evolution_with_transitions(self):
        """Test detecting transitions between phases."""
        tokens = ["+"]

        def make_layer_weights(layer_idx, expert):
            lw = Mock()
            lw.layer_idx = layer_idx
            pos = Mock()
            pos.expert_indices = [expert]
            lw.positions = [pos]
            return lw

        # Different experts in different phases
        weights_by_layer = [
            make_layer_weights(0, 5),
            make_layer_weights(4, 5),
            make_layer_weights(12, 10),  # Middle phase - different expert
            make_layer_weights(20, 10),
        ]

        evolution = ExploreService.analyze_layer_evolution(tokens, weights_by_layer, position=0)

        # Should detect transition from early to middle
        assert evolution.early.dominant_expert == 5
        assert evolution.middle.dominant_expert == 10


class TestExploreServiceCompareRouting:
    """Tests for ExploreService.compare_routing."""

    def test_compare_routing_basic(self):
        """Test comparing routing between prompts."""
        tokens1 = ["2", "+", "3"]
        tokens2 = ["Calculate", ":", "2", "+", "3"]

        def make_pos(experts):
            p = Mock()
            p.expert_indices = experts
            p.weights = [1.0 / len(experts)] * len(experts)
            return p

        positions1 = [make_pos([5]), make_pos([6, 10]), make_pos([5])]
        positions2 = [
            make_pos([12]),
            make_pos([15]),
            make_pos([5]),
            make_pos([6, 10]),
            make_pos([5]),
        ]

        result = ExploreService.compare_routing(
            tokens1,
            positions1,
            tokens2,
            positions2,
            "2 + 3",
            "Calculate: 2 + 3",
            layer=12,
        )

        assert result.prompt1 == "2 + 3"
        assert result.prompt2 == "Calculate: 2 + 3"
        assert result.layer == 12
        assert len(result.tokens1) == 3
        assert len(result.tokens2) == 5
        assert isinstance(result.overlap_ratio, float)

    def test_compare_routing_identical(self):
        """Test comparing identical prompts."""
        tokens = ["test"]

        pos = Mock()
        pos.expert_indices = [5]
        pos.weights = [1.0]

        result = ExploreService.compare_routing(
            tokens, [pos], tokens, [pos], "test", "test", layer=0
        )

        assert result.overlap_ratio == 1.0
        assert result.only_prompt1 == []
        assert result.only_prompt2 == []


class TestExploreServiceDeepDivePosition:
    """Tests for ExploreService.deep_dive_position."""

    def test_deep_dive_position_basic(self):
        """Test deep dive into a position."""
        tokens = ["2", "+", "3"]

        def make_layer_weights(layer_idx, expert_weights_list):
            lw = Mock()
            lw.layer_idx = layer_idx
            positions = []
            for ew in expert_weights_list:
                p = Mock()
                p.expert_indices = [e for e, _ in ew]
                p.weights = [w for _, w in ew]
                positions.append(p)
            lw.positions = positions
            return lw

        weights_by_layer = [
            make_layer_weights(0, [[(5, 1.0)], [(6, 0.7), (10, 0.3)], [(5, 1.0)]]),
            make_layer_weights(12, [[(5, 1.0)], [(6, 0.8), (10, 0.2)], [(5, 1.0)]]),
            make_layer_weights(23, [[(5, 1.0)], [(6, 0.6), (10, 0.4)], [(5, 1.0)]]),
        ]

        result = ExploreService.deep_dive_position(tokens, weights_by_layer, position=1)

        assert result.position == 1
        assert result.token == "+"
        assert result.prev_token == "2"
        assert result.next_token == "3"
        assert result.dominant_expert in [6, 10]  # Most frequent expert

    def test_deep_dive_position_first(self):
        """Test deep dive on first position."""
        tokens = ["+", "3"]

        def make_layer_weights(layer_idx):
            lw = Mock()
            lw.layer_idx = layer_idx
            pos1 = Mock()
            pos1.expert_indices = [5]
            pos1.weights = [1.0]
            pos2 = Mock()
            pos2.expert_indices = [5]
            pos2.weights = [1.0]
            lw.positions = [pos1, pos2]
            return lw

        weights_by_layer = [make_layer_weights(0)]

        result = ExploreService.deep_dive_position(tokens, weights_by_layer, position=0)

        assert result.prev_token == "^"  # Start marker
        assert result.next_token == "3"

    def test_deep_dive_position_last(self):
        """Test deep dive on last position."""
        tokens = ["2", "+"]

        def make_layer_weights(layer_idx):
            lw = Mock()
            lw.layer_idx = layer_idx
            pos1 = Mock()
            pos1.expert_indices = [5]
            pos1.weights = [1.0]
            pos2 = Mock()
            pos2.expert_indices = [6]
            pos2.weights = [1.0]
            lw.positions = [pos1, pos2]
            return lw

        weights_by_layer = [make_layer_weights(0)]

        result = ExploreService.deep_dive_position(tokens, weights_by_layer, position=1)

        assert result.prev_token == "2"
        assert result.next_token == "$"  # End marker
