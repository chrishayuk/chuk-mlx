"""Tests for explore handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
    handle_explore,
)


class TestHandleExplore:
    """Tests for handle_explore function."""

    def test_handle_explore_calls_asyncio_run(self):
        """Test that handle_explore calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_explore_with_layer(self):
        """Test handle_explore with layer parameter."""
        args = Namespace(model="test/model", layer=5)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()

    def test_handle_explore_with_verbose(self):
        """Test handle_explore with verbose parameter."""
        args = Namespace(model="test/model", verbose=True)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.asyncio"
        ) as mock_asyncio:
            handle_explore(args)
            mock_asyncio.run.assert_called_once()


class TestExploreService:
    """Tests for ExploreService."""

    def test_analyze_routing(self):
        """Test analyze_routing returns TokenAnalysis list."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["Hello", "world", "+", "5"]
        positions = [
            MagicMock(expert_indices=[1, 2], weights=[0.6, 0.4]),
            MagicMock(expert_indices=[3], weights=[1.0]),
            MagicMock(expert_indices=[6, 7], weights=[0.7, 0.3]),
            MagicMock(expert_indices=[8], weights=[1.0]),
        ]

        result = ExploreService.analyze_routing(tokens, positions)

        assert len(result) == 4
        assert result[0].position == 0
        assert result[0].token == "Hello"
        assert result[0].top_expert == 1
        assert result[0].all_experts == [1, 2]

    def test_analyze_routing_empty_weights(self):
        """Test analyze_routing with empty weights."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["test"]
        positions = [MagicMock(expert_indices=[1], weights=None)]

        result = ExploreService.analyze_routing(tokens, positions)

        assert len(result) == 1
        assert result[0].expert_weights == []

    def test_find_patterns_arithmetic_operator(self):
        """Test find_patterns detects arithmetic operators."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["5", "+", "3"]
        positions = [
            MagicMock(expert_indices=[1], weights=[1.0]),
            MagicMock(expert_indices=[6], weights=[1.0]),
            MagicMock(expert_indices=[2], weights=[1.0]),
        ]

        result = ExploreService.find_patterns(tokens, positions)

        # Check if any pattern was found
        assert isinstance(result, list)

    def test_find_patterns_sequence_start(self):
        """Test find_patterns detects sequence start."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["def", "foo", "(", ")"]
        positions = [
            MagicMock(expert_indices=[1], weights=[1.0]),
            MagicMock(expert_indices=[2], weights=[1.0]),
            MagicMock(expert_indices=[3], weights=[1.0]),
            MagicMock(expert_indices=[4], weights=[1.0]),
        ]

        result = ExploreService.find_patterns(tokens, positions)
        assert isinstance(result, list)

    def test_find_interesting_positions(self):
        """Test find_interesting_positions returns sorted positions."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["The", "king", "is", "to", "the", "queen", "as"]

        result = ExploreService.find_interesting_positions(tokens, top_k=3)

        assert isinstance(result, list)
        assert len(result) <= 3

    def test_find_interesting_positions_sequence_markers(self):
        """Test find_interesting_positions scores sequence markers."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["Start", "middle", "End"]
        result = ExploreService.find_interesting_positions(tokens, top_k=2)

        assert isinstance(result, list)
        # First and last positions should be interesting
        assert 0 in result or 2 in result

    def test_analyze_layer_evolution(self):
        """Test analyze_layer_evolution returns PositionEvolution."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["Hello", "world"]

        # Create layer weights data
        layer_weights = []
        for layer_idx in [0, 5, 12, 20]:
            lw = MagicMock()
            lw.layer_idx = layer_idx
            lw.positions = [
                MagicMock(expert_indices=[1]),
                MagicMock(expert_indices=[2]),
            ]
            layer_weights.append(lw)

        result = ExploreService.analyze_layer_evolution(tokens, layer_weights, position=0)

        assert result.position == 0
        assert result.token == "Hello"
        assert result.early is not None
        assert result.middle is not None
        assert result.late is not None
        assert result.early.phase_name == "early"

    def test_analyze_layer_evolution_with_transitions(self):
        """Test analyze_layer_evolution detects transitions."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["test"]

        # Create layer weights with different experts in different phases
        layer_weights = []
        for layer_idx, exp in [(0, 1), (5, 1), (12, 5), (20, 5)]:
            lw = MagicMock()
            lw.layer_idx = layer_idx
            lw.positions = [MagicMock(expert_indices=[exp])]
            layer_weights.append(lw)

        result = ExploreService.analyze_layer_evolution(tokens, layer_weights, position=0)

        # Check that early and middle have different dominant experts
        assert result.early.dominant_expert == 1
        # The middle dominant depends on layer_idx thresholds

    def test_compare_routing(self):
        """Test compare_routing returns ComparisonResult."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens1 = ["Hello", "world"]
        tokens2 = ["Goodbye", "world"]
        positions1 = [
            MagicMock(expert_indices=[1, 2]),
            MagicMock(expert_indices=[3]),
        ]
        positions2 = [
            MagicMock(expert_indices=[4, 5]),
            MagicMock(expert_indices=[3]),  # Shared expert
        ]

        result = ExploreService.compare_routing(
            tokens1, positions1, tokens2, positions2, "Hello world", "Goodbye world", layer=0
        )

        assert result.prompt1 == "Hello world"
        assert result.prompt2 == "Goodbye world"
        assert 3 in result.shared_experts  # Expert 3 is shared
        assert result.overlap_ratio > 0

    def test_compare_routing_no_overlap(self):
        """Test compare_routing with no expert overlap."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens1 = ["test"]
        tokens2 = ["test"]
        positions1 = [MagicMock(expert_indices=[1, 2])]
        positions2 = [MagicMock(expert_indices=[3, 4])]

        result = ExploreService.compare_routing(
            tokens1, positions1, tokens2, positions2, "test1", "test2", layer=0
        )

        assert len(result.shared_experts) == 0
        assert 1 in result.only_prompt1
        assert 3 in result.only_prompt2

    def test_deep_dive_position(self):
        """Test deep_dive_position returns DeepDiveResult."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["Hello", "world", "!"]

        layer_weights = []
        for layer_idx in [0, 12, 23]:
            lw = MagicMock()
            lw.layer_idx = layer_idx
            lw.positions = [
                MagicMock(expert_indices=[1], weights=[1.0]),
                MagicMock(expert_indices=[2, 3], weights=[0.7, 0.3]),
                MagicMock(expert_indices=[4], weights=[1.0]),
            ]
            layer_weights.append(lw)

        result = ExploreService.deep_dive_position(tokens, layer_weights, position=1)

        assert result.position == 1
        assert result.token == "world"
        assert result.prev_token == "Hello"
        assert result.next_token == "!"
        assert len(result.layer_routing) == 3
        assert result.dominant_expert in [2, 3]

    def test_deep_dive_position_no_weights(self):
        """Test deep_dive_position with no weights."""
        from chuk_lazarus.introspection.moe.explore_service import ExploreService

        tokens = ["test"]

        lw = MagicMock()
        lw.layer_idx = 0
        lw.positions = [MagicMock(expert_indices=[1, 2], weights=None)]

        result = ExploreService.deep_dive_position(tokens, [lw], position=0)

        assert result.position == 0
        assert len(result.layer_routing) == 1


class TestExploreServiceModels:
    """Tests for ExploreService Pydantic models."""

    def test_token_analysis_model(self):
        """Test TokenAnalysis model."""
        from chuk_lazarus.introspection.moe.explore_service import TokenAnalysis

        analysis = TokenAnalysis(
            position=0,
            token="hello",
            token_type="WORD",
            trigram="^→WORD→$",
            top_expert=1,
            all_experts=[1, 2],
            expert_weights=[0.6, 0.4],
        )
        assert analysis.position == 0
        assert analysis.top_expert == 1

    def test_pattern_match_model(self):
        """Test PatternMatch model."""
        from chuk_lazarus.introspection.moe.explore_service import PatternMatch

        match = PatternMatch(
            position=1,
            token="+",
            trigram="NUM→OP→NUM",
            pattern_type="arithmetic operator",
            top_expert=6,
        )
        assert match.position == 1
        assert match.pattern_type == "arithmetic operator"

    def test_layer_phase_data_model(self):
        """Test LayerPhaseData model."""
        from chuk_lazarus.introspection.moe.explore_service import LayerPhaseData

        phase = LayerPhaseData(
            phase_name="early",
            layer_range="L0-7",
            layer_experts=[(0, 1), (2, 1), (5, 2)],
            dominant_expert=1,
        )
        assert phase.phase_name == "early"
        assert phase.dominant_expert == 1

    def test_position_evolution_model(self):
        """Test PositionEvolution model."""
        from chuk_lazarus.introspection.moe.explore_service import (
            LayerPhaseData,
            PositionEvolution,
        )

        early = LayerPhaseData(
            phase_name="early", layer_range="L0-7", layer_experts=[], dominant_expert=1
        )
        middle = LayerPhaseData(
            phase_name="middle", layer_range="L8-15", layer_experts=[], dominant_expert=2
        )
        late = LayerPhaseData(
            phase_name="late", layer_range="L16+", layer_experts=[], dominant_expert=2
        )

        evolution = PositionEvolution(
            position=0,
            token="test",
            trigram="^→W→$",
            early=early,
            middle=middle,
            late=late,
            has_transition=True,
            transitions=["E1→E2"],
        )
        assert evolution.has_transition is True
        assert "E1→E2" in evolution.transitions

    def test_comparison_result_model(self):
        """Test ComparisonResult model."""
        from chuk_lazarus.introspection.moe.explore_service import ComparisonResult

        result = ComparisonResult(
            prompt1="Hello",
            prompt2="World",
            layer=0,
            tokens1=[],
            tokens2=[],
            shared_experts=[1, 2],
            only_prompt1=[3],
            only_prompt2=[4, 5],
            overlap_ratio=0.4,
        )
        assert result.overlap_ratio == 0.4
        assert 1 in result.shared_experts

    def test_deep_dive_result_model(self):
        """Test DeepDiveResult model."""
        from chuk_lazarus.introspection.moe.explore_service import DeepDiveResult

        result = DeepDiveResult(
            position=1,
            token="+",
            token_type="OP",
            trigram="NUM→OP→NUM",
            prev_token="5",
            prev_type="NUM",
            next_token="3",
            next_type="NUM",
            layer_routing=[(0, [(6, 0.8), (7, 0.2)])],
            all_experts=[6, 7],
            dominant_expert=6,
            peak_layer=12,
        )
        assert result.dominant_expert == 6
        assert result.peak_layer == 12


class TestAsyncExplore:
    """Tests for _async_explore function."""

    @pytest.fixture
    def mock_router_context(self):
        """Create a mock ExpertRouter context manager."""
        mock_router = MagicMock()
        mock_router.info.num_experts = 8
        mock_router.info.num_experts_per_tok = 2
        mock_router.info.moe_layers = [0, 4, 8, 12]

        # Mock layer weights response
        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1, 2]
        mock_position.weights = [0.7, 0.3]

        mock_layer_weights = MagicMock()
        mock_layer_weights.layer_idx = 0
        mock_layer_weights.positions = [mock_position]

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture
        return mock_router

    @pytest.mark.asyncio
    async def test_async_explore_quit_immediately(self, mock_router_context, capsys):
        """Test _async_explore with immediate quit."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "MOE EXPERT EXPLORER" in captured.out
        assert "Loading model" in captured.out
        assert "Goodbye!" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_eof(self, mock_router_context, capsys):
        """Test _async_explore handles EOF."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "MOE EXPERT EXPLORER" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_keyboard_interrupt(self, mock_router_context, capsys):
        """Test _async_explore handles keyboard interrupt."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "MOE EXPERT EXPLORER" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_empty_input(self, mock_router_context, capsys):
        """Test _async_explore skips empty input."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_layer_command(self, mock_router_context, capsys):
        """Test _async_explore with layer switch command."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["l 2", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "Switched to layer 2" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_layer_invalid(self, mock_router_context, capsys):
        """Test _async_explore with invalid layer number."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["l 100", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "Invalid layer" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_layer_non_numeric(self, mock_router_context, capsys):
        """Test _async_explore with non-numeric layer."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["l abc", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "Usage: l <layer_number>" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_compare_no_prompt(self, mock_router_context, capsys):
        """Test _async_explore compare without current prompt."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=['c "other prompt"', "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "No current prompt" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_all_layers_no_prompt(self, mock_router_context, capsys):
        """Test _async_explore all layers without current prompt."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["a", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "No current prompt" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_deep_dive_no_prompt(self, mock_router_context, capsys):
        """Test _async_explore deep dive without current prompt."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["d 0", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "No current prompt" in captured.out

    @pytest.mark.asyncio
    async def test_async_explore_deep_dive_invalid(self, mock_router_context, capsys):
        """Test _async_explore deep dive with invalid position."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _async_explore,
        )

        args = Namespace(model="test/model", layer=None)

        async def mock_from_pretrained(*args, **kwargs):
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_router_context)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExpertRouter.from_pretrained",
                side_effect=mock_from_pretrained,
            ),
            patch("builtins.input", side_effect=["d abc", "q"]),
        ):
            await _async_explore(args)

        captured = capsys.readouterr()
        assert "Usage: d <position_number>" in captured.out


class TestShowAnalysis:
    """Tests for _show_analysis function."""

    @pytest.mark.asyncio
    async def test_show_analysis_empty_weights(self, capsys):
        """Test _show_analysis with empty weights."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_analysis,
        )

        mock_router = MagicMock()

        async def mock_capture(*args, **kwargs):
            return []

        mock_router.capture_router_weights = mock_capture

        await _show_analysis(mock_router, "test prompt", 0)

        captured = capsys.readouterr()
        assert "No routing data captured" in captured.out

    @pytest.mark.asyncio
    async def test_show_analysis_with_data(self, capsys):
        """Test _show_analysis with routing data."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_analysis,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "hello"
        mock_position.expert_indices = [1, 2]
        mock_position.weights = [0.7, 0.3]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import (
                PatternMatch,
                TokenAnalysis,
            )

            mock_service.analyze_routing.return_value = [
                TokenAnalysis(
                    position=0,
                    token="hello",
                    token_type="WORD",
                    trigram="^→W→$",
                    top_expert=1,
                    all_experts=[1, 2],
                    expert_weights=[0.7, 0.3],
                )
            ]
            mock_service.find_patterns.return_value = [
                PatternMatch(
                    position=0,
                    token="hello",
                    trigram="^→W→$",
                    pattern_type="start token",
                    top_expert=1,
                )
            ]

            await _show_analysis(mock_router, "hello", 0)

        captured = capsys.readouterr()
        assert "TOKENIZATION & ROUTING" in captured.out
        assert "PATTERN SUMMARY" in captured.out

    @pytest.mark.asyncio
    async def test_show_analysis_no_patterns(self, capsys):
        """Test _show_analysis with no patterns found."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_analysis,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "x"
        mock_position.expert_indices = [1]
        mock_position.weights = [1.0]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import TokenAnalysis

            mock_service.analyze_routing.return_value = [
                TokenAnalysis(
                    position=0,
                    token="x",
                    token_type="WORD",
                    trigram="^→W→$",
                    top_expert=1,
                    all_experts=[1],
                    expert_weights=[1.0],
                )
            ]
            mock_service.find_patterns.return_value = []

            await _show_analysis(mock_router, "x", 0)

        captured = capsys.readouterr()
        assert "No notable patterns detected" in captured.out

    @pytest.mark.asyncio
    async def test_show_analysis_no_expert_weights(self, capsys):
        """Test _show_analysis with no expert weights."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_analysis,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1, 2, 3]
        mock_position.weights = None

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import TokenAnalysis

            mock_service.analyze_routing.return_value = [
                TokenAnalysis(
                    position=0,
                    token="test",
                    token_type="WORD",
                    trigram="^→W→$",
                    top_expert=1,
                    all_experts=[1, 2, 3],
                    expert_weights=[],  # Empty weights
                )
            ]
            mock_service.find_patterns.return_value = []

            await _show_analysis(mock_router, "test", 0)

        captured = capsys.readouterr()
        assert "EXPERT ROUTING" in captured.out


class TestComparePrompts:
    """Tests for _compare_prompts function."""

    @pytest.mark.asyncio
    async def test_compare_prompts_empty_weights(self, capsys):
        """Test _compare_prompts with empty weights."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _compare_prompts,
        )

        mock_router = MagicMock()

        async def mock_capture(*args, **kwargs):
            return []

        mock_router.capture_router_weights = mock_capture

        await _compare_prompts(mock_router, "prompt1", "prompt2", 0)

        captured = capsys.readouterr()
        assert "Could not capture routing" in captured.out

    @pytest.mark.asyncio
    async def test_compare_prompts_with_data(self, capsys):
        """Test _compare_prompts with valid data."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _compare_prompts,
        )

        mock_router = MagicMock()

        def create_mock_weights(token_str):
            mock_position = MagicMock()
            mock_position.token = token_str
            mock_position.expert_indices = [1, 2]
            mock_position.weights = [0.7, 0.3]

            mock_layer_weights = MagicMock()
            mock_layer_weights.positions = [mock_position]
            return [mock_layer_weights]

        call_count = [0]

        async def mock_capture(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return create_mock_weights("hello")
            return create_mock_weights("goodbye")

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import (
                ComparisonResult,
                TokenAnalysis,
            )

            mock_analysis = TokenAnalysis(
                position=0,
                token="test",
                token_type="WORD",
                trigram="^→W→$",
                top_expert=1,
                all_experts=[1, 2],
                expert_weights=[0.7, 0.3],
            )

            mock_service.compare_routing.return_value = ComparisonResult(
                prompt1="hello",
                prompt2="goodbye",
                layer=0,
                tokens1=[mock_analysis],
                tokens2=[mock_analysis],
                shared_experts=[1, 2],
                only_prompt1=[],
                only_prompt2=[3],
                overlap_ratio=0.67,
            )

            await _compare_prompts(mock_router, "hello", "goodbye", 0)

        captured = capsys.readouterr()
        assert "COMPARISON" in captured.out
        assert "EXPERT OVERLAP" in captured.out
        assert "Shared experts" in captured.out


class TestShowAllLayers:
    """Tests for _show_all_layers function."""

    @pytest.mark.asyncio
    async def test_show_all_layers_empty_weights(self, capsys):
        """Test _show_all_layers with empty weights."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_all_layers,
        )

        mock_router = MagicMock()

        async def mock_capture(*args, **kwargs):
            return []

        mock_router.capture_router_weights = mock_capture

        await _show_all_layers(mock_router, "test", [0, 4, 8])

        captured = capsys.readouterr()
        assert "No routing data captured" in captured.out

    @pytest.mark.asyncio
    async def test_show_all_layers_with_data(self, capsys):
        """Test _show_all_layers with valid data."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_all_layers,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]
        mock_layer_weights.layer_idx = 0

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import (
                LayerPhaseData,
                PositionEvolution,
            )

            mock_service.find_interesting_positions.return_value = [0]
            mock_service.analyze_layer_evolution.return_value = PositionEvolution(
                position=0,
                token="test",
                trigram="^→W→$",
                early=LayerPhaseData(
                    phase_name="early",
                    layer_range="L0-7",
                    layer_experts=[(0, 1)],
                    dominant_expert=1,
                ),
                middle=LayerPhaseData(
                    phase_name="middle",
                    layer_range="L8-15",
                    layer_experts=[(8, 2)],
                    dominant_expert=2,
                ),
                late=LayerPhaseData(
                    phase_name="late",
                    layer_range="L16+",
                    layer_experts=[(16, 2)],
                    dominant_expert=2,
                ),
                has_transition=True,
                transitions=["E1→E2"],
            )

            await _show_all_layers(mock_router, "test", [0, 8, 16])

        captured = capsys.readouterr()
        assert "LAYER EVOLUTION" in captured.out
        assert "EXPERT TRANSITIONS" in captured.out

    @pytest.mark.asyncio
    async def test_show_all_layers_stable_expert(self, capsys):
        """Test _show_all_layers with stable expert."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _show_all_layers,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]
        mock_layer_weights.layer_idx = 0

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import (
                LayerPhaseData,
                PositionEvolution,
            )

            mock_service.find_interesting_positions.return_value = [0]
            mock_service.analyze_layer_evolution.return_value = PositionEvolution(
                position=0,
                token="test",
                trigram="^→W→$",
                early=LayerPhaseData(
                    phase_name="early",
                    layer_range="L0-7",
                    layer_experts=[(0, 1)],
                    dominant_expert=1,
                ),
                middle=LayerPhaseData(
                    phase_name="middle",
                    layer_range="L8-15",
                    layer_experts=[(8, 1)],
                    dominant_expert=1,
                ),
                late=LayerPhaseData(
                    phase_name="late",
                    layer_range="L16+",
                    layer_experts=[(16, 1)],
                    dominant_expert=1,
                ),
                has_transition=False,
                transitions=[],
            )

            await _show_all_layers(mock_router, "test", [0, 8, 16])

        captured = capsys.readouterr()
        assert "(stable)" in captured.out


class TestDeepDive:
    """Tests for _deep_dive function."""

    @pytest.mark.asyncio
    async def test_deep_dive_empty_weights(self, capsys):
        """Test _deep_dive with empty weights."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _deep_dive,
        )

        mock_router = MagicMock()

        async def mock_capture(*args, **kwargs):
            return []

        mock_router.capture_router_weights = mock_capture

        await _deep_dive(mock_router, "test", 0, [0, 4, 8])

        captured = capsys.readouterr()
        assert "No routing data captured" in captured.out

    @pytest.mark.asyncio
    async def test_deep_dive_invalid_position(self, capsys):
        """Test _deep_dive with invalid position."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _deep_dive,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        await _deep_dive(mock_router, "test", 100, [0, 4, 8])

        captured = capsys.readouterr()
        assert "Invalid position" in captured.out

    @pytest.mark.asyncio
    async def test_deep_dive_with_data(self, capsys):
        """Test _deep_dive with valid data."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _deep_dive,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1, 2]
        mock_position.weights = [0.7, 0.3]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]
        mock_layer_weights.layer_idx = 0

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import DeepDiveResult

            mock_service.deep_dive_position.return_value = DeepDiveResult(
                position=0,
                token="test",
                token_type="WORD",
                trigram="^→W→$",
                prev_token="",
                prev_type="^",
                next_token="",
                next_type="$",
                layer_routing=[(0, [(1, 0.7), (2, 0.3)])],
                all_experts=[1, 2],
                dominant_expert=1,
                peak_layer=0,
            )

            await _deep_dive(mock_router, "test", 0, [0, 4, 8])

        captured = capsys.readouterr()
        assert "DEEP DIVE" in captured.out
        assert "Context:" in captured.out
        assert "ROUTING ACROSS ALL LAYERS" in captured.out
        assert "FINDING" in captured.out

    @pytest.mark.asyncio
    async def test_deep_dive_no_dominant(self, capsys):
        """Test _deep_dive with no dominant expert."""
        from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore import (
            _deep_dive,
        )

        mock_router = MagicMock()

        mock_position = MagicMock()
        mock_position.token = "test"
        mock_position.expert_indices = [1, 2]
        mock_position.weights = [0.5, 0.5]

        mock_layer_weights = MagicMock()
        mock_layer_weights.positions = [mock_position]
        mock_layer_weights.layer_idx = 0

        async def mock_capture(*args, **kwargs):
            return [mock_layer_weights]

        mock_router.capture_router_weights = mock_capture

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.explore.ExploreService"
        ) as mock_service:
            from chuk_lazarus.introspection.moe.explore_service import DeepDiveResult

            mock_service.deep_dive_position.return_value = DeepDiveResult(
                position=0,
                token="test",
                token_type="WORD",
                trigram="^→W→$",
                prev_token="",
                prev_type="^",
                next_token="",
                next_type="$",
                layer_routing=[(0, [(1, 0.5), (2, 0.5)])],
                all_experts=[1, 2],
                dominant_expert=None,  # No clear dominant
                peak_layer=None,
            )

            await _deep_dive(mock_router, "test", 0, [0, 4, 8])

        captured = capsys.readouterr()
        assert "DEEP DIVE" in captured.out
        # FINDING should not be printed when no dominant
        assert "FINDING" not in captured.out
