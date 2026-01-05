"""Additional tests for full_taxonomy handler to improve coverage."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy import (
    _async_full_taxonomy,
    classify_token,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestClassifyToken:
    """Tests for classify_token function."""

    def test_classify_number(self):
        """Test classification of numbers."""
        assert classify_token("123") == "NUM"
        assert classify_token("0") == "NUM"
        assert classify_token("3.14") == "NUM"
        assert classify_token("-5") == "NUM"  # Negative numbers

    def test_classify_operator(self):
        """Test classification of operators."""
        assert classify_token("+") == "OP"
        assert classify_token("-") == "OP"
        assert classify_token("*") == "OP"
        assert classify_token("/") == "OP"
        assert classify_token("=") == "OP"
        assert classify_token("==") == "OP"
        assert classify_token("!=") == "OP"
        assert classify_token("&&") == "OP"
        assert classify_token("||") == "OP"
        assert classify_token("%") == "OP"

    def test_classify_brackets(self):
        """Test classification of brackets."""
        assert classify_token("(") == "BR"
        assert classify_token(")") == "BR"
        assert classify_token("[") == "BR"
        assert classify_token("]") == "BR"
        assert classify_token("{") == "BR"
        assert classify_token("}") == "BR"

    def test_classify_quotes(self):
        """Test classification of quotes."""
        assert classify_token("'") == "QUOTE"
        assert classify_token('"') == "QUOTE"
        assert classify_token("`") == "QUOTE"

    def test_classify_code_keyword(self):
        """Test classification of code keywords."""
        assert classify_token("def") == "KW"
        assert classify_token("if") == "KW"
        assert classify_token("return") == "KW"
        assert classify_token("class") == "KW"
        assert classify_token("for") == "KW"
        assert classify_token("while") == "KW"
        assert classify_token("async") == "KW"
        assert classify_token("await") == "KW"

    def test_classify_bool_literals(self):
        """Test classification of boolean/null literals."""
        assert classify_token("true") == "BOOL"
        assert classify_token("false") == "BOOL"
        assert classify_token("True") == "BOOL"
        assert classify_token("False") == "BOOL"
        # Note: "None" lowercases to "none" which is in NEGATION_WORDS
        assert classify_token("null") == "BOOL"
        assert classify_token("nil") == "BOOL"

    def test_classify_type_keywords(self):
        """Test classification of type keywords."""
        assert classify_token("int") == "TYPE"
        assert classify_token("str") == "TYPE"
        assert classify_token("float") == "TYPE"
        assert classify_token("bool") == "TYPE"

    def test_classify_synonym_markers(self):
        """Test classification of synonym markers."""
        assert classify_token("means") == "SYN"
        assert classify_token("equals") == "SYN"
        assert classify_token("similar") == "SYN"

    def test_classify_antonym_markers(self):
        """Test classification of antonym markers."""
        assert classify_token("versus") == "ANT"
        assert classify_token("opposite") == "ANT"
        assert classify_token("against") == "ANT"

    def test_classify_as_marker(self):
        """Test classification of 'as' marker."""
        assert classify_token("as") == "AS"

    def test_classify_to_marker(self):
        """Test classification of standalone 'to'."""
        assert classify_token("to") == "TO"

    def test_classify_cause_markers(self):
        """Test classification of cause markers."""
        assert classify_token("because") == "CAUSE"
        assert classify_token("therefore") == "CAUSE"
        assert classify_token("thus") == "CAUSE"

    def test_classify_condition_markers(self):
        """Test classification of condition markers."""
        # Note: "if" is KW first, but "unless", "when" are COND
        assert classify_token("unless") == "COND"
        assert classify_token("although") == "COND"

    def test_classify_than_marker(self):
        """Test classification of 'than'."""
        assert classify_token("than") == "THAN"

    def test_classify_question_words(self):
        """Test classification of question words."""
        assert classify_token("what") == "QW"
        assert classify_token("who") == "QW"
        assert classify_token("where") == "QW"
        assert classify_token("why") == "QW"
        assert classify_token("how") == "QW"

    def test_classify_answer_words(self):
        """Test classification of answer words."""
        assert classify_token("yes") == "ANS"
        assert classify_token("no") == "ANS"
        assert classify_token("maybe") == "ANS"

    def test_classify_negation(self):
        """Test classification of negation words."""
        assert classify_token("not") == "NEG"
        assert classify_token("never") == "NEG"
        assert classify_token("nothing") == "NEG"

    def test_classify_temporal(self):
        """Test classification of temporal words."""
        assert classify_token("now") == "TIME"
        assert classify_token("then") == "TIME"
        assert classify_token("yesterday") == "TIME"

    def test_classify_quantifiers(self):
        """Test classification of quantifiers."""
        assert classify_token("all") == "QUANT"
        assert classify_token("some") == "QUANT"
        assert classify_token("many") == "QUANT"

    def test_classify_comparison(self):
        """Test classification of comparison words."""
        # "than" returns "THAN" (checked earlier)
        assert classify_token("more") == "COMP"
        assert classify_token("less") == "COMP"
        assert classify_token("better") == "COMP"

    def test_classify_coordination(self):
        """Test classification of coordination words."""
        assert classify_token("and") == "COORD"
        assert classify_token("or") == "COORD"
        assert classify_token("but") == "COORD"

    def test_classify_nouns(self):
        """Test classification of nouns."""
        assert classify_token("cat") == "NOUN"
        assert classify_token("dog") == "NOUN"
        assert classify_token("king") == "NOUN"
        assert classify_token("queen") == "NOUN"

    def test_classify_adjectives(self):
        """Test classification of adjectives."""
        assert classify_token("big") == "ADJ"
        assert classify_token("small") == "ADJ"
        assert classify_token("happy") == "ADJ"
        assert classify_token("fast") == "ADJ"

    def test_classify_verbs(self):
        """Test classification of verbs."""
        assert classify_token("run") == "VERB"
        assert classify_token("walk") == "VERB"
        assert classify_token("think") == "VERB"

    def test_classify_function_words(self):
        """Test classification of function words."""
        assert classify_token("the") == "FUNC"
        assert classify_token("a") == "FUNC"
        assert classify_token("is") == "FUNC"
        assert classify_token("was") == "FUNC"

    def test_classify_capitalized(self):
        """Test classification of capitalized words (proper nouns)."""
        assert classify_token("Paris") == "CAP"
        assert classify_token("London") == "CAP"

    def test_classify_single_letter_variable(self):
        """Test classification of single letter variables."""
        assert classify_token("x") == "VAR"
        assert classify_token("y") == "VAR"
        assert classify_token("i") == "VAR"

    def test_classify_punctuation(self):
        """Test classification of punctuation."""
        # The actual classifications used in full_taxonomy
        # Note: "!" is in operators, not punctuation
        assert classify_token(".") == "PN"
        assert classify_token(",") == "PN"
        assert classify_token(":") == "PN"
        assert classify_token(";") == "PN"
        assert classify_token("?") == "PN"

    def test_classify_whitespace(self):
        """Test classification of whitespace."""
        assert classify_token(" ") == "WS"
        assert classify_token("\n") == "WS"
        assert classify_token("\t") == "WS"
        assert classify_token("") == "WS"  # Empty string

    def test_classify_regular_word(self):
        """Test classification of regular words."""
        # Regular words get classified as CW (common word)
        result = classify_token("banana")
        assert result == "CW"


class TestFullTaxonomyAdditional:
    """Additional tests for _async_full_taxonomy function."""

    @pytest.mark.asyncio
    async def test_full_taxonomy_handles_exceptions(self, capsys):
        """Test that exceptions during prompt processing are propagated."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        # Raise exception on capture
        mock_router.capture_router_weights = AsyncMock(side_effect=Exception("Test error"))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            # Exceptions are propagated, not caught
            with pytest.raises(Exception, match="Test error"):
                await _async_full_taxonomy(args)

    @pytest.mark.asyncio
    async def test_full_taxonomy_with_empty_weights(self, capsys):
        """Test taxonomy generation when no weights are returned."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=[])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "SEMANTIC TRIGRAM TAXONOMY ANALYSIS" in captured.out

    @pytest.mark.asyncio
    async def test_full_taxonomy_with_multiple_layers(self, capsys):
        """Test taxonomy generation with multiple layers."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10, 15),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=20,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=layer,
                positions=(
                    RouterWeightCapture(
                        layer_idx=layer,
                        position_idx=0,
                        token="test",
                        expert_indices=(layer % 8,),
                        weights=(1.0,),
                    ),
                ),
            )
            for layer in [0, 5, 10, 15]
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "SEMANTIC TRIGRAM TAXONOMY ANALYSIS" in captured.out
            assert "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_full_taxonomy_verbose_output(self, capsys):
        """Test verbose output includes detailed information."""
        args = Namespace(model="test/model", verbose=True)

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="127",
                        expert_indices=(0, 1),
                        weights=(0.6, 0.4),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token="+",
                        expert_indices=(2,),
                        weights=(1.0,),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=2,
                        token="89",
                        expert_indices=(0, 1),
                        weights=(0.7, 0.3),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "SEMANTIC TRIGRAM TAXONOMY ANALYSIS" in captured.out

    @pytest.mark.asyncio
    async def test_full_taxonomy_with_categories_arg(self, capsys):
        """Test taxonomy with specific categories argument."""
        args = Namespace(model="test/model", categories="arithmetic,code")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="127",
                        expert_indices=(0,),
                        weights=(1.0,),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token="+",
                        expert_indices=(1,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "SEMANTIC TRIGRAM TAXONOMY ANALYSIS" in captured.out
            # Should include arithmetic and code categories
            assert "ARITHMETIC" in captured.out or "CODE" in captured.out
