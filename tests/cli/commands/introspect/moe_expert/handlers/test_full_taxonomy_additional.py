"""Additional tests for full_taxonomy handler to improve coverage."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy import (
    _async_full_taxonomy,
    classify_token,
)
from chuk_lazarus.cli.commands._constants import TokenType
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
        assert classify_token("123") == TokenType.NUM
        assert classify_token("0") == TokenType.NUM
        # Floats with dots are not recognized as NUM (isdigit returns False)
        assert classify_token("3.14") == TokenType.CW  # Falls through to default
        # Negative numbers with dash go to OP check first (single char)
        assert classify_token("-5") == TokenType.CW  # Falls through to default

    def test_classify_operator(self):
        """Test classification of operators."""
        assert classify_token("+") == TokenType.OP
        assert classify_token("-") == TokenType.OP
        assert classify_token("*") == TokenType.OP
        assert classify_token("/") == TokenType.OP
        assert classify_token("=") == TokenType.OP
        # Multi-char operators not in set
        assert classify_token("==") == TokenType.CW
        assert classify_token("!=") == TokenType.CW
        assert classify_token("&&") == TokenType.CW
        assert classify_token("||") == TokenType.CW
        assert classify_token("%") == TokenType.OP

    def test_classify_brackets(self):
        """Test classification of brackets."""
        assert classify_token("(") == TokenType.BR
        assert classify_token(")") == TokenType.BR
        assert classify_token("[") == TokenType.BR
        assert classify_token("]") == TokenType.BR
        assert classify_token("{") == TokenType.BR
        assert classify_token("}") == TokenType.BR

    def test_classify_punctuation(self):
        """Test classification of punctuation."""
        assert classify_token(".") == TokenType.PN
        assert classify_token(",") == TokenType.PN
        assert classify_token(":") == TokenType.PN
        assert classify_token(";") == TokenType.PN
        assert classify_token("?") == TokenType.PN
        # Single quote is in punctuation set
        assert classify_token("'") == TokenType.PN

    def test_classify_quotes(self):
        """Test classification of quotes."""
        # Double quote is in QUOTE set
        assert classify_token('"') == TokenType.QUOTE
        assert classify_token("`") == TokenType.QUOTE
        assert classify_token("'''") == TokenType.QUOTE
        assert classify_token('"""') == TokenType.QUOTE

    def test_classify_code_keyword(self):
        """Test classification of code keywords."""
        assert classify_token("def") == TokenType.KW
        assert classify_token("if") == TokenType.KW
        assert classify_token("return") == TokenType.KW
        assert classify_token("class") == TokenType.KW
        assert classify_token("for") == TokenType.KW
        assert classify_token("while") == TokenType.KW
        assert classify_token("async") == TokenType.KW
        assert classify_token("await") == TokenType.KW

    def test_classify_bool_literals(self):
        """Test classification of boolean/null literals."""
        assert classify_token("true") == TokenType.BOOL
        assert classify_token("false") == TokenType.BOOL
        assert classify_token("True") == TokenType.BOOL
        assert classify_token("False") == TokenType.BOOL
        # null and nil are not in BOOLEAN_LITERALS
        assert classify_token("null") == TokenType.CW
        assert classify_token("nil") == TokenType.CW

    def test_classify_type_keywords(self):
        """Test classification of type keywords."""
        assert classify_token("int") == TokenType.TYPE
        assert classify_token("str") == TokenType.TYPE
        assert classify_token("float") == TokenType.TYPE
        assert classify_token("bool") == TokenType.TYPE

    def test_classify_synonym_markers(self):
        """Test classification of synonym markers."""
        assert classify_token("means") == TokenType.SYN
        assert classify_token("equals") == TokenType.SYN
        assert classify_token("similar") == TokenType.SYN

    def test_classify_antonym_markers(self):
        """Test classification of antonym markers."""
        assert classify_token("versus") == TokenType.ANT
        assert classify_token("opposite") == TokenType.ANT
        # "against" is not in the antonym set
        assert classify_token("against") == TokenType.CW

    def test_classify_as_marker(self):
        """Test classification of 'as' - it's a code keyword."""
        # "as" is in CODE_KEYWORDS, so returns KW
        assert classify_token("as") == TokenType.KW

    def test_classify_to_marker(self):
        """Test classification of standalone 'to'."""
        assert classify_token("to") == TokenType.TO

    def test_classify_cause_markers(self):
        """Test classification of cause markers."""
        assert classify_token("because") == TokenType.CAUSE
        assert classify_token("therefore") == TokenType.CAUSE
        assert classify_token("thus") == TokenType.CAUSE

    def test_classify_condition_markers(self):
        """Test classification of condition markers."""
        # "if" is KW first (code keyword check happens before conditional)
        assert classify_token("if") == TokenType.KW
        # "unless" is in CONDITIONAL_WORDS
        assert classify_token("unless") == TokenType.COND
        # "although" is not in CONDITIONAL_WORDS
        assert classify_token("although") == TokenType.CW

    def test_classify_than_marker(self):
        """Test classification of 'than'."""
        assert classify_token("than") == TokenType.THAN

    def test_classify_question_words(self):
        """Test classification of question words."""
        assert classify_token("what") == TokenType.QW
        assert classify_token("who") == TokenType.QW
        assert classify_token("where") == TokenType.QW
        assert classify_token("why") == TokenType.QW
        assert classify_token("how") == TokenType.QW

    def test_classify_answer_words(self):
        """Test classification of answer words."""
        assert classify_token("yes") == TokenType.ANS
        # "no" is in ANSWER_WORDS which is checked before NEGATION_WORDS
        assert classify_token("no") == TokenType.ANS
        assert classify_token("maybe") == TokenType.ANS

    def test_classify_negation(self):
        """Test classification of negation words."""
        # "not" is in CODE_KEYWORDS, checked first
        assert classify_token("not") == TokenType.KW
        # "never" is in both TIME_WORDS and NEGATION_WORDS
        # NEGATION is checked before TIME in the function
        assert classify_token("never") == TokenType.NEG
        assert classify_token("nothing") == TokenType.NEG

    def test_classify_temporal(self):
        """Test classification of temporal words."""
        assert classify_token("now") == TokenType.TIME
        assert classify_token("then") == TokenType.TIME
        assert classify_token("yesterday") == TokenType.TIME

    def test_classify_quantifiers(self):
        """Test classification of quantifiers."""
        assert classify_token("all") == TokenType.QUANT
        assert classify_token("some") == TokenType.QUANT
        assert classify_token("many") == TokenType.QUANT

    def test_classify_comparison(self):
        """Test classification of comparison words."""
        assert classify_token("more") == TokenType.COMP
        assert classify_token("less") == TokenType.COMP
        assert classify_token("better") == TokenType.COMP

    def test_classify_coordination(self):
        """Test classification of coordination words."""
        # "and" is in CODE_KEYWORDS, checked first
        assert classify_token("and") == TokenType.KW
        # "or" is in CODE_KEYWORDS, checked first
        assert classify_token("or") == TokenType.KW
        # "but" is in COORDINATION_WORDS, which is checked before inline antonym set
        assert classify_token("but") == TokenType.COORD
        # Test a pure coordination word
        assert classify_token("yet") == TokenType.COORD

    def test_classify_capitalized(self):
        """Test classification of capitalized words (proper nouns)."""
        assert classify_token("Paris") == TokenType.CAP
        assert classify_token("London") == TokenType.CAP

    def test_classify_single_letter_variable(self):
        """Test classification of single letter variables."""
        # Single letters fall through to CW (no VAR category in classify_token)
        assert classify_token("x") == TokenType.CW
        assert classify_token("y") == TokenType.CW
        # "i" is considered single char but not capitalized
        assert classify_token("i") == TokenType.CW

    def test_classify_whitespace(self):
        """Test classification of whitespace."""
        assert classify_token(" ") == TokenType.WS
        assert classify_token("\n") == TokenType.WS
        assert classify_token("\t") == TokenType.WS
        assert classify_token("") == TokenType.WS  # Empty string

    def test_classify_regular_word(self):
        """Test classification of regular words."""
        # Regular words get classified as CW (common word)
        result = classify_token("banana")
        assert result == TokenType.CW

    def test_classify_general_nouns_as_cw(self):
        """Test that general nouns without word lists are CW."""
        # No noun word list exists, so nouns become CW
        assert classify_token("cat") == TokenType.CW
        assert classify_token("dog") == TokenType.CW
        # Capitalized nouns become CAP
        assert classify_token("King") == TokenType.CAP
        assert classify_token("Queen") == TokenType.CAP

    def test_classify_general_adjectives_as_cw(self):
        """Test that general adjectives without word lists are CW."""
        # No adjective word list exists, so adjectives become CW
        assert classify_token("big") == TokenType.CW
        assert classify_token("small") == TokenType.CW
        assert classify_token("happy") == TokenType.CW
        # "fast" is in COMPARISON_WORDS as "faster"? No, it's not
        assert classify_token("fast") == TokenType.CW

    def test_classify_general_verbs_as_cw(self):
        """Test that general verbs without word lists are CW."""
        # No verb word list exists, so verbs become CW
        assert classify_token("run") == TokenType.CW
        assert classify_token("walk") == TokenType.CW
        assert classify_token("think") == TokenType.CW

    def test_classify_function_words_as_cw(self):
        """Test that general function words become CW."""
        # No function word list exists
        assert classify_token("the") == TokenType.CW
        assert classify_token("a") == TokenType.CW
        # "is" is in CODE_KEYWORDS
        assert classify_token("is") == TokenType.KW
        assert classify_token("was") == TokenType.CW


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
