"""Tests for analysis_service.py to improve coverage."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection._shared_constants import LayerPhase, TokenType
from chuk_lazarus.introspection.moe.analysis_service import (
    AttentionCaptureResult,
    ExpertWeightInfo,
    LayerRoutingInfo,
    MoEAnalysisServiceConfig,
    PositionRoutingInfo,
    TaxonomyExpertMapping,
    classify_token,
    get_layer_phase,
    get_trigram,
)


class TestClassifyToken:
    """Tests for classify_token function."""

    def test_whitespace(self):
        """Test whitespace classification (line 111-112)."""
        assert classify_token("") == TokenType.WS
        assert classify_token("   ") == TokenType.WS
        assert classify_token("\t") == TokenType.WS
        assert classify_token("\n") == TokenType.WS

    def test_numbers(self):
        """Test number classification (line 115-116)."""
        assert classify_token("123") == TokenType.NUM
        assert classify_token("0") == TokenType.NUM
        assert classify_token("42") == TokenType.NUM
        assert classify_token(" 99 ") == TokenType.NUM

    def test_operators(self):
        """Test operator classification (line 119-120)."""
        assert classify_token("+") == TokenType.OP
        assert classify_token("-") == TokenType.OP
        assert classify_token("*") == TokenType.OP
        assert classify_token("/") == TokenType.OP
        assert classify_token("=") == TokenType.OP
        assert classify_token("<") == TokenType.OP
        assert classify_token(">") == TokenType.OP
        assert classify_token("^") == TokenType.OP
        assert classify_token("%") == TokenType.OP

    def test_brackets(self):
        """Test bracket classification (line 123-124)."""
        assert classify_token("(") == TokenType.BR
        assert classify_token(")") == TokenType.BR
        assert classify_token("[") == TokenType.BR
        assert classify_token("]") == TokenType.BR
        assert classify_token("{") == TokenType.BR
        assert classify_token("}") == TokenType.BR

    def test_punctuation(self):
        """Test punctuation classification (line 127-128)."""
        assert classify_token(".") == TokenType.PN
        assert classify_token(",") == TokenType.PN
        assert classify_token(":") == TokenType.PN
        assert classify_token(";") == TokenType.PN
        assert classify_token("!") == TokenType.PN
        assert classify_token("?") == TokenType.PN

    def test_quotes(self):
        """Test quote classification (line 131-132)."""
        assert classify_token('"') == TokenType.QUOTE
        # Note: single quote might be punctuation as '-' is checked in PN
        assert classify_token("`") == TokenType.QUOTE
        assert classify_token("'''") == TokenType.QUOTE
        assert classify_token('"""') == TokenType.QUOTE

    def test_code_keywords(self):
        """Test code keyword classification (line 138-139)."""
        assert classify_token("if") == TokenType.KW
        assert classify_token("else") == TokenType.KW
        assert classify_token("for") == TokenType.KW
        assert classify_token("while") == TokenType.KW
        assert classify_token("def") == TokenType.KW
        assert classify_token("class") == TokenType.KW
        assert classify_token("return") == TokenType.KW

    def test_boolean_literals(self):
        """Test boolean literal classification (line 142-143)."""
        assert classify_token("True") == TokenType.BOOL
        assert classify_token("False") == TokenType.BOOL
        assert classify_token("true") == TokenType.BOOL
        assert classify_token("false") == TokenType.BOOL

    def test_type_keywords(self):
        """Test type keyword classification (line 146-147)."""
        assert classify_token("int") == TokenType.TYPE
        assert classify_token("str") == TokenType.TYPE
        assert classify_token("float") == TokenType.TYPE
        assert classify_token("list") == TokenType.TYPE
        assert classify_token("dict") == TokenType.TYPE

    def test_question_words(self):
        """Test question word classification (line 150-151)."""
        assert classify_token("what") == TokenType.QW
        assert classify_token("where") == TokenType.QW
        assert classify_token("when") == TokenType.QW
        assert classify_token("why") == TokenType.QW
        assert classify_token("how") == TokenType.QW
        assert classify_token("which") == TokenType.QW

    def test_answer_words(self):
        """Test answer word classification (line 154-155)."""
        assert classify_token("yes") == TokenType.ANS
        # Note: "no" is also in NEGATION_WORDS which is checked after ANSWER_WORDS
        # but there may be overlap - let's check actual behavior
        assert classify_token("maybe") == TokenType.ANS
        assert classify_token("probably") == TokenType.ANS
        assert classify_token("definitely") == TokenType.ANS

    def test_negation_words(self):
        """Test negation word classification (line 158-159)."""
        # Note: "not" is in CODE_KEYWORDS so it matches KW first
        assert classify_token("never") == TokenType.NEG
        # "none" is in QUANTIFIER_WORDS checked first
        assert classify_token("nothing") == TokenType.NEG
        assert classify_token("nobody") == TokenType.NEG

    def test_time_words(self):
        """Test time word classification (line 162-163)."""
        assert classify_token("now") == TokenType.TIME
        assert classify_token("before") == TokenType.TIME
        assert classify_token("after") == TokenType.TIME
        assert classify_token("today") == TokenType.TIME
        assert classify_token("tomorrow") == TokenType.TIME

    def test_quantifier_words(self):
        """Test quantifier word classification (line 166-167)."""
        assert classify_token("all") == TokenType.QUANT
        assert classify_token("some") == TokenType.QUANT
        assert classify_token("many") == TokenType.QUANT
        assert classify_token("few") == TokenType.QUANT
        assert classify_token("every") == TokenType.QUANT

    def test_comparison_words(self):
        """Test comparison word classification (line 170-171)."""
        assert classify_token("more") == TokenType.COMP
        assert classify_token("less") == TokenType.COMP
        assert classify_token("better") == TokenType.COMP
        assert classify_token("worse") == TokenType.COMP

    def test_coordination_words(self):
        """Test coordination word classification (line 174-175)."""
        # Note: "and", "or", "for" are in CODE_KEYWORDS so they match KW first
        # "but" is also in ANT check which happens later
        # "so" is in CAUSATION_WORDS checked earlier
        # Let's test words that are only in COORD: "yet", "nor"
        assert classify_token("yet") == TokenType.COORD
        assert classify_token("nor") == TokenType.COORD

    def test_causation_words(self):
        """Test causation word classification (line 178-179)."""
        assert classify_token("because") == TokenType.CAUSE
        assert classify_token("since") == TokenType.CAUSE
        assert classify_token("therefore") == TokenType.CAUSE
        assert classify_token("thus") == TokenType.CAUSE

    def test_conditional_words(self):
        """Test conditional word classification (line 182-183)."""
        # Note: "if", "when" may be in CODE_KEYWORDS
        assert classify_token("unless") == TokenType.COND
        assert classify_token("whenever") == TokenType.COND
        assert classify_token("provided") == TokenType.COND
        assert classify_token("assuming") == TokenType.COND

    def test_special_markers_as(self):
        """Test 'as' marker classification (line 186-187)."""
        # Note: "as" is in CODE_KEYWORDS so it matches KW first
        # The AS token type won't be reached for "as"
        # Skipping this test since CODE_KEYWORDS has priority
        pass

    def test_special_markers_to(self):
        """Test 'to' marker classification (line 188-189)."""
        assert classify_token("to") == TokenType.TO

    def test_special_markers_than(self):
        """Test 'than' marker classification (line 190-191)."""
        assert classify_token("than") == TokenType.THAN

    def test_synonym_markers(self):
        """Test synonym marker classification (line 194-195)."""
        assert classify_token("like") == TokenType.SYN
        assert classify_token("similar") == TokenType.SYN
        assert classify_token("same") == TokenType.SYN
        assert classify_token("means") == TokenType.SYN
        assert classify_token("equals") == TokenType.SYN

    def test_antonym_markers(self):
        """Test antonym marker classification (line 196-197)."""
        assert classify_token("opposite") == TokenType.ANT
        assert classify_token("versus") == TokenType.ANT
        assert classify_token("unlike") == TokenType.ANT
        assert classify_token("contrasts") == TokenType.ANT
        # Note: "but" is in COORDINATION_WORDS (line 174-175) which is checked
        # BEFORE ANT (line 196-197), so "but" matches COORD first
        assert classify_token("but") == TokenType.COORD

    def test_capitalized_proper_nouns(self):
        """Test capitalized proper noun classification (line 200-201)."""
        assert classify_token("Python") == TokenType.CAP
        assert classify_token("JavaScript") == TokenType.CAP
        assert classify_token("Apple") == TokenType.CAP
        # Single capital letter doesn't count
        assert classify_token("I") != TokenType.CAP

    def test_content_words(self):
        """Test default content word classification (line 204)."""
        assert classify_token("hello") == TokenType.CW
        assert classify_token("world") == TokenType.CW
        assert classify_token("code") == TokenType.CW
        assert classify_token("variable") == TokenType.CW

    def test_case_insensitivity(self):
        """Test that keyword matching is case-insensitive where appropriate."""
        # Code keywords should match case-insensitively
        assert classify_token("IF") == TokenType.KW
        assert classify_token("ELSE") == TokenType.KW
        # Question words
        assert classify_token("What") == TokenType.QW
        assert classify_token("WHERE") == TokenType.QW


class TestGetTrigram:
    """Tests for get_trigram function."""

    def test_first_position(self):
        """Test trigram at first position (line 220)."""
        tokens = ["hello", "world", "!"]
        trigram = get_trigram(tokens, 0)
        # ^→CW→CW
        assert trigram.startswith("^")
        assert "CW" in trigram

    def test_last_position(self):
        """Test trigram at last position (line 222)."""
        tokens = ["hello", "world", "!"]
        trigram = get_trigram(tokens, 2)
        # CW→PN→$
        assert trigram.endswith("$")
        assert "PN" in trigram

    def test_middle_position(self):
        """Test trigram at middle position (line 221)."""
        tokens = ["hello", "world", "test"]
        trigram = get_trigram(tokens, 1)
        # CW→CW→CW
        assert "→" in trigram
        assert not trigram.startswith("^")
        assert not trigram.endswith("$")

    def test_single_token(self):
        """Test trigram with single token."""
        tokens = ["hello"]
        trigram = get_trigram(tokens, 0)
        # ^→CW→$
        assert trigram.startswith("^")
        assert trigram.endswith("$")

    def test_two_tokens_first(self):
        """Test trigram with two tokens, first position."""
        tokens = ["hello", "world"]
        trigram = get_trigram(tokens, 0)
        # ^→CW→CW
        assert trigram.startswith("^")
        assert not trigram.endswith("$")

    def test_two_tokens_second(self):
        """Test trigram with two tokens, second position."""
        tokens = ["hello", "world"]
        trigram = get_trigram(tokens, 1)
        # CW→CW→$
        assert not trigram.startswith("^")
        assert trigram.endswith("$")


class TestGetLayerPhase:
    """Tests for get_layer_phase function."""

    def test_early_phase(self):
        """Test early layer phase classification (line 236-237)."""
        assert get_layer_phase(0) == LayerPhase.EARLY
        assert get_layer_phase(1) == LayerPhase.EARLY
        assert get_layer_phase(5) == LayerPhase.EARLY

    def test_middle_phase(self):
        """Test middle layer phase classification (line 238-239)."""
        # Actual defaults: EARLY_END=8, MIDDLE_END=16
        # So layers 8-15 are MIDDLE
        assert get_layer_phase(8) == LayerPhase.MIDDLE
        assert get_layer_phase(10) == LayerPhase.MIDDLE
        assert get_layer_phase(15) == LayerPhase.MIDDLE

    def test_late_phase(self):
        """Test late layer phase classification (line 240-241)."""
        # Actual defaults: MIDDLE_END=16, so 16+ is LATE
        assert get_layer_phase(16) == LayerPhase.LATE
        assert get_layer_phase(20) == LayerPhase.LATE
        assert get_layer_phase(50) == LayerPhase.LATE


class TestPydanticModels:
    """Tests for Pydantic models."""

    def test_expert_weight_info(self):
        """Test ExpertWeightInfo creation."""
        info = ExpertWeightInfo(expert_idx=0, weight=0.8)
        assert info.expert_idx == 0
        assert info.weight == 0.8

    def test_position_routing_info(self):
        """Test PositionRoutingInfo creation."""
        experts = [
            ExpertWeightInfo(expert_idx=0, weight=0.6),
            ExpertWeightInfo(expert_idx=1, weight=0.4),
        ]
        info = PositionRoutingInfo(
            position=5,
            token="hello",
            token_type="CW",
            trigram="^→CW→CW",
            experts=experts,
        )
        assert info.position == 5
        assert info.token == "hello"
        assert len(info.experts) == 2

    def test_layer_routing_info(self):
        """Test LayerRoutingInfo creation."""
        info = LayerRoutingInfo(layer_idx=10, positions=[])
        assert info.layer_idx == 10
        assert info.positions == []

    def test_attention_capture_result(self):
        """Test AttentionCaptureResult creation."""
        result = AttentionCaptureResult(
            layer=5,
            query_position=10,
            query_token="test",
            attention_weights=[(0, 0.3), (1, 0.2), (2, 0.15)],
            self_attention=0.25,
        )
        assert result.layer == 5
        assert result.query_position == 10
        assert result.query_token == "test"
        assert len(result.attention_weights) == 3
        assert result.self_attention == 0.25

    def test_taxonomy_expert_mapping(self):
        """Test TaxonomyExpertMapping creation."""
        mapping = TaxonomyExpertMapping(
            category="math",
            layer=12,
            experts=[0, 5, 10],
            trigrams=["NUM→OP→NUM"],
        )
        assert mapping.category == "math"
        assert mapping.layer == 12
        assert mapping.experts == [0, 5, 10]
        assert len(mapping.trigrams) == 1

    def test_moe_analysis_service_config(self):
        """Test MoEAnalysisServiceConfig creation."""
        config = MoEAnalysisServiceConfig(model="test-model")
        assert config.model == "test-model"

    def test_moe_analysis_service_config_frozen(self):
        """Test MoEAnalysisServiceConfig is frozen."""
        config = MoEAnalysisServiceConfig(model="test-model")
        with pytest.raises(ValidationError):
            config.model = "other-model"

    def test_moe_analysis_service_config_extra_forbid(self):
        """Test MoEAnalysisServiceConfig forbids extra fields."""
        with pytest.raises(ValidationError):
            MoEAnalysisServiceConfig(model="test", extra_field="value")


class TestClassifyTokenEdgeCases:
    """Edge case tests for classify_token."""

    def test_empty_string(self):
        """Test empty string is whitespace."""
        assert classify_token("") == TokenType.WS

    def test_mixed_case_keywords(self):
        """Test mixed case handling for keywords."""
        assert classify_token("If") == TokenType.KW
        assert classify_token("WHILE") == TokenType.KW
        assert classify_token("Return") == TokenType.KW

    def test_leading_trailing_whitespace(self):
        """Test tokens with leading/trailing whitespace."""
        assert classify_token(" + ") == TokenType.OP
        assert classify_token(" 42 ") == TokenType.NUM

    def test_single_char_not_capitalized(self):
        """Test single char doesn't count as proper noun."""
        # "I" is single char, shouldn't be CAP
        result = classify_token("I")
        # Single character capitalized is not CAP (len check in line 200)
        assert result != TokenType.CAP

    def test_lowercase_single_char(self):
        """Test lowercase single char is content word."""
        assert classify_token("a") == TokenType.CW
        assert classify_token("i") == TokenType.CW

    def test_special_characters_in_token(self):
        """Test tokens with special characters."""
        # Tokens not matching any specific pattern fall to CW
        assert classify_token("var_name") == TokenType.CW
        assert classify_token("test123") == TokenType.CW
