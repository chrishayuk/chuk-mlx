"""Tests for MoE enums."""

import pytest

from chuk_lazarus.introspection.moe.enums import (
    ExpertCategory,
    ExpertRole,
    MoEArchitecture,
)


class TestMoEArchitecture:
    """Tests for MoEArchitecture enum."""

    def test_gpt_oss_value(self):
        """Test GPT_OSS enum value."""
        assert MoEArchitecture.GPT_OSS.value == "gpt_oss"
        assert MoEArchitecture.GPT_OSS == "gpt_oss"

    def test_llama4_value(self):
        """Test LLAMA4 enum value."""
        assert MoEArchitecture.LLAMA4.value == "llama4"
        assert MoEArchitecture.LLAMA4 == "llama4"

    def test_granite_hybrid_value(self):
        """Test GRANITE_HYBRID enum value."""
        assert MoEArchitecture.GRANITE_HYBRID.value == "granite_hybrid"
        assert MoEArchitecture.GRANITE_HYBRID == "granite_hybrid"

    def test_mixtral_value(self):
        """Test MIXTRAL enum value."""
        assert MoEArchitecture.MIXTRAL.value == "mixtral"
        assert MoEArchitecture.MIXTRAL == "mixtral"

    def test_generic_value(self):
        """Test GENERIC enum value."""
        assert MoEArchitecture.GENERIC.value == "generic"
        assert MoEArchitecture.GENERIC == "generic"

    def test_string_comparison(self):
        """Test string comparison works."""
        assert MoEArchitecture.GENERIC == "generic"
        assert "generic" == MoEArchitecture.GENERIC

    def test_all_values(self):
        """Test all enum values are accessible."""
        values = [a.value for a in MoEArchitecture]
        assert "gpt_oss" in values
        assert "llama4" in values
        assert "granite_hybrid" in values
        assert "mixtral" in values
        assert "generic" in values


class TestExpertCategory:
    """Tests for ExpertCategory enum."""

    def test_code_value(self):
        """Test CODE enum value."""
        assert ExpertCategory.CODE.value == "code"

    def test_math_value(self):
        """Test MATH enum value."""
        assert ExpertCategory.MATH.value == "math"

    def test_language_value(self):
        """Test LANGUAGE enum value."""
        assert ExpertCategory.LANGUAGE.value == "language"

    def test_punctuation_value(self):
        """Test PUNCTUATION enum value."""
        assert ExpertCategory.PUNCTUATION.value == "punctuation"

    def test_proper_nouns_value(self):
        """Test PROPER_NOUNS enum value."""
        assert ExpertCategory.PROPER_NOUNS.value == "proper_nouns"

    def test_function_words_value(self):
        """Test FUNCTION_WORDS enum value."""
        assert ExpertCategory.FUNCTION_WORDS.value == "function_words"

    def test_numbers_value(self):
        """Test NUMBERS enum value."""
        assert ExpertCategory.NUMBERS.value == "numbers"

    def test_position_first_value(self):
        """Test POSITION_FIRST enum value."""
        assert ExpertCategory.POSITION_FIRST.value == "position_first"

    def test_position_last_value(self):
        """Test POSITION_LAST enum value."""
        assert ExpertCategory.POSITION_LAST.value == "position_last"

    def test_generalist_value(self):
        """Test GENERALIST enum value."""
        assert ExpertCategory.GENERALIST.value == "generalist"

    def test_unknown_value(self):
        """Test UNKNOWN enum value."""
        assert ExpertCategory.UNKNOWN.value == "unknown"

    def test_all_categories(self):
        """Test all categories are defined."""
        categories = list(ExpertCategory)
        assert len(categories) == 11


class TestExpertRole:
    """Tests for ExpertRole enum."""

    def test_specialist_value(self):
        """Test SPECIALIST enum value."""
        assert ExpertRole.SPECIALIST.value == "specialist"

    def test_generalist_value(self):
        """Test GENERALIST enum value."""
        assert ExpertRole.GENERALIST.value == "generalist"

    def test_positional_value(self):
        """Test POSITIONAL enum value."""
        assert ExpertRole.POSITIONAL.value == "positional"

    def test_rare_value(self):
        """Test RARE enum value."""
        assert ExpertRole.RARE.value == "rare"

    def test_all_roles(self):
        """Test all roles are defined."""
        roles = list(ExpertRole)
        assert len(roles) == 4
