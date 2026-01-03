"""Tests for MoE datasets and prompts."""

from chuk_lazarus.introspection.moe.datasets import (
    CATEGORY_GROUPS,
    CategoryPrompts,
    PromptCategory,
    PromptCategoryGroup,
    get_all_prompts,
    get_category_prompts,
    get_grouped_prompts,
    get_prompts_by_group,
    get_prompts_flat,
)


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_python_category(self):
        """Test PYTHON category."""
        assert PromptCategory.PYTHON.value == "python"

    def test_arithmetic_category(self):
        """Test ARITHMETIC category."""
        assert PromptCategory.ARITHMETIC.value == "arithmetic"

    def test_geography_category(self):
        """Test GEOGRAPHY category."""
        assert PromptCategory.GEOGRAPHY.value == "geography"

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        categories = list(PromptCategory)
        assert len(categories) >= 20  # At least 20 categories


class TestPromptCategoryGroup:
    """Tests for PromptCategoryGroup enum."""

    def test_code_group(self):
        """Test CODE group exists."""
        assert PromptCategoryGroup.CODE.value == "code"

    def test_math_group(self):
        """Test MATH group exists."""
        assert PromptCategoryGroup.MATH.value == "math"

    def test_has_groups(self):
        """Test groups exist."""
        groups = list(PromptCategoryGroup)
        assert len(groups) >= 5


class TestCategoryPrompts:
    """Tests for CategoryPrompts model."""

    def test_creation(self):
        """Test model creation."""
        prompts = CategoryPrompts(
            category=PromptCategory.PYTHON,
            group=PromptCategoryGroup.CODE,
            prompts=("def foo():", "class Bar:"),
        )
        assert prompts.category == PromptCategory.PYTHON
        assert len(prompts.prompts) == 2

    def test_defaults(self):
        """Test default values."""
        prompts = CategoryPrompts(
            category=PromptCategory.PYTHON,
            group=PromptCategoryGroup.CODE,
        )
        assert prompts.prompts == ()
        assert prompts.description == ""


class TestCategoryGroups:
    """Tests for CATEGORY_GROUPS constant."""

    def test_is_dict(self):
        """Test CATEGORY_GROUPS is a dictionary."""
        assert isinstance(CATEGORY_GROUPS, dict)

    def test_has_code_group(self):
        """Test has CODE group."""
        assert PromptCategoryGroup.CODE in CATEGORY_GROUPS

    def test_code_group_has_python(self):
        """Test CODE group contains PYTHON."""
        assert PromptCategory.PYTHON in CATEGORY_GROUPS[PromptCategoryGroup.CODE]

    def test_math_group_has_arithmetic(self):
        """Test MATH group contains ARITHMETIC."""
        assert PromptCategory.ARITHMETIC in CATEGORY_GROUPS[PromptCategoryGroup.MATH]


class TestGetCategoryPrompts:
    """Tests for get_category_prompts function."""

    def test_returns_category_prompts(self):
        """Test returns CategoryPrompts for category."""
        prompts = get_category_prompts(PromptCategory.PYTHON)

        assert isinstance(prompts, CategoryPrompts)
        assert prompts.category == PromptCategory.PYTHON
        assert prompts.group == PromptCategoryGroup.CODE

    def test_different_categories(self):
        """Test different categories work."""
        python_prompts = get_category_prompts(PromptCategory.PYTHON)
        math_prompts = get_category_prompts(PromptCategory.ARITHMETIC)

        assert python_prompts.category != math_prompts.category


class TestGetAllPrompts:
    """Tests for get_all_prompts function."""

    def test_returns_dict(self):
        """Test returns dictionary."""
        all_prompts = get_all_prompts()

        assert isinstance(all_prompts, dict)
        assert len(all_prompts) > 0

    def test_values_are_category_prompts(self):
        """Test values are CategoryPrompts."""
        all_prompts = get_all_prompts()

        for category, prompts in all_prompts.items():
            assert isinstance(prompts, CategoryPrompts)
            assert prompts.category == category


class TestGetGroupedPrompts:
    """Tests for get_grouped_prompts function."""

    def test_returns_dict(self):
        """Test returns dictionary."""
        grouped = get_grouped_prompts()

        assert isinstance(grouped, dict)


class TestGetPromptsByGroup:
    """Tests for get_prompts_by_group function."""

    def test_returns_list(self):
        """Test returns list of CategoryPrompts."""
        prompts = get_prompts_by_group(PromptCategoryGroup.CODE)

        assert isinstance(prompts, list)
        for p in prompts:
            assert isinstance(p, CategoryPrompts)


class TestGetPromptsFlat:
    """Tests for get_prompts_flat function."""

    def test_returns_list(self):
        """Test returns flat list of tuples."""
        prompts = get_prompts_flat()

        assert isinstance(prompts, list)

    def test_tuples_have_category_and_prompt(self):
        """Test each item is (category, prompt) tuple."""
        prompts = get_prompts_flat()

        for item in prompts:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], PromptCategory)
            assert isinstance(item[1], str)

    def test_filter_by_categories(self):
        """Test filtering by specific categories."""
        prompts = get_prompts_flat(categories=[PromptCategory.PYTHON])

        for cat, _ in prompts:
            assert cat == PromptCategory.PYTHON
