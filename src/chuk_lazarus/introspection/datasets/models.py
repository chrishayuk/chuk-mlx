"""Pydantic models for dataset validation and loading."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Arithmetic Benchmarks
# =============================================================================


class ArithmeticProblem(BaseModel):
    """A single arithmetic problem with expected answer."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="The arithmetic prompt (e.g., '127 * 89 = ')")
    answer: int = Field(description="The expected numeric answer")
    operation: str = Field(
        description="The operation type: addition, subtraction, multiplication, division"
    )


class ArithmeticBenchmark(BaseModel):
    """Full arithmetic benchmark dataset organized by difficulty."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(description="Dataset version")
    description: str = Field(description="Dataset description")
    problems: dict[str, list[ArithmeticProblem]] = Field(
        description="Problems organized by difficulty (simple, medium, hard)"
    )

    def get_all_problems(self) -> list[ArithmeticProblem]:
        """Get all problems flattened across difficulties."""
        result: list[ArithmeticProblem] = []
        for difficulty_problems in self.problems.values():
            result.extend(difficulty_problems)
        return result

    def get_by_difficulty(self, difficulty: str) -> list[ArithmeticProblem]:
        """Get problems by difficulty level."""
        return self.problems.get(difficulty, [])

    def get_prompts(self, difficulty: str | None = None) -> list[str]:
        """Get just the prompt strings, optionally filtered by difficulty."""
        if difficulty:
            return [p.prompt for p in self.get_by_difficulty(difficulty)]
        return [p.prompt for p in self.get_all_problems()]


# =============================================================================
# Uncertainty Detection
# =============================================================================


class UncertaintyPromptsSection(BaseModel):
    """A section of prompts with description."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Description of this prompt set")
    prompts: list[str] = Field(description="List of prompts")


class UncertaintyDataset(BaseModel):
    """Calibration prompts for uncertainty detection."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(description="Dataset version")
    description: str = Field(description="Dataset description")
    working_prompts: UncertaintyPromptsSection = Field(
        description="Prompts that should trigger compute pathway"
    )
    broken_prompts: UncertaintyPromptsSection = Field(
        description="Prompts that may trigger refusal/uncertainty"
    )

    @property
    def working(self) -> list[str]:
        """Get working prompts list."""
        return self.working_prompts.prompts

    @property
    def broken(self) -> list[str]:
        """Get broken prompts list."""
        return self.broken_prompts.prompts


# =============================================================================
# Context Independence Tests
# =============================================================================


class ContextTest(BaseModel):
    """A single context test case."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="The test prompt")
    context_type: str = Field(description="The type of context (number, word, article, etc.)")
    description: str = Field(default="", description="Optional description of this test case")


class ContextTestDataset(BaseModel):
    """Context independence test prompts."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(description="Dataset version")
    description: str = Field(description="Dataset description")
    target_token: str = Field(description="The token being tested for context independence")
    tests: list[ContextTest] = Field(description="List of test cases")

    def get_by_context_type(self, context_type: str) -> list[ContextTest]:
        """Get tests filtered by context type."""
        return [t for t in self.tests if t.context_type == context_type]

    def get_prompts(self) -> list[str]:
        """Get just the prompt strings."""
        return [t.prompt for t in self.tests]


# =============================================================================
# Pattern Discovery
# =============================================================================


class PatternCategory(BaseModel):
    """A category of test prompts for pattern discovery."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Description of this pattern category")
    prompts: list[str] = Field(description="List of prompts in this category")


class PatternDiscoveryDataset(BaseModel):
    """Test prompts for expert pattern discovery."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(description="Dataset version")
    description: str = Field(description="Dataset description")
    categories: dict[str, PatternCategory] = Field(
        description="Categories of patterns (num_seq, word_seq, code_patterns, etc.)"
    )

    def get_category(self, name: str) -> PatternCategory | None:
        """Get a specific category by name."""
        return self.categories.get(name)

    def get_category_names(self) -> list[str]:
        """Get all category names."""
        return list(self.categories.keys())

    def get_all_prompts(self) -> list[tuple[str, str]]:
        """Get all (category_name, prompt) tuples."""
        result: list[tuple[str, str]] = []
        for cat_name, cat in self.categories.items():
            for prompt in cat.prompts:
                result.append((cat_name, prompt))
        return result

    def get_prompts_for_category(self, category: str) -> list[str]:
        """Get prompts for a specific category."""
        cat = self.categories.get(category)
        return cat.prompts if cat else []


# =============================================================================
# Layer Sweep Tests
# =============================================================================


class LayerSweepSubcategory(BaseModel):
    """A subcategory of test prompts within a layer sweep category."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Description of this subcategory")
    prompts: list[str] = Field(description="List of prompts in this subcategory")


class LayerSweepCategory(BaseModel):
    """A category of test prompts for layer sweep analysis."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Description of this category")
    subcategories: dict[str, LayerSweepSubcategory] = Field(description="Subcategories of prompts")

    def get_all_prompts(self) -> list[tuple[str, str]]:
        """Get all (subcategory_name, prompt) tuples."""
        result: list[tuple[str, str]] = []
        for subcat_name, subcat in self.subcategories.items():
            for prompt in subcat.prompts:
                result.append((subcat_name, prompt))
        return result

    def get_subcategory_names(self) -> list[str]:
        """Get all subcategory names."""
        return list(self.subcategories.keys())


class LayerExpectation(BaseModel):
    """Expected patterns for a layer range."""

    model_config = ConfigDict(frozen=True)

    layer_fraction: tuple[float, float] = Field(
        description="Layer fraction range [start, end] as fraction of total MoE layers"
    )
    expected_patterns: list[str] = Field(description="Expected pattern types for this layer range")
    description: str = Field(description="Description of what this layer range does")


class LayerSweepDataset(BaseModel):
    """Comprehensive test suite for layer sweep analysis."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(description="Dataset version")
    description: str = Field(description="Dataset description")
    categories: dict[str, LayerSweepCategory] = Field(
        description="Test categories (structural, task_type, magnitude, etc.)"
    )
    layer_expectations: dict[str, LayerExpectation] = Field(
        description="Expected patterns by layer position"
    )

    def get_category(self, name: str) -> LayerSweepCategory | None:
        """Get a specific category by name."""
        return self.categories.get(name)

    def get_category_names(self) -> list[str]:
        """Get all category names."""
        return list(self.categories.keys())

    def get_all_prompts(self) -> list[tuple[str, str, str]]:
        """Get all (category, subcategory, prompt) tuples."""
        result: list[tuple[str, str, str]] = []
        for cat_name, cat in self.categories.items():
            for subcat_name, subcat in cat.subcategories.items():
                for prompt in subcat.prompts:
                    result.append((cat_name, subcat_name, prompt))
        return result

    def get_layer_expectation(self, layer_fraction: float) -> LayerExpectation | None:
        """Get expected patterns for a layer at given fraction of total."""
        for exp in self.layer_expectations.values():
            start, end = exp.layer_fraction
            if start <= layer_fraction < end:
                return exp
        return None

    def get_prompts_by_category(self, category: str) -> list[tuple[str, str]]:
        """Get (subcategory, prompt) tuples for a category."""
        cat = self.categories.get(category)
        if not cat:
            return []
        return cat.get_all_prompts()

    def get_structural_prompts(self) -> list[tuple[str, str]]:
        """Get structural test prompts."""
        return self.get_prompts_by_category("structural")

    def get_task_prompts(self) -> list[tuple[str, str]]:
        """Get task type test prompts."""
        return self.get_prompts_by_category("task_type")

    def get_output_prompts(self) -> list[tuple[str, str]]:
        """Get output type test prompts."""
        return self.get_prompts_by_category("output_type")
