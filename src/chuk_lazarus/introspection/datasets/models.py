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
