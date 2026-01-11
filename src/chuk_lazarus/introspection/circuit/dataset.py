"""
Generic dataset for circuit analysis.

Provides structured datasets of prompts with arbitrary labels for:
- Binary classification (e.g., computation vs suppression)
- Multi-class classification (e.g., task type)
- Contrastive pairs (e.g., base vs instruct behavior)

This module is domain-agnostic - use it for tool-calling, arithmetic,
factual recall, safety, or any other circuit analysis task.

Example:
    >>> from chuk_lazarus.introspection.circuit import CircuitDataset, LabeledPrompt
    >>>
    >>> # Create arithmetic suppression dataset
    >>> dataset = CircuitDataset(name="arithmetic")
    >>> dataset.add(LabeledPrompt(
    ...     text="6 * 7 =",
    ...     label=1,  # Should compute
    ...     category="arithmetic",
    ...     metadata={"expected": "42"}
    ... ))
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LabeledPrompt(BaseModel):
    """A single prompt with labels for circuit analysis.

    This is the generic version - labels can represent anything:
    - Tool vs no-tool
    - Compute vs suppress
    - Factual vs false
    - Safe vs unsafe
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="The prompt text")
    label: int = Field(description="Primary label (0 or 1 for binary, 0-N for multi-class)")
    category: str = Field(default="default", description="Grouping category")
    label_name: str | None = Field(default=None, description="Human-readable label name")
    expected_output: str | None = Field(default=None, description="Expected model output")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabeledPrompt:
        return cls.model_validate(data)


class ContrastivePair(BaseModel):
    """A pair of prompts for contrastive analysis.

    Useful for comparing:
    - Same prompt on base vs instruct model
    - Prompts that should vs shouldn't trigger a behavior
    - Before/after some intervention
    """

    model_config = ConfigDict(frozen=True)

    positive: LabeledPrompt = Field(description="Should exhibit the behavior")
    negative: LabeledPrompt = Field(description="Should NOT exhibit the behavior")
    pair_name: str = Field(default="", description="Name of this pair")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContrastivePair:
        return cls(
            positive=LabeledPrompt.from_dict(data["positive"]),
            negative=LabeledPrompt.from_dict(data["negative"]),
            pair_name=data.get("pair_name", ""),
        )


class CircuitDataset(BaseModel):
    """Generic dataset for circuit analysis.

    Supports:
    - Binary and multi-class labels
    - Categories for grouping
    - Contrastive pairs
    - Arbitrary metadata
    """

    model_config = ConfigDict(validate_default=True)

    prompts: list[LabeledPrompt] = Field(default_factory=list, description="All prompts")
    contrastive_pairs: list[ContrastivePair] = Field(
        default_factory=list, description="Contrastive pairs"
    )
    name: str = Field(default="circuit_dataset", description="Dataset name")
    version: str = Field(default="1.0", description="Version string")
    label_names: dict[int, str] = Field(default_factory=dict, description="Maps label int -> name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[LabeledPrompt]:
        return iter(self.prompts)

    def __getitem__(self, idx: int) -> LabeledPrompt:
        return self.prompts[idx]

    def add(self, prompt: LabeledPrompt) -> None:
        """Add a prompt to the dataset."""
        self.prompts.append(prompt)

    def add_many(self, prompts: list[LabeledPrompt]) -> None:
        """Add multiple prompts."""
        self.prompts.extend(prompts)

    def add_pair(self, pair: ContrastivePair) -> None:
        """Add a contrastive pair."""
        self.contrastive_pairs.append(pair)
        # Also add to main prompts list
        self.prompts.append(pair.positive)
        self.prompts.append(pair.negative)

    def get_by_label(self, label: int) -> list[LabeledPrompt]:
        """Get all prompts with a specific label."""
        return [p for p in self.prompts if p.label == label]

    def get_by_category(self, category: str) -> list[LabeledPrompt]:
        """Get all prompts of a specific category."""
        return [p for p in self.prompts if p.category == category]

    def get_positive(self) -> list[LabeledPrompt]:
        """Get all prompts with label=1 (positive class)."""
        return self.get_by_label(1)

    def get_negative(self) -> list[LabeledPrompt]:
        """Get all prompts with label=0 (negative class)."""
        return self.get_by_label(0)

    def get_labels(self) -> list[int]:
        """Get all labels as a list."""
        return [p.label for p in self.prompts]

    def get_texts(self) -> list[str]:
        """Get all prompt texts."""
        return [p.text for p in self.prompts]

    def get_categories(self) -> list[str]:
        """Get all categories."""
        return [p.category for p in self.prompts]

    def unique_labels(self) -> set[int]:
        """Get unique label values."""
        return set(self.get_labels())

    def unique_categories(self) -> set[str]:
        """Get unique category values."""
        return set(self.get_categories())

    def sample(self, n: int, balanced: bool = True, seed: int | None = None) -> list[LabeledPrompt]:
        """Sample n prompts, optionally balancing by label."""
        if seed is not None:
            random.seed(seed)

        if not balanced:
            return random.sample(self.prompts, min(n, len(self.prompts)))

        # Balance by label
        labels = self.unique_labels()
        n_per_label = n // len(labels)
        sampled = []

        for label in labels:
            label_prompts = self.get_by_label(label)
            sampled.extend(random.sample(label_prompts, min(n_per_label, len(label_prompts))))

        random.shuffle(sampled)
        return sampled

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        by_label = {}
        for label in self.unique_labels():
            count = len(self.get_by_label(label))
            label_name = self.label_names.get(label, f"label_{label}")
            by_label[label_name] = count

        by_category = {}
        for cat in self.unique_categories():
            by_category[cat] = len(self.get_by_category(cat))

        return {
            "name": self.name,
            "total": len(self.prompts),
            "contrastive_pairs": len(self.contrastive_pairs),
            "by_label": by_label,
            "by_category": by_category,
        }

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        data = {
            "name": self.name,
            "version": self.version,
            "label_names": {str(k): v for k, v in self.label_names.items()},
            "metadata": self.metadata,
            "prompts": [p.to_dict() for p in self.prompts],
            "contrastive_pairs": [p.to_dict() for p in self.contrastive_pairs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved dataset to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> CircuitDataset:
        """Load dataset from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        prompts = [LabeledPrompt.from_dict(p) for p in data.get("prompts", [])]
        pairs = [ContrastivePair.from_dict(p) for p in data.get("contrastive_pairs", [])]
        label_names = {int(k): v for k, v in data.get("label_names", {}).items()}

        return cls(
            prompts=prompts,
            contrastive_pairs=pairs,
            name=data.get("name", "loaded"),
            version=data.get("version", "1.0"),
            label_names=label_names,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Dataset factory functions for common use cases
# =============================================================================


def create_binary_dataset(
    positive_prompts: list[str],
    negative_prompts: list[str],
    name: str = "binary",
    positive_label: str = "positive",
    negative_label: str = "negative",
    positive_category: str = "positive",
    negative_category: str = "negative",
) -> CircuitDataset:
    """
    Create a binary classification dataset from two lists of prompts.

    Args:
        positive_prompts: Prompts for label=1
        negative_prompts: Prompts for label=0
        name: Dataset name
        positive_label: Human-readable name for positive class
        negative_label: Human-readable name for negative class
        positive_category: Category for positive prompts
        negative_category: Category for negative prompts

    Returns:
        CircuitDataset with binary labels
    """
    dataset = CircuitDataset(
        name=name,
        label_names={0: negative_label, 1: positive_label},
    )

    for text in positive_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=1,
                category=positive_category,
                label_name=positive_label,
            )
        )

    for text in negative_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=0,
                category=negative_category,
                label_name=negative_label,
            )
        )

    return dataset


def create_contrastive_dataset(
    pairs: list[tuple[str, str]],
    name: str = "contrastive",
    positive_label: str = "positive",
    negative_label: str = "negative",
) -> CircuitDataset:
    """
    Create a dataset from contrastive pairs.

    Args:
        pairs: List of (positive_text, negative_text) tuples
        name: Dataset name
        positive_label: Name for positive class
        negative_label: Name for negative class

    Returns:
        CircuitDataset with contrastive pairs
    """
    dataset = CircuitDataset(
        name=name,
        label_names={0: negative_label, 1: positive_label},
    )

    for i, (pos_text, neg_text) in enumerate(pairs):
        pair = ContrastivePair(
            positive=LabeledPrompt(text=pos_text, label=1, label_name=positive_label),
            negative=LabeledPrompt(text=neg_text, label=0, label_name=negative_label),
            pair_name=f"pair_{i}",
        )
        dataset.add_pair(pair)

    return dataset


# =============================================================================
# Example datasets for common circuit analysis tasks
# =============================================================================


def create_arithmetic_dataset(seed: int = 42) -> CircuitDataset:
    """
    Create dataset for arithmetic computation analysis.

    Tests whether models compute vs suppress arithmetic.
    """
    random.seed(seed)

    dataset = CircuitDataset(
        name="arithmetic",
        label_names={0: "non_arithmetic", 1: "arithmetic"},
        metadata={"purpose": "Test arithmetic computation circuits"},
    )

    # Arithmetic prompts (should compute)
    arithmetic_prompts = [
        ("6 * 7 =", "42"),
        ("3 + 5 =", "8"),
        ("12 - 4 =", "8"),
        ("8 * 9 =", "72"),
        ("15 + 27 =", "42"),
        ("100 - 37 =", "63"),
        ("7 * 8 =", "56"),
        ("25 + 17 =", "42"),
        ("9 * 6 =", "54"),
        ("48 / 6 =", "8"),
        ("156 + 287 =", "443"),
        ("324 - 189 =", "135"),
        ("23 * 17 =", "391"),
        ("144 / 12 =", "12"),
        ("999 + 1 =", "1000"),
    ]

    for text, expected in arithmetic_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=1,
                category="arithmetic",
                label_name="arithmetic",
                expected_output=expected,
            )
        )

    # Non-arithmetic prompts (retrieval, not computation)
    non_arithmetic_prompts = [
        ("The capital of France is", "Paris"),
        ("The Eiffel Tower is in", "Paris"),
        ("Water boils at", "100"),
        ("The speed of light is", "299792458"),
        ("pi equals approximately", "3.14"),
        ("The year World War II ended was", "1945"),
        ("Shakespeare wrote", "plays"),
        ("The largest planet is", "Jupiter"),
        ("Oxygen's atomic number is", "8"),
        ("The Mona Lisa was painted by", "da Vinci"),
        ("Photosynthesis produces", "oxygen"),
        ("DNA stands for", "deoxyribonucleic acid"),
        ("The Great Wall is in", "China"),
        ("Newton discovered", "gravity"),
        ("The Amazon River is in", "South America"),
    ]

    for text, expected in non_arithmetic_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=0,
                category="factual_retrieval",
                label_name="non_arithmetic",
                expected_output=expected,
            )
        )

    return dataset


def create_code_execution_dataset(seed: int = 42) -> CircuitDataset:
    """
    Create dataset for code execution/tracing analysis.

    Tests whether models trace code vs generate code.
    """
    dataset = CircuitDataset(
        name="code_execution",
        label_names={0: "no_trace", 1: "trace"},
        metadata={"purpose": "Test code tracing circuits"},
    )

    # Code that needs tracing
    trace_prompts = [
        ("x = 5\ny = 3\nprint(x + y)  # outputs:", "8"),
        ("a = 10\nb = a * 2\nprint(b)  # prints:", "20"),
        ("items = [1, 2, 3]\nprint(len(items))  # outputs:", "3"),
        ("s = 'hello'\nprint(s.upper())  # outputs:", "HELLO"),
        ("x = 7\nx = x + 3\nprint(x)  # prints:", "10"),
        ("n = 4\nresult = n ** 2\nprint(result)  # outputs:", "16"),
        ("a, b = 3, 4\nprint(a + b)  # outputs:", "7"),
        ("x = [1, 2, 3]\nx.append(4)\nprint(x[-1])  # prints:", "4"),
    ]

    for text, expected in trace_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=1,
                category="code_trace",
                label_name="trace",
                expected_output=expected,
            )
        )

    # Code discussion (no execution needed)
    no_trace_prompts = [
        ("What does the print function do in Python?", None),
        ("Explain how for loops work", None),
        ("What is the difference between list and tuple?", None),
        ("How do you define a function in Python?", None),
        ("What are Python decorators?", None),
        ("Explain object-oriented programming", None),
        ("What is recursion?", None),
        ("How does garbage collection work?", None),
    ]

    for text, expected in no_trace_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=0,
                category="code_discussion",
                label_name="no_trace",
                expected_output=expected,
            )
        )

    return dataset


def create_factual_consistency_dataset(seed: int = 42) -> CircuitDataset:
    """
    Create dataset for testing factual consistency circuits.

    Tests whether models detect contradictions between context and knowledge.
    """
    dataset = CircuitDataset(
        name="factual_consistency",
        label_names={0: "contradiction", 1: "consistent"},
        metadata={"purpose": "Test fact-checking circuits"},
    )

    # Consistent context (matches parametric knowledge)
    consistent = [
        ("The Eiffel Tower is in Paris. The Eiffel Tower is in", "Paris"),
        ("Water is H2O. Water is made of", "hydrogen and oxygen"),
        ("The Earth orbits the Sun. The Earth orbits", "the Sun"),
        ("Shakespeare was English. Shakespeare was from", "England"),
        ("Tokyo is in Japan. Tokyo is located in", "Japan"),
    ]

    for text, expected in consistent:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=1,
                category="consistent",
                label_name="consistent",
                expected_output=expected,
            )
        )

    # Contradictory context (conflicts with parametric knowledge)
    contradictions = [
        (
            "The Eiffel Tower is in London. The Eiffel Tower is in",
            "Paris",
        ),  # Model should hesitate/hedge
        ("Water is made of iron. Water is composed of", "hydrogen and oxygen"),
        ("The Earth orbits Mars. The Earth orbits", "the Sun"),
        ("Shakespeare was Chinese. Shakespeare was from", "England"),
        ("Tokyo is in Brazil. Tokyo is located in", "Japan"),
    ]

    for text, expected in contradictions:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=0,
                category="contradiction",
                label_name="contradiction",
                expected_output=expected,
                metadata={
                    "ground_truth": expected,
                    "context_claim": text.split(".")[0],
                },
            )
        )

    return dataset


def create_tool_delegation_dataset(seed: int = 42) -> CircuitDataset:
    """
    Create dataset for tool delegation analysis.

    Tests whether models delegate to tools vs compute internally.
    """
    dataset = CircuitDataset(
        name="tool_delegation",
        label_names={0: "internal", 1: "delegate"},
        metadata={"purpose": "Test tool delegation circuits"},
    )

    # Should delegate to tools
    delegate_prompts = [
        ("What's the weather in Tokyo?", "get_weather"),
        ("Search for Italian restaurants", "search"),
        ("Send an email to John", "send_email"),
        ("Calculate 156 + 287", "calculator"),
        ("Set a timer for 5 minutes", "timer"),
        ("Create a meeting for tomorrow", "calendar"),
        ("Find hotels in Paris", "search"),
        ("What time is it in London?", "get_time"),
    ]

    for text, tool in delegate_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=1,
                category="delegate",
                label_name="delegate",
                metadata={"expected_tool": tool},
            )
        )

    # Should use internal knowledge (no tools needed)
    internal_prompts = [
        ("What is the capital of France?", None),
        ("Explain photosynthesis", None),
        ("Write a haiku about the ocean", None),
        ("What is 2 + 2?", None),  # Simple enough to compute
        ("Who wrote Romeo and Juliet?", None),
        ("What is the meaning of life?", None),
        ("Tell me a joke", None),
        ("How do computers work?", None),
    ]

    for text, _ in internal_prompts:
        dataset.add(
            LabeledPrompt(
                text=text,
                label=0,
                category="internal",
                label_name="internal",
            )
        )

    return dataset


# =============================================================================
# Backwards compatibility: Keep the old tool-calling interface
# =============================================================================

# Re-export with old names for backwards compatibility
from enum import Enum  # noqa: E402


class PromptCategory(str, Enum):
    """Category of prompt (for backwards compatibility)."""

    WEATHER = "weather"
    CALENDAR = "calendar"
    SEARCH = "search"
    EMAIL = "email"
    CALCULATOR = "calculator"
    TIMER = "timer"
    FACTUAL = "factual"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"
    CODING = "coding"
    AMBIGUOUS = "ambiguous"
    MULTI_TOOL = "multi_tool"


class ToolPrompt(BaseModel):
    """A prompt for tool-calling analysis (backwards compatibility wrapper)."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="The prompt text")
    category: PromptCategory = Field(description="Prompt category")
    expected_tool: str | None = Field(default=None, description="Expected tool to call")
    should_call_tool: bool = Field(default=True, description="Whether tool should be called")
    difficulty: str = Field(default="normal", description="Difficulty level")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_labeled_prompt(self) -> LabeledPrompt:
        """Convert to generic LabeledPrompt."""
        return LabeledPrompt(
            text=self.text,
            label=1 if self.should_call_tool else 0,
            category=self.category.value,
            label_name="tool" if self.should_call_tool else "no_tool",
            metadata={
                "expected_tool": self.expected_tool,
                "difficulty": self.difficulty,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPrompt:
        data_copy = dict(data)
        if isinstance(data_copy.get("category"), str):
            data_copy["category"] = PromptCategory(data_copy["category"])
        return cls.model_validate(data_copy)


class ToolPromptDataset:
    """Wrapper for backwards compatibility with tool-calling code."""

    def __init__(self, circuit_dataset: CircuitDataset | None = None):
        self._dataset = circuit_dataset or CircuitDataset(name="tool_prompts")
        self._prompts: list[ToolPrompt] = []

    @property
    def prompts(self) -> list[ToolPrompt]:
        return self._prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self) -> Iterator[ToolPrompt]:
        return iter(self._prompts)

    def add(self, prompt: ToolPrompt) -> None:
        self._prompts.append(prompt)
        self._dataset.add(prompt.to_labeled_prompt())

    def get_tool_prompts(self) -> list[ToolPrompt]:
        return [p for p in self._prompts if p.should_call_tool]

    def get_no_tool_prompts(self) -> list[ToolPrompt]:
        return [p for p in self._prompts if not p.should_call_tool]

    def to_circuit_dataset(self) -> CircuitDataset:
        """Convert to generic CircuitDataset."""
        return self._dataset


def create_tool_calling_dataset(
    prompts_per_tool: int = 20,
    no_tool_prompts: int = 50,
    include_edge_cases: bool = True,
    seed: int = 42,
) -> ToolPromptDataset:
    """Create tool-calling dataset (backwards compatibility)."""
    # Import creates the actual dataset
    return _create_tool_calling_dataset_impl(
        prompts_per_tool, no_tool_prompts, include_edge_cases, seed
    )


def _create_tool_calling_dataset_impl(
    prompts_per_tool: int,
    no_tool_prompts: int,
    include_edge_cases: bool,
    seed: int,
) -> ToolPromptDataset:
    """Implementation of tool-calling dataset creation."""
    random.seed(seed)
    dataset = ToolPromptDataset()

    # Weather prompts
    weather_prompts = [
        "What's the weather in Tokyo?",
        "Is it raining in London?",
        "Temperature in New York",
        "Weather forecast for Paris",
        "Will it rain tomorrow in Berlin?",
    ]
    for text in weather_prompts[:prompts_per_tool]:
        dataset.add(
            ToolPrompt(
                text=text,
                category=PromptCategory.WEATHER,
                expected_tool="get_weather",
                should_call_tool=True,
            )
        )

    # Factual prompts (no tool)
    factual_prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "Explain quantum computing",
        "What is the speed of light?",
    ]
    for text in factual_prompts[:no_tool_prompts]:
        dataset.add(
            ToolPrompt(
                text=text,
                category=PromptCategory.FACTUAL,
                expected_tool=None,
                should_call_tool=False,
            )
        )

    return dataset
