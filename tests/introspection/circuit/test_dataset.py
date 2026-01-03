"""Tests for circuit dataset module."""

import tempfile
from pathlib import Path

from chuk_lazarus.introspection.circuit.dataset import (
    CircuitDataset,
    ContrastivePair,
    LabeledPrompt,
    PromptCategory,
    ToolPrompt,
    ToolPromptDataset,
    create_arithmetic_dataset,
    create_binary_dataset,
    create_code_execution_dataset,
    create_contrastive_dataset,
    create_factual_consistency_dataset,
    create_tool_calling_dataset,
    create_tool_delegation_dataset,
)


class TestLabeledPrompt:
    """Tests for LabeledPrompt dataclass."""

    def test_create_with_defaults(self):
        """Test creating a labeled prompt with default values."""
        prompt = LabeledPrompt(text="Test prompt", label=1)
        assert prompt.text == "Test prompt"
        assert prompt.label == 1
        assert prompt.category == "default"
        assert prompt.label_name is None
        assert prompt.expected_output is None
        assert prompt.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating a labeled prompt with all fields."""
        prompt = LabeledPrompt(
            text="6 * 7 =",
            label=1,
            category="arithmetic",
            label_name="compute",
            expected_output="42",
            metadata={"difficulty": "easy"},
        )
        assert prompt.text == "6 * 7 ="
        assert prompt.label == 1
        assert prompt.category == "arithmetic"
        assert prompt.label_name == "compute"
        assert prompt.expected_output == "42"
        assert prompt.metadata["difficulty"] == "easy"

    def test_to_dict(self):
        """Test converting to dictionary."""
        prompt = LabeledPrompt(
            text="Test",
            label=0,
            category="test",
            label_name="negative",
            expected_output="output",
            metadata={"key": "value"},
        )
        data = prompt.to_dict()
        assert data["text"] == "Test"
        assert data["label"] == 0
        assert data["category"] == "test"
        assert data["label_name"] == "negative"
        assert data["expected_output"] == "output"
        assert data["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "text": "Test",
            "label": 1,
            "category": "test",
            "label_name": "positive",
            "expected_output": "output",
            "metadata": {"key": "value"},
        }
        prompt = LabeledPrompt.from_dict(data)
        assert prompt.text == "Test"
        assert prompt.label == 1
        assert prompt.category == "test"
        assert prompt.label_name == "positive"
        assert prompt.expected_output == "output"
        assert prompt.metadata["key"] == "value"

    def test_from_dict_minimal(self):
        """Test creating from dictionary with minimal fields."""
        data = {"text": "Test", "label": 1}
        prompt = LabeledPrompt.from_dict(data)
        assert prompt.text == "Test"
        assert prompt.label == 1
        assert prompt.category == "default"
        assert prompt.label_name is None


class TestContrastivePair:
    """Tests for ContrastivePair dataclass."""

    def test_create(self):
        """Test creating a contrastive pair."""
        pos = LabeledPrompt(text="Positive", label=1)
        neg = LabeledPrompt(text="Negative", label=0)
        pair = ContrastivePair(positive=pos, negative=neg, pair_name="test_pair")
        assert pair.positive == pos
        assert pair.negative == neg
        assert pair.pair_name == "test_pair"

    def test_to_dict(self):
        """Test converting to dictionary."""
        pos = LabeledPrompt(text="Positive", label=1)
        neg = LabeledPrompt(text="Negative", label=0)
        pair = ContrastivePair(positive=pos, negative=neg, pair_name="test")
        data = pair.to_dict()
        assert "positive" in data
        assert "negative" in data
        assert data["pair_name"] == "test"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "positive": {"text": "Pos", "label": 1},
            "negative": {"text": "Neg", "label": 0},
            "pair_name": "test",
        }
        pair = ContrastivePair.from_dict(data)
        assert pair.positive.text == "Pos"
        assert pair.negative.text == "Neg"
        assert pair.pair_name == "test"


class TestCircuitDataset:
    """Tests for CircuitDataset."""

    def test_create_empty(self):
        """Test creating an empty dataset."""
        dataset = CircuitDataset()
        assert len(dataset) == 0
        assert dataset.name == "circuit_dataset"
        assert dataset.version == "1.0"

    def test_add_prompt(self):
        """Test adding a prompt."""
        dataset = CircuitDataset()
        prompt = LabeledPrompt(text="Test", label=1)
        dataset.add(prompt)
        assert len(dataset) == 1
        assert dataset.prompts[0] == prompt

    def test_add_many_prompts(self):
        """Test adding multiple prompts."""
        dataset = CircuitDataset()
        prompts = [LabeledPrompt(text=f"Test {i}", label=i % 2) for i in range(5)]
        dataset.add_many(prompts)
        assert len(dataset) == 5

    def test_add_pair(self):
        """Test adding a contrastive pair."""
        dataset = CircuitDataset()
        pos = LabeledPrompt(text="Pos", label=1)
        neg = LabeledPrompt(text="Neg", label=0)
        pair = ContrastivePair(positive=pos, negative=neg)
        dataset.add_pair(pair)
        assert len(dataset) == 2
        assert len(dataset.contrastive_pairs) == 1

    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = CircuitDataset()
        prompts = [LabeledPrompt(text=f"Test {i}", label=i) for i in range(3)]
        dataset.add_many(prompts)
        for i, prompt in enumerate(dataset):
            assert prompt.text == f"Test {i}"

    def test_getitem(self):
        """Test indexing dataset."""
        dataset = CircuitDataset()
        prompts = [LabeledPrompt(text=f"Test {i}", label=i) for i in range(3)]
        dataset.add_many(prompts)
        assert dataset[0].text == "Test 0"
        assert dataset[2].text == "Test 2"

    def test_get_by_label(self):
        """Test getting prompts by label."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1))
        dataset.add(LabeledPrompt(text="P0", label=0))
        dataset.add(LabeledPrompt(text="P1_2", label=1))
        label_1 = dataset.get_by_label(1)
        assert len(label_1) == 2
        assert all(p.label == 1 for p in label_1)

    def test_get_by_category(self):
        """Test getting prompts by category."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="A1", label=1, category="cat_a"))
        dataset.add(LabeledPrompt(text="B1", label=0, category="cat_b"))
        dataset.add(LabeledPrompt(text="A2", label=1, category="cat_a"))
        cat_a = dataset.get_by_category("cat_a")
        assert len(cat_a) == 2
        assert all(p.category == "cat_a" for p in cat_a)

    def test_get_positive_negative(self):
        """Test getting positive and negative prompts."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1))
        dataset.add(LabeledPrompt(text="N1", label=0))
        dataset.add(LabeledPrompt(text="P2", label=1))
        positive = dataset.get_positive()
        negative = dataset.get_negative()
        assert len(positive) == 2
        assert len(negative) == 1

    def test_get_labels(self):
        """Test getting all labels."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1))
        dataset.add(LabeledPrompt(text="P0", label=0))
        labels = dataset.get_labels()
        assert labels == [1, 0]

    def test_get_texts(self):
        """Test getting all texts."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="Text 1", label=1))
        dataset.add(LabeledPrompt(text="Text 2", label=0))
        texts = dataset.get_texts()
        assert texts == ["Text 1", "Text 2"]

    def test_get_categories(self):
        """Test getting all categories."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1, category="cat1"))
        dataset.add(LabeledPrompt(text="P2", label=0, category="cat2"))
        categories = dataset.get_categories()
        assert categories == ["cat1", "cat2"]

    def test_unique_labels(self):
        """Test getting unique labels."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1))
        dataset.add(LabeledPrompt(text="P2", label=1))
        dataset.add(LabeledPrompt(text="P3", label=0))
        unique = dataset.unique_labels()
        assert unique == {0, 1}

    def test_unique_categories(self):
        """Test getting unique categories."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1, category="cat1"))
        dataset.add(LabeledPrompt(text="P2", label=0, category="cat2"))
        dataset.add(LabeledPrompt(text="P3", label=1, category="cat1"))
        unique = dataset.unique_categories()
        assert unique == {"cat1", "cat2"}

    def test_sample_unbalanced(self):
        """Test sampling without balancing."""
        dataset = CircuitDataset()
        for i in range(10):
            dataset.add(LabeledPrompt(text=f"P{i}", label=i % 2))
        sample = dataset.sample(5, balanced=False, seed=42)
        assert len(sample) == 5

    def test_sample_balanced(self):
        """Test balanced sampling."""
        dataset = CircuitDataset()
        for i in range(10):
            dataset.add(LabeledPrompt(text=f"P{i}", label=i % 2))
        sample = dataset.sample(4, balanced=True, seed=42)
        assert len(sample) == 4

    def test_sample_more_than_available(self):
        """Test sampling more items than available."""
        dataset = CircuitDataset()
        dataset.add(LabeledPrompt(text="P1", label=1))
        sample = dataset.sample(10, balanced=False, seed=42)
        assert len(sample) == 1

    def test_summary(self):
        """Test dataset summary."""
        dataset = CircuitDataset(name="test", label_names={0: "neg", 1: "pos"})
        dataset.add(LabeledPrompt(text="P1", label=1, category="cat1"))
        dataset.add(LabeledPrompt(text="P2", label=0, category="cat2"))
        dataset.add(LabeledPrompt(text="P3", label=1, category="cat1"))
        summary = dataset.summary()
        assert summary["name"] == "test"
        assert summary["total"] == 3
        assert summary["by_label"]["pos"] == 2
        assert summary["by_label"]["neg"] == 1
        assert summary["by_category"]["cat1"] == 2
        assert summary["by_category"]["cat2"] == 1

    def test_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = CircuitDataset(name="test", version="1.0", label_names={0: "neg", 1: "pos"})
        dataset.add(LabeledPrompt(text="Test", label=1, category="test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            dataset.save(path)
            loaded = CircuitDataset.load(path)
            assert loaded.name == dataset.name
            assert len(loaded) == len(dataset)
            assert loaded.label_names == dataset.label_names

    def test_save_with_contrastive_pairs(self):
        """Test saving dataset with contrastive pairs."""
        dataset = CircuitDataset()
        pos = LabeledPrompt(text="Pos", label=1)
        neg = LabeledPrompt(text="Neg", label=0)
        pair = ContrastivePair(positive=pos, negative=neg)
        dataset.add_pair(pair)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            dataset.save(path)
            loaded = CircuitDataset.load(path)
            assert len(loaded.contrastive_pairs) == 1


class TestBinaryDatasetFactory:
    """Tests for create_binary_dataset."""

    def test_create_binary_dataset(self):
        """Test creating a binary dataset."""
        pos = ["Pos 1", "Pos 2"]
        neg = ["Neg 1", "Neg 2"]
        dataset = create_binary_dataset(pos, neg, name="test")
        assert dataset.name == "test"
        assert len(dataset) == 4
        assert len(dataset.get_positive()) == 2
        assert len(dataset.get_negative()) == 2

    def test_create_binary_dataset_with_labels(self):
        """Test creating binary dataset with custom labels."""
        pos = ["Yes"]
        neg = ["No"]
        dataset = create_binary_dataset(pos, neg, positive_label="yes", negative_label="no")
        assert dataset.label_names[1] == "yes"
        assert dataset.label_names[0] == "no"

    def test_create_binary_dataset_with_categories(self):
        """Test creating binary dataset with custom categories."""
        pos = ["Pos"]
        neg = ["Neg"]
        dataset = create_binary_dataset(
            pos, neg, positive_category="pos_cat", negative_category="neg_cat"
        )
        assert dataset[0].category == "pos_cat"
        assert dataset[1].category == "neg_cat"


class TestContrastiveDatasetFactory:
    """Tests for create_contrastive_dataset."""

    def test_create_contrastive_dataset(self):
        """Test creating a contrastive dataset."""
        pairs = [("Pos 1", "Neg 1"), ("Pos 2", "Neg 2")]
        dataset = create_contrastive_dataset(pairs, name="test")
        assert dataset.name == "test"
        assert len(dataset) == 4
        assert len(dataset.contrastive_pairs) == 2

    def test_create_contrastive_dataset_with_labels(self):
        """Test creating contrastive dataset with custom labels."""
        pairs = [("A", "B")]
        dataset = create_contrastive_dataset(
            pairs, positive_label="option_a", negative_label="option_b"
        )
        assert dataset.label_names[1] == "option_a"
        assert dataset.label_names[0] == "option_b"


class TestArithmeticDataset:
    """Tests for create_arithmetic_dataset."""

    def test_create_arithmetic_dataset(self):
        """Test creating arithmetic dataset."""
        dataset = create_arithmetic_dataset(seed=42)
        assert dataset.name == "arithmetic"
        assert len(dataset) > 0
        assert 0 in dataset.label_names
        assert 1 in dataset.label_names

    def test_arithmetic_dataset_has_both_labels(self):
        """Test arithmetic dataset has both arithmetic and non-arithmetic."""
        dataset = create_arithmetic_dataset()
        labels = dataset.unique_labels()
        assert 0 in labels
        assert 1 in labels

    def test_arithmetic_dataset_has_expected_output(self):
        """Test arithmetic prompts have expected outputs."""
        dataset = create_arithmetic_dataset()
        for prompt in dataset:
            assert prompt.expected_output is not None


class TestCodeExecutionDataset:
    """Tests for create_code_execution_dataset."""

    def test_create_code_execution_dataset(self):
        """Test creating code execution dataset."""
        dataset = create_code_execution_dataset(seed=42)
        assert dataset.name == "code_execution"
        assert len(dataset) > 0

    def test_code_dataset_has_both_labels(self):
        """Test code dataset has both trace and no-trace."""
        dataset = create_code_execution_dataset()
        labels = dataset.unique_labels()
        assert 0 in labels
        assert 1 in labels


class TestFactualConsistencyDataset:
    """Tests for create_factual_consistency_dataset."""

    def test_create_factual_consistency_dataset(self):
        """Test creating factual consistency dataset."""
        dataset = create_factual_consistency_dataset(seed=42)
        assert dataset.name == "factual_consistency"
        assert len(dataset) > 0

    def test_factual_dataset_has_both_labels(self):
        """Test factual dataset has both consistent and contradictory."""
        dataset = create_factual_consistency_dataset()
        labels = dataset.unique_labels()
        assert 0 in labels
        assert 1 in labels


class TestToolDelegationDataset:
    """Tests for create_tool_delegation_dataset."""

    def test_create_tool_delegation_dataset(self):
        """Test creating tool delegation dataset."""
        dataset = create_tool_delegation_dataset(seed=42)
        assert dataset.name == "tool_delegation"
        assert len(dataset) > 0

    def test_tool_delegation_has_both_labels(self):
        """Test tool delegation dataset has both delegate and internal."""
        dataset = create_tool_delegation_dataset()
        labels = dataset.unique_labels()
        assert 0 in labels
        assert 1 in labels


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_prompt_category_values(self):
        """Test prompt category enum values."""
        assert PromptCategory.WEATHER.value == "weather"
        assert PromptCategory.CALENDAR.value == "calendar"
        assert PromptCategory.SEARCH.value == "search"


class TestToolPrompt:
    """Tests for ToolPrompt (backwards compatibility)."""

    def test_create_tool_prompt(self):
        """Test creating a tool prompt."""
        prompt = ToolPrompt(
            text="What's the weather?",
            category=PromptCategory.WEATHER,
            expected_tool="get_weather",
            should_call_tool=True,
        )
        assert prompt.text == "What's the weather?"
        assert prompt.category == PromptCategory.WEATHER
        assert prompt.expected_tool == "get_weather"
        assert prompt.should_call_tool is True

    def test_to_labeled_prompt(self):
        """Test converting to LabeledPrompt."""
        prompt = ToolPrompt(text="Test", category=PromptCategory.FACTUAL, should_call_tool=False)
        labeled = prompt.to_labeled_prompt()
        assert labeled.text == "Test"
        assert labeled.label == 0
        assert labeled.category == "factual"

    def test_to_dict(self):
        """Test converting to dictionary."""
        prompt = ToolPrompt(
            text="Test",
            category=PromptCategory.WEATHER,
            expected_tool="tool",
            should_call_tool=True,
        )
        data = prompt.to_dict()
        assert data["text"] == "Test"
        assert data["category"] == "weather"
        assert data["expected_tool"] == "tool"
        assert data["should_call_tool"] is True

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "text": "Test",
            "category": "weather",
            "expected_tool": "tool",
            "should_call_tool": True,
        }
        prompt = ToolPrompt.from_dict(data)
        assert prompt.text == "Test"
        assert prompt.category == PromptCategory.WEATHER


class TestToolPromptDataset:
    """Tests for ToolPromptDataset (backwards compatibility)."""

    def test_create_empty(self):
        """Test creating empty tool prompt dataset."""
        dataset = ToolPromptDataset()
        assert len(dataset) == 0

    def test_add_prompt(self):
        """Test adding a tool prompt."""
        dataset = ToolPromptDataset()
        prompt = ToolPrompt(text="Test", category=PromptCategory.FACTUAL, should_call_tool=False)
        dataset.add(prompt)
        assert len(dataset) == 1

    def test_iteration(self):
        """Test iterating over tool prompts."""
        dataset = ToolPromptDataset()
        prompt = ToolPrompt(text="Test", category=PromptCategory.FACTUAL, should_call_tool=False)
        dataset.add(prompt)
        for p in dataset:
            assert p.text == "Test"

    def test_get_tool_prompts(self):
        """Test getting only tool-calling prompts."""
        dataset = ToolPromptDataset()
        dataset.add(ToolPrompt(text="Tool", category=PromptCategory.WEATHER, should_call_tool=True))
        dataset.add(
            ToolPrompt(text="NoTool", category=PromptCategory.FACTUAL, should_call_tool=False)
        )
        tool_prompts = dataset.get_tool_prompts()
        assert len(tool_prompts) == 1
        assert tool_prompts[0].should_call_tool is True

    def test_get_no_tool_prompts(self):
        """Test getting only no-tool prompts."""
        dataset = ToolPromptDataset()
        dataset.add(ToolPrompt(text="Tool", category=PromptCategory.WEATHER, should_call_tool=True))
        dataset.add(
            ToolPrompt(text="NoTool", category=PromptCategory.FACTUAL, should_call_tool=False)
        )
        no_tool = dataset.get_no_tool_prompts()
        assert len(no_tool) == 1
        assert no_tool[0].should_call_tool is False

    def test_to_circuit_dataset(self):
        """Test converting to CircuitDataset."""
        dataset = ToolPromptDataset()
        dataset.add(ToolPrompt(text="Test", category=PromptCategory.FACTUAL, should_call_tool=True))
        circuit_ds = dataset.to_circuit_dataset()
        assert isinstance(circuit_ds, CircuitDataset)
        assert len(circuit_ds) == 1


class TestCreateToolCallingDataset:
    """Tests for create_tool_calling_dataset."""

    def test_create_tool_calling_dataset(self):
        """Test creating tool calling dataset."""
        dataset = create_tool_calling_dataset(prompts_per_tool=5, no_tool_prompts=10, seed=42)
        assert isinstance(dataset, ToolPromptDataset)
        assert len(dataset) > 0

    def test_has_both_tool_and_no_tool(self):
        """Test dataset has both tool and no-tool prompts."""
        dataset = create_tool_calling_dataset()
        tool = dataset.get_tool_prompts()
        no_tool = dataset.get_no_tool_prompts()
        assert len(tool) > 0
        assert len(no_tool) > 0

    def test_exclude_edge_cases(self):
        """Test excluding edge cases."""
        dataset = create_tool_calling_dataset(include_edge_cases=False)
        assert len(dataset) > 0
