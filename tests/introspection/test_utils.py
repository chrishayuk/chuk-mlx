"""Tests for introspection utils module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chuk_lazarus.introspection.utils import (
    analyze_orthogonality,
    apply_chat_template,
    compute_similarity_matrix,
    cosine_similarity,
    extract_expected_answer,
    find_answer_onset,
    find_discriminative_neurons,
    generate_arithmetic_prompts,
    load_external_chat_template,
    normalize_number_string,
    parse_layers_arg,
    parse_prompts_from_arg,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, chat_template: str | None = None):
        self.chat_template = chat_template
        self._vocab = {}
        self._id_counter = 0

    def encode(self, text: str) -> list[int]:
        """Simple character-based encoding."""
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Simple character-based decoding."""
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        if not self.chat_template:
            raise ValueError("No chat template")
        # Simple mock implementation
        result = ""
        for msg in messages:
            result += f"[{msg['role']}]: {msg['content']}\n"
        if add_generation_prompt:
            result += "[assistant]: "
        return result


class TestApplyChatTemplate:
    """Tests for apply_chat_template function."""

    def test_with_chat_template(self):
        tokenizer = MockTokenizer(chat_template="template")
        result = apply_chat_template(tokenizer, "Hello world")
        assert "[user]: Hello world" in result
        assert "[assistant]:" in result

    def test_without_chat_template(self):
        tokenizer = MockTokenizer()
        result = apply_chat_template(tokenizer, "Hello world")
        assert result == "Hello world"

    def test_no_add_generation_prompt(self):
        tokenizer = MockTokenizer(chat_template="template")
        result = apply_chat_template(tokenizer, "Test", add_generation_prompt=False)
        assert "[user]: Test" in result

    def test_tokenizer_error(self):
        # Tokenizer that raises error on apply_chat_template
        tokenizer = MockTokenizer(chat_template="template")

        def bad_apply(*args, **kwargs):
            raise RuntimeError("Template error")

        tokenizer.apply_chat_template = bad_apply
        result = apply_chat_template(tokenizer, "Test")
        # Should fallback to original prompt
        assert result == "Test"


class TestLoadExternalChatTemplate:
    """Tests for load_external_chat_template function."""

    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chat_template.jinja
            template_path = Path(tmpdir) / "chat_template.jinja"
            template_path.write_text("{% for msg in messages %}{{ msg.content }}{% endfor %}")

            tokenizer = MockTokenizer()
            load_external_chat_template(tokenizer, tmpdir)

            assert tokenizer.chat_template is not None
            assert "msg in messages" in tokenizer.chat_template

    def test_no_template_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = MockTokenizer()
            original_template = tokenizer.chat_template
            load_external_chat_template(tokenizer, tmpdir)
            # Should not change template
            assert tokenizer.chat_template == original_template

    def test_already_has_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = Path(tmpdir) / "chat_template.jinja"
            template_path.write_text("new template")

            tokenizer = MockTokenizer(chat_template="existing")
            load_external_chat_template(tokenizer, tmpdir)
            # Should not override existing template
            assert tokenizer.chat_template == "existing"


class TestExtractExpectedAnswer:
    """Tests for extract_expected_answer function."""

    def test_addition(self):
        assert extract_expected_answer("5 + 3 = ") == "8"
        assert extract_expected_answer("100 + 200 = ") == "300"

    def test_subtraction(self):
        assert extract_expected_answer("10 - 3 = ") == "7"
        assert extract_expected_answer("100 - 37 = ") == "63"

    def test_multiplication(self):
        assert extract_expected_answer("7 * 8 = ") == "56"
        assert extract_expected_answer("12 * 5 = ") == "60"

    def test_multiplication_aliases(self):
        assert extract_expected_answer("7 x 8 = ") == "56"
        assert extract_expected_answer("7×8 = ") == "56"

    def test_division(self):
        assert extract_expected_answer("20 / 4 = ") == "5"
        assert extract_expected_answer("100÷10 = ") == "10"

    def test_no_spaces(self):
        assert extract_expected_answer("7*8=") == "56"

    def test_extra_spaces(self):
        assert extract_expected_answer("  10   +   5   =  ") == "15"

    def test_invalid_format(self):
        assert extract_expected_answer("not a math problem") is None
        assert extract_expected_answer("5 + = ") is None
        assert extract_expected_answer("abc * def = ") is None

    def test_division_by_zero(self):
        assert extract_expected_answer("10 / 0 = ") is None


class TestFindAnswerOnset:
    """Tests for find_answer_onset function."""

    def test_answer_found_first(self):
        tokenizer = MockTokenizer()
        result = find_answer_onset("56", "56", tokenizer)
        assert result["answer_found"] is True
        assert result["is_answer_first"] is True
        assert result["onset_index"] is not None

    def test_answer_found_later(self):
        tokenizer = MockTokenizer()
        result = find_answer_onset("The answer is 56", "56", tokenizer)
        assert result["answer_found"] is True
        assert result["onset_index"] is not None

    def test_answer_not_found(self):
        tokenizer = MockTokenizer()
        result = find_answer_onset("42", "56", tokenizer)
        assert result["answer_found"] is False
        assert result["onset_index"] is None

    def test_no_expected_answer(self):
        tokenizer = MockTokenizer()
        result = find_answer_onset("any output", None, tokenizer)
        assert result["answer_found"] is False
        assert result["onset_index"] is None
        assert result["is_answer_first"] is None


class TestGenerateArithmeticPrompts:
    """Tests for generate_arithmetic_prompts function."""

    def test_multiplication_default(self):
        prompts = generate_arithmetic_prompts(operation="*", digit_range=(2, 3))
        assert len(prompts) == 4  # 2x2, 2x3, 3x2, 3x3
        assert all(p["prompt"].endswith("=") for p in prompts)
        assert all("*" in p["prompt"] for p in prompts)

    def test_addition(self):
        prompts = generate_arithmetic_prompts(operation="+", digit_range=(2, 3))
        assert len(prompts) == 4
        assert all("+" in p["prompt"] for p in prompts)
        # Check one example
        p = next(p for p in prompts if p["operand_a"] == 2 and p["operand_b"] == 3)
        assert p["result"] == 5

    def test_subtraction(self):
        prompts = generate_arithmetic_prompts(operation="-", digit_range=(5, 6))
        assert len(prompts) == 4
        # Check one example (operand_a=5, operand_b=5 gives result=0)
        p = next((p for p in prompts if p["operand_a"] == 6 and p["operand_b"] == 5), None)
        assert p is not None
        assert p["result"] == 1

    def test_division(self):
        # Use range that has some integer divisions
        prompts = generate_arithmetic_prompts(operation="/", digit_range=(2, 9))
        # Should skip non-integer divisions
        assert all(p["operand_a"] % p["operand_b"] == 0 for p in prompts)

    def test_include_answer(self):
        prompts = generate_arithmetic_prompts(
            operation="*", digit_range=(2, 3), include_answer=True
        )
        # Should include answer in prompt
        assert all(str(p["result"]) in p["prompt"] for p in prompts)

    def test_difficulty_easy(self):
        prompts = generate_arithmetic_prompts(operation="*", digit_range=(2, 9), difficulty="easy")
        # Easy: at least one operand <= 3
        assert all(p["operand_a"] <= 3 or p["operand_b"] <= 3 for p in prompts)

    def test_difficulty_hard(self):
        prompts = generate_arithmetic_prompts(operation="*", digit_range=(2, 9), difficulty="hard")
        # Hard: both operands >= 7
        assert all(p["operand_a"] >= 7 and p["operand_b"] >= 7 for p in prompts)

    def test_difficulty_medium(self):
        prompts = generate_arithmetic_prompts(
            operation="*", digit_range=(2, 9), difficulty="medium"
        )
        # Medium: not easy and not hard
        assert len(prompts) > 0

    def test_invalid_operation(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            generate_arithmetic_prompts(operation="%")


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - 0.0) < 1e-6

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([-1.0, -2.0])
        sim = cosine_similarity(v1, v2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([0.0, 0.0])
        # Should not divide by zero
        sim = cosine_similarity(v1, v2)
        assert not np.isnan(sim)


class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix function."""

    def test_single_vector(self):
        vectors = [np.array([1.0, 2.0, 3.0])]
        matrix = compute_similarity_matrix(vectors)
        assert matrix.shape == (1, 1)
        assert abs(matrix[0, 0] - 1.0) < 1e-6

    def test_multiple_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v3 = np.array([1.0, 1.0])
        vectors = [v1, v2, v3]
        matrix = compute_similarity_matrix(vectors)

        assert matrix.shape == (3, 3)
        # Diagonal should be 1
        assert abs(matrix[0, 0] - 1.0) < 1e-6
        # v1 and v2 are orthogonal
        assert abs(matrix[0, 1] - 0.0) < 1e-6
        # Matrix should be symmetric
        assert abs(matrix[0, 1] - matrix[1, 0]) < 1e-6


class TestAnalyzeOrthogonality:
    """Tests for analyze_orthogonality function."""

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        result = analyze_orthogonality([v1, v2, v3], threshold=0.1)

        assert len(result["orthogonal_pairs"]) == 3
        assert len(result["aligned_pairs"]) == 0
        assert result["mean_abs_similarity"] < 0.1

    def test_aligned_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])  # Same direction, different magnitude
        result = analyze_orthogonality([v1, v2], threshold=0.1)

        assert len(result["aligned_pairs"]) == 1
        assert len(result["orthogonal_pairs"]) == 0

    def test_with_names(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        result = analyze_orthogonality([v1, v2], names=["x", "y"])

        assert result["names"] == ["x", "y"]
        assert ("x", "y", pytest.approx(0.0, abs=1e-6)) in result["orthogonal_pairs"]

    def test_threshold_parameter(self):
        v1 = np.array([1.0, 0.1])
        v2 = np.array([0.1, 1.0])
        # Small non-zero similarity

        # Strict threshold
        result1 = analyze_orthogonality([v1, v2], threshold=0.01)
        assert len(result1["orthogonal_pairs"]) == 0

        # Loose threshold
        result2 = analyze_orthogonality([v1, v2], threshold=0.5)
        assert len(result2["orthogonal_pairs"]) == 1


class TestFindDiscriminativeNeurons:
    """Tests for find_discriminative_neurons function."""

    def test_single_discriminative_neuron(self):
        # Create activations where neuron 0 discriminates
        activations = np.array(
            [
                [10.0, 0.0, 0.0],  # Class A
                [11.0, 0.0, 0.0],  # Class A (slight variation)
                [0.0, 0.0, 0.0],  # Class B
                [1.0, 0.0, 0.0],  # Class B (slight variation)
            ]
        )
        labels = ["A", "A", "B", "B"]

        neurons = find_discriminative_neurons(activations, labels, top_k=3)

        assert len(neurons) <= 3
        # Neuron 0 should be most discriminative
        assert neurons[0]["idx"] == 0
        # With some std, separation should be positive
        assert neurons[0]["separation"] >= 0

    def test_top_k_parameter(self):
        activations = np.random.randn(10, 50)
        labels = ["A"] * 5 + ["B"] * 5

        neurons = find_discriminative_neurons(activations, labels, top_k=5)
        assert len(neurons) == 5

    def test_single_sample_per_group(self):
        activations = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        labels = ["A", "B"]

        neurons = find_discriminative_neurons(activations, labels, top_k=2)
        # Should handle single samples
        assert len(neurons) == 2

    def test_group_means(self):
        activations = np.array(
            [
                [10.0, 5.0],
                [12.0, 5.0],  # Class A
                [0.0, 20.0],
                [0.0, 22.0],  # Class B
            ]
        )
        labels = ["A", "A", "B", "B"]

        neurons = find_discriminative_neurons(activations, labels, top_k=2)

        # Check that group means are calculated
        neuron = neurons[0]
        assert "group_means" in neuron
        assert "A" in neuron["group_means"]
        assert "B" in neuron["group_means"]


class TestNormalizeNumberString:
    """Tests for normalize_number_string function."""

    def test_remove_spaces(self):
        assert normalize_number_string("1 234 567") == "1234567"

    def test_remove_commas(self):
        assert normalize_number_string("1,234,567") == "1234567"

    def test_remove_thin_space(self):
        # U+202F thin space
        assert normalize_number_string("1\u202f234") == "1234"

    def test_remove_nbsp(self):
        # U+00A0 non-breaking space
        assert normalize_number_string("1\u00a0234") == "1234"

    def test_mixed_separators(self):
        assert normalize_number_string("1,234 567") == "1234567"

    def test_no_separators(self):
        assert normalize_number_string("1234567") == "1234567"


class TestParsePromptsFromArg:
    """Tests for parse_prompts_from_arg function."""

    def test_pipe_separated(self):
        prompts = parse_prompts_from_arg("prompt1|prompt2|prompt3")
        assert prompts == ["prompt1", "prompt2", "prompt3"]

    def test_with_spaces(self):
        prompts = parse_prompts_from_arg("  prompt1  |  prompt2  ")
        assert prompts == ["prompt1", "prompt2"]

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("prompt1\nprompt2\nprompt3\n")
            f.flush()
            filepath = f.name

        try:
            prompts = parse_prompts_from_arg(f"@{filepath}")
            assert prompts == ["prompt1", "prompt2", "prompt3"]
        finally:
            Path(filepath).unlink()

    def test_from_file_with_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("prompt1\n\nprompt2\n  \nprompt3\n")
            f.flush()
            filepath = f.name

        try:
            prompts = parse_prompts_from_arg(f"@{filepath}")
            # Should skip blank lines
            assert prompts == ["prompt1", "prompt2", "prompt3"]
        finally:
            Path(filepath).unlink()


class TestParseLayersArg:
    """Tests for parse_layers_arg function."""

    def test_comma_separated(self):
        layers = parse_layers_arg("0,1,2,5")
        assert layers == [0, 1, 2, 5]

    def test_range(self):
        layers = parse_layers_arg("0-3")
        assert layers == [0, 1, 2, 3]

    def test_mixed(self):
        layers = parse_layers_arg("0,2-4,8")
        assert layers == [0, 2, 3, 4, 8]

    def test_with_spaces(self):
        layers = parse_layers_arg("  0 ,  2 - 4  ,  8  ")
        assert layers == [0, 2, 3, 4, 8]

    def test_none_input(self):
        layers = parse_layers_arg(None)
        assert layers is None

    def test_empty_string(self):
        layers = parse_layers_arg("")
        assert layers is None

    def test_single_layer(self):
        layers = parse_layers_arg("5")
        assert layers == [5]
