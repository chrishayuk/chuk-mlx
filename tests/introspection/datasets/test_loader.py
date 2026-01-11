"""Tests for DatasetLoader and convenience functions."""

import pytest

from chuk_lazarus.introspection.datasets import (
    DatasetLoader,
    get_arithmetic_benchmarks,
    get_context_tests,
    get_pattern_discovery_prompts,
    get_uncertainty_prompts,
)


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_load_json_caches_result(self):
        """Test that JSON loading is cached."""
        DatasetLoader.clear_cache()

        # First load
        data1 = DatasetLoader.load_json("benchmarks/arithmetic.json")
        # Second load should return same object (cached)
        data2 = DatasetLoader.load_json("benchmarks/arithmetic.json")

        assert data1 is data2

    def test_load_json_invalid_path(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader.load_json("nonexistent/file.json")

    def test_clear_cache(self):
        """Test cache clearing."""
        DatasetLoader.clear_cache()
        # Should not raise
        DatasetLoader.load_json("benchmarks/arithmetic.json")
        DatasetLoader.clear_cache()


class TestConvenienceFunctions:
    """Tests for convenience loading functions."""

    def test_get_arithmetic_benchmarks(self):
        """Test loading arithmetic benchmarks."""
        benchmarks = get_arithmetic_benchmarks()

        assert benchmarks.version == "1.0.0"
        assert "simple" in benchmarks.problems
        assert "medium" in benchmarks.problems
        assert "hard" in benchmarks.problems

        # Check we have actual problems
        all_problems = benchmarks.get_all_problems()
        assert len(all_problems) > 0

        # Check a specific problem
        hard = benchmarks.get_by_difficulty("hard")
        assert any(p.prompt == "127 * 89 = " for p in hard)

    def test_get_uncertainty_prompts(self):
        """Test loading uncertainty prompts."""
        dataset = get_uncertainty_prompts()

        assert dataset.version == "1.0.0"
        assert len(dataset.working) > 0
        assert len(dataset.broken) > 0

        # Working prompts should have trailing space
        for prompt in dataset.working:
            assert prompt.endswith(" "), f"Working prompt should end with space: {prompt}"

        # Broken prompts should NOT have trailing space
        for prompt in dataset.broken:
            assert not prompt.endswith(" "), f"Broken prompt should not end with space: {prompt}"

    def test_get_context_tests(self):
        """Test loading context tests."""
        dataset = get_context_tests()

        assert dataset.version == "1.0.0"
        assert dataset.target_token == "127"
        assert len(dataset.tests) > 0

        # Check we have different context types
        context_types = {t.context_type for t in dataset.tests}
        assert "number" in context_types
        assert "word" in context_types

    def test_get_pattern_discovery_prompts(self):
        """Test loading pattern discovery prompts."""
        dataset = get_pattern_discovery_prompts()

        assert dataset.version == "1.0.0"

        # Check we have expected categories
        category_names = dataset.get_category_names()
        assert "num_seq" in category_names
        assert "word_seq" in category_names
        assert "code_patterns" in category_names

        # Check each category has prompts
        for name in category_names:
            category = dataset.get_category(name)
            assert category is not None
            assert len(category.prompts) > 0


class TestDatasetIntegrity:
    """Tests for dataset content integrity."""

    def test_arithmetic_answers_are_correct(self):
        """Verify arithmetic problem answers are correct."""
        benchmarks = get_arithmetic_benchmarks()

        for problem in benchmarks.get_all_problems():
            # Parse the expression
            prompt = problem.prompt.strip()
            if prompt.endswith("= "):
                prompt = prompt[:-2]
            elif prompt.endswith("="):
                prompt = prompt[:-1]

            # Very basic evaluation for simple expressions
            try:
                if "+" in prompt:
                    parts = prompt.split("+")
                    expected = int(parts[0].strip()) + int(parts[1].strip())
                elif "-" in prompt:
                    parts = prompt.split("-")
                    expected = int(parts[0].strip()) - int(parts[1].strip())
                elif "*" in prompt:
                    parts = prompt.split("*")
                    expected = int(parts[0].strip()) * int(parts[1].strip())
                elif "/" in prompt:
                    parts = prompt.split("/")
                    expected = int(parts[0].strip()) // int(parts[1].strip())
                else:
                    continue

                assert problem.answer == expected, (
                    f"Wrong answer for {problem.prompt}: got {problem.answer}, expected {expected}"
                )
            except (ValueError, IndexError):
                # Complex expression, skip
                pass

    def test_context_tests_contain_target_token(self):
        """Verify context tests contain the target token."""
        dataset = get_context_tests()

        for test in dataset.tests:
            assert dataset.target_token in test.prompt, (
                f"Test prompt '{test.prompt}' should contain target token '{dataset.target_token}'"
            )
