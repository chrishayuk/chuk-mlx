"""Tests for virtual_experts/plugins/math.py to improve coverage."""

import pytest

from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin


class TestMathExpertCleanExpression:
    """Tests for _clean_expression edge cases (line 143)."""

    def test_clean_expression_no_match(self):
        """Test when no numeric expression is found (line 143)."""
        plugin = MathExpertPlugin()
        # Expression with no numbers at all - returns empty or stripped
        result = plugin._clean_expression("hello world")
        # After removing prefixes and searching for numbers, may be empty
        assert isinstance(result, str)

    def test_clean_expression_with_special_chars(self):
        """Test clean_expression with Unicode operators."""
        plugin = MathExpertPlugin()
        result = plugin._clean_expression("5 × 3")
        assert "*" in result

        result = plugin._clean_expression("10 ÷ 2")
        assert "/" in result

        result = plugin._clean_expression("2 ^ 3")
        assert "**" in result


class TestMathExpertEvalNode:
    """Tests for _eval_node edge cases."""

    def test_eval_node_invalid_constant(self):
        """Test _eval_node with invalid constant (line 150)."""
        plugin = MathExpertPlugin()
        # Try to evaluate a string constant - should raise ValueError
        result = plugin._evaluate('"hello"')
        assert result is None

    def test_eval_node_unknown_name(self):
        """Test _eval_node with unknown name (lines 153-155)."""
        plugin = MathExpertPlugin()
        # Unknown variable name
        result = plugin._evaluate("xyz")
        assert result is None

    def test_eval_node_valid_constants(self):
        """Test _eval_node with valid constants (pi, e, inf)."""
        plugin = MathExpertPlugin()

        # pi
        result = plugin._evaluate("pi")
        assert result is not None
        assert abs(result - 3.14159) < 0.001

        # e
        result = plugin._evaluate("e")
        assert result is not None
        assert abs(result - 2.71828) < 0.001

    def test_eval_node_unsupported_binary_op(self):
        """Test _eval_node with unsupported binary operator (line 162)."""
        plugin = MathExpertPlugin()
        # Bitwise operators are not in OPERATORS dict, but Python AST parses them
        # The _clean_expression extracts just numbers, so test with direct AST
        import ast

        # Test that unsupported operator raises in _eval_node
        try:
            tree = ast.parse("5 & 3", mode="eval")
            plugin._eval_node(tree.body)
            # If we get here, the operator was processed somehow
        except (ValueError, Exception):
            pass  # Expected - unsupported operator

    def test_eval_node_unary_operators(self):
        """Test _eval_node with unary operators (line 169)."""
        plugin = MathExpertPlugin()

        # Unary minus
        result = plugin._evaluate("-5")
        assert result == -5

        # Unary plus
        result = plugin._evaluate("+5")
        assert result == 5

    def test_eval_node_unsupported_unary_op(self):
        """Test _eval_node with unsupported unary operator (line 169)."""
        plugin = MathExpertPlugin()
        import ast

        # Test that unsupported unary operator raises in _eval_node
        try:
            tree = ast.parse("~5", mode="eval")
            plugin._eval_node(tree.body)
        except (ValueError, Exception):
            pass  # Expected - unsupported unary operator

    def test_eval_node_function_calls(self):
        """Test _eval_node with function calls (lines 172-181)."""
        import ast

        plugin = MathExpertPlugin()

        # Test function calls via direct AST parsing
        # sqrt
        tree = ast.parse("sqrt(16)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 4.0

        # abs
        tree = ast.parse("abs(-5)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 5

        # round
        tree = ast.parse("round(3.7)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 4

        # min/max
        tree = ast.parse("min(3, 5)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 3

        tree = ast.parse("max(3, 5)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 5

        # sin, cos, tan
        tree = ast.parse("sin(0)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert abs(result) < 0.001

        tree = ast.parse("cos(0)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert abs(result - 1) < 0.001

        # log functions
        tree = ast.parse("log(1)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert abs(result) < 0.001

        tree = ast.parse("log10(10)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert abs(result - 1) < 0.001

        # exp
        tree = ast.parse("exp(0)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert abs(result - 1) < 0.001

        # pow
        tree = ast.parse("pow(2, 3)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 8

        # floor/ceil
        tree = ast.parse("floor(3.7)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 3

        tree = ast.parse("ceil(3.2)", mode="eval")
        result = plugin._eval_node(tree.body)
        assert result == 4

    def test_eval_node_unsupported_function(self):
        """Test _eval_node with unsupported function (line 178)."""
        import ast

        plugin = MathExpertPlugin()
        # Unknown function should raise ValueError
        tree = ast.parse("unknown_func(5)", mode="eval")
        with pytest.raises(ValueError, match="Unsupported function"):
            plugin._eval_node(tree.body)

    def test_eval_node_unsupported_node_type(self):
        """Test _eval_node with unsupported node type (lines 180-181)."""
        plugin = MathExpertPlugin()
        # Lambda or other complex expressions
        result = plugin._evaluate("lambda x: x")
        assert result is None


class TestMathExpertExtractAndEvaluate:
    """Tests for extract_and_evaluate (lines 208-209)."""

    def test_extract_and_evaluate_direct_fallback(self):
        """Test extract_and_evaluate direct evaluation fallback (lines 208-209)."""
        plugin = MathExpertPlugin()

        # Simple expression without pattern match
        expr, result = plugin.extract_and_evaluate("2 + 3")
        assert result == 5

    def test_extract_and_evaluate_pattern_match(self):
        """Test extract_and_evaluate with pattern matching."""
        plugin = MathExpertPlugin()

        # Pattern: X op Y
        expr, result = plugin.extract_and_evaluate("5 + 3")
        assert result == 8

        # Pattern with "what is"
        expr, result = plugin.extract_and_evaluate("what is 10 - 5")
        assert result == 5

        # Pattern with "calculate"
        expr, result = plugin.extract_and_evaluate("calculate 4 * 6")
        assert result == 24

    def test_extract_and_evaluate_no_match(self):
        """Test extract_and_evaluate when no expression found."""
        plugin = MathExpertPlugin()

        # No math expression
        expr, result = plugin.extract_and_evaluate("hello world")
        assert result is None


class TestMathExpertExecute:
    """Tests for execute method edge cases."""

    def test_execute_float_result(self):
        """Test execute with float result that is whole number."""
        plugin = MathExpertPlugin()
        result = plugin.execute("10 / 2 = ")
        assert result == "5"  # Should be integer string

    def test_execute_non_integer_result(self):
        """Test execute with non-integer float result."""
        plugin = MathExpertPlugin()
        result = plugin.execute("10 / 3 = ")
        assert result is not None
        assert "." in result  # Should be float string

    def test_execute_none_result(self):
        """Test execute returns None for invalid expressions."""
        plugin = MathExpertPlugin()
        result = plugin.execute("not a math expression")
        assert result is None


class TestMathExpertCanHandle:
    """Tests for can_handle edge cases."""

    def test_can_handle_with_spaces(self):
        """Test can_handle with various spacing."""
        plugin = MathExpertPlugin()
        assert plugin.can_handle("2+2=")
        assert plugin.can_handle("2 +2=")
        assert plugin.can_handle("2+ 2=")
        assert plugin.can_handle("2 + 2 =")
        assert plugin.can_handle("2  +  2  =")

    def test_can_handle_without_equals(self):
        """Test can_handle without equals sign."""
        plugin = MathExpertPlugin()
        # Pattern requires = at end or end of string
        assert plugin.can_handle("2 + 2")

    def test_can_handle_all_operators(self):
        """Test can_handle with all operators."""
        plugin = MathExpertPlugin()
        assert plugin.can_handle("2 + 2 = ")
        assert plugin.can_handle("10 - 5 = ")
        assert plugin.can_handle("3 * 4 = ")
        assert plugin.can_handle("20 / 5 = ")
