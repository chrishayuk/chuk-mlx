"""
Invocation format detection and classification.

Provides tools for:
- Detecting invocation patterns in text
- Classifying output types (numeric, boolean, text)
- Extracting pattern signatures for comparison
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    import mlx.nn as nn


class OutputType(Enum):
    """Types of output a circuit can produce."""

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    TEXT = "text"
    ERROR = "error"


class OperationType(Enum):
    """Types of operations."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CMP_GT = "cmp_gt"
    CMP_LT = "cmp_lt"
    CMP_EQ = "cmp_eq"
    UNKNOWN = "unknown"


@dataclass
class InvocationFormat:
    """A detected invocation format pattern."""

    raw_format: str
    pattern: str  # Abstract pattern like "NUM + NUM ="
    operation: OperationType
    operands: list[int | float]
    invoke_token: str  # The token that triggers invocation (=, is, etc.)
    is_complete: bool  # Has all required components


@dataclass
class InvocationResult:
    """Result of probing an invocation format."""

    format: InvocationFormat
    output: str
    output_type: OutputType
    confidence: float
    invokes_circuit: bool
    top_predictions: list[tuple[str, float]]


class InvocationDetector:
    """
    Detects and analyzes invocation formats.

    Usage:
        detector = InvocationDetector(model, tokenizer)
        result = detector.probe_format("5 + 3 =")
        print(result.invokes_circuit)  # True
        print(result.output_type)  # OutputType.NUMERIC
    """

    # Known invocation suffixes
    INVOKE_SUFFIXES = ["=", "is", "?", ":"]

    # Pattern regexes
    ARITHMETIC_PATTERN = re.compile(
        r"(-?\d+\.?\d*)\s*([+\-*/×÷])\s*(-?\d+\.?\d*)\s*([=]?)"
    )
    COMPARISON_PATTERN = re.compile(
        r"(-?\d+\.?\d*)\s*([<>=!]+)\s*(-?\d+\.?\d*)\s*([=]|is|[?])?",
        re.IGNORECASE,
    )
    FUNCTIONAL_PATTERN = re.compile(
        r"(add|sub|mul|div|sum|compare)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*([=]?)",
        re.IGNORECASE,
    )

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def parse_format(self, text: str) -> InvocationFormat:
        """
        Parse a text string into an InvocationFormat.

        Identifies:
        - Operation type
        - Operands
        - Invoke token
        - Pattern signature
        """
        text = text.strip()

        # Try arithmetic pattern
        match = self.ARITHMETIC_PATTERN.match(text)
        if match:
            a, op, b, suffix = match.groups()
            operation = self._op_to_type(op)
            return InvocationFormat(
                raw_format=text,
                pattern=f"NUM {op} NUM {suffix}".strip(),
                operation=operation,
                operands=[float(a), float(b)],
                invoke_token=suffix or "",
                is_complete=bool(suffix),
            )

        # Try comparison pattern
        match = self.COMPARISON_PATTERN.match(text)
        if match:
            a, cmp, b, suffix = match.groups()
            operation = self._cmp_to_type(cmp)
            return InvocationFormat(
                raw_format=text,
                pattern=f"NUM {cmp} NUM {suffix or ''}".strip(),
                operation=operation,
                operands=[float(a), float(b)],
                invoke_token=suffix or "",
                is_complete=bool(suffix),
            )

        # Try functional pattern
        match = self.FUNCTIONAL_PATTERN.match(text)
        if match:
            func, a, b, suffix = match.groups()
            operation = self._func_to_type(func)
            return InvocationFormat(
                raw_format=text,
                pattern=f"{func}(NUM, NUM) {suffix}".strip(),
                operation=operation,
                operands=[float(a), float(b)],
                invoke_token=suffix or "",
                is_complete=bool(suffix),
            )

        # Unknown format
        return InvocationFormat(
            raw_format=text,
            pattern="UNKNOWN",
            operation=OperationType.UNKNOWN,
            operands=[],
            invoke_token="",
            is_complete=False,
        )

    def probe_format(self, format_string: str) -> InvocationResult:
        """
        Probe whether a format invokes a circuit.

        Args:
            format_string: The format to test (e.g., "5 + 3 =")

        Returns:
            InvocationResult with output type, confidence, etc.
        """
        # Parse the format
        fmt = self.parse_format(format_string)

        # Get model prediction
        tokens = self.tokenizer.encode(format_string)
        input_ids = mx.array([tokens])

        output = self.model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        probs = mx.softmax(logits[0, -1, :])
        mx.eval(probs)

        # Get top predictions
        top_k = 5
        top_indices = mx.argsort(probs)[-top_k:][::-1].tolist()
        top_predictions = [
            (self.tokenizer.decode([i]).strip() or f"[{i}]", float(probs[i]))
            for i in top_indices
        ]

        # Classify output
        top_token = top_predictions[0][0]
        confidence = top_predictions[0][1]
        output_type = self._classify_output(top_token)

        # Determine if it invokes a circuit
        invokes_circuit = output_type in [OutputType.NUMERIC, OutputType.BOOLEAN]

        return InvocationResult(
            format=fmt,
            output=top_token,
            output_type=output_type,
            confidence=confidence,
            invokes_circuit=invokes_circuit,
            top_predictions=top_predictions,
        )

    def _op_to_type(self, op: str) -> OperationType:
        """Convert operator string to OperationType."""
        mapping = {
            "+": OperationType.ADD,
            "-": OperationType.SUB,
            "*": OperationType.MUL,
            "×": OperationType.MUL,
            "/": OperationType.DIV,
            "÷": OperationType.DIV,
        }
        return mapping.get(op, OperationType.UNKNOWN)

    def _cmp_to_type(self, cmp: str) -> OperationType:
        """Convert comparison operator to OperationType."""
        if ">" in cmp:
            return OperationType.CMP_GT
        elif "<" in cmp:
            return OperationType.CMP_LT
        elif "=" in cmp:
            return OperationType.CMP_EQ
        return OperationType.UNKNOWN

    def _func_to_type(self, func: str) -> OperationType:
        """Convert function name to OperationType."""
        mapping = {
            "add": OperationType.ADD,
            "sub": OperationType.SUB,
            "mul": OperationType.MUL,
            "div": OperationType.DIV,
            "sum": OperationType.ADD,
            "compare": OperationType.CMP_GT,
        }
        return mapping.get(func.lower(), OperationType.UNKNOWN)

    def _classify_output(self, token: str) -> OutputType:
        """Classify output token type."""
        token = token.strip()

        # Numeric
        if re.match(r"^-?\d+\.?\d*$", token):
            return OutputType.NUMERIC

        # Boolean
        if token.lower() in ["true", "false", "0", "1", "yes", "no"]:
            return OutputType.BOOLEAN

        # Error indicators
        if token.lower() in ["nan", "inf", "error", "undefined"]:
            return OutputType.ERROR

        return OutputType.TEXT

    def build_vocabulary(
        self, format_tests: dict[str, list[str]]
    ) -> dict[str, list[InvocationResult]]:
        """
        Build an invocation vocabulary by testing multiple formats.

        Args:
            format_tests: Dict mapping operation name to list of format strings

        Returns:
            Dict mapping operation name to list of InvocationResults
        """
        vocabulary = {}

        for operation, formats in format_tests.items():
            results = []
            for fmt in formats:
                result = self.probe_format(fmt)
                results.append(result)
            vocabulary[operation] = results

        return vocabulary

    def extract_working_formats(
        self, vocabulary: dict[str, list[InvocationResult]]
    ) -> dict[str, list[str]]:
        """
        Extract only the formats that successfully invoke circuits.
        """
        working = {}

        for operation, results in vocabulary.items():
            working_formats = [
                r.format.raw_format
                for r in results
                if r.invokes_circuit
            ]
            if working_formats:
                working[operation] = working_formats

        return working
