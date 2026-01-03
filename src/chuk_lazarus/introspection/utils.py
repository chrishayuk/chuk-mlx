"""Shared utilities for introspection operations.

This module contains reusable functions that support CLI commands
and programmatic introspection workflows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    """Apply chat template to a prompt if available.

    Args:
        tokenizer: The tokenizer with optional chat_template
        prompt: The user prompt
        add_generation_prompt: Whether to add generation prompt marker

    Returns:
        Formatted prompt (original if no template available)
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass
    return prompt


def load_external_chat_template(tokenizer: Any, model_path: str) -> None:
    """Load external chat template from model directory if available.

    Some models (like GPT-OSS) store the chat template in a separate
    chat_template.jinja file rather than in tokenizer_config.json.

    Args:
        tokenizer: The tokenizer to update
        model_path: Path or HuggingFace model ID
    """
    from huggingface_hub import snapshot_download

    try:
        local_path = Path(snapshot_download(model_path, allow_patterns=["chat_template.jinja"]))
    except Exception:
        local_path = Path(model_path)

    chat_template_path = local_path / "chat_template.jinja"
    if chat_template_path.exists() and not tokenizer.chat_template:
        try:
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
        except Exception:
            pass


def extract_expected_answer(prompt: str) -> str | None:
    """Try to compute expected answer from arithmetic prompt.

    Args:
        prompt: An arithmetic prompt like "100 - 37 = " or "7 * 8 = "

    Returns:
        The computed answer as a string, or None if not parseable
    """
    from .enums import ArithmeticOperator

    match = re.match(r"(\d+)\s*([+\-*/x×÷])\s*(\d+)\s*=\s*$", prompt.strip())
    if not match:
        return None

    a, op_str, b = int(match.group(1)), match.group(2), int(match.group(3))
    try:
        op = ArithmeticOperator.from_string(op_str)
        result = op.compute(a, b)
        return str(int(result))
    except (ValueError, ZeroDivisionError):
        return None


def find_answer_onset(
    output: str,
    expected_answer: str | None,
    tokenizer: Any,
) -> dict[str, Any]:
    """Find where the answer first appears in the output.

    Args:
        output: The generated output string
        expected_answer: The expected answer string
        tokenizer: Tokenizer for token-level analysis

    Returns:
        Dict with onset_index, onset_token, is_answer_first, answer_found
    """
    if expected_answer is None:
        return {
            "onset_index": None,
            "onset_token": None,
            "is_answer_first": None,
            "answer_found": False,
        }

    # Normalize expected answer
    expected_normalized = normalize_number_string(expected_answer)

    # Tokenize output
    tokens = []
    output_ids = tokenizer.encode(output)
    for tid in output_ids:
        tokens.append(tokenizer.decode([tid]))

    # Find first position where expected answer appears
    cumulative = ""
    for i, tok in enumerate(tokens):
        cumulative += tok
        if expected_normalized in normalize_number_string(cumulative):
            return {
                "onset_index": i,
                "onset_token": tok,
                "is_answer_first": i <= 1,
                "answer_found": True,
            }

    return {
        "onset_index": None,
        "onset_token": None,
        "is_answer_first": False,
        "answer_found": False,
    }


def generate_arithmetic_prompts(
    operation: str = "*",
    digit_range: tuple[int, int] = (2, 9),
    difficulty: str | None = None,
    include_answer: bool = False,
) -> list[dict[str, Any]]:
    """Generate arithmetic test prompts.

    Args:
        operation: The operation to use (*, +, -, /)
        digit_range: (min, max) range for operands (inclusive)
        difficulty: Filter by difficulty (easy, medium, hard) or None for all
        include_answer: Whether to include the answer in the prompt

    Returns:
        List of dicts with prompt, operand_a, operand_b, result, difficulty
    """
    min_digit, max_digit = digit_range
    prompts = []

    for a in range(min_digit, max_digit + 1):
        for b in range(min_digit, max_digit + 1):
            # Calculate result
            if operation in ["*", "x", "×"]:
                result = a * b
            elif operation == "+":
                result = a + b
            elif operation == "-":
                result = a - b
            elif operation == "/":
                # Skip non-integer divisions
                if b == 0 or a % b != 0:
                    continue
                result = a // b
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Determine difficulty
            if operation in ["*", "x", "×"]:
                if a <= 3 or b <= 3:
                    diff = "easy"
                elif a >= 7 and b >= 7:
                    diff = "hard"
                else:
                    diff = "medium"
            else:
                # For addition/subtraction
                if result <= 10:
                    diff = "easy"
                elif result >= 100:
                    diff = "hard"
                else:
                    diff = "medium"

            # Filter by difficulty if specified
            if difficulty and diff != difficulty:
                continue

            # Build prompt
            if include_answer:
                prompt = f"{a}{operation}{b}={result}"
            else:
                prompt = f"{a}{operation}{b}="

            prompts.append(
                {
                    "prompt": prompt,
                    "operand_a": a,
                    "operand_b": b,
                    "result": result,
                    "difficulty": diff,
                }
            )

    return prompts


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return float(dot / (norm1 * norm2 + 1e-8))


def compute_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    n = len(vectors)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity[i, j] = cosine_similarity(vectors[i], vectors[j])
    return similarity


def analyze_orthogonality(
    vectors: list[np.ndarray],
    names: list[str] | None = None,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Analyze orthogonality between a set of direction vectors.

    Args:
        vectors: List of direction vectors
        names: Optional names for each vector
        threshold: Threshold below which vectors are considered orthogonal

    Returns:
        Dict with similarity matrix, orthogonal pairs, aligned pairs, and summary
    """
    n = len(vectors)
    if names is None:
        names = [f"v{i}" for i in range(n)]

    similarity = compute_similarity_matrix(vectors)

    # Find orthogonal and aligned pairs
    orthogonal_pairs = []
    aligned_pairs = []
    off_diag = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity[i, j]
            off_diag.append((names[i], names[j], sim))

            if abs(sim) < threshold:
                orthogonal_pairs.append((names[i], names[j], sim))
            elif abs(sim) > 0.5:
                aligned_pairs.append((names[i], names[j], sim))

    mean_abs_sim = np.mean([abs(s) for _, _, s in off_diag]) if off_diag else 0.0

    return {
        "similarity_matrix": similarity,
        "names": names,
        "orthogonal_pairs": orthogonal_pairs,
        "aligned_pairs": aligned_pairs,
        "mean_abs_similarity": mean_abs_sim,
        "threshold": threshold,
    }


def find_discriminative_neurons(
    activations: np.ndarray,
    labels: list[str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find neurons that best discriminate between label groups.

    Args:
        activations: Shape (num_prompts, hidden_size) - activation vectors
        labels: Labels for each prompt
        top_k: Number of top neurons to return

    Returns:
        List of dicts with neuron idx, separation score, and group means
    """
    unique_labels = sorted(set(labels))
    num_neurons = activations.shape[1]

    # Group activations by label
    label_groups = {lbl: [] for lbl in unique_labels}
    for i, lbl in enumerate(labels):
        label_groups[lbl].append(activations[i])

    for lbl in unique_labels:
        label_groups[lbl] = np.array(label_groups[lbl])

    # Check if single sample per group
    single_sample_mode = all(len(label_groups[lbl]) == 1 for lbl in unique_labels)

    # Score each neuron
    neuron_scores = []
    for neuron_idx in range(num_neurons):
        group_means = []
        group_stds = []
        for lbl in unique_labels:
            vals = label_groups[lbl][:, neuron_idx]
            group_means.append(np.mean(vals))
            group_stds.append(np.std(vals))

        # Overall std across all prompts
        overall_std = np.std(activations[:, neuron_idx])

        # Max pairwise separation (Cohen's d style)
        max_separation = 0.0
        best_pair = None
        for i, lbl1 in enumerate(unique_labels):
            for j, lbl2 in enumerate(unique_labels):
                if i >= j:
                    continue
                mean_diff = abs(group_means[i] - group_means[j])

                if single_sample_mode:
                    if overall_std > 1e-6:
                        separation = mean_diff / overall_std
                    else:
                        separation = 0.0
                else:
                    pooled_std = np.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                    if pooled_std > 1e-6:
                        separation = mean_diff / pooled_std
                    else:
                        separation = 0.0

                if separation > max_separation:
                    max_separation = separation
                    best_pair = (lbl1, lbl2)

        mean_range = max(group_means) - min(group_means)

        neuron_scores.append(
            {
                "idx": neuron_idx,
                "separation": max_separation,
                "best_pair": best_pair,
                "overall_std": overall_std,
                "mean_range": mean_range,
                "group_means": {lbl: group_means[i] for i, lbl in enumerate(unique_labels)},
            }
        )

    # Sort by separation score and take top-k
    neuron_scores.sort(key=lambda x: -x["separation"])
    return neuron_scores[:top_k]


def normalize_number_string(s: str) -> str:
    """Normalize a number string by removing formatting characters.

    Removes commas, thin spaces, regular spaces, and other separators.
    """
    import re

    return re.sub(r"[\s,\u202f\u00a0]+", "", s)


def parse_prompts_from_arg(prompts_arg: str) -> list[str]:
    """Parse prompts from argument string or file.

    Args:
        prompts_arg: Either a pipe-separated string or @filename

    Returns:
        List of prompt strings
    """
    if prompts_arg.startswith("@"):
        with open(prompts_arg[1:]) as f:
            return [line.strip() for line in f if line.strip()]
    return [p.strip() for p in prompts_arg.split("|")]


def parse_layers_arg(layers_str: str | None, num_layers: int | None = None) -> list[int] | None:
    """Parse comma-separated layer list with support for ranges.

    Examples:
        "0,1,2" -> [0, 1, 2]
        "0-5" -> [0, 1, 2, 3, 4, 5]
        "0-5,10,15-20" -> [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19, 20]
    """
    if not layers_str:
        return None

    layers = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers
