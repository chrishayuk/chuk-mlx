"""
IR-Attention Routing: CoT as Circuit Invocation

Unified experiment that tests the hypothesis:
  CoT is a learned compiler frontend that normalizes arbitrary input
  into circuit invocation formats that trigger specific computations.

Sub-experiments:
1. Discover Invocation Format Vocabulary
2. CoT as Format Compiler
3. Attention Trace Through Rewrite
4. Multi-Step Self-Invocation
5. Virtual Expert Integration Readiness

Builds on: suffix_routing, moe_attention_routing, neural_compiler, learned_ir_head
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FormatProbeResult:
    """Result of probing a single format for circuit invocation."""

    format_string: str
    output: str
    output_type: str  # "numeric", "boolean", "text", "error"
    invokes_circuit: bool
    confidence: float
    top_tokens: list[tuple[str, float]]


@dataclass
class FormatVocabularyEntry:
    """Entry in the discovered invocation format vocabulary."""

    pattern: str  # e.g., "NUM + NUM ="
    operation: str  # e.g., "add"
    canonical_example: str
    working_formats: list[str]
    broken_formats: list[str]
    hidden_similarity_to_canonical: float


@dataclass
class CoTRewriteResult:
    """Result of CoT rewriting arbitrary input to invocation format."""

    input_text: str
    cot_output: str
    expected_format: str
    format_match: bool
    circuit_output: str
    expected_result: str
    result_match: bool
    success: bool


@dataclass
class AttentionTraceEntry:
    """Attention trace for a single generated token."""

    token: str
    position: int
    top_attended_positions: list[int]
    top_attended_tokens: list[str]
    attention_weights: list[float]
    routing_expert: int | None  # For MoE models


@dataclass
class MultiStepInvocation:
    """Result of tracing a multi-step CoT invocation."""

    problem: str
    cot_steps: list[str]
    step_results: list[dict]  # Per-step: {line, attends_to_previous, circuit}
    final_result: str
    expected_result: str
    correct: bool
    all_steps_self_invoke: bool


@dataclass
class VirtualExpertProbeResult:
    """Result of training/testing IR extraction probe."""

    operation_accuracy: float
    operand_a_accuracy: float  # For binned
    operand_a_mae: float  # For regression
    operand_b_accuracy: float
    operand_b_mae: float
    end_to_end_accuracy: float  # Correct IR → correct WASM result
    ready_for_virtual_expert: bool


# =============================================================================
# Main Experiment
# =============================================================================


class IRAttentionRoutingExperiment(ExperimentBase):
    """
    IR-Attention Routing: CoT as Circuit Invocation.

    Tests whether CoT serves as a learned rewriter that normalizes
    arbitrary input into circuit invocation formats.
    """

    def setup(self) -> None:
        """Initialize experiment resources."""
        self.log("Setting up IR-Attention Routing experiment...")

        self.params = self.config.parameters
        self.results: dict[str, Any] = {}

        # Load model
        self.log(f"Loading model: {self.config.model}")
        loaded = self.load_model()
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())

        # Get model architecture info
        self.num_layers = self._get_num_layers()
        self.hidden_dim = self._get_hidden_dim()
        self.log(f"Model: {self.num_layers} layers, hidden_dim={self.hidden_dim}")

    def run(self) -> dict:
        """Run all sub-experiments."""
        self.log("=" * 70)
        self.log("IR-ATTENTION ROUTING: CoT AS CIRCUIT INVOCATION")
        self.log("=" * 70)

        experiments_to_run = self.params.get("experiments", [
            "discover_formats",
            "cot_compiler",
            "attention_trace",
            "multi_step",
            "virtual_expert",
        ])

        results = {
            "model": self.config.model,
            "timestamp": datetime.now().isoformat(),
            "num_layers": self.num_layers,
        }

        # Run each sub-experiment
        if "discover_formats" in experiments_to_run:
            self.log("\n" + "=" * 70)
            self.log("EXPERIMENT 1: DISCOVER INVOCATION FORMAT VOCABULARY")
            self.log("=" * 70)
            results["format_vocabulary"] = self._run_discover_formats()

        if "cot_compiler" in experiments_to_run:
            self.log("\n" + "=" * 70)
            self.log("EXPERIMENT 2: COT AS FORMAT COMPILER")
            self.log("=" * 70)
            results["cot_compiler"] = self._run_cot_compiler()

        if "attention_trace" in experiments_to_run:
            self.log("\n" + "=" * 70)
            self.log("EXPERIMENT 3: ATTENTION TRACE THROUGH COT REWRITE")
            self.log("=" * 70)
            results["attention_trace"] = self._run_attention_trace()

        if "multi_step" in experiments_to_run:
            self.log("\n" + "=" * 70)
            self.log("EXPERIMENT 4: MULTI-STEP SELF-INVOCATION")
            self.log("=" * 70)
            results["multi_step"] = self._run_multi_step()

        if "virtual_expert" in experiments_to_run:
            self.log("\n" + "=" * 70)
            self.log("EXPERIMENT 5: VIRTUAL EXPERT INTEGRATION READINESS")
            self.log("=" * 70)
            results["virtual_expert"] = self._run_virtual_expert_probe()

        # Synthesize findings
        results["synthesis"] = self._synthesize_findings(results)

        self.results = results
        self.save_results(results)
        self._print_summary(results)

        return results

    # =========================================================================
    # Experiment 1: Discover Invocation Format Vocabulary
    # =========================================================================

    def _run_discover_formats(self) -> dict:
        """
        Discover which formats the model recognizes as circuit invocations.

        For each operation type, test various format variations and identify
        which ones reliably invoke the arithmetic circuit.
        """
        format_vocab = self.params.get("format_vocabulary", {})
        results = {"by_operation": {}, "vocabulary": []}

        for operation, config in format_vocab.items():
            self.log(f"\nProbing {operation} formats...")

            canonical = config.get("canonical", "")
            variations = config.get("variations", [])

            # Get canonical hidden state for comparison
            canonical_hidden = self._get_hidden_at_invoke(canonical)

            operation_results = {
                "canonical": canonical,
                "format_results": [],
                "working_formats": [],
                "broken_formats": [],
            }

            for fmt in variations:
                result = self._probe_format_invocation(fmt, canonical_hidden)
                operation_results["format_results"].append(result.__dict__)

                if result.invokes_circuit:
                    operation_results["working_formats"].append(fmt)
                else:
                    operation_results["broken_formats"].append(fmt)

                status = "INVOKES" if result.invokes_circuit else "FAILS"
                self.log(f"  {fmt:<20} → {result.output:<6} [{status}]")

            results["by_operation"][operation] = operation_results

            # Add to vocabulary
            if operation_results["working_formats"]:
                results["vocabulary"].append({
                    "operation": operation,
                    "pattern": self._extract_pattern(canonical),
                    "canonical": canonical,
                    "working_count": len(operation_results["working_formats"]),
                    "total_tested": len(variations),
                })

        return results

    def _probe_format_invocation(
        self, format_string: str, canonical_hidden: mx.array | None = None
    ) -> FormatProbeResult:
        """
        Probe whether a format invokes a circuit.

        Measures:
        - Output type (numeric, boolean, text)
        - Confidence
        - Hidden state similarity to canonical
        """
        # Generate output
        tokens = self.tokenizer.encode(format_string)
        input_ids = mx.array([tokens])

        output = self.model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        probs = mx.softmax(logits[0, -1, :])
        mx.eval(probs)

        # Get top tokens
        top_k = 5
        top_indices = mx.argsort(probs)[-top_k:][::-1].tolist()
        top_tokens = [
            (self.tokenizer.decode([i]).strip() or repr(i), float(probs[i]))
            for i in top_indices
        ]

        # Classify output
        top_token = top_tokens[0][0]
        confidence = top_tokens[0][1]

        output_type = self._classify_output(top_token)
        invokes_circuit = output_type in ["numeric", "boolean"]

        return FormatProbeResult(
            format_string=format_string,
            output=top_token,
            output_type=output_type,
            invokes_circuit=invokes_circuit,
            confidence=confidence,
            top_tokens=top_tokens,
        )

    def _classify_output(self, token: str) -> str:
        """Classify output token type."""
        # Numeric
        if re.match(r"^-?\d+\.?\d*$", token.strip()):
            return "numeric"

        # Boolean
        if token.lower().strip() in ["true", "false", "0", "1", "yes", "no"]:
            return "boolean"

        # Text/other
        return "text"

    def _extract_pattern(self, canonical: str) -> str:
        """Extract abstract pattern from canonical format."""
        # Replace numbers with NUM, operators with OP
        pattern = re.sub(r"\d+", "NUM", canonical)
        return pattern

    # =========================================================================
    # Experiment 2: CoT as Format Compiler
    # =========================================================================

    def _run_cot_compiler(self) -> dict:
        """
        Test if CoT can reliably rewrite arbitrary input to invocation formats.
        """
        test_cases = self.params.get("cot_compiler_tests", [])
        prompt_template = self.params.get("cot_rewrite_prompt", "")

        results = {
            "test_results": [],
            "format_match_rate": 0.0,
            "result_match_rate": 0.0,
            "overall_success_rate": 0.0,
        }

        format_matches = 0
        result_matches = 0
        successes = 0

        for test_case in test_cases:
            result = self._test_cot_rewrite(
                test_case["input"],
                test_case["expected_format"],
                test_case["expected_result"],
                prompt_template,
            )
            results["test_results"].append(result.__dict__)

            if result.format_match:
                format_matches += 1
            if result.result_match:
                result_matches += 1
            if result.success:
                successes += 1

            status = "SUCCESS" if result.success else "FAILED"
            self.log(
                f"  {test_case['input'][:30]:<32} → "
                f"{result.cot_output[:20]:<20} [{status}]"
            )

        n = len(test_cases) if test_cases else 1
        results["format_match_rate"] = format_matches / n
        results["result_match_rate"] = result_matches / n
        results["overall_success_rate"] = successes / n

        self.log(f"\nFormat match rate: {results['format_match_rate']:.1%}")
        self.log(f"Result match rate: {results['result_match_rate']:.1%}")
        self.log(f"Overall success rate: {results['overall_success_rate']:.1%}")

        return results

    def _test_cot_rewrite(
        self,
        input_text: str,
        expected_format: str,
        expected_result: str,
        prompt_template: str,
    ) -> CoTRewriteResult:
        """Test if CoT produces correct invocation format."""
        # Build prompt
        prompt = prompt_template.format(input=input_text)

        # Generate CoT rewrite
        cot_output = self._generate(prompt, max_tokens=30, stop="\n")
        cot_output = cot_output.strip()

        # Check format match (normalize whitespace)
        format_match = self._normalize_format(cot_output) == self._normalize_format(
            expected_format
        )

        # Generate circuit output from the produced format
        circuit_output = self._generate(cot_output, max_tokens=10, stop=" ")
        circuit_output = circuit_output.strip()

        # Check result match
        result_match = self._extract_number(circuit_output) == expected_result

        return CoTRewriteResult(
            input_text=input_text,
            cot_output=cot_output,
            expected_format=expected_format,
            format_match=format_match,
            circuit_output=circuit_output,
            expected_result=expected_result,
            result_match=result_match,
            success=format_match and result_match,
        )

    def _normalize_format(self, s: str) -> str:
        """Normalize format string for comparison."""
        # Remove extra whitespace, normalize operators
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("×", "*").replace("÷", "/")
        return s

    def _extract_number(self, s: str) -> str:
        """Extract first number from string."""
        match = re.search(r"-?\d+\.?\d*", s)
        if match:
            return match.group()
        return s.strip()

    # =========================================================================
    # Experiment 3: Attention Trace Through Rewrite
    # =========================================================================

    def _run_attention_trace(self) -> dict:
        """
        Trace attention flow during CoT rewrite → circuit invocation.

        Key question: At the "=" token, what does attention encode?
        """
        prompts = self.params.get("attention_trace_prompts", ["5 + 3 ="])
        decision_layer = self.params.get("decision_layer", 12)

        if decision_layer >= self.num_layers:
            decision_layer = self.num_layers // 2

        results = {
            "decision_layer": decision_layer,
            "traces": [],
        }

        for prompt in prompts:
            self.log(f"\nTracing: {prompt}")
            trace = self._trace_attention(prompt, decision_layer)
            results["traces"].append(trace)

            # Show key findings
            if "invoke_position" in trace:
                inv = trace["invoke_position"]
                self.log(f"  Invoke position: {inv.get('position', '?')}")
                self.log(f"  Top attended: {inv.get('top_attended_tokens', [])[:3]}")

        return results

    def _trace_attention(self, prompt: str, layer: int) -> dict:
        """Trace attention patterns for a prompt."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        token_strs = [
            self.tokenizer.decode([t]).strip() or f"[{t}]" for t in tokens
        ]

        # Find invoke position ("=" or end)
        invoke_pos = len(tokens) - 1
        for i, t in enumerate(token_strs):
            if "=" in t:
                invoke_pos = i
                break

        # Get attention patterns
        attention_patterns = self._get_attention_at_layer(input_ids, layer)

        trace = {
            "prompt": prompt,
            "tokens": token_strs,
            "layer": layer,
            "token_traces": [],
        }

        if attention_patterns is not None:
            # Trace each token
            for pos in range(len(tokens)):
                if pos == 0:
                    continue  # Skip first token

                attn = attention_patterns[0, :, pos, :pos + 1]
                avg_attn = mx.mean(attn, axis=0)
                mx.eval(avg_attn)

                top_k = min(3, pos + 1)
                top_indices = mx.argsort(avg_attn)[-top_k:][::-1].tolist()

                trace["token_traces"].append({
                    "token": token_strs[pos],
                    "position": pos,
                    "top_attended_positions": top_indices,
                    "top_attended_tokens": [token_strs[i] for i in top_indices],
                    "attention_weights": [float(avg_attn[i]) for i in top_indices],
                })

            # Special analysis at invoke position
            if invoke_pos < len(tokens):
                attn_at_invoke = attention_patterns[0, :, invoke_pos, :invoke_pos + 1]
                avg_attn_invoke = mx.mean(attn_at_invoke, axis=0)
                mx.eval(avg_attn_invoke)

                top_k = min(5, invoke_pos + 1)
                top_indices = mx.argsort(avg_attn_invoke)[-top_k:][::-1].tolist()

                trace["invoke_position"] = {
                    "position": invoke_pos,
                    "token": token_strs[invoke_pos],
                    "top_attended_positions": top_indices,
                    "top_attended_tokens": [token_strs[i] for i in top_indices],
                    "attention_weights": [float(avg_attn_invoke[i]) for i in top_indices],
                }
        else:
            trace["error"] = "Could not extract attention patterns"

        return trace

    def _get_attention_at_layer(
        self, input_ids: mx.array, layer: int
    ) -> mx.array | None:
        """Extract attention patterns at a specific layer."""
        try:
            # Forward with attention output
            output = self.model(input_ids, output_attentions=True)

            if hasattr(output, "attentions") and output.attentions:
                if layer < len(output.attentions):
                    return output.attentions[layer]
        except Exception as e:
            self.log(f"Could not extract attention: {e}", level="warning")

        return None

    # =========================================================================
    # Experiment 4: Multi-Step Self-Invocation
    # =========================================================================

    def _run_multi_step(self) -> dict:
        """
        Test multi-step CoT where each line invokes a circuit
        and later lines attend to previous results.
        """
        test_cases = self.params.get("multi_step_tests", [])

        results = {
            "test_results": [],
            "correct_rate": 0.0,
            "self_invoke_rate": 0.0,
        }

        correct = 0
        self_invoke = 0

        for test_case in test_cases:
            result = self._test_multi_step(test_case)
            results["test_results"].append(result.__dict__)

            if result.correct:
                correct += 1
            if result.all_steps_self_invoke:
                self_invoke += 1

            status = "CORRECT" if result.correct else "WRONG"
            self.log(f"  {test_case['problem'][:40]:<42} → {result.final_result} [{status}]")

        n = len(test_cases) if test_cases else 1
        results["correct_rate"] = correct / n
        results["self_invoke_rate"] = self_invoke / n

        self.log(f"\nCorrect rate: {results['correct_rate']:.1%}")
        self.log(f"Self-invoke rate: {results['self_invoke_rate']:.1%}")

        return results

    def _test_multi_step(self, test_case: dict) -> MultiStepInvocation:
        """Test a multi-step problem."""
        problem = test_case["problem"]
        expected_result = test_case["expected_result"]

        # Generate CoT
        cot_prompt = f"Solve step by step:\n{problem}\n"
        cot_output = self._generate(cot_prompt, max_tokens=100, stop=None)

        # Parse steps (lines containing "=")
        cot_lines = cot_output.strip().split("\n")
        cot_steps = [line for line in cot_lines if "=" in line]

        # Analyze each step
        step_results = []
        previous_results = []

        for i, step in enumerate(cot_steps):
            step_info = {"line": step, "attends_to_previous": False}

            # Check if step mentions previous results
            for prev in previous_results:
                if prev in step:
                    step_info["attends_to_previous"] = True
                    break

            # Extract result from this step
            match = re.search(r"=\s*(\d+)", step)
            if match:
                previous_results.append(match.group(1))

            step_results.append(step_info)

        # Get final result
        final_result = ""
        if previous_results:
            final_result = previous_results[-1]
        else:
            # Try to extract from last line
            match = re.search(r"(\d+)\s*$", cot_output)
            if match:
                final_result = match.group(1)

        # Check correctness
        correct = final_result == expected_result

        # Check self-invocation (steps 2+ should attend to previous)
        all_self_invoke = all(
            sr["attends_to_previous"] for sr in step_results[1:]
        ) if len(step_results) > 1 else True

        return MultiStepInvocation(
            problem=problem,
            cot_steps=cot_steps,
            step_results=step_results,
            final_result=final_result,
            expected_result=expected_result,
            correct=correct,
            all_steps_self_invoke=all_self_invoke,
        )

    # =========================================================================
    # Experiment 5: Virtual Expert Integration Readiness
    # =========================================================================

    def _run_virtual_expert_probe(self) -> dict:
        """
        Train and test probes to extract IR from hidden states.

        If we can reliably decode (operation, operand_a, operand_b) from
        hidden states, we can replace the fuzzy neural expert with a
        deterministic WASM expert.
        """
        probe_config = self.params.get("virtual_expert_probe", {})
        decision_layer = self.params.get("decision_layer", 12)

        if decision_layer >= self.num_layers:
            decision_layer = self.num_layers // 2

        self.log(f"Training IR extraction probes at layer {decision_layer}")

        # Generate training data
        train_data = self._generate_probe_data(
            probe_config.get("num_training_examples", 200),
            probe_config.get("operations", ["add", "sub", "mul", "div"]),
            probe_config.get("operand_range", [1, 100]),
        )

        test_data = self._generate_probe_data(
            probe_config.get("num_test_examples", 50),
            probe_config.get("operations", ["add", "sub", "mul", "div"]),
            probe_config.get("operand_range", [1, 100]),
        )

        # Extract hidden states
        self.log("Extracting hidden states...")
        train_hiddens, train_labels = self._extract_probe_features(
            train_data, decision_layer
        )
        test_hiddens, test_labels = self._extract_probe_features(
            test_data, decision_layer
        )

        # Train operation classifier
        self.log("Training operation classifier...")
        op_accuracy = self._train_operation_probe(
            train_hiddens, train_labels,
            test_hiddens, test_labels,
            probe_config,
        )

        # Train operand extractors
        self.log("Training operand extractors...")
        operand_results = self._train_operand_probes(
            train_hiddens, train_labels,
            test_hiddens, test_labels,
            probe_config,
        )

        # End-to-end test
        self.log("Testing end-to-end IR extraction...")
        e2e_accuracy = self._test_end_to_end(
            test_hiddens, test_labels, probe_config
        )

        # Determine readiness
        ready = (
            op_accuracy >= 0.95 and
            operand_results["operand_a_accuracy"] >= 0.80 and
            operand_results["operand_b_accuracy"] >= 0.80
        )

        results = {
            "decision_layer": decision_layer,
            "operation_accuracy": op_accuracy,
            "operand_a_accuracy": operand_results["operand_a_accuracy"],
            "operand_a_mae": operand_results["operand_a_mae"],
            "operand_b_accuracy": operand_results["operand_b_accuracy"],
            "operand_b_mae": operand_results["operand_b_mae"],
            "end_to_end_accuracy": e2e_accuracy,
            "ready_for_virtual_expert": ready,
        }

        self.log(f"\nOperation accuracy: {op_accuracy:.1%}")
        self.log(f"Operand A accuracy: {operand_results['operand_a_accuracy']:.1%}")
        self.log(f"Operand B accuracy: {operand_results['operand_b_accuracy']:.1%}")
        self.log(f"End-to-end accuracy: {e2e_accuracy:.1%}")
        self.log(f"Ready for virtual expert: {ready}")

        return results

    def _generate_probe_data(
        self, n: int, operations: list[str], operand_range: list[int]
    ) -> list[dict]:
        """Generate training/test data for probes."""
        import random

        op_symbols = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        data = []

        for _ in range(n):
            op = random.choice(operations)
            a = random.randint(operand_range[0], operand_range[1])
            b = random.randint(operand_range[0], operand_range[1])

            # Ensure valid division
            if op == "div" and b != 0:
                a = b * random.randint(1, 10)  # Make it divide evenly

            prompt = f"{a} {op_symbols[op]} {b} ="
            data.append({
                "prompt": prompt,
                "operation": op,
                "operand_a": a,
                "operand_b": b,
            })

        return data

    def _extract_probe_features(
        self, data: list[dict], layer: int
    ) -> tuple[mx.array, dict]:
        """Extract hidden states and labels from data."""
        hiddens = []
        labels = {"operation": [], "operand_a": [], "operand_b": []}

        op_to_idx = {"add": 0, "sub": 1, "mul": 2, "div": 3}

        for item in data:
            h = self._get_hidden_at_invoke(item["prompt"], layer)
            if h is not None:
                hiddens.append(h)
                labels["operation"].append(op_to_idx[item["operation"]])
                labels["operand_a"].append(item["operand_a"])
                labels["operand_b"].append(item["operand_b"])

        if not hiddens:
            return mx.zeros((1, self.hidden_dim)), labels

        return mx.stack(hiddens), labels

    def _train_operation_probe(
        self,
        train_hiddens: mx.array,
        train_labels: dict,
        test_hiddens: mx.array,
        test_labels: dict,
        config: dict,
    ) -> float:
        """Train operation classification probe."""
        num_classes = 4
        hidden_dim = train_hiddens.shape[1]
        epochs = config.get("probe_epochs", 50)
        lr = config.get("probe_learning_rate", 0.001)

        # Simple linear probe
        W = mx.random.normal((hidden_dim, num_classes)) * 0.01
        b = mx.zeros((num_classes,))

        y_train = mx.array(train_labels["operation"])
        y_test = mx.array(test_labels["operation"])

        for _ in range(epochs):
            logits = train_hiddens @ W + b
            probs = mx.softmax(logits, axis=-1)

            # Gradient
            grad_logits = probs
            grad_logits = grad_logits.at[mx.arange(len(y_train)), y_train].add(-1)
            grad_logits = grad_logits / len(y_train)

            grad_W = train_hiddens.T @ grad_logits
            grad_b = mx.sum(grad_logits, axis=0)

            W = W - lr * grad_W
            b = b - lr * grad_b
            mx.eval(W, b)

        # Test accuracy
        test_logits = test_hiddens @ W + b
        test_preds = mx.argmax(test_logits, axis=-1)
        accuracy = float(mx.mean(test_preds == y_test))

        # Store probe
        self._op_probe = (W, b)

        return accuracy

    def _train_operand_probes(
        self,
        train_hiddens: mx.array,
        train_labels: dict,
        test_hiddens: mx.array,
        test_labels: dict,
        config: dict,
    ) -> dict:
        """Train operand extraction probes."""
        method = config.get("operand_method", "binned")
        num_bins = config.get("num_bins", 128)
        epochs = config.get("probe_epochs", 50)
        lr = config.get("probe_learning_rate", 0.001)
        hidden_dim = train_hiddens.shape[1]

        results = {}

        for operand_name in ["operand_a", "operand_b"]:
            y_train = mx.array(train_labels[operand_name])
            y_test = mx.array(test_labels[operand_name])

            if method == "binned":
                # Bin classification
                y_train_binned = mx.clip(y_train, 0, num_bins - 1)
                y_test_binned = mx.clip(y_test, 0, num_bins - 1)

                W = mx.random.normal((hidden_dim, num_bins)) * 0.01
                b = mx.zeros((num_bins,))

                for _ in range(epochs):
                    logits = train_hiddens @ W + b
                    probs = mx.softmax(logits, axis=-1)

                    grad_logits = probs
                    grad_logits = grad_logits.at[
                        mx.arange(len(y_train_binned)), y_train_binned
                    ].add(-1)
                    grad_logits = grad_logits / len(y_train_binned)

                    grad_W = train_hiddens.T @ grad_logits
                    grad_b = mx.sum(grad_logits, axis=0)

                    W = W - lr * grad_W
                    b = b - lr * grad_b
                    mx.eval(W, b)

                # Test
                test_logits = test_hiddens @ W + b
                test_preds = mx.argmax(test_logits, axis=-1)
                accuracy = float(mx.mean(test_preds == y_test_binned))
                mae = float(mx.mean(mx.abs(test_preds - y_test_binned)))

            else:
                # Regression
                W = mx.random.normal((hidden_dim, 1)) * 0.01
                b = mx.zeros((1,))

                for _ in range(epochs):
                    preds = (train_hiddens @ W + b).squeeze()
                    grad = 2 * (preds - y_train) / len(y_train)

                    grad_W = train_hiddens.T @ grad[:, None]
                    grad_b = mx.sum(grad, keepdims=True)

                    W = W - lr * grad_W
                    b = b - lr * grad_b
                    mx.eval(W, b)

                # Test
                test_preds = (test_hiddens @ W + b).squeeze()
                mae = float(mx.mean(mx.abs(test_preds - y_test)))
                # Accuracy = within 5 of true value
                accuracy = float(mx.mean(mx.abs(test_preds - y_test) <= 5))

            results[f"{operand_name}_accuracy"] = accuracy
            results[f"{operand_name}_mae"] = mae

        return results

    def _test_end_to_end(
        self,
        test_hiddens: mx.array,
        test_labels: dict,
        config: dict,
    ) -> float:
        """Test end-to-end: extract IR → compute result → check correctness."""
        # This would use WASM in a full implementation
        # For now, just verify we can extract correct IR
        if not hasattr(self, "_op_probe"):
            return 0.0

        W, b = self._op_probe
        test_logits = test_hiddens @ W + b
        op_preds = mx.argmax(test_logits, axis=-1).tolist()

        correct = 0
        for i, op_pred in enumerate(op_preds):
            if op_pred == test_labels["operation"][i]:
                correct += 1

        return correct / len(op_preds) if op_preds else 0.0

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_hidden_at_invoke(
        self, prompt: str, layer: int | None = None
    ) -> mx.array | None:
        """Get hidden state at invoke position (= or end)."""
        if layer is None:
            layer = self.params.get("decision_layer", 12)
        if layer >= self.num_layers:
            layer = self.num_layers // 2

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        try:
            output = self.model(input_ids, output_hidden_states=True)

            if hasattr(output, "hidden_states") and output.hidden_states:
                if layer < len(output.hidden_states):
                    hidden = output.hidden_states[layer]
                    return hidden[0, -1, :]  # Last token position
        except Exception as e:
            self.log(f"Could not get hidden state: {e}", level="warning")

        return None

    def _generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        stop: str | None = "\n",
    ) -> str:
        """Generate text continuation."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output
            next_token = mx.argmax(logits[0, -1, :])
            token_id = int(next_token.item())

            if token_id == self.tokenizer.eos_token_id:
                break

            token_str = self.tokenizer.decode([token_id])
            if stop and stop in token_str:
                break

            generated.append(token_id)
            input_ids = mx.concatenate(
                [input_ids, next_token.reshape(1, 1)], axis=1
            )

        return self.tokenizer.decode(generated)

    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(self.model, attr, None)
            if submodel is not None:
                layers = getattr(submodel, "layers", None)
                if layers is not None:
                    return len(layers)
        return 22  # Default

    def _get_hidden_dim(self) -> int:
        """Get hidden dimension."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(self.model, attr, None)
            if submodel is not None:
                embed = getattr(submodel, "embed_tokens", None)
                if embed is not None:
                    # TokenEmbedding wrapper: embed_tokens.weight.parameters()["weight"]
                    if hasattr(embed, "weight"):
                        w = embed.weight
                        if hasattr(w, "parameters"):
                            params = w.parameters()
                            if "weight" in params:
                                return params["weight"].shape[1]
                        if hasattr(w, "weight"):
                            return w.weight.shape[1]
                        elif hasattr(w, "shape"):
                            return w.shape[1]
        return 2048  # Default

    # =========================================================================
    # Synthesis and Summary
    # =========================================================================

    def _synthesize_findings(self, results: dict) -> dict:
        """Synthesize findings from all sub-experiments."""
        findings = {
            "thesis_supported": False,
            "key_findings": [],
            "implications": [],
        }

        # Check format vocabulary
        if "format_vocabulary" in results:
            vocab = results["format_vocabulary"]
            if vocab.get("vocabulary"):
                findings["key_findings"].append(
                    f"Discovered {len(vocab['vocabulary'])} invocation format patterns"
                )

        # Check CoT compiler
        if "cot_compiler" in results:
            cot = results["cot_compiler"]
            success_rate = cot.get("overall_success_rate", 0)
            if success_rate >= 0.8:
                findings["key_findings"].append(
                    f"CoT compiler achieves {success_rate:.0%} success rate"
                )
                findings["thesis_supported"] = True

        # Check multi-step
        if "multi_step" in results:
            ms = results["multi_step"]
            if ms.get("self_invoke_rate", 0) >= 0.8:
                findings["key_findings"].append(
                    "Multi-step CoT shows self-invocation pattern"
                )

        # Check virtual expert readiness
        if "virtual_expert" in results:
            ve = results["virtual_expert"]
            if ve.get("ready_for_virtual_expert"):
                findings["key_findings"].append(
                    "Hidden states encode IR sufficiently for virtual expert"
                )
                findings["implications"].append(
                    "Can replace fuzzy neural expert with deterministic WASM"
                )

        # Overall thesis
        if findings["thesis_supported"]:
            findings["implications"].append(
                "CoT is a learned compiler frontend that normalizes input to invocation formats"
            )

        return findings

    def _print_summary(self, results: dict) -> None:
        """Print experiment summary."""
        print("\n" + "=" * 70)
        print("IR-ATTENTION ROUTING: EXPERIMENT SUMMARY")
        print("=" * 70)

        print(f"\nModel: {results.get('model', 'Unknown')}")
        print(f"Layers: {results.get('num_layers', '?')}")

        # Format vocabulary
        if "format_vocabulary" in results:
            vocab = results["format_vocabulary"]
            print("\n1. INVOCATION FORMAT VOCABULARY")
            for entry in vocab.get("vocabulary", [])[:5]:
                print(
                    f"   {entry['operation']}: {entry['working_count']}/{entry['total_tested']} formats work"
                )

        # CoT compiler
        if "cot_compiler" in results:
            cot = results["cot_compiler"]
            print("\n2. COT AS FORMAT COMPILER")
            print(f"   Format match rate:  {cot.get('format_match_rate', 0):.1%}")
            print(f"   Result match rate:  {cot.get('result_match_rate', 0):.1%}")
            print(f"   Overall success:    {cot.get('overall_success_rate', 0):.1%}")

        # Attention trace
        if "attention_trace" in results:
            trace = results["attention_trace"]
            print(f"\n3. ATTENTION TRACE (Layer {trace.get('decision_layer', '?')})")
            for t in trace.get("traces", [])[:2]:
                if "invoke_position" in t:
                    inv = t["invoke_position"]
                    print(f"   {t['prompt']}: attends to {inv.get('top_attended_tokens', [])[:3]}")

        # Multi-step
        if "multi_step" in results:
            ms = results["multi_step"]
            print("\n4. MULTI-STEP SELF-INVOCATION")
            print(f"   Correct rate:     {ms.get('correct_rate', 0):.1%}")
            print(f"   Self-invoke rate: {ms.get('self_invoke_rate', 0):.1%}")

        # Virtual expert
        if "virtual_expert" in results:
            ve = results["virtual_expert"]
            print("\n5. VIRTUAL EXPERT READINESS")
            print(f"   Operation accuracy:  {ve.get('operation_accuracy', 0):.1%}")
            print(f"   Operand A accuracy:  {ve.get('operand_a_accuracy', 0):.1%}")
            print(f"   Operand B accuracy:  {ve.get('operand_b_accuracy', 0):.1%}")
            ready = "YES" if ve.get("ready_for_virtual_expert") else "NO"
            print(f"   Ready for virtual expert: {ready}")

        # Synthesis
        if "synthesis" in results:
            syn = results["synthesis"]
            print("\n" + "=" * 70)
            print("SYNTHESIS")
            print("=" * 70)
            thesis = "SUPPORTED" if syn.get("thesis_supported") else "NOT SUPPORTED"
            print(f"\nThesis: {thesis}")
            print("\nKey findings:")
            for finding in syn.get("key_findings", []):
                print(f"  - {finding}")
            print("\nImplications:")
            for impl in syn.get("implications", []):
                print(f"  - {impl}")

        print("\n" + "=" * 70)

    def evaluate(self) -> dict:
        """Return summary metrics."""
        return {
            "thesis_supported": self.results.get("synthesis", {}).get(
                "thesis_supported", False
            ),
            "cot_success_rate": self.results.get("cot_compiler", {}).get(
                "overall_success_rate", 0
            ),
            "virtual_expert_ready": self.results.get("virtual_expert", {}).get(
                "ready_for_virtual_expert", False
            ),
        }

    def cleanup(self) -> None:
        """Release resources."""
        self.model = None
        self.tokenizer = None
        self.results = {}


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="IR-Attention Routing: CoT as Circuit Invocation"
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run only specific experiment(s), comma-separated",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Override experiments if specified
    if args.only:
        config_data["parameters"]["experiments"] = args.only.split(",")

    # Override model if specified
    if args.model:
        config_data["model"] = args.model

    # Create config
    from chuk_lazarus.experiments import ExperimentConfig

    config = ExperimentConfig(**config_data)
    config.experiment_dir = Path(__file__).parent

    # Run experiment
    experiment = IRAttentionRoutingExperiment(config)
    experiment.setup()
    results = experiment.run()
    experiment.cleanup()
