"""
IR extraction probes for virtual expert integration.

Provides tools for:
- Training operation classification probes
- Training operand extraction probes (regression or binned)
- End-to-end IR extraction from hidden states

The goal is to decode (operation, operand_a, operand_b) directly from
hidden states, enabling replacement of fuzzy neural experts with
deterministic WASM execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    pass


class Operation(IntEnum):
    """Operation types for classification."""

    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


@dataclass
class IRInstruction:
    """Extracted IR instruction."""

    operation: Operation
    operand_a: int
    operand_b: int
    confidence: float

    def execute(self) -> int | float:
        """Execute the IR instruction."""
        if self.operation == Operation.ADD:
            return self.operand_a + self.operand_b
        elif self.operation == Operation.SUB:
            return self.operand_a - self.operand_b
        elif self.operation == Operation.MUL:
            return self.operand_a * self.operand_b
        elif self.operation == Operation.DIV:
            if self.operand_b != 0:
                return self.operand_a // self.operand_b
            return 0
        return 0


class OperationProbe(nn.Module):
    """
    Linear probe for operation classification.

    Maps hidden state -> 4-class operation prediction.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)

    def predict(self, hidden: mx.array) -> tuple[Operation, float]:
        """
        Predict operation from hidden state.

        Returns:
            (predicted_operation, confidence)
        """
        logits = self(hidden)
        probs = mx.softmax(logits)
        pred_idx = int(mx.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        return Operation(pred_idx), confidence


class OperandProbe(nn.Module):
    """
    Probe for operand extraction.

    Supports two modes:
    - Regression: Direct prediction of operand value
    - Binned: Classification into bins (0-255, etc.)
    """

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "binned",
        num_bins: int = 256,
    ):
        super().__init__()
        self.mode = mode
        self.num_bins = num_bins

        if mode == "binned":
            self.linear = nn.Linear(hidden_dim, num_bins)
        else:  # regression
            self.linear = nn.Linear(hidden_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)

    def predict(self, hidden: mx.array) -> tuple[int, float]:
        """
        Predict operand value from hidden state.

        Returns:
            (predicted_value, confidence_or_variance)
        """
        output = self(hidden)

        if self.mode == "binned":
            probs = mx.softmax(output)
            pred_idx = int(mx.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            return pred_idx, confidence
        else:
            value = int(output.item())
            # For regression, return 1.0 as "confidence"
            return value, 1.0


class IRProbe(nn.Module):
    """
    Combined IR extraction probe.

    Extracts (operation, operand_a, operand_b) from a hidden state.
    """

    def __init__(
        self,
        hidden_dim: int,
        operand_mode: str = "binned",
        num_bins: int = 256,
    ):
        super().__init__()
        self.op_probe = OperationProbe(hidden_dim, num_classes=4)
        self.operand_a_probe = OperandProbe(hidden_dim, operand_mode, num_bins)
        self.operand_b_probe = OperandProbe(hidden_dim, operand_mode, num_bins)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """
        Forward pass returning raw outputs.

        Returns:
            (op_logits, operand_a_output, operand_b_output)
        """
        return self.op_probe(x), self.operand_a_probe(x), self.operand_b_probe(x)

    def extract_ir(self, hidden: mx.array) -> IRInstruction:
        """
        Extract IR instruction from hidden state.

        Args:
            hidden: Hidden state at invoke position [hidden_dim]

        Returns:
            IRInstruction with operation and operands
        """
        op, op_conf = self.op_probe.predict(hidden)
        a, a_conf = self.operand_a_probe.predict(hidden)
        b, b_conf = self.operand_b_probe.predict(hidden)

        # Combined confidence
        confidence = op_conf * a_conf * b_conf

        return IRInstruction(
            operation=op,
            operand_a=a,
            operand_b=b,
            confidence=confidence,
        )


class IRProbeTrainer:
    """
    Trainer for IR extraction probes.

    Trains operation and operand probes on (hidden_state, IR) pairs.
    """

    def __init__(
        self,
        hidden_dim: int,
        operand_mode: str = "binned",
        num_bins: int = 256,
        learning_rate: float = 0.001,
    ):
        self.probe = IRProbe(hidden_dim, operand_mode, num_bins)
        self.lr = learning_rate
        self.operand_mode = operand_mode
        self.num_bins = num_bins

    def train(
        self,
        hiddens: mx.array,
        operations: list[int],
        operands_a: list[int],
        operands_b: list[int],
        epochs: int = 50,
    ) -> dict[str, list[float]]:
        """
        Train the IR probe.

        Args:
            hiddens: Hidden states [N, hidden_dim]
            operations: Operation labels [N]
            operands_a: First operand labels [N]
            operands_b: Second operand labels [N]
            epochs: Number of training epochs

        Returns:
            Dict with loss histories
        """
        y_op = mx.array(operations)
        y_a = mx.array(operands_a)
        y_b = mx.array(operands_b)

        if self.operand_mode == "binned":
            y_a = mx.clip(y_a, 0, self.num_bins - 1)
            y_b = mx.clip(y_b, 0, self.num_bins - 1)

        losses = {"op": [], "a": [], "b": []}

        for epoch in range(epochs):
            # Operation loss (cross-entropy)
            op_logits = self.probe.op_probe(hiddens)
            op_probs = mx.softmax(op_logits, axis=-1)
            op_loss = -mx.mean(mx.log(op_probs[mx.arange(len(y_op)), y_op] + 1e-10))

            # Operand A loss
            a_out = self.probe.operand_a_probe(hiddens)
            if self.operand_mode == "binned":
                a_probs = mx.softmax(a_out, axis=-1)
                a_loss = -mx.mean(mx.log(a_probs[mx.arange(len(y_a)), y_a] + 1e-10))
            else:
                a_loss = mx.mean((a_out.squeeze() - y_a) ** 2)

            # Operand B loss
            b_out = self.probe.operand_b_probe(hiddens)
            if self.operand_mode == "binned":
                b_probs = mx.softmax(b_out, axis=-1)
                b_loss = -mx.mean(mx.log(b_probs[mx.arange(len(y_b)), y_b] + 1e-10))
            else:
                b_loss = mx.mean((b_out.squeeze() - y_b) ** 2)

            # Total loss
            total_loss = op_loss + a_loss + b_loss

            # Compute gradients (manual for simplicity)
            # In practice, would use mx.grad and optimizer
            self._update_probe(hiddens, y_op, y_a, y_b)

            losses["op"].append(float(op_loss.item()))
            losses["a"].append(float(a_loss.item()))
            losses["b"].append(float(b_loss.item()))

        return losses

    def _update_probe(
        self,
        hiddens: mx.array,
        y_op: mx.array,
        y_a: mx.array,
        y_b: mx.array,
    ) -> None:
        """Update probe weights via gradient descent."""
        n = len(y_op)

        # Operation probe update
        op_logits = self.probe.op_probe(hiddens)
        op_probs = mx.softmax(op_logits, axis=-1)
        op_grad = op_probs
        op_grad = op_grad.at[mx.arange(n), y_op].add(-1)
        op_grad = op_grad / n

        # Get current weights
        W_op = self.probe.op_probe.linear.weight
        b_op = self.probe.op_probe.linear.bias

        grad_W_op = hiddens.T @ op_grad
        grad_b_op = mx.sum(op_grad, axis=0)

        self.probe.op_probe.linear.weight = W_op - self.lr * grad_W_op.T
        self.probe.op_probe.linear.bias = b_op - self.lr * grad_b_op

        # Operand probes (similar updates)
        if self.operand_mode == "binned":
            # Operand A
            a_out = self.probe.operand_a_probe(hiddens)
            a_probs = mx.softmax(a_out, axis=-1)
            a_grad = a_probs
            a_grad = a_grad.at[mx.arange(n), y_a].add(-1)
            a_grad = a_grad / n

            W_a = self.probe.operand_a_probe.linear.weight
            b_a = self.probe.operand_a_probe.linear.bias

            grad_W_a = hiddens.T @ a_grad
            grad_b_a = mx.sum(a_grad, axis=0)

            self.probe.operand_a_probe.linear.weight = W_a - self.lr * grad_W_a.T
            self.probe.operand_a_probe.linear.bias = b_a - self.lr * grad_b_a

            # Operand B
            b_out = self.probe.operand_b_probe(hiddens)
            b_probs = mx.softmax(b_out, axis=-1)
            b_grad = b_probs
            b_grad = b_grad.at[mx.arange(n), y_b].add(-1)
            b_grad = b_grad / n

            W_b = self.probe.operand_b_probe.linear.weight
            b_b = self.probe.operand_b_probe.linear.bias

            grad_W_b = hiddens.T @ b_grad
            grad_b_b = mx.sum(b_grad, axis=0)

            self.probe.operand_b_probe.linear.weight = W_b - self.lr * grad_W_b.T
            self.probe.operand_b_probe.linear.bias = b_b - self.lr * grad_b_b

        mx.eval(self.probe.parameters())

    def evaluate(
        self,
        hiddens: mx.array,
        operations: list[int],
        operands_a: list[int],
        operands_b: list[int],
    ) -> dict[str, float]:
        """
        Evaluate probe accuracy.

        Returns:
            Dict with op_accuracy, a_accuracy, b_accuracy, e2e_accuracy
        """
        n = len(operations)

        # Operation accuracy
        op_logits = self.probe.op_probe(hiddens)
        op_preds = mx.argmax(op_logits, axis=-1).tolist()
        op_correct = sum(1 for i in range(n) if op_preds[i] == operations[i])
        op_accuracy = op_correct / n

        # Operand A accuracy
        a_out = self.probe.operand_a_probe(hiddens)
        if self.operand_mode == "binned":
            a_preds = mx.argmax(a_out, axis=-1).tolist()
            a_targets = [min(x, self.num_bins - 1) for x in operands_a]
            a_correct = sum(1 for i in range(n) if a_preds[i] == a_targets[i])
        else:
            a_preds = a_out.squeeze().tolist()
            a_correct = sum(1 for i in range(n) if abs(a_preds[i] - operands_a[i]) <= 5)
        a_accuracy = a_correct / n

        # Operand B accuracy
        b_out = self.probe.operand_b_probe(hiddens)
        if self.operand_mode == "binned":
            b_preds = mx.argmax(b_out, axis=-1).tolist()
            b_targets = [min(x, self.num_bins - 1) for x in operands_b]
            b_correct = sum(1 for i in range(n) if b_preds[i] == b_targets[i])
        else:
            b_preds = b_out.squeeze().tolist()
            b_correct = sum(1 for i in range(n) if abs(b_preds[i] - operands_b[i]) <= 5)
        b_accuracy = b_correct / n

        # End-to-end accuracy (all three correct)
        e2e_correct = 0
        for i in range(n):
            op_ok = op_preds[i] == operations[i]
            if self.operand_mode == "binned":
                a_ok = a_preds[i] == min(operands_a[i], self.num_bins - 1)
                b_ok = b_preds[i] == min(operands_b[i], self.num_bins - 1)
            else:
                a_ok = abs(a_preds[i] - operands_a[i]) <= 5
                b_ok = abs(b_preds[i] - operands_b[i]) <= 5
            if op_ok and a_ok and b_ok:
                e2e_correct += 1
        e2e_accuracy = e2e_correct / n

        return {
            "op_accuracy": op_accuracy,
            "a_accuracy": a_accuracy,
            "b_accuracy": b_accuracy,
            "e2e_accuracy": e2e_accuracy,
        }
