"""
IR Head Architectures.

Different approaches to extracting IR structure from hidden states:
- Regression: Predict operands as continuous values
- Binned: Classify operands into discrete bins
- Hybrid: Classify operation, regress operands
"""

import mlx.core as mx
import mlx.nn as nn


class IRHead(nn.Module):
    """Base class for IR extraction heads."""

    # Operation vocabulary
    OP_VOCAB = ["add", "subtract", "multiply", "divide"]
    OP_TO_IDX = {op: i for i, op in enumerate(OP_VOCAB)}
    IDX_TO_OP = {i: op for i, op in enumerate(OP_VOCAB)}

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """
        Extract IR structure from hidden states.

        Args:
            h: Hidden states, shape (batch, hidden_dim) - pooled representation

        Returns:
            op_logits: Operation logits, shape (batch, 4)
            a: First operand (format depends on subclass)
            b: Second operand (format depends on subclass)
        """
        raise NotImplementedError

    def decode(self, op_logits: mx.array, a: mx.array, b: mx.array) -> list[dict]:
        """Decode model outputs to IR instructions."""
        raise NotImplementedError


class IRHeadRegression(IRHead):
    """
    IR head using regression for operands.

    Simple but may struggle with integer precision.
    """

    def __init__(self, hidden_dim: int = 2048):
        super().__init__(hidden_dim)

        # Operation classifier
        self.op_proj = nn.Linear(hidden_dim, 4)

        # Operand regression (through hidden layer for capacity)
        self.operand_hidden = nn.Linear(hidden_dim, 256)
        self.operand_a = nn.Linear(256, 1)
        self.operand_b = nn.Linear(256, 1)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Forward pass."""
        # Operation classification
        op_logits = self.op_proj(h)

        # Operand regression
        operand_h = nn.relu(self.operand_hidden(h))
        a = self.operand_a(operand_h).squeeze(-1)
        b = self.operand_b(operand_h).squeeze(-1)

        return op_logits, a, b

    def decode(self, op_logits: mx.array, a: mx.array, b: mx.array) -> list[dict]:
        """Decode to IR instructions."""
        mx.eval(op_logits, a, b)

        batch_size = op_logits.shape[0]
        results = []

        for i in range(batch_size):
            op_idx = int(mx.argmax(op_logits[i]).item())
            results.append({
                "op": self.IDX_TO_OP[op_idx],
                "a": int(round(float(a[i].item()))),
                "b": int(round(float(b[i].item()))),
            })

        return results


class IRHeadBinned(IRHead):
    """
    IR head using binned classification for operands.

    Treats operand prediction as classification into discrete bins.
    More aligned with how transformers naturally work.
    """

    def __init__(self, hidden_dim: int = 2048, num_bins: int = 256, max_value: int = 255):
        super().__init__(hidden_dim)

        self.num_bins = num_bins
        self.max_value = max_value

        # Shared encoder with more capacity
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        # Operation classifier
        self.op_proj = nn.Linear(512, 4)

        # Operand classifiers
        self.operand_a = nn.Linear(512, num_bins)
        self.operand_b = nn.Linear(512, num_bins)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Forward pass."""
        # Shared encoding
        encoded = self.encoder(h)

        # All predictions from shared representation
        op_logits = self.op_proj(encoded)
        a_logits = self.operand_a(encoded)
        b_logits = self.operand_b(encoded)

        return op_logits, a_logits, b_logits

    def decode(self, op_logits: mx.array, a_logits: mx.array, b_logits: mx.array) -> list[dict]:
        """Decode to IR instructions."""
        mx.eval(op_logits, a_logits, b_logits)

        batch_size = op_logits.shape[0]
        results = []

        for i in range(batch_size):
            op_idx = int(mx.argmax(op_logits[i]).item())
            a_bin = int(mx.argmax(a_logits[i]).item())
            b_bin = int(mx.argmax(b_logits[i]).item())

            results.append({
                "op": self.IDX_TO_OP[op_idx],
                "a": a_bin,  # Bin index IS the value (0-255)
                "b": b_bin,
            })

        return results

    def value_to_bin(self, value: int) -> int:
        """Convert value to bin index."""
        return min(max(0, value), self.num_bins - 1)


class IRHeadHybrid(IRHead):
    """
    Hybrid head: classification for operation, separate treatment for operands.

    Uses a small MLP to predict operands with both classification and regression
    signals, then combines them.
    """

    def __init__(self, hidden_dim: int = 2048, num_bins: int = 256):
        super().__init__(hidden_dim)

        self.num_bins = num_bins

        # Operation classifier
        self.op_proj = nn.Linear(hidden_dim, 4)

        # Shared operand encoder
        self.operand_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Separate heads for each operand
        self.a_classifier = nn.Linear(256, num_bins)
        self.b_classifier = nn.Linear(256, num_bins)

        # Regression heads for fine-tuning
        self.a_regressor = nn.Linear(256, 1)
        self.b_regressor = nn.Linear(256, 1)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Forward pass with both classification and regression."""
        op_logits = self.op_proj(h)

        operand_h = self.operand_encoder(h)

        a_logits = self.a_classifier(operand_h)
        b_logits = self.b_classifier(operand_h)

        a_reg = self.a_regressor(operand_h).squeeze(-1)
        b_reg = self.b_regressor(operand_h).squeeze(-1)

        return op_logits, a_logits, b_logits, a_reg, b_reg

    def decode(
        self,
        op_logits: mx.array,
        a_logits: mx.array,
        b_logits: mx.array,
        a_reg: mx.array = None,
        b_reg: mx.array = None,
    ) -> list[dict]:
        """Decode to IR instructions (uses classification by default)."""
        mx.eval(op_logits, a_logits, b_logits)

        batch_size = op_logits.shape[0]
        results = []

        for i in range(batch_size):
            op_idx = int(mx.argmax(op_logits[i]).item())
            a_val = int(mx.argmax(a_logits[i]).item())
            b_val = int(mx.argmax(b_logits[i]).item())

            results.append({
                "op": self.IDX_TO_OP[op_idx],
                "a": a_val,
                "b": b_val,
            })

        return results


def create_ir_head(
    head_type: str = "binned",
    hidden_dim: int = 2048,
    **kwargs
) -> IRHead:
    """Factory function to create IR heads."""
    if head_type == "regression":
        return IRHeadRegression(hidden_dim)
    elif head_type == "binned":
        return IRHeadBinned(hidden_dim, **kwargs)
    elif head_type == "hybrid":
        return IRHeadHybrid(hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
