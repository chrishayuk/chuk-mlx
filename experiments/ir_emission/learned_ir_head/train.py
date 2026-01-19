"""
Train the Learned IR Head.

Trains a projection head to extract IR structure directly from layer 12 hidden states.
No text generation, no regex parsing - just learned structure extraction.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model

from .dataset import IRDataset, create_eval_data, create_training_data
from .heads import IRHeadBinned, create_ir_head


def get_layer_hidden_states(
    model,
    input_ids: mx.array,
    layer_idx: int = 12,
    normalize: bool = True,
) -> mx.array:
    """
    Extract hidden states from a specific layer.

    Args:
        model: The language model
        input_ids: Input token IDs, shape (batch, seq_len)
        layer_idx: Which layer to extract from
        normalize: Whether to apply final norm (important for projection!)

    Returns:
        Hidden states, shape (batch, seq_len, hidden_dim)
    """
    backbone = model.model

    # Embed
    h = backbone.embed_tokens(input_ids)

    # Create causal mask
    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Run through layers up to layer_idx
    for i, layer in enumerate(backbone.layers):
        if i >= layer_idx:
            break
        output = layer(h, mask=mask)
        h = output.hidden_states if hasattr(output, "hidden_states") else output

    # Apply normalization (critical for stable projections!)
    if normalize:
        h = backbone.norm(h)

    return h


def compute_loss(
    ir_head: IRHeadBinned,
    op_logits: mx.array,
    a_logits: mx.array,
    b_logits: mx.array,
    target_op: mx.array,
    target_a: mx.array,
    target_b: mx.array,
) -> tuple[mx.array, dict]:
    """
    Compute training loss.

    Returns:
        total_loss: Combined loss for backprop
        metrics: Dict of individual loss components
    """
    # Operation classification loss
    op_loss = nn.losses.cross_entropy(op_logits, target_op, reduction="mean")

    # Operand classification loss (binned)
    a_loss = nn.losses.cross_entropy(a_logits, target_a, reduction="mean")
    b_loss = nn.losses.cross_entropy(b_logits, target_b, reduction="mean")

    # Weighted combination (operation is most important for structure)
    total_loss = 0.4 * op_loss + 0.3 * a_loss + 0.3 * b_loss

    metrics = {
        "op_loss": float(op_loss.item()),
        "a_loss": float(a_loss.item()),
        "b_loss": float(b_loss.item()),
        "total_loss": float(total_loss.item()),
    }

    return total_loss, metrics


def compute_accuracy(
    ir_head: IRHeadBinned,
    op_logits: mx.array,
    a_logits: mx.array,
    b_logits: mx.array,
    target_op: mx.array,
    target_a: mx.array,
    target_b: mx.array,
) -> dict:
    """Compute accuracy metrics."""
    mx.eval(op_logits, a_logits, b_logits)

    # Operation accuracy
    pred_op = mx.argmax(op_logits, axis=-1)
    op_correct = mx.sum(pred_op == target_op).item()

    # Operand accuracy
    pred_a = mx.argmax(a_logits, axis=-1)
    pred_b = mx.argmax(b_logits, axis=-1)
    a_correct = mx.sum(pred_a == target_a).item()
    b_correct = mx.sum(pred_b == target_b).item()

    # Full IR accuracy (all three correct)
    full_correct = mx.sum(
        (pred_op == target_op) & (pred_a == target_a) & (pred_b == target_b)
    ).item()

    batch_size = op_logits.shape[0]

    return {
        "op_acc": op_correct / batch_size,
        "a_acc": a_correct / batch_size,
        "b_acc": b_correct / batch_size,
        "full_acc": full_correct / batch_size,
    }


def train(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    layer_idx: int = 12,
    head_type: str = "binned",
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_train_examples: int = 2000,
    save_path: str = None,
):
    """
    Train the IR head.

    Args:
        model_name: HuggingFace model to use as backbone
        layer_idx: Which layer to extract hidden states from
        head_type: Type of IR head (binned, regression, hybrid)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        num_train_examples: Number of training examples to generate
        save_path: Where to save trained head
    """
    print(f"Loading {model_name}...")
    result = load_model(model_name)
    model, tokenizer = result.model, result.tokenizer

    # Freeze backbone
    model.freeze()

    # Get hidden dimension from model
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"Hidden dim: {hidden_dim}")

    # Create IR head
    print(f"Creating {head_type} IR head...")
    ir_head = create_ir_head(head_type, hidden_dim=hidden_dim)

    # Create datasets
    print("Generating training data...")
    train_examples = create_training_data(num_train_examples, max_value=200)
    eval_examples = create_eval_data()

    train_dataset = IRDataset(train_examples, tokenizer)
    eval_dataset = IRDataset(eval_examples, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    def loss_fn(ir_head, h_pooled, target_op, target_a, target_b):
        op_logits, a_logits, b_logits = ir_head(h_pooled)
        loss, _ = compute_loss(
            ir_head, op_logits, a_logits, b_logits,
            target_op, target_a, target_b
        )
        return loss

    loss_and_grad = nn.value_and_grad(ir_head, loss_fn)

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accs = []

        for batch in train_dataset.iter_batches(batch_size):
            # Get hidden states from frozen backbone (no grad needed - backbone is frozen)
            h = get_layer_hidden_states(model, batch["input_ids"], layer_idx)
            h_pooled = h[:, -1, :]  # Last token
            mx.eval(h_pooled)

            # Forward and backward (only IR head is trained)
            loss, grads = loss_and_grad(
                ir_head, h_pooled,
                batch["op"], batch["a"], batch["b"]
            )

            # Update
            optimizer.update(ir_head, grads)
            mx.eval(ir_head.parameters(), optimizer.state)

            epoch_losses.append(float(loss.item()))

            # Compute accuracy
            op_logits, a_logits, b_logits = ir_head(h_pooled)
            acc = compute_accuracy(
                ir_head, op_logits, a_logits, b_logits,
                batch["op"], batch["a"], batch["b"]
            )
            epoch_accs.append(acc["full_acc"])

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = sum(epoch_accs) / len(epoch_accs)
        print(f"Epoch {epoch + 1:2d}: loss={avg_loss:.4f}, train_acc={avg_acc:.2%}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Held-Out Set")
    print("=" * 60)

    all_preds = []
    all_targets = []

    for batch in eval_dataset.iter_batches(batch_size=len(eval_dataset), shuffle=False):
        h = get_layer_hidden_states(model, batch["input_ids"], layer_idx)
        h_pooled = h[:, -1, :]
        mx.eval(h_pooled)

        op_logits, a_logits, b_logits = ir_head(h_pooled)
        preds = ir_head.decode(op_logits, a_logits, b_logits)

        for i, (pred, text) in enumerate(zip(preds, batch["texts"])):
            target = {"op": ir_head.IDX_TO_OP[int(batch["op"][i].item())],
                      "a": int(batch["a"][i].item()),
                      "b": int(batch["b"][i].item())}
            all_preds.append(pred)
            all_targets.append(target)

            # Check correctness
            correct = (pred["op"] == target["op"] and
                       pred["a"] == target["a"] and
                       pred["b"] == target["b"])
            status = "✓" if correct else "✗"
            print(f"{status} {text[:40]:<40} pred={pred} target={target}")

    # Summary
    correct_count = sum(
        1 for p, t in zip(all_preds, all_targets)
        if p["op"] == t["op"] and p["a"] == t["a"] and p["b"] == t["b"]
    )
    print(f"\nFinal accuracy: {correct_count}/{len(all_preds)} = {correct_count/len(all_preds):.1%}")

    # Save if requested
    if save_path:
        print(f"\nSaving IR head to {save_path}")
        ir_head.save_weights(save_path)

    return ir_head


if __name__ == "__main__":
    train(
        num_epochs=50,
        batch_size=32,
        num_train_examples=10000,
        learning_rate=1e-3,
    )
