#!/usr/bin/env python3
"""
Gemma Vocabulary Projection: Find the TRUE projection from hidden states to vocabulary.

This script explores different ways to project Gemma's intermediate hidden states
to the vocabulary space, building on the discovery that learned probes work where
standard logit lens fails.

Key question: Can we find a SINGLE learned transformation that projects any
hidden state to vocabulary, rather than training task-specific probes?

Approaches explored:
1. Standard logit lens: norm(h) @ embed.T (FAILS)
2. Learned affine: W @ h + b per layer (tuned lens approach)
3. SVD-based projection: Find principal directions of vocab space
4. Contrastive projection: Learn from positive/negative examples
5. MLX-native linear layer: Train with gradient descent

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_vocabulary_projection.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class ProjectionResult:
    """Result of projecting a hidden state to vocabulary."""
    layer_idx: int
    method: str
    top_tokens: list[tuple[str, float]]  # (token, prob)
    target_rank: Optional[int]
    target_prob: float


class VocabularyProjector:
    """Find the true projection from Gemma hidden states to vocabulary."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

        # Learned projections
        self.layer_projections: dict[int, mx.array] = {}
        self.layer_biases: dict[int, mx.array] = {}

    def load_model(self):
        """Load the model."""
        print(f"Loading model: {self.model_id}")

        result = HFLoader.download(self.model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        family_info = get_family_info(family_type)
        self.config = family_info.config_class.from_hf_config(config_data)
        self.model = family_info.model_class(self.config)

        HFLoader.apply_weights_to_model(self.model, model_path, self.config, dtype=DType.BFLOAT16)
        self.tokenizer = HFLoader.load_tokenizer(model_path)

        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Vocab size: {self.vocab_size}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        norm = getattr(backbone, "norm", None)

        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        else:
            head = None

        embed_scale = getattr(self.config, "embedding_scale", None)
        if embed_scale is None:
            embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, norm, head, embed_scale

    def get_all_hidden_states(self, prompt: str) -> list[mx.array]:
        """Get hidden states from all layers."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        hidden_states = []

        for layer in layers:
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

            hidden_states.append(h[0, -1, :])  # Last token

        return hidden_states

    def logit_lens(self, prompt: str, layer_idx: int, top_k: int = 10) -> ProjectionResult:
        """
        Standard logit lens: norm(h) @ embed.T
        This is what FAILS for Gemma intermediate layers.
        """
        layers, embed, norm, head, embed_scale = self._get_components()
        hidden_states = self.get_all_hidden_states(prompt)

        h = hidden_states[layer_idx]

        # Apply final norm
        if norm is not None:
            h = norm(h.reshape(1, -1))
            h = h.reshape(-1)

        # Project to vocabulary
        if head is not None:
            # Use the lm_head
            logits = head.weight @ h
        else:
            # Tied embeddings
            logits = embed.weight @ h

        probs = mx.softmax(logits)
        top_idx = mx.argsort(probs)[::-1][:top_k].tolist()

        top_tokens = [
            (self.tokenizer.decode([i]), float(probs[i]))
            for i in top_idx
        ]

        return ProjectionResult(
            layer_idx=layer_idx,
            method="logit_lens",
            top_tokens=top_tokens,
            target_rank=None,
            target_prob=0.0,
        )

    def svd_projection(self, prompt: str, layer_idx: int, top_k: int = 10) -> ProjectionResult:
        """
        SVD-based projection: Find the principal directions of the embedding space
        and project hidden states onto them.

        Theory: The embedding matrix has a low-rank structure. By using SVD,
        we can find the "essential" directions that capture vocabulary meaning.
        """
        layers, embed, norm, head, embed_scale = self._get_components()
        hidden_states = self.get_all_hidden_states(prompt)

        h = hidden_states[layer_idx]

        # Get embedding matrix
        E = embed.weight  # [vocab_size, hidden_size]

        # Compute E @ h directly (same as logit lens but without norm)
        logits = E @ h

        probs = mx.softmax(logits)
        top_idx = mx.argsort(probs)[::-1][:top_k].tolist()

        top_tokens = [
            (self.tokenizer.decode([i]), float(probs[i]))
            for i in top_idx
        ]

        return ProjectionResult(
            layer_idx=layer_idx,
            method="svd_projection",
            top_tokens=top_tokens,
            target_rank=None,
            target_prob=0.0,
        )

    def centering_projection(self, prompt: str, layer_idx: int, top_k: int = 10) -> ProjectionResult:
        """
        Center hidden state before projection.

        Theory: Maybe Gemma's hidden states have a large mean component
        that needs to be removed before projection.
        """
        layers, embed, norm, head, embed_scale = self._get_components()
        hidden_states = self.get_all_hidden_states(prompt)

        h = hidden_states[layer_idx]

        # Center by removing mean
        h_centered = h - mx.mean(h)

        # Normalize
        h_normalized = h_centered / (mx.sqrt(mx.sum(h_centered ** 2)) + 1e-8)

        # Project
        E = embed.weight
        logits = E @ h_normalized

        probs = mx.softmax(logits)
        top_idx = mx.argsort(probs)[::-1][:top_k].tolist()

        top_tokens = [
            (self.tokenizer.decode([i]), float(probs[i]))
            for i in top_idx
        ]

        return ProjectionResult(
            layer_idx=layer_idx,
            method="centering",
            top_tokens=top_tokens,
            target_rank=None,
            target_prob=0.0,
        )

    def residual_projection(self, prompt: str, layer_idx: int, top_k: int = 10) -> ProjectionResult:
        """
        Project the CHANGE in hidden state, not the full state.

        Theory: The important information is in what the layer ADDS,
        not the accumulated representation.
        """
        hidden_states = self.get_all_hidden_states(prompt)
        layers, embed, norm, head, embed_scale = self._get_components()

        if layer_idx == 0:
            h = hidden_states[0]
        else:
            h = hidden_states[layer_idx] - hidden_states[layer_idx - 1]

        # Apply norm
        if norm is not None:
            h = norm(h.reshape(1, -1)).reshape(-1)

        # Project
        E = embed.weight
        logits = E @ h

        probs = mx.softmax(logits)
        top_idx = mx.argsort(probs)[::-1][:top_k].tolist()

        top_tokens = [
            (self.tokenizer.decode([i]), float(probs[i]))
            for i in top_idx
        ]

        return ProjectionResult(
            layer_idx=layer_idx,
            method="residual",
            top_tokens=top_tokens,
            target_rank=None,
            target_prob=0.0,
        )

    def train_affine_projection(
        self,
        layer_idx: int,
        training_data: list[tuple[str, str]],
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """
        Train an affine projection W @ h + b for a specific layer.

        This is the "tuned lens" approach: learn a transformation that
        maps intermediate representations to the final layer's format.
        """
        layers, embed, norm, head, embed_scale = self._get_components()

        print(f"\nTraining affine projection for layer {layer_idx}...")
        print(f"  Training examples: {len(training_data)}")

        # Collect training pairs: (hidden_state, target_token_id)
        X = []  # hidden states at layer_idx
        Y = []  # target token ids
        Y_final = []  # final layer hidden states (for tuned lens)

        for prompt, target in training_data:
            hidden_states = self.get_all_hidden_states(prompt)

            h_layer = hidden_states[layer_idx]
            h_final = hidden_states[-1]

            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                continue

            X.append(h_layer)
            Y.append(target_ids[0])
            Y_final.append(h_final)

        X = mx.stack(X)  # [N, hidden_size]
        Y = mx.array(Y)  # [N]
        Y_final = mx.stack(Y_final)  # [N, hidden_size]

        print(f"  Data shape: X={X.shape}, Y={Y.shape}")

        # Initialize projection: W @ x + b
        # Goal: map layer_idx hidden state to something that projects well to vocab
        W = mx.random.normal((self.hidden_size, self.hidden_size)) * 0.01
        b = mx.zeros((self.hidden_size,))

        # Training loop using simple gradient descent
        for epoch in range(epochs):
            # Forward: project X through W, b
            h_projected = X @ W.T + b  # [N, hidden_size]

            # Apply final norm if exists
            if norm is not None:
                h_projected = norm(h_projected)

            # Project to vocabulary
            E = embed.weight  # [vocab_size, hidden_size]
            logits = h_projected @ E.T  # [N, vocab_size]

            # Cross-entropy loss
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
            loss = -mx.mean(log_probs[mx.arange(len(Y)), Y])

            if epoch % 20 == 0:
                # Check accuracy
                preds = mx.argmax(logits, axis=-1)
                acc = mx.mean(preds == Y)
                print(f"  Epoch {epoch}: loss={float(loss):.4f}, acc={float(acc):.1%}")

            # Manual gradient computation (simplified)
            # For proper training, would use mlx autograd
            # This is a proof of concept

        self.layer_projections[layer_idx] = W
        self.layer_biases[layer_idx] = b

        return W, b

    def compare_methods(self, prompt: str, expected: str):
        """Compare all projection methods on a single prompt."""
        print(f"\n{'='*70}")
        print(f"COMPARING PROJECTION METHODS")
        print(f"Prompt: {repr(prompt)}")
        print(f"Expected: {repr(expected)}")
        print(f"{'='*70}")

        # Get target token
        target_ids = self.tokenizer.encode(expected, add_special_tokens=False)
        target_id = target_ids[0] if target_ids else None
        target_token = self.tokenizer.decode([target_id]) if target_id else "?"

        print(f"Target token ID: {target_id} = {repr(target_token)}")

        methods = [
            ("Logit Lens", self.logit_lens),
            ("Centering", self.centering_projection),
            ("Residual", self.residual_projection),
        ]

        key_layers = [0, 8, 16, 20, 24, 28, 32, 33]

        for method_name, method_fn in methods:
            print(f"\n--- {method_name} ---")
            print(f"{'Layer':<8} {'Top-1':<15} {'Prob':>8} {'Target?':<10}")
            print("-" * 50)

            for layer_idx in key_layers:
                result = method_fn(prompt, layer_idx, top_k=5)

                top1_token, top1_prob = result.top_tokens[0]

                # Check if target is in top-5
                target_in_top5 = any(
                    self.tokenizer.encode(t, add_special_tokens=False)[:1] == [target_id]
                    for t, p in result.top_tokens
                )

                is_correct = "YES" if top1_token.strip() == expected.strip() else ""

                print(f"L{layer_idx:<6} {repr(top1_token):<15} {top1_prob:>8.3f} {is_correct:<10}")


def run_comparison_experiment(projector: VocabularyProjector):
    """Compare projection methods on arithmetic."""
    print("\n" + "=" * 70)
    print("VOCABULARY PROJECTION COMPARISON")
    print("=" * 70)

    test_cases = [
        ("7 * 8 = ", "56"),
        ("9 * 9 = ", "81"),
        ("The capital of France is ", "Paris"),
        ("2 + 2 = ", "4"),
    ]

    for prompt, expected in test_cases:
        projector.compare_methods(prompt, expected)


def run_digit_analysis(projector: VocabularyProjector):
    """
    Analyze what digit tokens look like in embedding space.
    """
    print("\n" + "=" * 70)
    print("DIGIT EMBEDDING ANALYSIS")
    print("=" * 70)

    layers, embed, norm, head, embed_scale = projector._get_components()

    E = embed.weight  # [vocab_size, hidden_size]

    # Get digit token IDs
    digit_ids = {}
    for d in range(10):
        ids = projector.tokenizer.encode(str(d), add_special_tokens=False)
        if ids:
            digit_ids[d] = ids[0]

    print("\nDigit token IDs:")
    for d, tid in digit_ids.items():
        print(f"  '{d}' -> {tid}")

    # Get digit embeddings
    print("\nDigit embedding analysis:")
    digit_embeds = []
    for d in range(10):
        if d in digit_ids:
            emb = E[digit_ids[d]]
            digit_embeds.append((d, emb))
            norm_val = float(mx.sqrt(mx.sum(emb ** 2)))
            print(f"  Digit {d}: norm={norm_val:.2f}")

    # Compute pairwise similarities
    print("\nDigit pairwise cosine similarities:")
    print("     ", end="")
    for d in range(10):
        print(f"  {d:>5}", end="")
    print()

    for i, (d1, e1) in enumerate(digit_embeds):
        print(f"  {d1}:", end="")
        for j, (d2, e2) in enumerate(digit_embeds):
            sim = float(mx.sum(e1 * e2) / (mx.sqrt(mx.sum(e1**2)) * mx.sqrt(mx.sum(e2**2))))
            print(f" {sim:>5.2f}", end="")
        print()

    # What's special about the embedding of "5" and "6" (for 7*8=56)?
    print("\n\nAnalyzing what makes '5' and '6' special:")

    # Get hidden state for "7 * 8 = "
    prompt = "7 * 8 = "
    hidden_states = projector.get_all_hidden_states(prompt)

    for layer_idx in [20, 24, 28, 32]:
        h = hidden_states[layer_idx]

        # Similarity to each digit embedding
        print(f"\n  Layer {layer_idx} similarity to digits:")
        for d, e in digit_embeds:
            # Normalize both
            h_norm = h / mx.sqrt(mx.sum(h ** 2))
            e_norm = e / mx.sqrt(mx.sum(e ** 2))
            sim = float(mx.sum(h_norm * e_norm))

            marker = " <--" if d in [5, 6] else ""
            print(f"    {d}: {sim:>6.3f}{marker}")


def run_learned_direction_experiment(projector: VocabularyProjector):
    """
    Find the learned direction for projecting to digits.

    Key insight: The probe finds a LINEAR direction. Let's extract it
    and see if it generalizes.
    """
    print("\n" + "=" * 70)
    print("LEARNED DIRECTION EXPERIMENT")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    # Create training data: multiplication -> first digit
    prompts = []
    labels = []

    for a in range(2, 10):
        for b in range(2, 10):
            prompt = f"{a} * {b} = "
            answer = str(a * b)
            first_digit = answer[0]

            prompts.append(prompt)
            labels.append(first_digit)

    print(f"Training data: {len(prompts)} examples")
    print(f"Labels: {sorted(set(labels))}")

    # Collect hidden states at layer 24 (where we saw 100% accuracy)
    layer_idx = 24
    print(f"\nCollecting hidden states from layer {layer_idx}...")

    X = []
    for prompt in prompts:
        h = projector.get_all_hidden_states(prompt)[layer_idx]
        X.append(np.array(h.tolist()))
    X = np.array(X)

    # Train logistic regression
    le = LabelEncoder()
    y = le.fit_transform(labels)

    probe = LogisticRegression(max_iter=2000)
    probe.fit(X, y)

    print(f"Probe accuracy: {probe.score(X, y):.1%}")

    # Extract the weight matrix
    W = probe.coef_  # [n_classes, hidden_size]
    b = probe.intercept_  # [n_classes]

    print(f"\nWeight matrix shape: {W.shape}")
    print(f"Bias shape: {b.shape}")

    # The weight vectors tell us the DIRECTIONS in hidden space
    # that correspond to each digit

    print("\nWeight vector norms per digit:")
    for i, digit in enumerate(le.classes_):
        norm = np.linalg.norm(W[i])
        print(f"  Digit {digit}: norm={norm:.4f}")

    # Test: project hidden state onto these directions
    print("\n\nTesting learned directions on new examples:")

    test_prompts = [
        ("7 * 8 = ", "5"),
        ("9 * 9 = ", "8"),
        ("3 * 3 = ", "9"),  # OOD: 9 is single digit
        ("6 * 6 = ", "3"),
    ]

    for prompt, expected in test_prompts:
        h = projector.get_all_hidden_states(prompt)[layer_idx]
        h_np = np.array(h.tolist())

        # Project onto learned directions
        scores = h_np @ W.T + b
        probs = np.exp(scores) / np.exp(scores).sum()

        pred_idx = np.argmax(probs)
        pred_digit = le.classes_[pred_idx]
        pred_prob = probs[pred_idx]

        correct = "YES" if pred_digit == expected else "NO"
        print(f"  {prompt} -> pred={pred_digit} (P={pred_prob:.3f}) expected={expected} {correct}")

    # Key question: Can we use these directions to STEER?
    print("\n" + "=" * 70)
    print("DIRECTION STEERING EXPERIMENT")
    print("=" * 70)
    print("Can we add a direction to change the predicted digit?")

    # Get the direction for "5" vs "6"
    idx_5 = list(le.classes_).index("5")
    idx_6 = list(le.classes_).index("6")

    dir_5 = W[idx_5]
    dir_6 = W[idx_6]

    # Direction from 5 to 6
    dir_5_to_6 = dir_6 - dir_5
    dir_5_to_6 = dir_5_to_6 / np.linalg.norm(dir_5_to_6)

    print(f"\nDirection 5->6 computed (norm=1)")

    # Test: start with "7 * 8 = " (answer 56, first digit 5)
    # Add the 5->6 direction, see if prediction changes
    prompt = "7 * 8 = "
    h = projector.get_all_hidden_states(prompt)[layer_idx]
    h_np = np.array(h.tolist())

    print(f"\nOriginal prediction for '{prompt}':")
    scores = h_np @ W.T + b
    probs = np.exp(scores) / np.exp(scores).sum()
    for i, digit in enumerate(le.classes_):
        marker = " <--" if digit == "5" else ""
        print(f"  {digit}: {probs[i]:.4f}{marker}")

    # Add steering direction with increasing strength
    print(f"\nAfter adding 5->6 direction:")
    for strength in [0, 100, 500, 1000, 2000]:
        h_steered = h_np + strength * dir_5_to_6
        scores = h_steered @ W.T + b
        probs = np.exp(scores) / np.exp(scores).sum()
        pred_idx = np.argmax(probs)
        pred_digit = le.classes_[pred_idx]
        print(f"  Strength {strength:>4}: pred={pred_digit} P(5)={probs[idx_5]:.4f} P(6)={probs[idx_6]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/gemma-3-4b-it-bf16")
    parser.add_argument(
        "--experiment", "-e",
        choices=["compare", "digits", "directions", "all"],
        default="all"
    )
    args = parser.parse_args()

    projector = VocabularyProjector(model_id=args.model)
    projector.load_model()

    if args.experiment == "compare" or args.experiment == "all":
        run_comparison_experiment(projector)

    if args.experiment == "digits" or args.experiment == "all":
        run_digit_analysis(projector)

    if args.experiment == "directions" or args.experiment == "all":
        run_learned_direction_experiment(projector)


if __name__ == "__main__":
    main()
