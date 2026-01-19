#!/usr/bin/env python3
"""
layer_distillation.py

Test if we can distill multiple early layers into a single smaller network.

Key insight from previous experiments:
- Skipping ANY layer causes catastrophic accuracy loss
- But the layers may be COMPRESSIBLE via distillation

Approach:
1. Use layers 0-5 as "teacher"
2. Train a small student network to match teacher output
3. Test if student + layers 6-17 maintains accuracy

Run: uv run python examples/introspection/layer_distillation.py
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

warnings.filterwarnings("ignore")


class StudentNetwork(nn.Module):
    """
    Small student network to replace multiple teacher layers.

    Architecture: embedding → 2 transformer blocks → output
    Goal: Match output of layers 0-5 (or 0-3, etc.)
    """

    def __init__(
        self,
        hidden_size: int = 640,
        num_heads: int = 4,
        intermediate_size: int = 1024,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Input projection (learnable)
        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Student blocks
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(StudentBlock(hidden_size, num_heads, intermediate_size))

        # Output projection to match teacher
        self.output_norm = nn.RMSNorm(hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Project input
        x = self.input_proj(x)

        # Apply blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.output_norm(x)
        return x


class StudentBlock(nn.Module):
    """Single transformer block for student network."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention
        self.input_norm = nn.RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = nn.RoPE(dims=self.head_dim, base=10000.0)

        # MLP (use gated for better capacity)
        self.post_attn_norm = nn.RMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

        # Attention with residual
        normed = self.input_norm(x)
        q = self.q_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(normed).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        scale = self.head_dim**-0.5
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        x = x + self.o_proj(attn_out)

        # Gated MLP with residual
        normed = self.post_attn_norm(x)
        gate_out = nn.gelu(self.gate(normed))
        up_out = self.up(normed)
        x = x + self.down(gate_out * up_out)

        return x


class LayerDistillation:
    """Distill multiple layers into a smaller student network."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the teacher model."""
        print(f"Loading teacher: {self.model_id}")
        from chuk_lazarus.introspection.ablation import AblationStudy

        study = AblationStudy.from_pretrained(self.model_id)
        self.model = study.adapter.model
        self.tokenizer = study.adapter.tokenizer

        self.embed_layer = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.final_norm = self.model.model.norm
        self.hidden_size = self.model.model.hidden_size
        self.embed_scale = self.hidden_size**0.5
        self.lm_head = self.model.lm_head
        self.num_layers = len(self.layers)

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def get_teacher_output(self, tokens: list[int], up_to_layer: int) -> mx.array:
        """Get output of layers 0..up_to_layer."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(up_to_layer + 1):
            layer = self.layers[layer_idx]
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        return h.astype(mx.float32)

    def get_embeddings(self, tokens: list[int]) -> mx.array:
        """Get scaled embeddings."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale
        return emb.astype(mx.float32)

    def run_remaining_layers(self, h: mx.array, from_layer: int) -> tuple[mx.array, mx.array]:
        """Run remaining layers and get logits."""
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer_idx in range(from_layer, self.num_layers):
            layer = self.layers[layer_idx]
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return h.astype(mx.float32), logits

    def full_forward(self, tokens: list[int]) -> tuple[mx.array, mx.array]:
        """Full model forward."""
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale

        h = emb
        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for layer in self.layers:
            try:
                layer_out = layer(h, mask=mask)
            except TypeError:
                layer_out = layer(h)

            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return h.astype(mx.float32), logits

    def train_student(
        self,
        student: StudentNetwork,
        train_prompts: list[str],
        teacher_up_to: int,
        num_epochs: int = 100,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train student to match teacher output."""

        print(f"\n  Training student to match L0-{teacher_up_to}...")

        # Prepare data
        train_data = []
        for prompt in train_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            emb = self.get_embeddings(tokens)
            teacher_out = self.get_teacher_output(tokens, teacher_up_to)
            train_data.append((emb, teacher_out))

        # Optimizer
        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(student, emb, target):
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            pred = student(emb, mask)

            # MSE on all positions (not just last)
            mse = mx.mean((pred - target) ** 2)

            # Cosine loss on last token
            pred_last = pred[:, -1, :]
            target_last = target[:, -1, :]
            pred_norm = mx.sqrt(mx.sum(pred_last * pred_last, axis=-1, keepdims=True) + 1e-8)
            target_norm = mx.sqrt(mx.sum(target_last * target_last, axis=-1, keepdims=True) + 1e-8)
            cosine = mx.sum((pred_last / pred_norm) * (target_last / target_norm), axis=-1)
            cosine_loss = 1.0 - mx.mean(cosine)

            return mse + cosine_loss * 100.0  # Weight cosine more

        loss_and_grad = nn.value_and_grad(student, loss_fn)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for emb, target in train_data:
                loss, grads = loss_and_grad(student, emb, target)
                optimizer.update(student, grads)
                mx.eval(student.parameters())
                epoch_loss += float(loss)

            avg_loss = epoch_loss / len(train_data)
            losses.append(avg_loss)

            if epoch % 20 == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch}: loss = {avg_loss:.2f}")

        return losses

    def evaluate_student(
        self,
        student: StudentNetwork,
        test_prompts: list[str],
        teacher_up_to: int,
    ) -> dict:
        """Evaluate student + remaining layers vs full model."""

        similarities = []
        cosines = []
        top1_matches = 0
        top5_matches = 0

        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Student path: embedding → student → remaining layers
            emb = self.get_embeddings(tokens)
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            student_out = student(emb, mask)
            student_final, student_logits = self.run_remaining_layers(
                student_out, teacher_up_to + 1
            )

            # Full model path
            full_final, full_logits = self.full_forward(tokens)

            # Compute similarity
            student_last = student_final[0, -1, :]
            full_last = full_final[0, -1, :]

            student_norm = mx.sqrt(mx.sum(student_last * student_last))
            full_norm = mx.sqrt(mx.sum(full_last * full_last))

            if student_norm > 0 and full_norm > 0:
                cos = mx.sum(student_last * full_last) / (student_norm * full_norm)
                cosines.append(float(cos))

            # Top-k accuracy
            full_top1 = int(mx.argmax(full_logits[0, -1, :]))
            student_top1 = int(mx.argmax(student_logits[0, -1, :]))

            if full_top1 == student_top1:
                top1_matches += 1

            student_top5 = set(mx.argsort(student_logits[0, -1, :])[-5:].tolist())
            if full_top1 in student_top5:
                top5_matches += 1

        return {
            "cosine_similarity": np.mean(cosines) if cosines else 0,
            "top1_accuracy": top1_matches / len(test_prompts),
            "top5_accuracy": top5_matches / len(test_prompts),
        }

    def count_params(self, module) -> int:
        """Count parameters in module."""
        total = 0
        for param in module.parameters().values():
            if isinstance(param, mx.array):
                total += param.size
            elif isinstance(param, dict):
                for v in param.values():
                    if isinstance(v, mx.array):
                        total += v.size
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                for vv in item.values():
                                    if isinstance(vv, mx.array):
                                        total += vv.size
            elif isinstance(param, list):
                for item in param:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, mx.array):
                                total += v.size
        return total

    def run_experiment(self):
        """Run full distillation experiment."""
        print("=" * 60)
        print("LAYER DISTILLATION EXPERIMENT")
        print("=" * 60)

        self.load_model()

        # Training prompts
        train_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "What is the capital of France?",
            "Explain quantum physics",
            "Create a calendar event",
            "Write a poem about the ocean",
            "Search for restaurants nearby",
            "What is 2 + 2?",
            "Set a timer for 10 minutes",
            "Tell me about Einstein",
            "Get the stock price of Apple",
            "How do I learn Python?",
            "Book a flight to Paris",
            "What is the meaning of life?",
            "Check my schedule",
            "Tell me a joke",
            "What color is the sky?",
            "Who invented the telephone?",
            "Calculate the area of a circle",
            "Translate hello to Spanish",
        ]

        # Test prompts (separate)
        test_prompts = [
            "Find hotels in London",
            "Order a pizza",
            "What is photosynthesis?",
            "Play some music",
            "Describe the moon",
            "Turn on the lights",
            "What is democracy?",
            "Schedule a meeting",
        ]

        # Test distilling different layer ranges
        distill_configs = [
            {"up_to": 2, "student_blocks": 1, "desc": "Distill L0-2 → 1 block"},
            {"up_to": 5, "student_blocks": 2, "desc": "Distill L0-5 → 2 blocks"},
            {"up_to": 8, "student_blocks": 3, "desc": "Distill L0-8 → 3 blocks"},
        ]

        results = []

        for config in distill_configs:
            print(f"\n{'=' * 50}")
            print(f"{config['desc']}")
            print("=" * 50)

            # Create student
            student = StudentNetwork(
                hidden_size=self.hidden_size,
                num_blocks=config["student_blocks"],
            )

            # Count parameters
            teacher_layers = sum(
                self.count_params(self.layers[i]) for i in range(config["up_to"] + 1)
            )
            student_params = self.count_params(student)

            print(f"  Teacher layers 0-{config['up_to']}: ~{teacher_layers:,} params")
            print(f"  Student ({config['student_blocks']} blocks): ~{student_params:,} params")
            print(f"  Compression ratio: {teacher_layers / student_params:.1f}x")

            # Train
            losses = self.train_student(
                student,
                train_prompts,
                config["up_to"],
                num_epochs=100,
            )

            # Evaluate
            metrics = self.evaluate_student(student, test_prompts, config["up_to"])

            print("\n  Results:")
            print(f"    Cosine similarity to full: {metrics['cosine_similarity']:.3f}")
            print(f"    Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
            print(f"    Top-5 accuracy: {metrics['top5_accuracy']:.1%}")

            results.append(
                {
                    "config": config,
                    "metrics": metrics,
                    "final_loss": losses[-1],
                    "compression": teacher_layers / student_params,
                }
            )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n{'Config':<25} {'Cosine':>10} {'Top-1':>10} {'Compress':>10}")
        print("-" * 58)

        for r in results:
            config_str = r["config"]["desc"].split("→")[0].strip()
            print(
                f"{config_str:<25} {r['metrics']['cosine_similarity']:>9.3f} {r['metrics']['top1_accuracy']:>9.1%} {r['compression']:>9.1f}x"
            )

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
If distillation works (cosine > 0.9, top-1 > 80%):
- Early layers can be compressed with minimal accuracy loss
- "Train deep, deploy shallow" hypothesis is validated

If distillation fails:
- Each layer contains unique, non-redundant computation
- Layers can't be approximated by smaller networks

Key insight: The gap between "skipping layers" (0% accuracy) and
"distilling layers" shows whether the information can be compressed
or is inherently complex.
""")

        return results


def main():
    experiment = LayerDistillation()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
