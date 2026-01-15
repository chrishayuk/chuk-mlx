#!/usr/bin/env python3
"""
normalized_distillation.py

Fixed distillation with proper normalization.

The problem: Layer outputs have norms of 1,000-30,000.
MSE on these is in the millions, gradient signal is poor.

Solution: Normalize both teacher and student outputs before comparison.
Match the DIRECTION, not the magnitude.

Run: uv run python examples/introspection/normalized_distillation.py
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

warnings.filterwarnings("ignore")


class NormalizedStudent(nn.Module):
    """
    Student network with normalized output.
    Matches direction of teacher, learns a scale factor.
    """

    def __init__(
        self,
        hidden_size: int = 640,
        num_heads: int = 4,
        intermediate_size: int = 2560,  # Same as Gemma's 4x
        num_blocks: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Learnable input scale
        self.input_scale = mx.array([1.0])

        # Pre-norm
        self.input_norm = nn.RMSNorm(hidden_size)

        # Student blocks
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(StudentBlock(hidden_size, num_heads, intermediate_size))

        # Output norm
        self.output_norm = nn.RMSNorm(hidden_size)

        # Learnable output scale (to match teacher magnitude)
        self.output_scale = mx.array([1.0])

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Scale and norm input
        x = x * self.input_scale
        x = self.input_norm(x)

        # Apply blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output norm and scale
        x = self.output_norm(x)
        x = x * self.output_scale

        return x


class StudentBlock(nn.Module):
    """Single transformer block."""

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

        # Gated MLP
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


class NormalizedDistillation:
    """Distill with normalized targets."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id

    def load_model(self):
        """Load teacher."""
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

        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}")

    def get_embeddings(self, tokens: list[int]) -> mx.array:
        input_ids = mx.array([tokens])
        return (self.embed_layer(input_ids) * self.embed_scale).astype(mx.float32)

    def get_teacher_output(self, tokens: list[int], up_to_layer: int) -> mx.array:
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

    def run_remaining(self, h: mx.array, from_layer: int) -> tuple[mx.array, mx.array]:
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
        student: NormalizedStudent,
        train_prompts: list[str],
        teacher_up_to: int,
        num_epochs: int = 200,
        lr: float = 3e-4,
    ) -> list[float]:
        """Train with normalized cosine loss."""

        print(f"\n  Training student to match L0-{teacher_up_to} (normalized)...")

        # Compute target norms for scale initialization
        sample_tokens = self.tokenizer.encode(train_prompts[0])
        if isinstance(sample_tokens, np.ndarray):
            sample_tokens = sample_tokens.flatten().tolist()
        teacher_out = self.get_teacher_output(sample_tokens, teacher_up_to)
        target_norm = float(mx.mean(mx.sqrt(mx.sum(teacher_out * teacher_out, axis=-1))))

        emb = self.get_embeddings(sample_tokens)
        input_norm = float(mx.mean(mx.sqrt(mx.sum(emb * emb, axis=-1))))

        print(f"    Input norm: {input_norm:.1f}")
        print(f"    Target norm: {target_norm:.1f}")
        print(f"    Scale needed: {target_norm / input_norm:.1f}x")

        # Initialize output scale
        student.output_scale = mx.array([target_norm])

        # Prepare data
        train_data = []
        for prompt in train_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            emb = self.get_embeddings(tokens)
            teacher = self.get_teacher_output(tokens, teacher_up_to)
            train_data.append((emb, teacher))

        optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

        def loss_fn(student, emb, target):
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            pred = student(emb, mask)

            # Normalize both for direction matching
            pred_norm = mx.sqrt(mx.sum(pred * pred, axis=-1, keepdims=True) + 1e-8)
            target_norm = mx.sqrt(mx.sum(target * target, axis=-1, keepdims=True) + 1e-8)

            pred_dir = pred / pred_norm
            target_dir = target / target_norm

            # Cosine similarity loss (1 - cos)
            cosine = mx.sum(pred_dir * target_dir, axis=-1)
            cos_loss = mx.mean(1.0 - cosine)

            # Also MSE on normalized (helps with sign)
            mse_norm = mx.mean((pred_dir - target_dir) ** 2)

            # Scale matching loss
            scale_loss = mx.mean((pred_norm - target_norm) ** 2) / (target_norm.mean() ** 2)

            return cos_loss + mse_norm * 0.1 + scale_loss * 0.01

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

            if epoch % 40 == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch}: loss = {avg_loss:.4f}")

        return losses

    def evaluate(
        self,
        student: NormalizedStudent,
        test_prompts: list[str],
        teacher_up_to: int,
    ) -> dict:
        """Evaluate student + remaining layers."""

        cosines = []
        top1_matches = 0
        top5_matches = 0

        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()

            # Student path
            emb = self.get_embeddings(tokens)
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            student_out = student(emb, mask)
            student_final, student_logits = self.run_remaining(student_out, teacher_up_to + 1)

            # Full path
            full_final, full_logits = self.full_forward(tokens)

            # Cosine similarity
            s_last = student_final[0, -1, :]
            f_last = full_final[0, -1, :]

            s_norm = mx.sqrt(mx.sum(s_last * s_last))
            f_norm = mx.sqrt(mx.sum(f_last * f_last))

            if s_norm > 0 and f_norm > 0:
                cos = mx.sum(s_last * f_last) / (s_norm * f_norm)
                cosines.append(float(cos))

            # Accuracy
            full_top1 = int(mx.argmax(full_logits[0, -1, :]))
            student_top1 = int(mx.argmax(student_logits[0, -1, :]))

            if full_top1 == student_top1:
                top1_matches += 1

            student_top5 = set(mx.argsort(student_logits[0, -1, :])[-5:].tolist())
            if full_top1 in student_top5:
                top5_matches += 1

        return {
            "cosine": np.mean(cosines) if cosines else 0,
            "top1": top1_matches / len(test_prompts),
            "top5": top5_matches / len(test_prompts),
        }

    def run_experiment(self):
        print("=" * 60)
        print("NORMALIZED DISTILLATION")
        print("=" * 60)

        self.load_model()

        train_prompts = [
            "What is the weather in Tokyo?",
            "Send an email to John",
            "What is the capital of France?",
            "Explain quantum physics briefly",
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
            "Check my schedule tomorrow",
            "Tell me a joke about programming",
            "What color is the sky?",
            "Who invented the telephone?",
            "Calculate the area of a circle",
            "Translate hello to Spanish",
            "What day is it today?",
            "Find a recipe for pasta",
            "How tall is Mount Everest?",
            "What is machine learning?",
        ]

        test_prompts = [
            "Find hotels in London",
            "Order a pizza for dinner",
            "What is photosynthesis?",
            "Play some relaxing music",
            "Describe the moon in detail",
            "Turn on the living room lights",
            "What is democracy?",
            "Schedule a meeting for Friday",
            "How does gravity work?",
            "Send a message to Sarah",
        ]

        # Test different configurations
        configs = [
            {"up_to": 3, "blocks": 1, "desc": "L0-3 → 1 block"},
            {"up_to": 5, "blocks": 2, "desc": "L0-5 → 2 blocks"},
            {"up_to": 8, "blocks": 2, "desc": "L0-8 → 2 blocks"},
            {"up_to": 8, "blocks": 3, "desc": "L0-8 → 3 blocks"},
        ]

        results = []

        for config in configs:
            print(f"\n{'=' * 50}")
            print(f"{config['desc']}")
            print("=" * 50)

            student = NormalizedStudent(
                hidden_size=self.hidden_size,
                num_blocks=config["blocks"],
            )

            self.train_student(
                student,
                train_prompts,
                config["up_to"],
                num_epochs=200,
            )

            metrics = self.evaluate(student, test_prompts, config["up_to"])

            print("\n  Results:")
            print(f"    Cosine to full: {metrics['cosine']:.3f}")
            print(f"    Top-1 match: {metrics['top1']:.1%}")
            print(f"    Top-5 match: {metrics['top5']:.1%}")

            # Theoretical speedup
            layers_replaced = config["up_to"] + 1
            remaining = self.num_layers - layers_replaced
            speedup = self.num_layers / (config["blocks"] + remaining)

            results.append(
                {
                    **config,
                    **metrics,
                    "speedup": speedup,
                }
            )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n{'Config':<20} {'Cosine':>8} {'Top-1':>8} {'Top-5':>8} {'Speedup':>8}")
        print("-" * 56)

        for r in results:
            print(
                f"{r['desc']:<20} {r['cosine']:>7.3f} {r['top1']:>7.1%} {r['top5']:>7.1%} {r['speedup']:>7.2f}x"
            )

        # Best result
        best = max(results, key=lambda x: x["top1"])
        print(f"\nBest config: {best['desc']}")
        print(f"  Top-1: {best['top1']:.1%}, Speedup: {best['speedup']:.2f}x")

        return results


def main():
    exp = NormalizedDistillation()
    exp.run_experiment()


if __name__ == "__main__":
    main()
