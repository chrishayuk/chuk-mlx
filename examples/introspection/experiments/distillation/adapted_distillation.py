#!/usr/bin/env python3
"""
adapted_distillation.py

Fixed distillation with:
1. Per-channel scale+bias adapter at splice point
2. KL divergence loss on logits (behavioral alignment)
3. Proper dtype handling (bf16 for teacher compatibility)
4. Diagnostic metrics at splice point

Run: uv run python examples/introspection/adapted_distillation.py
"""

import warnings

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

warnings.filterwarnings("ignore")


class SpliceAdapter(nn.Module):
    """
    Learnable adapter between student output and teacher remainder.

    Learns per-channel scale and bias to align student output
    to the exact manifold the teacher expects.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Per-channel scale (init to 1)
        self.scale = mx.ones((hidden_size,))
        # Per-channel bias (init to 0)
        self.bias = mx.zeros((hidden_size,))
        # Optional: linear adapter (init to identity)
        self.use_linear = False  # Start simple
        if self.use_linear:
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
            # Initialize to identity
            self.linear.weight = mx.eye(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_linear:
            x = self.linear(x)
        return x * self.scale + self.bias


class AdaptedStudent(nn.Module):
    """
    Student network with per-channel output scaling.
    """

    def __init__(
        self,
        hidden_size: int = 640,
        num_heads: int = 4,
        intermediate_size: int = 2560,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Input norm
        self.input_norm = nn.RMSNorm(hidden_size)

        # Student blocks
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(StudentBlock(hidden_size, num_heads, intermediate_size))

        # Output norm
        self.output_norm = nn.RMSNorm(hidden_size)

        # Per-channel output scale and bias (the key fix!)
        self.output_scale = mx.ones((hidden_size,))
        self.output_bias = mx.zeros((hidden_size,))

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.output_norm(x)

        # Per-channel affine transform
        x = x * self.output_scale + self.output_bias

        return x


class StudentBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_norm = nn.RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = nn.RoPE(dims=self.head_dim, base=10000.0)

        self.post_attn_norm = nn.RMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape

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

        normed = self.post_attn_norm(x)
        gate_out = nn.gelu(self.gate(normed))
        up_out = self.up(normed)
        x = x + self.down(gate_out * up_out)

        return x


class AdaptedDistillation:
    """Distillation with adapter and KL loss."""

    def __init__(self, model_id: str = "mlx-community/functiongemma-270m-it-bf16"):
        self.model_id = model_id

    def load_model(self):
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

        # Get teacher's native dtype
        sample_weight = self.layers[0].self_attn.q_proj.weight
        self.teacher_dtype = sample_weight.dtype
        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}")
        print(f"  Teacher dtype: {self.teacher_dtype}")

    def get_embeddings(self, tokens: list[int]) -> mx.array:
        input_ids = mx.array([tokens])
        emb = self.embed_layer(input_ids) * self.embed_scale
        return emb  # Keep in teacher dtype

    def get_teacher_boundary(self, tokens: list[int], up_to_layer: int) -> mx.array:
        """Get teacher's activation at boundary (output of layer up_to_layer)."""
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

        return h

    def run_remaining(self, h: mx.array, from_layer: int) -> tuple[mx.array, mx.array]:
        """Run remaining layers and return hidden states + logits."""
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
        return h, logits

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
        return h, logits

    def compute_diagnostics(
        self,
        student_boundary: mx.array,
        teacher_boundary: mx.array,
    ) -> dict:
        """Compute diagnostic metrics at splice point."""
        # Cast to float32 for stable computation
        s = student_boundary.astype(mx.float32)
        t = teacher_boundary.astype(mx.float32)

        # Per-dimension mean
        s_mean = mx.mean(s, axis=(0, 1))  # [hidden_size]
        t_mean = mx.mean(t, axis=(0, 1))
        mean_diff = float(mx.mean(mx.abs(s_mean - t_mean)))

        # Per-dimension std
        s_std = mx.std(s, axis=(0, 1))
        t_std = mx.std(t, axis=(0, 1))
        std_ratio = float(mx.mean(s_std / (t_std + 1e-8)))

        # Cosine (last token)
        s_last = s[0, -1, :]
        t_last = t[0, -1, :]
        s_norm = mx.sqrt(mx.sum(s_last * s_last))
        t_norm = mx.sqrt(mx.sum(t_last * t_last))
        cosine = float(mx.sum(s_last * t_last) / (s_norm * t_norm + 1e-8))

        # Centered cosine (after subtracting mean)
        s_centered = s_last - s_mean
        t_centered = t_last - t_mean
        s_c_norm = mx.sqrt(mx.sum(s_centered * s_centered))
        t_c_norm = mx.sqrt(mx.sum(t_centered * t_centered))
        centered_cosine = float(mx.sum(s_centered * t_centered) / (s_c_norm * t_c_norm + 1e-8))

        # L2 distance (normalized by teacher norm)
        l2_dist = float(mx.sqrt(mx.sum((s_last - t_last) ** 2)) / (t_norm + 1e-8))

        return {
            "mean_diff": mean_diff,
            "std_ratio": std_ratio,
            "cosine": cosine,
            "centered_cosine": centered_cosine,
            "l2_dist_normalized": l2_dist,
        }

    def kl_divergence(
        self, teacher_logits: mx.array, student_logits: mx.array, temperature: float = 2.0
    ) -> mx.array:
        """Compute KL divergence with temperature."""
        # Apply temperature
        t_scaled = teacher_logits / temperature
        s_scaled = student_logits / temperature

        # Softmax and log_softmax (use mx.softmax, compute log manually)
        t_probs = mx.softmax(t_scaled, axis=-1)
        s_probs = mx.softmax(s_scaled, axis=-1)

        # Compute log probabilities with numerical stability
        t_log_probs = mx.log(t_probs + 1e-10)
        s_log_probs = mx.log(s_probs + 1e-10)

        # KL = sum(p * log(p/q)) = sum(p * (log_p - log_q))
        kl = mx.sum(t_probs * (t_log_probs - s_log_probs), axis=-1)

        return kl

    def train_student(
        self,
        student: AdaptedStudent,
        adapter: SpliceAdapter,
        train_prompts: list[str],
        teacher_up_to: int,
        num_epochs: int = 200,
        lr: float = 3e-4,
    ) -> list[dict]:
        """Train with hidden + KL loss."""

        print(f"\n  Training student to match L0-{teacher_up_to}...")
        print("  Using: per-channel scale+bias + KL on logits")

        # Initialize scale/bias from teacher statistics
        sample_tokens = self.tokenizer.encode(train_prompts[0])
        if isinstance(sample_tokens, np.ndarray):
            sample_tokens = sample_tokens.flatten().tolist()

        teacher_boundary = self.get_teacher_boundary(sample_tokens, teacher_up_to)
        t_mean = mx.mean(teacher_boundary.astype(mx.float32), axis=(0, 1))
        t_std = mx.std(teacher_boundary.astype(mx.float32), axis=(0, 1))

        print(f"    Teacher boundary mean norm: {float(mx.sqrt(mx.sum(t_mean * t_mean))):.1f}")
        print(f"    Teacher boundary std mean: {float(mx.mean(t_std)):.1f}")

        # Initialize adapter: scale to teacher std, bias to 0
        # (The student output_norm already handles mean centering)
        adapter.scale = t_std * 3.0  # Start higher since std_ratio was 0.29
        adapter.bias = mx.zeros_like(t_mean)

        # Prepare data
        train_data = []
        for prompt in train_prompts:
            tokens = self.tokenizer.encode(prompt)
            if isinstance(tokens, np.ndarray):
                tokens = tokens.flatten().tolist()
            train_data.append(tokens)

        # Optimizer for student + adapter
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

        def loss_fn(student, adapter, tokens):
            emb = self.get_embeddings(tokens)
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            # Student forward
            student_out = student(emb, mask)

            # Apply adapter
            adapted = adapter(student_out)

            # Get teacher boundary for hidden loss
            teacher_boundary = self.get_teacher_boundary(tokens, teacher_up_to)

            # Hidden state loss (per-channel MSE)
            hidden_mse = mx.mean((adapted - teacher_boundary) ** 2)

            # Run remaining layers on adapted output
            _, student_logits = self.run_remaining(adapted, teacher_up_to + 1)

            # Get full teacher logits
            _, teacher_logits = self.full_forward(tokens)

            # KL divergence on logits (last token)
            kl = self.kl_divergence(
                teacher_logits[:, -1, :], student_logits[:, -1, :], temperature=2.0
            )
            kl_loss = mx.mean(kl)

            # Combined loss
            # Weight KL more heavily since that's what matters for behavior
            total_loss = hidden_mse * 0.1 + kl_loss * 1.0

            return total_loss, hidden_mse, kl_loss

        # Create combined parameters for optimizer
        def get_all_params():
            return {
                "student": student.parameters(),
                "adapter": adapter.parameters(),
            }

        # Wrapper class to hold both student and adapter for joint optimization
        class JointModel(nn.Module):
            def __init__(self, student, adapter):
                super().__init__()
                self.student = student
                self.adapter = adapter

        joint = JointModel(student, adapter)

        def joint_loss_fn(joint, tokens):
            emb = self.get_embeddings(tokens)
            seq_len = emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(emb.dtype)

            # Student forward
            student_out = joint.student(emb, mask)
            adapted = joint.adapter(student_out)

            # Get teacher boundary for hidden loss
            teacher_boundary = self.get_teacher_boundary(tokens, teacher_up_to)

            # Cast both to float32 for stable loss computation
            adapted_f32 = adapted.astype(mx.float32)
            teacher_f32 = teacher_boundary.astype(mx.float32)

            # Hidden state loss (per-channel MSE)
            hidden_mse = mx.mean((adapted_f32 - teacher_f32) ** 2)

            # Per-dimension std matching loss
            adapted_std = mx.std(adapted_f32, axis=(0, 1))
            teacher_std = mx.std(teacher_f32, axis=(0, 1))
            std_loss = mx.mean((adapted_std - teacher_std) ** 2)

            # Per-dimension mean matching loss
            adapted_mean = mx.mean(adapted_f32, axis=(0, 1))
            teacher_mean = mx.mean(teacher_f32, axis=(0, 1))
            mean_loss = mx.mean((adapted_mean - teacher_mean) ** 2)

            # Run remaining layers on adapted output
            _, student_logits = self.run_remaining(adapted, teacher_up_to + 1)

            # Get full teacher logits
            _, teacher_logits = self.full_forward(tokens)

            # KL divergence on logits (last token)
            kl = self.kl_divergence(
                teacher_logits[:, -1, :], student_logits[:, -1, :], temperature=2.0
            )
            kl_loss = mx.mean(kl)

            # Combined loss - include std and mean matching
            total_loss = (
                hidden_mse * 0.01
                + std_loss * 1.0
                + mean_loss * 1.0
                + kl_loss * 10.0  # Weight KL even more heavily
            )

            return total_loss

        loss_and_grad = nn.value_and_grad(joint, joint_loss_fn)

        history = []
        for epoch in range(num_epochs):
            epoch_total = 0.0
            epoch_hidden = 0.0
            epoch_kl = 0.0

            for tokens in train_data:
                # Compute loss and gradients
                loss_val, grads = loss_and_grad(joint, tokens)

                # Update
                optimizer.update(joint, grads)
                mx.eval(joint.parameters())

                # Track individual losses for logging
                total, hidden, kl = loss_fn(student, adapter, tokens)
                epoch_total += float(total)
                epoch_hidden += float(hidden)
                epoch_kl += float(kl)

            n = len(train_data)
            avg_total = epoch_total / n
            avg_hidden = epoch_hidden / n
            avg_kl = epoch_kl / n

            history.append(
                {
                    "epoch": epoch,
                    "total": avg_total,
                    "hidden": avg_hidden,
                    "kl": avg_kl,
                }
            )

            if epoch % 40 == 0 or epoch == num_epochs - 1:
                print(
                    f"    Epoch {epoch}: total={avg_total:.4f}, hidden={avg_hidden:.4f}, kl={avg_kl:.4f}"
                )

        return history

    def evaluate(
        self,
        student: AdaptedStudent,
        adapter: SpliceAdapter,
        test_prompts: list[str],
        teacher_up_to: int,
    ) -> dict:
        """Evaluate with detailed metrics."""

        cosines = []
        kls = []
        top1_matches = 0
        top5_matches = 0
        top10_matches = 0
        top50_matches = 0

        all_diagnostics = []

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
            adapted = adapter(student_out)
            student_final, student_logits = self.run_remaining(adapted, teacher_up_to + 1)

            # Teacher path
            teacher_boundary = self.get_teacher_boundary(tokens, teacher_up_to)
            full_final, full_logits = self.full_forward(tokens)

            # Diagnostics at boundary
            diag = self.compute_diagnostics(adapted, teacher_boundary)
            all_diagnostics.append(diag)

            # Final cosine
            s_last = student_final[0, -1, :].astype(mx.float32)
            f_last = full_final[0, -1, :].astype(mx.float32)
            s_norm = mx.sqrt(mx.sum(s_last * s_last))
            f_norm = mx.sqrt(mx.sum(f_last * f_last))
            cos = float(mx.sum(s_last * f_last) / (s_norm * f_norm + 1e-8))
            cosines.append(cos)

            # KL divergence
            kl = self.kl_divergence(
                full_logits[:, -1, :],
                student_logits[:, -1, :],
                temperature=1.0,  # Use T=1 for eval
            )
            kls.append(float(mx.mean(kl)))

            # Top-k accuracy
            full_top1 = int(mx.argmax(full_logits[0, -1, :]))
            student_sorted = mx.argsort(student_logits[0, -1, :])

            student_top1 = int(student_sorted[-1])
            student_top5 = set(student_sorted[-5:].tolist())
            student_top10 = set(student_sorted[-10:].tolist())
            student_top50 = set(student_sorted[-50:].tolist())

            if full_top1 == student_top1:
                top1_matches += 1
            if full_top1 in student_top5:
                top5_matches += 1
            if full_top1 in student_top10:
                top10_matches += 1
            if full_top1 in student_top50:
                top50_matches += 1

        n = len(test_prompts)

        # Aggregate diagnostics
        avg_diag = {k: np.mean([d[k] for d in all_diagnostics]) for k in all_diagnostics[0].keys()}

        return {
            "cosine_final": np.mean(cosines),
            "kl_mean": np.mean(kls),
            "top1": top1_matches / n,
            "top5": top5_matches / n,
            "top10": top10_matches / n,
            "top50": top50_matches / n,
            "boundary_diagnostics": avg_diag,
        }

    def run_experiment(self):
        print("=" * 60)
        print("ADAPTED DISTILLATION")
        print("=" * 60)
        print("\nFixes applied:")
        print("  1. Per-channel scale+bias adapter at splice point")
        print("  2. KL divergence loss on logits")
        print("  3. Proper dtype handling")
        print("  4. Diagnostic metrics at boundary")

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

        # Test configuration
        configs = [
            {"up_to": 5, "blocks": 2, "desc": "L0-5 → 2 blocks"},
        ]

        results = []

        for config in configs:
            print(f"\n{'=' * 50}")
            print(f"{config['desc']}")
            print("=" * 50)

            student = AdaptedStudent(
                hidden_size=self.hidden_size,
                num_blocks=config["blocks"],
            )
            adapter = SpliceAdapter(self.hidden_size)

            history = self.train_student(
                student,
                adapter,
                train_prompts,
                config["up_to"],
                num_epochs=400,
            )

            metrics = self.evaluate(student, adapter, test_prompts, config["up_to"])

            print("\n  Results:")
            print(f"    Cosine to full (final): {metrics['cosine_final']:.3f}")
            print(f"    KL divergence: {metrics['kl_mean']:.3f}")
            print(f"    Top-1 match: {metrics['top1']:.1%}")
            print(f"    Top-5 match: {metrics['top5']:.1%}")
            print(f"    Top-10 match: {metrics['top10']:.1%}")
            print(f"    Top-50 match: {metrics['top50']:.1%}")

            print("\n  Boundary diagnostics:")
            for k, v in metrics["boundary_diagnostics"].items():
                print(f"    {k}: {v:.4f}")

            results.append(
                {
                    **config,
                    **metrics,
                }
            )

        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        print("""
With adapter and KL loss, we should see:
- Top-k moving off 0% (behavioral alignment)
- Lower KL (student matches teacher distribution)
- Better boundary diagnostics (per-dim alignment)

If top-50 > 0% but top-1 = 0%, the student is "close but not exact".
If all top-k = 0%, we may need:
- More student capacity
- End-to-end fine-tuning of some teacher layers
- Different adapter architecture
""")

        return results


def main():
    exp = AdaptedDistillation()
    exp.run_experiment()


if __name__ == "__main__":
    main()
