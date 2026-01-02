#!/usr/bin/env python3
"""
Gemma Embedding Analysis.

Test the hypothesis that RLVF backprop has baked task-relevant information
directly into the embeddings, making early layers primarily "format converters".

We probe the RAW EMBEDDINGS (before any layer processing) for:
1. Task type (arithmetic vs language)
2. Operation type (*, +, -)
3. Operand values (a, b)
4. Answer (a*b, a+b, a-b)

If these are detectable from embeddings alone, it supports the hypothesis
that early layers are format converters, not information creators.

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_embedding_analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


class EmbeddingAnalyzer:
    """Analyze what information is encoded in raw embeddings."""

    def __init__(self, model_id: str = "mlx-community/gemma-3-4b-it-bf16"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None

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

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")

    def _get_components(self):
        """Get model components."""
        if hasattr(self.model, "model"):
            backbone = self.model.model
        else:
            backbone = self.model

        layers = list(backbone.layers)
        embed = backbone.embed_tokens
        embed_scale = float(self.hidden_size ** 0.5)

        return layers, embed, embed_scale

    def get_raw_embedding(self, prompt: str) -> np.ndarray:
        """Get the raw embedding (before any layer processing)."""
        layers, embed, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        # Raw embedding (scaled, but before any layer)
        h = embed(input_ids) * embed_scale

        # Return last token position
        return np.array(h[0, -1, :].tolist())

    def get_embedding_and_layer0(self, prompt: str) -> tuple:
        """Get both raw embedding and after L0."""
        layers, embed, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        raw_emb = np.array(h[0, -1, :].tolist())

        # Process through L0
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        try:
            out = layers[0](h, mask=mask)
        except TypeError:
            out = layers[0](h)

        if hasattr(out, "hidden_states"):
            h = out.hidden_states
        elif isinstance(out, tuple):
            h = out[0]
        else:
            h = out

        after_l0 = np.array(h[0, -1, :].tolist())

        return raw_emb, after_l0

    # =========================================================================
    # EXPERIMENT 1: Task Type in Embeddings
    # =========================================================================
    def test_task_type_in_embeddings(self) -> dict:
        """Test if arithmetic vs language is detectable from raw embeddings."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: TASK TYPE IN RAW EMBEDDINGS")
        print("=" * 70)
        print("Can we detect arithmetic vs language from embeddings alone?")

        # Create dataset
        arithmetic_prompts = []
        language_prompts = []

        for a in range(2, 10):
            for b in range(2, 10):
                arithmetic_prompts.append(f"{a} * {b} = ")
                arithmetic_prompts.append(f"{a} + {b} = ")

        language_templates = [
            "The cat sat on the",
            "I went to the store to",
            "The weather today is very",
            "She picked up the book and",
            "The dog barked at the",
            "He walked down the street and",
            "The sun was shining on the",
            "We decided to go to the",
            "They found a treasure in the",
            "The music played softly in the",
            "Birds fly through the",
            "The ocean waves crashed on the",
            "Children played in the",
            "The clock struck midnight and",
            "Stars twinkled in the",
        ]

        for template in language_templates:
            for _ in range(10):
                language_prompts.append(template)

        # Balance and sample
        np.random.seed(42)
        n_samples = min(len(arithmetic_prompts), len(language_prompts), 100)
        arithmetic_prompts = list(np.random.choice(arithmetic_prompts, n_samples, replace=False))
        language_prompts = list(np.random.choice(language_prompts, n_samples, replace=False))

        all_prompts = arithmetic_prompts + language_prompts
        labels = [1] * len(arithmetic_prompts) + [0] * len(language_prompts)

        # Shuffle
        combined = list(zip(all_prompts, labels))
        np.random.shuffle(combined)
        all_prompts, labels = zip(*combined)
        all_prompts = list(all_prompts)
        labels = list(labels)

        print(f"\nDataset: {n_samples} arithmetic + {n_samples} language = {len(all_prompts)} total")

        # Collect raw embeddings
        print("Collecting raw embeddings...")
        raw_embeddings = []
        after_l0_embeddings = []

        for prompt in all_prompts:
            raw, l0 = self.get_embedding_and_layer0(prompt)
            raw_embeddings.append(raw)
            after_l0_embeddings.append(l0)

        X_raw = np.array(raw_embeddings)
        X_l0 = np.array(after_l0_embeddings)
        y = np.array(labels)

        # Train/test split
        n_test = max(1, len(X_raw) // 5)

        # Test raw embeddings
        X_train, X_test = X_raw[:-n_test], X_raw[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        probe_raw = LogisticRegression(max_iter=2000)
        probe_raw.fit(X_train, y_train)
        acc_raw = probe_raw.score(X_test, y_test)

        # Test after L0
        X_train_l0, X_test_l0 = X_l0[:-n_test], X_l0[-n_test:]
        probe_l0 = LogisticRegression(max_iter=2000)
        probe_l0.fit(X_train_l0, y_train)
        acc_l0 = probe_l0.score(X_test_l0, y_test)

        print(f"\n{'Representation':<20} {'Accuracy':<15} {'Interpretation'}")
        print("-" * 55)
        print(f"{'Raw Embedding':<20} {acc_raw:>12.1%}   {'BAKED IN!' if acc_raw > 0.9 else 'Partial' if acc_raw > 0.7 else 'Not encoded'}")
        print(f"{'After L0':<20} {acc_l0:>12.1%}   {'Already there' if acc_l0 > 0.9 else 'Enhanced'}")

        return {
            'raw_embedding_accuracy': acc_raw,
            'after_l0_accuracy': acc_l0,
            'interpretation': 'Task type baked into embeddings' if acc_raw > 0.9 else 'L0 adds task info'
        }

    # =========================================================================
    # EXPERIMENT 2: Operation Type in Embeddings
    # =========================================================================
    def test_operation_type_in_embeddings(self) -> dict:
        """Test if operation type (*, +, -) is detectable from raw embeddings."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: OPERATION TYPE IN RAW EMBEDDINGS")
        print("=" * 70)
        print("Can we detect *, +, - from embeddings alone?")

        # Create dataset
        mult_prompts = []
        add_prompts = []
        sub_prompts = []

        for a in range(2, 10):
            for b in range(2, 10):
                mult_prompts.append(f"{a} * {b} = ")
                add_prompts.append(f"{a} + {b} = ")
                sub_prompts.append(f"{a} - {b} = ")

        np.random.seed(42)
        n_samples = 50
        mult_prompts = list(np.random.choice(mult_prompts, n_samples, replace=False))
        add_prompts = list(np.random.choice(add_prompts, n_samples, replace=False))
        sub_prompts = list(np.random.choice(sub_prompts, n_samples, replace=False))

        all_prompts = mult_prompts + add_prompts + sub_prompts
        labels = [0] * n_samples + [1] * n_samples + [2] * n_samples  # 0=*, 1=+, 2=-

        # Shuffle
        combined = list(zip(all_prompts, labels))
        np.random.shuffle(combined)
        all_prompts, labels = zip(*combined)
        all_prompts = list(all_prompts)
        labels = list(labels)

        print(f"\nDataset: {n_samples} each of *, +, - = {len(all_prompts)} total")

        # Collect embeddings
        print("Collecting embeddings...")
        raw_embeddings = []
        after_l0_embeddings = []

        for prompt in all_prompts:
            raw, l0 = self.get_embedding_and_layer0(prompt)
            raw_embeddings.append(raw)
            after_l0_embeddings.append(l0)

        X_raw = np.array(raw_embeddings)
        X_l0 = np.array(after_l0_embeddings)
        y = np.array(labels)

        # Train/test split
        n_test = max(1, len(X_raw) // 5)

        # Test raw embeddings
        X_train, X_test = X_raw[:-n_test], X_raw[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        probe_raw = LogisticRegression(max_iter=2000)
        probe_raw.fit(X_train, y_train)
        acc_raw = probe_raw.score(X_test, y_test)

        # Test after L0
        X_train_l0, X_test_l0 = X_l0[:-n_test], X_l0[-n_test:]
        probe_l0 = LogisticRegression(max_iter=2000)
        probe_l0.fit(X_train_l0, y_train)
        acc_l0 = probe_l0.score(X_test_l0, y_test)

        print(f"\n{'Representation':<20} {'Accuracy':<15} {'Interpretation'}")
        print("-" * 55)
        random_baseline = 0.333
        print(f"{'Random baseline':<20} {random_baseline:>12.1%}")
        print(f"{'Raw Embedding':<20} {acc_raw:>12.1%}   {'BAKED IN!' if acc_raw > 0.9 else 'Partial' if acc_raw > 0.6 else 'Not encoded'}")
        print(f"{'After L0':<20} {acc_l0:>12.1%}   {'Already there' if acc_l0 > 0.9 else 'Enhanced'}")

        return {
            'raw_embedding_accuracy': acc_raw,
            'after_l0_accuracy': acc_l0,
            'interpretation': 'Operation baked into embeddings' if acc_raw > 0.9 else 'L0 adds operation info'
        }

    # =========================================================================
    # EXPERIMENT 3: Operand Values in Embeddings
    # =========================================================================
    def test_operands_in_embeddings(self) -> dict:
        """Test if operand values are detectable from raw embeddings."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: OPERAND VALUES IN RAW EMBEDDINGS")
        print("=" * 70)
        print("Can we decode operand 1 (a) and operand 2 (b) from embeddings?")

        # Create dataset
        prompts = []
        op1_values = []
        op2_values = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompts.append(f"{a} * {b} = ")
                op1_values.append(a)
                op2_values.append(b)

        np.random.seed(42)
        n_samples = 64
        indices = np.random.choice(len(prompts), n_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        op1_values = [op1_values[i] for i in indices]
        op2_values = [op2_values[i] for i in indices]

        print(f"\nDataset: {n_samples} multiplication problems")

        # Collect embeddings
        print("Collecting embeddings...")
        raw_embeddings = []
        after_l0_embeddings = []

        for prompt in prompts:
            raw, l0 = self.get_embedding_and_layer0(prompt)
            raw_embeddings.append(raw)
            after_l0_embeddings.append(l0)

        X_raw = np.array(raw_embeddings)
        X_l0 = np.array(after_l0_embeddings)

        # Train/test split
        n_test = max(1, len(X_raw) // 5)

        results = {}

        for name, values in [('Operand 1 (a)', op1_values), ('Operand 2 (b)', op2_values)]:
            y = np.array(values)

            # Raw embeddings
            X_train, X_test = X_raw[:-n_test], X_raw[-n_test:]
            y_train, y_test = y[:-n_test], y[-n_test:]

            probe_raw = LogisticRegression(max_iter=2000)
            try:
                probe_raw.fit(X_train, y_train)
                acc_raw = probe_raw.score(X_test, y_test)
            except:
                acc_raw = 0.125

            # After L0
            X_train_l0, X_test_l0 = X_l0[:-n_test], X_l0[-n_test:]
            probe_l0 = LogisticRegression(max_iter=2000)
            try:
                probe_l0.fit(X_train_l0, y_train)
                acc_l0 = probe_l0.score(X_test_l0, y_test)
            except:
                acc_l0 = 0.125

            results[name] = {'raw': acc_raw, 'l0': acc_l0}

        print(f"\n{'Operand':<15} {'Raw Embed':<12} {'After L0':<12} {'L0 Gain'}")
        print("-" * 50)
        random_baseline = 0.125  # 1/8 for digits 2-9

        for name, acc in results.items():
            gain = acc['l0'] - acc['raw']
            print(f"{name:<15} {acc['raw']:>10.1%} {acc['l0']:>10.1%} {gain:>+10.1%}")

        print(f"\nRandom baseline: {random_baseline:.1%}")

        return results

    # =========================================================================
    # EXPERIMENT 4: Answer in Embeddings
    # =========================================================================
    def test_answer_in_embeddings(self) -> dict:
        """Test if the answer is somehow encoded in raw embeddings."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: ANSWER IN RAW EMBEDDINGS")
        print("=" * 70)
        print("Is the answer (a*b) detectable from embeddings? (This would be surprising!)")

        # Create dataset
        prompts = []
        answers = []

        for a in range(2, 10):
            for b in range(2, 10):
                prompts.append(f"{a} * {b} = ")
                answers.append(a * b)

        np.random.seed(42)
        n_samples = 64
        indices = np.random.choice(len(prompts), n_samples, replace=False)
        prompts = [prompts[i] for i in indices]
        answers = [answers[i] for i in indices]

        print(f"\nDataset: {n_samples} multiplication problems")
        print(f"Unique answers: {len(set(answers))}")

        # Collect embeddings
        print("Collecting embeddings...")
        raw_embeddings = []
        after_l0_embeddings = []

        for prompt in prompts:
            raw, l0 = self.get_embedding_and_layer0(prompt)
            raw_embeddings.append(raw)
            after_l0_embeddings.append(l0)

        X_raw = np.array(raw_embeddings)
        X_l0 = np.array(after_l0_embeddings)
        y = np.array(answers)

        # Train/test split
        n_test = max(1, len(X_raw) // 5)

        # Classification (exact answer)
        X_train, X_test = X_raw[:-n_test], X_raw[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        probe_raw = LogisticRegression(max_iter=2000)
        try:
            probe_raw.fit(X_train, y_train)
            acc_raw = probe_raw.score(X_test, y_test)
        except:
            acc_raw = 0.0

        X_train_l0, X_test_l0 = X_l0[:-n_test], X_l0[-n_test:]
        probe_l0 = LogisticRegression(max_iter=2000)
        try:
            probe_l0.fit(X_train_l0, y_train)
            acc_l0 = probe_l0.score(X_test_l0, y_test)
        except:
            acc_l0 = 0.0

        # Regression (continuous prediction)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reg_raw = Ridge()
        reg_raw.fit(X_train_scaled, y_train)
        r2_raw = reg_raw.score(X_test_scaled, y_test)
        pred_raw = reg_raw.predict(X_test_scaled)
        mae_raw = np.mean(np.abs(pred_raw - y_test))

        X_train_l0_scaled = scaler.fit_transform(X_train_l0)
        X_test_l0_scaled = scaler.transform(X_test_l0)

        reg_l0 = Ridge()
        reg_l0.fit(X_train_l0_scaled, y_train)
        r2_l0 = reg_l0.score(X_test_l0_scaled, y_test)
        pred_l0 = reg_l0.predict(X_test_l0_scaled)
        mae_l0 = np.mean(np.abs(pred_l0 - y_test))

        print(f"\n{'Metric':<25} {'Raw Embed':<15} {'After L0':<15}")
        print("-" * 55)
        print(f"{'Classification Accuracy':<25} {acc_raw:>12.1%} {acc_l0:>12.1%}")
        print(f"{'Regression R²':<25} {r2_raw:>12.3f} {r2_l0:>12.3f}")
        print(f"{'Regression MAE':<25} {mae_raw:>12.2f} {mae_l0:>12.2f}")

        if acc_raw > 0.5 or r2_raw > 0.5:
            print("\n⚠️  SURPRISING: Answer partially encoded in raw embeddings!")
        else:
            print("\n✓ Expected: Answer NOT in raw embeddings (computed later)")

        return {
            'raw_classification': acc_raw,
            'l0_classification': acc_l0,
            'raw_r2': r2_raw,
            'l0_r2': r2_l0,
            'raw_mae': mae_raw,
            'l0_mae': mae_l0,
        }

    # =========================================================================
    # EXPERIMENT 5: Token-by-Token Embedding Analysis
    # =========================================================================
    def test_token_embeddings(self) -> dict:
        """Analyze what each token position contributes."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: TOKEN-BY-TOKEN EMBEDDING ANALYSIS")
        print("=" * 70)
        print("Which token positions encode task-relevant information?")

        layers, embed, embed_scale = self._get_components()

        # Test prompt
        prompt = "7 * 8 = "
        input_ids = self.tokenizer.encode(prompt)
        tokens = [self.tokenizer.decode([t]) for t in input_ids]

        print(f"\nPrompt: '{prompt}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {input_ids}")

        # Get embeddings for each token position
        input_ids_mx = mx.array(input_ids)[None, :]
        h = embed(input_ids_mx) * embed_scale

        print(f"\nEmbedding shape: {h.shape}")

        # Analyze each token
        print(f"\n{'Position':<10} {'Token':<10} {'Norm':<12} {'First 5 dims'}")
        print("-" * 60)

        for i, tok in enumerate(tokens):
            emb = h[0, i, :]
            norm = float(mx.sqrt(mx.sum(emb ** 2)))
            first_5 = [f"{float(x):.2f}" for x in emb[:5].tolist()]
            print(f"{i:<10} {tok:<10} {norm:>10.2f}   {first_5}")

        # Compare last token embedding for different prompts
        print("\n" + "-" * 60)
        print("Comparing last token embeddings for different prompts:")

        test_prompts = [
            "7 * 8 = ",
            "8 * 7 = ",
            "7 + 8 = ",
            "The cat = ",
        ]

        embeddings = {}
        for p in test_prompts:
            emb = self.get_raw_embedding(p)
            embeddings[p] = emb

        print(f"\n{'Prompt A':<15} {'Prompt B':<15} {'Cosine Sim'}")
        print("-" * 45)

        pairs = [
            ("7 * 8 = ", "8 * 7 = "),  # Commutative
            ("7 * 8 = ", "7 + 8 = "),  # Different op
            ("7 * 8 = ", "The cat = "),  # Arithmetic vs language
        ]

        for p1, p2 in pairs:
            e1, e2 = embeddings[p1], embeddings[p2]
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            print(f"{p1:<15} {p2:<15} {cos_sim:>10.4f}")

        return {'tokens': tokens, 'embeddings_compared': pairs}

    # =========================================================================
    # SUMMARY
    # =========================================================================
    def run_all_experiments(self) -> dict:
        """Run all embedding analysis experiments."""
        results = {}

        results['task_type'] = self.test_task_type_in_embeddings()
        results['operation_type'] = self.test_operation_type_in_embeddings()
        results['operands'] = self.test_operands_in_embeddings()
        results['answer'] = self.test_answer_in_embeddings()
        results['tokens'] = self.test_token_embeddings()

        # Summary
        print("\n" + "=" * 70)
        print("EMBEDDING ANALYSIS SUMMARY")
        print("=" * 70)

        print("""
WHAT'S BAKED INTO EMBEDDINGS (before any layer processing):

| Information      | Raw Embedding | After L0 | Interpretation           |
|------------------|---------------|----------|--------------------------|
| Task Type        | See above     | See above| Arithmetic vs Language   |
| Operation Type   | See above     | See above| *, +, -                  |
| Operand 1 (a)    | See above     | See above| First number             |
| Operand 2 (b)    | See above     | See above| Second number            |
| Answer (a*b)     | See above     | See above| Product                  |

IMPLICATIONS FOR "FORMAT CONVERTER" HYPOTHESIS:

If task/operation/operands are already in embeddings:
  → Early layers (L0-L4) are FORMAT CONVERTERS, not information creators
  → They transform embedding-space info into computation-ready format
  → This explains why layer SKIP breaks things (wrong format)
  → But component ABLATION doesn't (format conversion is distributed)

If answer is NOT in embeddings but IS after L0:
  → L0 begins the lookup/computation process
  → Even earliest layer contributes to answer formation
""")

        # Save results
        output_path = Path("gemma_discovery_cache/embedding_analysis.json")
        output_path.parent.mkdir(exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    analyzer = EmbeddingAnalyzer()
    analyzer.load_model()
    analyzer.run_all_experiments()


if __name__ == "__main__":
    main()
