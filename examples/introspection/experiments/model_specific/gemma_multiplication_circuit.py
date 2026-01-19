#!/usr/bin/env python3
"""
Gemma Multiplication Circuit Identification.

This script performs a comprehensive analysis to identify how Gemma
computes multiplication, including:

1. Layer-by-layer probing: When does the answer emerge?
2. Attention pattern analysis: Which tokens attend to which?
3. Activation patching: What information flows where?
4. Critical path identification: Minimum circuit for multiplication

Based on previous findings:
- L0, L1, L4, L21 are critical layers
- L29-L33 are dispensable
- Components within layers are redundant
- Steering works, ablation doesn't

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_multiplication_circuit.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

from chuk_lazarus.inference.loader import DType, HFLoader
from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info


@dataclass
class CircuitComponent:
    """A component of the multiplication circuit."""

    layer: int
    component_type: str  # 'attention', 'mlp', 'residual'
    role: str  # Description of what it does
    importance: float  # 0-1 importance score


class MultiplicationCircuitAnalyzer:
    """Comprehensive analysis of Gemma's multiplication circuit."""

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
        self.num_heads = self.config.num_attention_heads

        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Attention heads: {self.num_heads}")

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

        embed_scale = float(self.hidden_size**0.5)

        return layers, embed, norm, head, embed_scale

    def collect_layer_activations(self, prompt: str) -> dict[int, mx.array]:
        """Collect hidden states at each layer for a prompt."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        activations = {-1: mx.array(h)}  # -1 = embedding layer

        for i, layer in enumerate(layers):
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

            activations[i] = mx.array(h)

        return activations

    def collect_attention_patterns(self, prompt: str) -> dict[int, mx.array]:
        """Collect attention patterns at each layer."""
        layers, embed, norm, head, embed_scale = self._get_components()

        input_ids = self.tokenizer.encode(prompt)
        tokens = [self.tokenizer.decode([t]) for t in input_ids]
        input_ids = mx.array(input_ids)[None, :]

        h = embed(input_ids) * embed_scale
        mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
        mask = mask.astype(h.dtype)

        attention_patterns = {}

        for i, layer in enumerate(layers):
            # Get attention weights manually
            attn = layer.self_attn
            batch_size, seq_len, _ = h.shape

            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            else:
                h_normed = h

            # Q, K, V
            queries = attn.q_proj(h_normed)
            keys = attn.k_proj(h_normed)
            values = attn.v_proj(h_normed)

            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads
            head_dim = attn.head_dim

            queries = queries.reshape(batch_size, seq_len, num_heads, head_dim).transpose(
                0, 2, 1, 3
            )
            keys = keys.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Normalize
            queries = attn.q_norm(queries)
            keys = attn.k_norm(keys)

            # RoPE
            queries = attn.rope(queries)
            keys = attn.rope(keys)

            # Repeat KV for GQA
            n_rep = num_heads // num_kv_heads
            if n_rep > 1:
                keys = mx.repeat(keys, n_rep, axis=1)

            # Attention weights
            scale = attn.scale
            attn_weights = (queries @ keys.transpose(0, 1, 3, 2)) * scale
            if mask is not None:
                attn_weights = attn_weights + mask
            attn_weights = mx.softmax(attn_weights, axis=-1)

            attention_patterns[i] = attn_weights  # (1, num_heads, seq, seq)

            # Continue forward pass
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

        return attention_patterns, tokens

    def probe_answer_emergence(self) -> dict:
        """Probe each layer to find when the answer emerges."""
        print("\n" + "=" * 70)
        print("PROBING ANSWER EMERGENCE")
        print("=" * 70)

        # Create multiplication dataset
        prompts = []
        first_digits = []
        full_answers = []

        for a in range(2, 10):
            for b in range(2, 10):
                product = a * b
                prompts.append(f"{a} * {b} = ")
                first_digits.append(int(str(product)[0]))
                full_answers.append(product)

        print(f"\nDataset: {len(prompts)} multiplication problems")

        # Collect activations at each layer
        print("\nCollecting activations...")
        layer_activations = defaultdict(list)

        for prompt in prompts[:50]:  # Use subset for speed
            acts = self.collect_layer_activations(prompt)
            for layer, h in acts.items():
                # Get last token position
                layer_activations[layer].append(np.array(h[0, -1, :].tolist()))

        # Train probes at each layer
        print("\nTraining answer probes...")
        print(f"\n{'Layer':<8} {'First Digit':<15} {'Tens Digit':<15} {'Ones Digit':<15}")
        print("-" * 55)

        X_labels = {
            "first_digit": first_digits[:50],
            "tens_digit": [ans // 10 for ans in full_answers[:50]],
            "ones_digit": [ans % 10 for ans in full_answers[:50]],
        }

        probe_results = {}

        for layer in sorted(layer_activations.keys()):
            X = np.array(layer_activations[layer])

            accuracies = {}
            for label_name, labels in X_labels.items():
                y = np.array(labels)

                # Train/test split
                n_test = max(1, len(X) // 5)
                X_train, X_test = X[:-n_test], X[-n_test:]
                y_train, y_test = y[:-n_test], y[-n_test:]

                if len(np.unique(y_train)) < 2:
                    accuracies[label_name] = 0.0
                    continue

                probe = LogisticRegression(max_iter=1000)
                try:
                    probe.fit(X_train, y_train)
                    accuracies[label_name] = probe.score(X_test, y_test)
                except:
                    accuracies[label_name] = 0.0

            layer_name = "Embed" if layer == -1 else f"L{layer}"
            print(
                f"{layer_name:<8} {accuracies['first_digit']:>12.1%}   {accuracies['tens_digit']:>12.1%}   {accuracies['ones_digit']:>12.1%}"
            )

            probe_results[layer] = accuracies

        return probe_results

    def analyze_attention_patterns(self) -> dict:
        """Analyze which tokens attend to which in multiplication."""
        print("\n" + "=" * 70)
        print("ATTENTION PATTERN ANALYSIS")
        print("=" * 70)

        # Analyze a specific problem
        prompt = "7 * 8 = "
        attn_patterns, tokens = self.collect_attention_patterns(prompt)

        print(f"\nPrompt: '{prompt}'")
        print(f"Tokens: {tokens}")

        # Find the = token and operand tokens
        eq_pos = None
        op1_pos = None
        op2_pos = None
        star_pos = None

        for i, tok in enumerate(tokens):
            if "=" in tok:
                eq_pos = i
            elif "*" in tok:
                star_pos = i
            elif "7" in tok:
                op1_pos = i
            elif "8" in tok:
                op2_pos = i

        print(f"\nToken positions: op1={op1_pos}, star={star_pos}, op2={op2_pos}, eq={eq_pos}")

        # Analyze attention from = token to operands
        print("\nAttention from '=' to operands at each layer:")
        print(f"{'Layer':<8} {'to op1 (7)':<15} {'to op2 (8)':<15} {'to *':<15}")
        print("-" * 55)

        critical_layers = []

        for layer in sorted(attn_patterns.keys()):
            if layer % 4 != 0 and layer not in [0, 1, 4, 21, 22, 33]:
                continue  # Skip some layers for brevity

            attn = attn_patterns[layer]  # (1, heads, seq, seq)

            # Average across heads, look at attention FROM eq_pos
            if eq_pos is not None:
                avg_attn = np.array(attn[0, :, eq_pos, :].tolist()).mean(axis=0)

                attn_to_op1 = avg_attn[op1_pos] if op1_pos is not None else 0
                attn_to_op2 = avg_attn[op2_pos] if op2_pos is not None else 0
                attn_to_star = avg_attn[star_pos] if star_pos is not None else 0

                print(
                    f"L{layer:<7} {attn_to_op1:>12.3f}   {attn_to_op2:>12.3f}   {attn_to_star:>12.3f}"
                )

                # Track layers with high operand attention
                if attn_to_op1 > 0.1 or attn_to_op2 > 0.1:
                    critical_layers.append(
                        {
                            "layer": layer,
                            "attn_to_op1": attn_to_op1,
                            "attn_to_op2": attn_to_op2,
                        }
                    )

        return {"critical_layers": critical_layers, "tokens": tokens}

    def activation_patching(self) -> dict:
        """Test what information flows through the circuit using patching."""
        print("\n" + "=" * 70)
        print("ACTIVATION PATCHING ANALYSIS")
        print("=" * 70)

        layers, embed, norm, head, embed_scale = self._get_components()

        # Source: 7 * 8 = 56
        # Target: 3 * 4 = 12
        source_prompt = "7 * 8 = "
        target_prompt = "3 * 4 = "

        source_acts = self.collect_layer_activations(source_prompt)
        target_acts = self.collect_layer_activations(target_prompt)

        print(f"\nSource: {source_prompt} (answer: 56)")
        print(f"Target: {target_prompt} (answer: 12)")
        print("\nPatching source activations into target computation:")

        # For each layer, patch source activation and see what output we get
        print(f"\n{'Patched Layer':<15} {'Output':<20} {'Interpretation'}")
        print("-" * 55)

        patching_results = []

        for patch_layer in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
            # Run target up to patch_layer, then inject source activation
            input_ids = self.tokenizer.encode(target_prompt)
            input_ids = mx.array(input_ids)[None, :]

            h = embed(input_ids) * embed_scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
            mask = mask.astype(h.dtype)

            for i, layer in enumerate(layers):
                if i == patch_layer:
                    # PATCH: Replace with source activation
                    h = source_acts[patch_layer].astype(h.dtype)

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

            # Generate output
            if norm is not None:
                h = norm(h)

            if head is not None:
                logits = head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ embed.weight.T

            next_token = mx.argmax(logits[0, -1, :])
            output = self.tokenizer.decode([int(next_token)])

            # Interpret
            if "5" in output or "56" in output:
                interp = "Source answer (56) transferred!"
            elif "1" in output or "12" in output:
                interp = "Target answer (12) preserved"
            else:
                interp = f"Unexpected: {output}"

            print(f"L{patch_layer:<14} {output:<20} {interp}")

            patching_results.append(
                {
                    "layer": patch_layer,
                    "output": output,
                    "interpretation": interp,
                }
            )

        return {"results": patching_results}

    def identify_circuit_phases(self) -> dict:
        """Identify the phases of the multiplication circuit."""
        print("\n" + "=" * 70)
        print("CIRCUIT PHASE IDENTIFICATION")
        print("=" * 70)

        # Based on all our findings, map the circuit
        phases = {
            "phase_1_encoding": {
                "layers": [0, 1, 2, 3],
                "description": "Operand encoding",
                "evidence": "L0, L1 critical for accuracy (90-100% drop when skipped)",
                "what_happens": "Tokenized operands transformed into semantic representations",
            },
            "phase_2_recognition": {
                "layers": [4, 5, 6, 7],
                "description": "Arithmetic task recognition",
                "evidence": "L4 critical (100% drop), probes show arithmetic detection here",
                "what_happens": "Model recognizes this is a multiplication task",
            },
            "phase_3_retrieval": {
                "layers": [8, 9, 10, 11, 12, 13, 14, 15, 16],
                "description": "Lookup table retrieval",
                "evidence": "Probe accuracy reaches 100% by L16, commutativity perfect in early layers",
                "what_happens": "Model retrieves multiplication fact from lookup-like structure",
            },
            "phase_4_computation": {
                "layers": [17, 18, 19, 20, 21, 22],
                "description": "Answer computation/refinement",
                "evidence": "L21 shows 70% drop when skipped, steering effective at L20/L24",
                "what_happens": "Final answer computed and prepared for output",
            },
            "phase_5_formatting": {
                "layers": [23, 24, 25, 26, 27, 28],
                "description": "Output formatting",
                "evidence": "Skipping L24-L33 causes 60% drop",
                "what_happens": "Answer formatted for token generation",
            },
            "phase_6_optional": {
                "layers": [29, 30, 31, 32, 33],
                "description": "Non-essential processing",
                "evidence": "Skipping these layers causes 0% drop!",
                "what_happens": "General language modeling (not needed for simple arithmetic)",
            },
        }

        for phase_name, phase_info in phases.items():
            print(f"\n{phase_name.upper()}")
            print(f"  Layers: {phase_info['layers']}")
            print(f"  Role: {phase_info['description']}")
            print(f"  Evidence: {phase_info['evidence']}")

        return phases

    def run_full_analysis(self):
        """Run the complete circuit analysis."""
        self.load_model()

        results = {}

        # 1. Probe answer emergence
        results["probe_results"] = self.probe_answer_emergence()

        # 2. Attention patterns
        results["attention"] = self.analyze_attention_patterns()

        # 3. Activation patching
        results["patching"] = self.activation_patching()

        # 4. Circuit phases
        results["phases"] = self.identify_circuit_phases()

        # Summary
        print("\n" + "=" * 70)
        print("MULTIPLICATION CIRCUIT SUMMARY")
        print("=" * 70)

        print("""
GEMMA-3-4B MULTIPLICATION CIRCUIT
═════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│  INPUT: "7 * 8 = "                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: ENCODING (L0-L3)                                      │
│  • Tokenize "7", "*", "8", "="                                  │
│  • L0, L1 CRITICAL (90-100% drop if skipped)                    │
│  • Embed operands into semantic space                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: TASK RECOGNITION (L4-L7)                              │
│  • L4 CRITICAL (100% drop if skipped)                           │
│  • Recognize: "this is multiplication"                          │
│  • Route to arithmetic processing                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: LOOKUP/RETRIEVAL (L8-L16)                             │
│  • Answer emerges: probe accuracy → 100% by L16                 │
│  • Perfect commutativity: 7*8 ≈ 8*7 in activation space         │
│  • Lookup table structure (not step-by-step computation)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: COMPUTATION (L17-L22)                                 │
│  • L21 important (70% drop if skipped)                          │
│  • Steering effective at L20, L24                               │
│  • Refine answer representation                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: OUTPUT PREP (L23-L28)                                 │
│  • Format answer for generation                                 │
│  • Prepare logits                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6: OPTIONAL (L29-L33)                                    │
│  • CAN BE SKIPPED with 0% accuracy loss!                        │
│  • General language modeling (not arithmetic-specific)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: "56"                                                   │
└─────────────────────────────────────────────────────────────────┘

KEY FINDINGS:
─────────────
1. CRITICAL LAYERS: L0, L1, L4, L21 (cannot skip)
2. DISPENSABLE LAYERS: L29-L33 (can skip with 0% loss)
3. ANSWER EMERGENCE: By L16, answer is fully encoded
4. LOOKUP NOT COMPUTE: Uses table lookup, not step-by-step
5. DISTRIBUTED WITHIN: No single neuron/head is critical
6. SEQUENTIAL ACROSS: Must process through layer sequence
""")

        # Save results
        output_path = Path("gemma_discovery_cache/multiplication_circuit.json")
        output_path.parent.mkdir(exist_ok=True)

        # Convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    analyzer = MultiplicationCircuitAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
