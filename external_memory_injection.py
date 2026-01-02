#!/usr/bin/env python3
"""
External Memory Injection Prototype

Circuit-guided memory externalization: Replace the model's internal
fact retrieval with an external key-value store.

Hypothesis: If we inject the correct value representation at layer 20-21,
the model will output the correct answer even for "broken" facts like 7*8.

Architecture:
1. Extract query representation at layer 16-18 (before interference)
2. Match against external store (cosine similarity)
3. Inject retrieved value into residual stream at layer 20-21
4. Continue forward pass through retrieval layers 22-23

This tests whether the retrieval MACHINERY is intact, with only the
stored VALUES being corrupt.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class MemoryEntry:
    """A single entry in the external memory store."""
    query: str  # e.g., "7*8="
    answer: str  # e.g., "56"
    query_vector: mx.array  # representation at query layer
    value_vector: mx.array  # representation at value layer


@dataclass
class ExternalMemoryStore:
    """External key-value store for fact retrieval."""
    entries: list[MemoryEntry]
    query_layer: int
    value_layer: int

    def match(self, query_vec: mx.array, top_k: int = 1) -> list[tuple[MemoryEntry, float]]:
        """Find closest matches by cosine similarity."""
        query_norm = query_vec / mx.linalg.norm(query_vec)

        similarities = []
        for entry in self.entries:
            entry_norm = entry.query_vector / mx.linalg.norm(entry.query_vector)
            sim = float(mx.sum(query_norm * entry_norm))
            similarities.append((entry, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]


class ExternalMemoryModel:
    """Model wrapper with external memory injection capability."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        memory_store: ExternalMemoryStore | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_store = memory_store

    @classmethod
    def from_pretrained(cls, model_id: str) -> "ExternalMemoryModel":
        """Load model from pretrained."""
        from chuk_lazarus.inference.loader import DType, HFLoader
        from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden size: {config.hidden_size}")

        return cls(model, tokenizer, config)

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        return list(self.model.layers)

    def _get_embed(self):
        if hasattr(self.model, "model"):
            return self.model.model.embed_tokens
        return self.model.embed_tokens

    def _get_norm(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return None

    def _get_scale(self):
        return getattr(self.config, "embedding_scale", None)

    def extract_representation(self, prompt: str, layer: int) -> mx.array:
        """Extract hidden state at specified layer (last position)."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        layers = self._get_layers()
        embed = self._get_embed()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)
            if idx == layer:
                break

        return h[0, -1, :]  # Last position

    def forward_with_injection(
        self,
        prompt: str,
        inject_layer: int,
        inject_vector: mx.array,
        blend: float = 1.0,
    ) -> tuple[str, float, dict]:
        """
        Forward pass with vector injection at specified layer.

        Args:
            prompt: Input prompt
            inject_layer: Layer to inject at
            inject_vector: Vector to inject (replaces/blends with residual)
            blend: 0.0 = no injection, 1.0 = full replacement

        Returns:
            (top_prediction, probability, layer_predictions)
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        lm_head = self._get_lm_head()
        scale = self._get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        layer_preds = {}

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

            # Inject at specified layer
            if idx == inject_layer:
                # Blend: h = (1-blend)*h + blend*inject
                h_last = h[0, -1, :]
                blended = (1 - blend) * h_last + blend * inject_vector
                # Replace last position
                h = mx.concatenate([h[:, :-1, :], blended[None, None, :]], axis=1)

            # Capture predictions at key layers
            if idx in [18, 20, 22, 23]:
                h_probe = mx.array(h)  # Create a copy
                if norm is not None:
                    h_probe = norm(h_probe)
                if lm_head is not None:
                    outputs = lm_head(h_probe)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                else:
                    logits = h_probe @ embed.weight.T

                probs = mx.softmax(logits[0, -1, :], axis=-1)
                top_idx = mx.argmax(probs).item()
                top_prob = float(probs[top_idx])
                top_token = self.tokenizer.decode([top_idx])
                layer_preds[idx] = (top_token, top_prob)

        # Final prediction
        if norm is not None:
            h = norm(h)
        if lm_head is not None:
            outputs = lm_head(h)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            logits = h @ embed.weight.T

        probs = mx.softmax(logits[0, -1, :], axis=-1)
        top_idx = mx.argmax(probs).item()
        top_prob = float(probs[top_idx])
        top_token = self.tokenizer.decode([top_idx])

        return top_token, top_prob, layer_preds

    def build_memory_store(
        self,
        facts: list[dict],
        query_layer: int = 16,
        value_layer: int = 22,
    ) -> ExternalMemoryStore:
        """
        Build external memory store from facts.

        Args:
            facts: List of {"query": "7*8=", "answer": "56"}
            query_layer: Layer to extract query representation
            value_layer: Layer to extract value representation
        """
        print(f"Building memory store: {len(facts)} facts")
        print(f"  Query layer: {query_layer}")
        print(f"  Value layer: {value_layer}")

        entries = []
        for i, fact in enumerate(facts):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(facts)}...")

            query = fact["query"]
            answer = fact["answer"]

            # Extract representations
            query_vec = self.extract_representation(query, query_layer)
            value_vec = self.extract_representation(query, value_layer)

            entries.append(MemoryEntry(
                query=query,
                answer=answer,
                query_vector=query_vec,
                value_vector=value_vec,
            ))

        store = ExternalMemoryStore(
            entries=entries,
            query_layer=query_layer,
            value_layer=value_layer,
        )

        self.memory_store = store
        return store

    def query_with_external_memory(
        self,
        prompt: str,
        use_injection: bool = True,
        blend: float = 1.0,
    ) -> dict:
        """
        Query with optional external memory injection.

        Returns comparison of baseline vs injected predictions.
        """
        if self.memory_store is None:
            raise ValueError("No memory store configured")

        # Get baseline prediction (no injection)
        baseline_token, baseline_prob, baseline_layers = self.forward_with_injection(
            prompt,
            inject_layer=999,  # No injection
            inject_vector=mx.zeros(self.config.hidden_size),
            blend=0.0,
        )

        if not use_injection:
            return {
                "prompt": prompt,
                "baseline": {"token": baseline_token, "prob": baseline_prob},
                "injected": None,
                "matched_entry": None,
            }

        # Extract query representation
        query_vec = self.extract_representation(prompt, self.memory_store.query_layer)

        # Match against memory store
        matches = self.memory_store.match(query_vec, top_k=3)
        best_match, match_sim = matches[0]

        # Inject value from matched entry
        injected_token, injected_prob, injected_layers = self.forward_with_injection(
            prompt,
            inject_layer=self.memory_store.value_layer - 1,  # Inject before value layer
            inject_vector=best_match.value_vector,
            blend=blend,
        )

        return {
            "prompt": prompt,
            "baseline": {
                "token": baseline_token,
                "prob": baseline_prob,
                "layers": baseline_layers,
            },
            "injected": {
                "token": injected_token,
                "prob": injected_prob,
                "layers": injected_layers,
            },
            "matched_entry": {
                "query": best_match.query,
                "answer": best_match.answer,
                "similarity": match_sim,
            },
            "top_matches": [
                {"query": m.query, "answer": m.answer, "sim": s}
                for m, s in matches
            ],
        }


def main():
    """Test external memory injection on multiplication facts."""

    # Create model
    em_model = ExternalMemoryModel.from_pretrained("openai/gpt-oss-20b")

    # Build memory store with KNOWN CORRECT facts
    # Key insight: We need to store by ANSWER, not by query pattern
    # For each unique answer, pick one good exemplar
    answer_exemplars = {
        # Products from 2x and 3x tables (reliable)
        "4": "2*2=",
        "6": "2*3=",
        "8": "2*4=",
        "10": "2*5=",
        "12": "3*4=",
        "14": "2*7=",
        "16": "2*8=",
        "18": "2*9=",
        "9": "3*3=",
        "15": "3*5=",
        "21": "3*7=",
        "24": "3*8=",
        "27": "3*9=",
        # Add correct exemplars for 7x products
        "56": "7*8=",  # Use the fact itself as exemplar
        "42": "6*7=",
        "63": "9*7=",
        "49": "7*7=",
        "35": "5*7=",
        "28": "4*7=",
        # Other useful products
        "20": "4*5=",
        "25": "5*5=",
        "30": "5*6=",
        "36": "6*6=",
        "40": "5*8=",
        "45": "5*9=",
        "48": "6*8=",
        "54": "6*9=",
        "64": "8*8=",
        "72": "8*9=",
        "81": "9*9=",
    }

    good_facts = [{"query": q, "answer": a} for a, q in answer_exemplars.items()]

    # Build store with layer 22 as BOTH query and value layer
    # At layer 22, representations are organized by ANSWER
    store = em_model.build_memory_store(
        facts=good_facts,
        query_layer=22,  # Match in answer-space
        value_layer=22,
    )

    print("\n" + "=" * 70)
    print("EXTERNAL MEMORY INJECTION TEST")
    print("=" * 70)

    # Test on "broken" facts
    test_queries = [
        "7*8=",  # Previously hard
        "8*7=",
        "6*7=",
        "9*7=",
        "7*9=",
        "3*4=",  # Should work anyway
        "5*5=",  # Control
    ]

    print("\nTesting injection with blend=1.0 (full replacement):")
    print("-" * 70)

    for query in test_queries:
        result = em_model.query_with_external_memory(query, use_injection=True, blend=1.0)

        baseline = result["baseline"]
        injected = result["injected"]
        match = result["matched_entry"]

        # Get expected answer
        parts = query.rstrip("=").split("*")
        expected = str(int(parts[0]) * int(parts[1]))

        baseline_correct = baseline["token"].strip() == expected
        injected_correct = injected["token"].strip() == expected if injected else False

        print(f"\n{query} (expected: {expected})")
        print(f"  Baseline: '{baseline['token']}' ({baseline['prob']:.3f}) {'✓' if baseline_correct else '✗'}")
        if injected:
            print(f"  Injected: '{injected['token']}' ({injected['prob']:.3f}) {'✓' if injected_correct else '✗'}")
            print(f"  Matched:  '{match['query']}' → {match['answer']} (sim={match['similarity']:.4f})")

    # Test with varying blend factors
    print("\n" + "=" * 70)
    print("BLEND FACTOR ANALYSIS (7*8=)")
    print("=" * 70)

    for blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = em_model.query_with_external_memory("7*8=", use_injection=True, blend=blend)
        token = result["injected"]["token"] if result["injected"] else result["baseline"]["token"]
        prob = result["injected"]["prob"] if result["injected"] else result["baseline"]["prob"]
        print(f"  blend={blend:.2f}: '{token}' ({prob:.3f})")

    # CRITICAL TEST: Can we OVERRIDE the model's answer by injecting wrong values?
    print("\n" + "=" * 70)
    print("OVERRIDE TEST: Inject WRONG answers")
    print("=" * 70)
    print("If injection works, we should be able to make 7*8 output 12")

    # Find the entry for answer "12" (from 3*4=)
    entry_12 = next(e for e in store.entries if e.answer == "12")

    # Manually inject "12" value when querying 7*8
    result = em_model.forward_with_injection(
        "7*8=",
        inject_layer=21,  # Inject before layer 22
        inject_vector=entry_12.value_vector,
        blend=1.0,
    )
    print(f"  7*8= with '12' injection: '{result[0]}' ({result[1]:.3f})")
    print(f"    Layer predictions: {result[2]}")

    # Try different wrong answers
    for wrong_answer in ["12", "24", "42", "9"]:
        entry = next((e for e in store.entries if e.answer == wrong_answer), None)
        if entry:
            result = em_model.forward_with_injection(
                "7*8=",
                inject_layer=21,
                inject_vector=entry.value_vector,
                blend=1.0,
            )
            print(f"  7*8= with '{wrong_answer}' injection: '{result[0]}' ({result[1]:.3f})")

    print("\n" + "=" * 70)
    print("RESCUE TEST: Can external memory fix a deliberately broken query?")
    print("=" * 70)

    # Test with unusual formats that might confuse the model
    unusual_queries = [
        "7×8=",   # multiplication sign
        "7 * 8 =",  # spaces
        "seven times eight equals",
        "7*8",   # no equals
    ]

    for query in unusual_queries:
        try:
            # Get baseline
            baseline = em_model.forward_with_injection(
                query,
                inject_layer=999,
                inject_vector=mx.zeros(em_model.config.hidden_size),
                blend=0.0,
            )

            # Get query representation at layer 22
            query_vec = em_model.extract_representation(query, 22)

            # Find closest match
            matches = store.match(query_vec, top_k=1)
            best_match, sim = matches[0]

            # Inject
            injected = em_model.forward_with_injection(
                query,
                inject_layer=21,
                inject_vector=best_match.value_vector,
                blend=1.0,
            )

            print(f"\n  '{query}'")
            print(f"    Baseline: '{baseline[0]}' ({baseline[1]:.3f})")
            print(f"    Matched:  '{best_match.query}' → {best_match.answer} (sim={sim:.4f})")
            print(f"    Injected: '{injected[0]}' ({injected[1]:.3f})")
        except Exception as e:
            print(f"  '{query}': Error - {e}")


if __name__ == "__main__":
    main()
