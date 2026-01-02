"""
External Memory Module for Circuit-Guided Memory Injection

This module provides the ability to externalize LLM memory by:
1. Extracting query/value representations from known facts
2. Building a searchable memory store
3. Injecting retrieved values at inference time

This enables:
- Overriding incorrect model answers
- Rescuing out-of-distribution query formats
- Adding new facts without fine-tuning

Usage:
    >>> from chuk_lazarus.introspection.external_memory import ExternalMemory
    >>>
    >>> # Build memory from facts
    >>> memory = ExternalMemory.from_pretrained("openai/gpt-oss-20b")
    >>> memory.add_facts([
    ...     {"query": "7*8=", "answer": "56"},
    ...     {"query": "6*7=", "answer": "42"},
    ... ])
    >>>
    >>> # Query with injection
    >>> result = memory.query("seven times eight equals")
    >>> print(result)
    # {'answer': '56', 'confidence': 0.69, 'matched': '7*8=', 'similarity': 0.71}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class MemoryEntry:
    """A single fact stored in external memory."""

    query: str
    """The query string (e.g., '7*8=')"""

    answer: str
    """The expected answer (e.g., '56')"""

    query_vector: mx.array | None = None
    """Representation at query layer"""

    value_vector: mx.array | None = None
    """Representation at value layer (for injection)"""

    metadata: dict = field(default_factory=dict)
    """Optional metadata (category, source, etc.)"""


@dataclass
class MemoryConfig:
    """Configuration for external memory."""

    query_layer: int = 22
    """Layer to extract query representations (for matching)"""

    inject_layer: int = 21
    """Layer to inject value vectors (before crystallization)"""

    value_layer: int = 22
    """Layer to extract value representations"""

    similarity_threshold: float = 0.7
    """Minimum similarity to use external memory"""

    blend: float = 1.0
    """Blend factor: 0=no injection, 1=full replacement"""


@dataclass
class QueryResult:
    """Result of querying with external memory."""

    query: str
    """Input query"""

    baseline_answer: str
    """Model's answer without injection"""

    baseline_confidence: float
    """Confidence without injection"""

    injected_answer: str | None
    """Answer with injection (if used)"""

    injected_confidence: float | None
    """Confidence with injection"""

    matched_entry: MemoryEntry | None
    """The memory entry that was matched"""

    similarity: float
    """Similarity to matched entry"""

    used_injection: bool
    """Whether injection was actually used"""

    layer_predictions: dict[int, tuple[str, float]] = field(default_factory=dict)
    """Predictions at key layers (for debugging)"""


class ExternalMemory:
    """
    External memory system for LLM fact injection.

    Provides circuit-guided memory externalization by:
    1. Building a store of (query, value) vector pairs
    2. Matching input queries to stored entries
    3. Injecting retrieved values into the residual stream
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        memory_config: MemoryConfig | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._memory_config = memory_config or MemoryConfig()
        self._entries: list[MemoryEntry] = []

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        memory_config: MemoryConfig | None = None,
    ) -> ExternalMemory:
        """Load model and create external memory system."""
        from ..inference.loader import DType, HFLoader
        from ..models_v2.families.registry import detect_model_family, get_family_info

        print(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        config_path = model_path / "config.json"
        with open(config_path) as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden size: {config.hidden_size}")

        # Auto-configure memory layers based on model depth
        if memory_config is None:
            n_layers = config.num_hidden_layers
            memory_config = MemoryConfig(
                query_layer=int(n_layers * 0.92),  # ~92% depth for matching
                inject_layer=int(n_layers * 0.88),  # ~88% depth for injection
                value_layer=int(n_layers * 0.92),  # ~92% depth for value extraction
            )
            print(
                f"  Auto-configured: query_layer={memory_config.query_layer}, "
                f"inject_layer={memory_config.inject_layer}"
            )

        return cls(model, tokenizer, config, memory_config)

    @property
    def num_entries(self) -> int:
        """Number of facts in memory."""
        return len(self._entries)

    @property
    def hidden_size(self) -> int:
        """Model hidden dimension."""
        return self._config.hidden_size

    def _get_layers(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return list(self._model.model.layers)
        return list(self._model.layers)

    def _get_embed(self):
        if hasattr(self._model, "model"):
            return self._model.model.embed_tokens
        return self._model.embed_tokens

    def _get_norm(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "norm"):
            return self._model.model.norm
        if hasattr(self._model, "norm"):
            return self._model.norm
        return None

    def _get_lm_head(self):
        return self._model.lm_head if hasattr(self._model, "lm_head") else None

    def _get_scale(self):
        return getattr(self._config, "embedding_scale", None)

    def _extract_representation(self, prompt: str, layer: int) -> mx.array:
        """Extract hidden state at specified layer (last position)."""
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
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
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            if idx == layer:
                break

        return h[0, -1, :]

    def _forward_with_injection(
        self,
        prompt: str,
        inject_layer: int | None = None,
        inject_vector: mx.array | None = None,
        blend: float = 1.0,
        capture_layers: list[int] | None = None,
    ) -> tuple[str, float, dict[int, tuple[str, float]]]:
        """
        Forward pass with optional vector injection.

        Returns: (top_token, probability, layer_predictions)
        """
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        lm_head = self._get_lm_head()
        scale = self._get_scale()

        if capture_layers is None:
            capture_layers = [18, 20, 22, 23]

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
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )

            # Inject at specified layer
            if inject_layer is not None and idx == inject_layer and inject_vector is not None:
                h_last = h[0, -1, :]
                blended = (1 - blend) * h_last + blend * inject_vector
                h = mx.concatenate([h[:, :-1, :], blended[None, None, :]], axis=1)

            # Capture predictions at key layers
            if idx in capture_layers:
                h_probe = mx.array(h)
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
                top_token = self._tokenizer.decode([top_idx])
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
        top_token = self._tokenizer.decode([top_idx])

        return top_token, top_prob, layer_preds

    def add_fact(
        self,
        query: str,
        answer: str,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """
        Add a single fact to external memory.

        Args:
            query: The query string (e.g., "7*8=")
            answer: The expected answer (e.g., "56")
            metadata: Optional metadata

        Returns:
            The created MemoryEntry
        """
        mc = self._memory_config

        query_vec = self._extract_representation(query, mc.query_layer)
        value_vec = self._extract_representation(query, mc.value_layer)

        entry = MemoryEntry(
            query=query,
            answer=answer,
            query_vector=query_vec,
            value_vector=value_vec,
            metadata=metadata or {},
        )

        self._entries.append(entry)
        return entry

    def add_facts(
        self,
        facts: list[dict],
        verbose: bool = True,
    ) -> list[MemoryEntry]:
        """
        Add multiple facts to external memory.

        Args:
            facts: List of {"query": str, "answer": str, "metadata": dict}
            verbose: Print progress

        Returns:
            List of created MemoryEntry objects
        """
        entries = []

        for i, fact in enumerate(facts):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Adding fact {i + 1}/{len(facts)}...")

            entry = self.add_fact(
                query=fact["query"],
                answer=fact["answer"],
                metadata=fact.get("metadata", {}),
            )
            entries.append(entry)

        if verbose:
            print(f"  Added {len(entries)} facts to memory")

        return entries

    def add_multiplication_table(self, min_val: int = 2, max_val: int = 9) -> list[MemoryEntry]:
        """Add multiplication table facts."""
        facts = []
        for a in range(min_val, max_val + 1):
            for b in range(min_val, max_val + 1):
                facts.append(
                    {
                        "query": f"{a}*{b}=",
                        "answer": str(a * b),
                        "metadata": {"type": "multiplication", "a": a, "b": b},
                    }
                )

        print(f"Adding {len(facts)} multiplication facts...")
        return self.add_facts(facts)

    def match(self, query_vec: mx.array, top_k: int = 3) -> list[tuple[MemoryEntry, float]]:
        """
        Find closest matches in memory by cosine similarity.

        Args:
            query_vec: Query representation vector
            top_k: Number of matches to return

        Returns:
            List of (entry, similarity) tuples, sorted by similarity
        """
        if not self._entries:
            return []

        query_norm = query_vec / mx.linalg.norm(query_vec)

        similarities = []
        for entry in self._entries:
            if entry.query_vector is not None:
                entry_norm = entry.query_vector / mx.linalg.norm(entry.query_vector)
                sim = float(mx.sum(query_norm * entry_norm))
                similarities.append((entry, sim))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def query(
        self,
        prompt: str,
        use_injection: bool = True,
        force_injection: bool = False,
    ) -> QueryResult:
        """
        Query with optional external memory injection.

        Args:
            prompt: Input query
            use_injection: Whether to use external memory
            force_injection: Inject even if similarity is below threshold

        Returns:
            QueryResult with baseline and injected predictions
        """
        mc = self._memory_config

        # Get baseline prediction
        baseline_token, baseline_prob, baseline_layers = self._forward_with_injection(
            prompt,
            inject_layer=None,
            inject_vector=None,
        )

        # Find match in memory
        query_vec = self._extract_representation(prompt, mc.query_layer)
        matches = self.match(query_vec, top_k=1)

        matched_entry = None
        similarity = 0.0
        injected_token = None
        injected_prob = None
        used_injection = False
        injected_layers = {}

        if matches:
            matched_entry, similarity = matches[0]

            should_inject = (
                use_injection
                and matched_entry.value_vector is not None
                and (force_injection or similarity >= mc.similarity_threshold)
            )

            if should_inject:
                injected_token, injected_prob, injected_layers = self._forward_with_injection(
                    prompt,
                    inject_layer=mc.inject_layer,
                    inject_vector=matched_entry.value_vector,
                    blend=mc.blend,
                )
                used_injection = True

        return QueryResult(
            query=prompt,
            baseline_answer=baseline_token,
            baseline_confidence=baseline_prob,
            injected_answer=injected_token,
            injected_confidence=injected_prob,
            matched_entry=matched_entry,
            similarity=similarity,
            used_injection=used_injection,
            layer_predictions=injected_layers if used_injection else baseline_layers,
        )

    def batch_query(
        self,
        prompts: list[str],
        use_injection: bool = True,
        verbose: bool = True,
    ) -> list[QueryResult]:
        """Query multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Querying {i + 1}/{len(prompts)}...")
            results.append(self.query(prompt, use_injection=use_injection))
        return results

    def save(self, path: str | Path) -> None:
        """Save memory store to disk."""
        path = Path(path)

        # Save vectors
        vectors = {
            f"query_{i}": np.array(e.query_vector)
            for i, e in enumerate(self._entries)
            if e.query_vector is not None
        }
        vectors.update(
            {
                f"value_{i}": np.array(e.value_vector)
                for i, e in enumerate(self._entries)
                if e.value_vector is not None
            }
        )

        np.savez(path.with_suffix(".npz"), **vectors)

        # Save metadata
        metadata = {
            "config": {
                "query_layer": self._memory_config.query_layer,
                "inject_layer": self._memory_config.inject_layer,
                "value_layer": self._memory_config.value_layer,
                "similarity_threshold": self._memory_config.similarity_threshold,
                "blend": self._memory_config.blend,
            },
            "entries": [
                {
                    "query": e.query,
                    "answer": e.answer,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(self._entries)} entries to {path}")

    def load(self, path: str | Path) -> None:
        """Load memory store from disk."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)

        # Update config
        cfg = metadata["config"]
        self._memory_config = MemoryConfig(
            query_layer=cfg["query_layer"],
            inject_layer=cfg["inject_layer"],
            value_layer=cfg["value_layer"],
            similarity_threshold=cfg["similarity_threshold"],
            blend=cfg["blend"],
        )

        # Load vectors
        vectors = np.load(path.with_suffix(".npz"))

        # Reconstruct entries
        self._entries = []
        for i, entry_data in enumerate(metadata["entries"]):
            entry = MemoryEntry(
                query=entry_data["query"],
                answer=entry_data["answer"],
                metadata=entry_data.get("metadata", {}),
            )

            if f"query_{i}" in vectors:
                entry.query_vector = mx.array(vectors[f"query_{i}"])
            if f"value_{i}" in vectors:
                entry.value_vector = mx.array(vectors[f"value_{i}"])

            self._entries.append(entry)

        print(f"Loaded {len(self._entries)} entries from {path}")

    def evaluate(
        self,
        test_facts: list[dict],
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate memory injection on test facts.

        Args:
            test_facts: List of {"query": str, "answer": str}
            verbose: Print results

        Returns:
            Evaluation metrics
        """
        results = {
            "total": len(test_facts),
            "baseline_correct": 0,
            "injected_correct": 0,
            "rescued": 0,  # Wrong baseline, correct injected
            "broken": 0,  # Correct baseline, wrong injected
        }

        for fact in test_facts:
            result = self.query(fact["query"], use_injection=True, force_injection=True)
            expected = fact["answer"]

            baseline_correct = result.baseline_answer.strip() == expected
            injected_correct = (
                result.injected_answer is not None and result.injected_answer.strip() == expected
            )

            if baseline_correct:
                results["baseline_correct"] += 1
            if injected_correct:
                results["injected_correct"] += 1
            if not baseline_correct and injected_correct:
                results["rescued"] += 1
            if baseline_correct and not injected_correct:
                results["broken"] += 1

            if verbose:
                status = ""
                if not baseline_correct and injected_correct:
                    status = " [RESCUED]"
                elif baseline_correct and not injected_correct:
                    status = " [BROKEN]"

                print(
                    f"  {fact['query']} -> expected={expected}, "
                    f"baseline={result.baseline_answer.strip()}, "
                    f"injected={result.injected_answer.strip() if result.injected_answer else 'N/A'}"
                    f"{status}"
                )

        results["baseline_accuracy"] = results["baseline_correct"] / results["total"]
        results["injected_accuracy"] = results["injected_correct"] / results["total"]

        return results


def demo():
    """Quick demo of external memory."""
    print("=" * 70)
    print("External Memory Injection Demo")
    print("=" * 70)

    # Create memory system
    memory = ExternalMemory.from_pretrained("openai/gpt-oss-20b")

    # Add multiplication table
    memory.add_multiplication_table(2, 9)

    print("\n" + "=" * 70)
    print("Testing Standard Queries")
    print("=" * 70)

    standard_queries = ["7*8=", "6*7=", "9*9=", "3*4="]
    for q in standard_queries:
        result = memory.query(q)
        print(f"\n{q}")
        print(f"  Baseline: {result.baseline_answer} ({result.baseline_confidence:.1%})")
        if result.used_injection:
            print(f"  Injected: {result.injected_answer} ({result.injected_confidence:.1%})")
            print(f"  Match: {result.matched_entry.query} (sim={result.similarity:.3f})")

    print("\n" + "=" * 70)
    print("Testing Non-Standard Queries (Rescue Test)")
    print("=" * 70)

    rescue_queries = [
        "7 * 8 =",
        "seven times eight equals",
        "7×8=",
    ]

    for q in rescue_queries:
        result = memory.query(q, force_injection=True)
        print(f"\n'{q}'")
        print(f"  Baseline: {result.baseline_answer} ({result.baseline_confidence:.1%})")
        if result.used_injection:
            print(f"  Injected: {result.injected_answer} ({result.injected_confidence:.1%})")
            print(
                f"  Match: {result.matched_entry.query} -> {result.matched_entry.answer} (sim={result.similarity:.3f})"
            )

    print("\n" + "=" * 70)
    print("Override Test (Inject Wrong Answers)")
    print("=" * 70)

    # Manually inject wrong answer
    entry_12 = next(e for e in memory._entries if e.answer == "12")
    result = memory._forward_with_injection(
        "7*8=",
        inject_layer=memory._memory_config.inject_layer,
        inject_vector=entry_12.value_vector,
        blend=1.0,
    )
    print(f"\n7*8= with '12' vector injected: {result[0]} ({result[1]:.1%})")


if __name__ == "__main__":
    demo()
