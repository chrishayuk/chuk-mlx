"""
Attention-as-Router Decomposition Experiment.

Tests: Does attention dominate MoE routing universally (True MoE and Pseudo-MoE)?

Hypothesis space:
- H1: Attention dominates universally (>85% for both True and Pseudo MoE)
- H2: True MoE is more balanced (50-80% attention vs 96% in Pseudo MoE)
- H3: True MoE is fundamentally different (token embedding dominates)

Method:
1. Decompose router input: hidden = embed + attention_delta
2. Project each component through router weights
3. Measure relative contribution to routing decision
4. Test context sensitivity (same token, different context -> different expert?)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.experiments import ExperimentBase
from chuk_lazarus.introspection.moe.detector import (
    detect_moe_architecture,
    get_moe_layers,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Result of decomposing router signal for a single prompt."""

    prompt: str
    layer: int
    position: int  # Token position analyzed

    # Norms in router space
    embed_norm: float  # ||embed @ router_weights.T||
    attention_norm: float  # ||attention_delta @ router_weights.T||
    total_norm: float  # ||hidden @ router_weights.T||

    # Ratios
    attention_ratio: float  # attention_norm / (embed_norm + attention_norm)

    # Cosine similarity between components
    embed_attention_cosine: float  # How aligned are the components?

    # Top expert from each component
    embed_top_expert: int
    attention_top_expert: int
    combined_top_expert: int

    # Agreement (do components agree on routing?)
    components_agree: bool


@dataclass
class ContextSensitivityResult:
    """Result of testing context sensitivity for a token."""

    target_token: str
    contexts: list[tuple[str, str]]  # (context_name, full_prompt)
    layer: int

    # Per-context results
    context_experts: dict[str, int]  # context_name -> top expert
    context_weights: dict[str, list[float]]  # context_name -> expert weights

    # Analysis
    unique_experts: int  # How many different experts were selected
    is_context_sensitive: bool  # Different contexts -> different experts?


@dataclass
class LayerDecomposition:
    """Aggregated decomposition results for a layer."""

    layer: int
    num_prompts: int

    # Average ratios
    mean_attention_ratio: float
    std_attention_ratio: float

    # Average norms
    mean_embed_norm: float
    mean_attention_norm: float

    # Agreement rate
    component_agreement_rate: float

    # Per-prompt results
    results: list[DecompositionResult] = field(default_factory=list)


class AttentionRouterExperiment(ExperimentBase):
    """
    Test whether attention dominates MoE routing universally.

    Compares True MoE (OLMoE) to Pseudo-MoE (GPT-OSS) to determine if
    the 96% attention contribution found in GPT-OSS is universal or
    architecture-specific.
    """

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up attention-router decomposition experiment...")
        self.params = self.config.parameters
        self.results: dict[str, list[LayerDecomposition]] = {}
        self.context_results: dict[str, list[ContextSensitivityResult]] = {}

    def run(self) -> dict:
        """Run the experiment."""
        self.log("=" * 70)
        self.log("ATTENTION-AS-ROUTER DECOMPOSITION EXPERIMENT")
        self.log("Does attention drive MoE routing universally?")
        self.log("=" * 70)

        # Get models to test
        models = self.params.get("models", [self.config.model])

        all_results = {}

        for model_name in models:
            self.log(f"\n{'=' * 70}")
            self.log(f"ANALYZING: {model_name}")
            self.log("=" * 70)

            try:
                decomp_results, context_results = self._analyze_model(model_name)
                self.results[model_name] = decomp_results
                self.context_results[model_name] = context_results
                all_results[model_name] = {
                    "decomposition": self._summarize_decomposition(decomp_results),
                    "context_sensitivity": self._summarize_context_sensitivity(context_results),
                }
            except Exception as e:
                self.log(f"ERROR analyzing {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}

        # Compare models
        comparison = self._compare_models(all_results)

        return {
            "models": all_results,
            "comparison": comparison,
            "hypothesis_evaluation": self._evaluate_hypotheses(all_results),
        }

    def _analyze_model(
        self, model_name: str
    ) -> tuple[list[LayerDecomposition], list[ContextSensitivityResult]]:
        """Analyze a single model."""
        from chuk_lazarus.models_v2.loader import load_model

        self.log(f"Loading {model_name}...")
        loaded = load_model(model_name)
        model = loaded.model
        tokenizer = loaded.tokenizer

        # Detect MoE architecture
        architecture = detect_moe_architecture(model)
        moe_layers = get_moe_layers(model)

        self.log(f"Architecture: {architecture}")
        self.log(f"MoE layers: {moe_layers}")

        if not moe_layers:
            raise ValueError("No MoE layers detected")

        # Select layers to analyze
        target_layers = self._select_layers(moe_layers)
        self.log(f"Target layers for analysis: {target_layers}")

        # Run decomposition analysis
        decomp_results = self._run_decomposition(model, tokenizer, target_layers, model_name)

        # Run context sensitivity analysis
        context_results = self._run_context_sensitivity(model, tokenizer, target_layers, model_name)

        return decomp_results, context_results

    def _select_layers(self, moe_layers: list[int]) -> list[int]:
        """Select representative layers for analysis."""
        explicit_layers = self.params.get("layers")
        if explicit_layers:
            return [l for l in explicit_layers if l in moe_layers]

        # Default: early, middle, late
        if len(moe_layers) >= 3:
            return [
                moe_layers[0],
                moe_layers[len(moe_layers) // 2],
                moe_layers[-1],
            ]
        return list(moe_layers)

    def _run_decomposition(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: list[int],
        model_name: str,
    ) -> list[LayerDecomposition]:
        """Run router signal decomposition across prompts and layers."""
        test_prompts = self.params.get(
            "test_prompts",
            [
                # Arithmetic
                "127 + 45 =",
                "999 * 3 =",
                # Code
                "def fibonacci(n):",
                "import numpy as np",
                # Language
                "The capital of France is",
                "A synonym for happy is",
                # Mixed
                "Calculate: 100 + 200 + 300 =",
                "In Python, compute 45 * 37:",
            ],
        )

        results_by_layer: dict[int, list[DecompositionResult]] = defaultdict(list)

        for prompt in test_prompts:
            self.log(f"  Decomposing: {prompt[:40]}...")

            for layer_idx in target_layers:
                result = self._decompose_router_signal(model, tokenizer, prompt, layer_idx)
                if result:
                    results_by_layer[layer_idx].append(result)
                    self.log(
                        f"    L{layer_idx}: attention={result.attention_ratio:.1%}, "
                        f"agree={result.components_agree}"
                    )

        # Aggregate by layer
        layer_decompositions = []
        for layer_idx in sorted(results_by_layer.keys()):
            layer_results = results_by_layer[layer_idx]
            if not layer_results:
                continue

            attention_ratios = [r.attention_ratio for r in layer_results]
            embed_norms = [r.embed_norm for r in layer_results]
            attention_norms = [r.attention_norm for r in layer_results]
            agreements = [r.components_agree for r in layer_results]

            layer_decompositions.append(
                LayerDecomposition(
                    layer=layer_idx,
                    num_prompts=len(layer_results),
                    mean_attention_ratio=sum(attention_ratios) / len(attention_ratios),
                    std_attention_ratio=self._std(attention_ratios),
                    mean_embed_norm=sum(embed_norms) / len(embed_norms),
                    mean_attention_norm=sum(attention_norms) / len(attention_norms),
                    component_agreement_rate=sum(agreements) / len(agreements),
                    results=layer_results,
                )
            )

        return layer_decompositions

    def _decompose_router_signal(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        layer_idx: int,
    ) -> DecompositionResult | None:
        """Decompose router input into embedding vs attention contribution."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        seq_len = input_ids.shape[1]

        # Get model components
        model_layers = self._get_model_layers(model)
        if layer_idx >= len(model_layers):
            return None

        # Get token embeddings
        embed = self._get_embeddings(model, input_ids)
        if embed is None:
            return None

        # Forward through layers up to target layer to get hidden state
        hidden = embed
        for i, layer in enumerate(model_layers):
            if i == layer_idx:
                break
            layer_out = layer(hidden, mask=None, cache=None)
            if hasattr(layer_out, "hidden_states"):
                hidden = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out

        # Get router weights from target layer
        layer = model_layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None

        router = getattr(mlp, "router", None) or getattr(mlp, "gate", None)
        if router is None or not hasattr(router, "weight"):
            return None

        router_weight = router.weight  # [num_experts, hidden_dim]

        # Analyze last token position (most relevant for next-token prediction)
        pos = seq_len - 1

        # Extract vectors at position
        embed_vec = embed[0, pos, :]  # [hidden_dim]
        hidden_vec = hidden[0, pos, :]  # [hidden_dim]
        attention_delta = hidden_vec - embed_vec  # What layers 0..L-1 added

        # Project through router
        router_from_embed = embed_vec @ router_weight.T  # [num_experts]
        router_from_attention = attention_delta @ router_weight.T  # [num_experts]
        router_from_hidden = hidden_vec @ router_weight.T  # [num_experts]

        # Compute norms
        embed_norm = float(mx.linalg.norm(router_from_embed))
        attention_norm = float(mx.linalg.norm(router_from_attention))
        total_norm = float(mx.linalg.norm(router_from_hidden))

        # Attention ratio
        denom = embed_norm + attention_norm
        attention_ratio = attention_norm / denom if denom > 0 else 0.5

        # Cosine similarity between components in router space
        if embed_norm > 0 and attention_norm > 0:
            cosine = float(
                mx.sum(router_from_embed * router_from_attention) / (embed_norm * attention_norm)
            )
        else:
            cosine = 0.0

        # Top experts from each component
        embed_top = int(mx.argmax(router_from_embed))
        attention_top = int(mx.argmax(router_from_attention))
        combined_top = int(mx.argmax(router_from_hidden))

        mx.eval(
            router_from_embed,
            router_from_attention,
            router_from_hidden,
        )

        return DecompositionResult(
            prompt=prompt,
            layer=layer_idx,
            position=pos,
            embed_norm=embed_norm,
            attention_norm=attention_norm,
            total_norm=total_norm,
            attention_ratio=attention_ratio,
            embed_attention_cosine=cosine,
            embed_top_expert=embed_top,
            attention_top_expert=attention_top,
            combined_top_expert=combined_top,
            components_agree=(embed_top == attention_top),
        )

    def _run_context_sensitivity(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: list[int],
        model_name: str,
    ) -> list[ContextSensitivityResult]:
        """Test if same token routes differently based on context."""
        # Test contexts for specific tokens
        context_tests = self.params.get(
            "context_tests",
            [
                {
                    "target": "127",
                    "contexts": [
                        ("numbers", "111 127"),
                        ("letters", "abc 127"),
                        ("words", "The number 127"),
                        ("operator", "= 127"),
                    ],
                },
                {
                    "target": "+",
                    "contexts": [
                        ("math", "3 + 5"),
                        ("code", "x + y"),
                        ("concat", '"a" + "b"'),
                        ("increment", "count +="),
                    ],
                },
            ],
        )

        results = []

        for test in context_tests:
            target = test["target"]
            contexts = test["contexts"]

            self.log(f"  Context sensitivity for '{target}'...")

            for layer_idx in target_layers:
                context_experts: dict[str, int] = {}
                context_weights: dict[str, list[float]] = {}

                for context_name, full_prompt in contexts:
                    expert, weights = self._get_routing_for_token(
                        model, tokenizer, full_prompt, target, layer_idx
                    )
                    if expert is not None:
                        context_experts[context_name] = expert
                        context_weights[context_name] = weights

                if context_experts:
                    unique_experts = len(set(context_experts.values()))
                    is_sensitive = unique_experts > 1

                    results.append(
                        ContextSensitivityResult(
                            target_token=target,
                            contexts=contexts,
                            layer=layer_idx,
                            context_experts=context_experts,
                            context_weights=context_weights,
                            unique_experts=unique_experts,
                            is_context_sensitive=is_sensitive,
                        )
                    )

                    self.log(
                        f"    L{layer_idx}: {unique_experts} unique experts, "
                        f"sensitive={is_sensitive}"
                    )

        return results

    def _get_routing_for_token(
        self,
        model: nn.Module,
        tokenizer,
        prompt: str,
        target_token: str,
        layer_idx: int,
    ) -> tuple[int | None, list[float]]:
        """Get routing decision for a specific token in a prompt."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        # Find target token position
        target_pos = None
        for i, tok in enumerate(tokens):
            if target_token in tok:
                target_pos = i
                break

        if target_pos is None:
            return None, []

        # Forward to layer and get routing
        model_layers = self._get_model_layers(model)
        if layer_idx >= len(model_layers):
            return None, []

        hidden = self._get_embeddings(model, input_ids)
        if hidden is None:
            return None, []

        for i, layer in enumerate(model_layers):
            if i == layer_idx:
                break
            layer_out = layer(hidden, mask=None, cache=None)
            if hasattr(layer_out, "hidden_states"):
                hidden = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                hidden = layer_out[0]
            else:
                hidden = layer_out

        # Get router
        layer = model_layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return None, []

        router = getattr(mlp, "router", None) or getattr(mlp, "gate", None)
        if router is None or not hasattr(router, "weight"):
            return None, []

        # Compute routing
        hidden_vec = hidden[0, target_pos, :]
        logits = hidden_vec @ router.weight.T
        if hasattr(router, "bias") and router.bias is not None:
            logits = logits + router.bias

        probs = mx.softmax(logits, axis=-1)
        top_expert = int(mx.argmax(probs))
        weights = probs.tolist()

        mx.eval(probs)

        return top_expert, weights

    def _get_model_layers(self, model) -> list:
        """Get transformer layers from model."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(model, attr, None)
            if submodel is not None:
                layers = getattr(submodel, "layers", None)
                if layers is not None:
                    return list(layers)
        return list(getattr(model, "layers", []))

    def _get_embeddings(self, model, input_ids: mx.array) -> mx.array | None:
        """Get token embeddings from model."""
        # Try common paths
        for path in [
            ("model", "embed_tokens"),
            ("transformer", "wte"),
            ("decoder", "embed_tokens"),
        ]:
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    return obj(input_ids)
                except Exception:
                    pass

        # Direct attribute
        embed = getattr(model, "embed_tokens", None)
        if embed is not None:
            try:
                return embed(input_ids)
            except Exception:
                pass

        return None

    def _std(self, values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance**0.5

    def _summarize_decomposition(self, results: list[LayerDecomposition]) -> dict:
        """Summarize decomposition results."""
        if not results:
            return {}

        summary = {
            "num_layers": len(results),
            "layers": {},
            "overall_attention_ratio": 0.0,
        }

        total_ratio = 0.0
        for layer in results:
            summary["layers"][f"L{layer.layer}"] = {
                "attention_ratio": layer.mean_attention_ratio,
                "attention_ratio_std": layer.std_attention_ratio,
                "embed_norm": layer.mean_embed_norm,
                "attention_norm": layer.mean_attention_norm,
                "agreement_rate": layer.component_agreement_rate,
            }
            total_ratio += layer.mean_attention_ratio

        summary["overall_attention_ratio"] = total_ratio / len(results)
        return summary

    def _summarize_context_sensitivity(self, results: list[ContextSensitivityResult]) -> dict:
        """Summarize context sensitivity results."""
        if not results:
            return {}

        sensitive_count = sum(1 for r in results if r.is_context_sensitive)
        total = len(results)

        by_token: dict[str, list[ContextSensitivityResult]] = defaultdict(list)
        for r in results:
            by_token[r.target_token].append(r)

        return {
            "sensitivity_rate": sensitive_count / total if total > 0 else 0,
            "sensitive_count": sensitive_count,
            "total_tests": total,
            "by_token": {
                token: {
                    "layers_tested": len(tests),
                    "sensitive_layers": sum(1 for t in tests if t.is_context_sensitive),
                }
                for token, tests in by_token.items()
            },
        }

    def _compare_models(self, all_results: dict) -> dict:
        """Compare results across models."""
        comparison = {"models": {}}

        for model_name, results in all_results.items():
            if "error" in results:
                comparison["models"][model_name] = {"error": results["error"]}
                continue

            decomp = results.get("decomposition", {})
            context = results.get("context_sensitivity", {})

            comparison["models"][model_name] = {
                "attention_ratio": decomp.get("overall_attention_ratio", 0),
                "context_sensitivity": context.get("sensitivity_rate", 0),
            }

        # Determine pattern
        ratios = [
            v["attention_ratio"] for v in comparison["models"].values() if "attention_ratio" in v
        ]

        if ratios:
            comparison["min_attention_ratio"] = min(ratios)
            comparison["max_attention_ratio"] = max(ratios)
            comparison["range"] = max(ratios) - min(ratios)

        return comparison

    def _evaluate_hypotheses(self, all_results: dict) -> dict:
        """Evaluate which hypothesis is supported."""
        # Extract attention ratios
        model_ratios = {}
        for model_name, results in all_results.items():
            if "error" in results:
                continue
            decomp = results.get("decomposition", {})
            ratio = decomp.get("overall_attention_ratio", 0)
            model_ratios[model_name] = ratio

        if not model_ratios:
            return {"conclusion": "INSUFFICIENT_DATA", "reason": "No valid results"}

        min_ratio = min(model_ratios.values())
        max_ratio = max(model_ratios.values())

        # H1: Attention dominates universally (>85% for all)
        if min_ratio > 0.85:
            return {
                "conclusion": "H1_SUPPORTED",
                "hypothesis": "Attention dominates universally",
                "evidence": f"All models show >{int(min_ratio * 100)}% attention contribution",
                "implication": (
                    "Router is redundant in ALL MoE architectures. "
                    "Attention-gated-subspace applies everywhere."
                ),
                "model_ratios": model_ratios,
            }

        # H2: True MoE more balanced (50-80% attention)
        if 0.50 <= min_ratio <= 0.80:
            return {
                "conclusion": "H2_SUPPORTED",
                "hypothesis": "True MoE uses more token embedding signal",
                "evidence": (
                    f"True MoE shows {int(min_ratio * 100)}% attention "
                    f"vs Pseudo-MoE {int(max_ratio * 100)}%"
                ),
                "implication": (
                    "Different optimization strategies needed per architecture type. "
                    "Pseudo-MoE can simplify routing; True MoE router has purpose."
                ),
                "model_ratios": model_ratios,
            }

        # H3: True MoE fundamentally different (token dominates, <50%)
        if min_ratio < 0.50:
            return {
                "conclusion": "H3_SUPPORTED",
                "hypothesis": "True MoE uses fundamentally different routing",
                "evidence": (
                    f"Token embedding dominates in True MoE ({int(min_ratio * 100)}% attention)"
                ),
                "implication": (
                    "Cannot generalize Pseudo-MoE findings. "
                    "Different architectures, different routing dynamics."
                ),
                "model_ratios": model_ratios,
            }

        # Unclear case
        return {
            "conclusion": "UNCLEAR",
            "reason": f"Attention ratios ({min_ratio:.1%}-{max_ratio:.1%}) don't clearly support any hypothesis",
            "model_ratios": model_ratios,
        }

    def evaluate(self) -> dict:
        """Return summary metrics."""
        latest = self.load_latest_results("results")
        if not latest:
            return {"error": "No results"}

        return {
            "conclusion": latest.get("hypothesis_evaluation", {}).get("conclusion", "Unknown"),
            "comparison": latest.get("comparison", {}),
        }

    def cleanup(self) -> None:
        """Cleanup."""
        self.results = {}
        self.context_results = {}
