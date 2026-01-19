"""
MoE Routing Correlation Experiment.

Tests: Does MoE architecture create pressure for vocabulary-aligned task representations?

Hypothesis: MoE routing requires discrete decisions, which might force vocabulary-aligned
task representations to emerge naturally - no special training objective needed.

Key measurements:
1. Logit lens on OLMoE at intermediate layers - do operation tokens surface?
2. Correlation between expert selection and task token probability
3. Comparison to dense baseline (Llama-3.2-1B shows ~0% at intermediate layers)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Analysis results for a single prompt."""

    input: str
    task: str
    expected: str

    # Logit lens results per layer
    # layer_idx -> {token: probability}
    vocab_probs: dict[int, dict[str, float]] = field(default_factory=dict)

    # Max task token probability per layer
    max_task_prob: dict[int, float] = field(default_factory=dict)

    # Expert routing per layer
    # layer_idx -> list of (position, [expert_indices], [weights])
    routing: dict[int, list[tuple[int, list[int], list[float]]]] = field(default_factory=dict)

    # Generated output
    generated: str = ""
    answer_correct: bool = False


@dataclass
class ExpertTaskCorrelation:
    """Correlation between expert and task type."""

    expert_idx: int
    task: str
    activation_count: int
    total_task_prompts: int
    activation_rate: float
    avg_task_prob_when_active: float


class MoERoutingCorrelationExperiment(ExperimentBase):
    """
    Test if MoE routing creates vocabulary-aligned classifiers.

    Compares OLMoE-1B-7B (MoE) to Llama-3.2-1B (dense) to isolate
    the effect of MoE architecture on vocabulary alignment.
    """

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up MoE routing correlation experiment...")
        self.params = self.config.parameters
        self.results: list[PromptAnalysis] = []
        self.baseline_results: list[PromptAnalysis] = []

    def run(self) -> dict:
        """Run the experiment."""
        self.log("=" * 70)
        self.log("MOE ROUTING CORRELATION EXPERIMENT")
        self.log("Does MoE architecture create vocabulary-aligned classifiers?")
        self.log("=" * 70)

        moe_results = {"error": "skipped"}
        baseline_results = {"error": "skipped"}

        # Check if we should skip MoE (large model)
        skip_moe = self.params.get("skip_moe", False)

        if not skip_moe:
            # Run on MoE model
            self.log("\n" + "=" * 70)
            self.log(f"PHASE 1: {self.config.model} (MoE model)")
            self.log("=" * 70)
            moe_results = self._analyze_model(
                self.config.model,
                self.params.get("target_layers", [4, 6, 8, 10, 12, 14]),
                is_moe=True,
            )
        else:
            self.log("\n" + "=" * 70)
            self.log("PHASE 1: SKIPPED (skip_moe=True)")
            self.log("=" * 70)

        # Run on baseline (dense) model
        baseline_model = self.params.get("baseline_model", "meta-llama/Llama-3.2-1B")
        self.log("\n" + "=" * 70)
        self.log(f"PHASE 2: {baseline_model} (Dense baseline)")
        self.log("=" * 70)
        baseline_results = self._analyze_model(
            baseline_model,
            self.params.get("baseline_layers", [4, 6, 8, 10, 12, 14]),
            is_moe=False,
        )

        # Build comparison
        return self._build_comparison(moe_results, baseline_results)

    def _analyze_model(
        self,
        model_name: str,
        layers: list[int],
        is_moe: bool,
    ) -> dict:
        """Analyze a single model."""
        from chuk_lazarus.models_v2.loader import load_model

        self.log(f"\nLoading {model_name}...")

        try:
            loaded = load_model(model_name)
        except Exception as e:
            self.log(f"Failed to load model: {e}")
            return {"error": str(e)}

        model = loaded.model
        tokenizer = loaded.tokenizer
        num_layers = loaded.config.num_hidden_layers

        self.log(f"Model loaded: {num_layers} layers")

        # Check if this is actually MoE
        has_moe = self._detect_moe(model)
        self.log(f"MoE detected: {has_moe}")

        if is_moe and not has_moe:
            self.log("WARNING: Expected MoE model but no MoE layers detected!")

        # Get embed tokens for logit lens
        embed_weight = self._get_embed_weight(model)
        if embed_weight is None:
            self.log("ERROR: Could not find embedding weights")
            return {"error": "No embedding weights"}

        # Prepare task tokens
        task_tokens = self.params.get("task_tokens", {})
        task_token_ids = self._resolve_task_tokens(tokenizer, task_tokens)

        # Analyze prompts
        prompts = self.params.get("test_prompts", [])
        self.log(f"\nAnalyzing {len(prompts)} prompts across layers {layers}...")

        results = []
        for prompt_info in prompts:
            input_text = prompt_info["input"]
            task = prompt_info["task"]
            expected = prompt_info["expected"]

            self.log(f"\n  [{task}] {input_text}")

            analysis = PromptAnalysis(
                input=input_text,
                task=task,
                expected=expected,
            )

            # Run logit lens
            vocab_probs, routing = self._run_logit_lens(
                model,
                tokenizer,
                embed_weight,
                input_text,
                layers,
                task_token_ids,
                capture_routing=is_moe and has_moe,
            )

            analysis.vocab_probs = vocab_probs
            analysis.routing = routing

            # Compute max task token probability per layer
            task_tokens_for_this = task_token_ids.get(task, {})
            for layer_idx, probs in vocab_probs.items():
                max_prob = 0.0
                for token, prob in probs.items():
                    if token in task_tokens_for_this.values():
                        max_prob = max(max_prob, prob)
                    # Also check by token string
                    for tok_str, tok_id in task_tokens_for_this.items():
                        if token == tok_str:
                            max_prob = max(max_prob, prob)
                analysis.max_task_prob[layer_idx] = max_prob
                self.log(f"    L{layer_idx}: max task prob = {max_prob:.1%}")

            # Generate and check answer
            analysis.generated = self._generate(model, tokenizer, input_text)
            analysis.answer_correct = expected in analysis.generated
            self.log(f"    Generated: {analysis.generated[:40]}...")
            self.log(f"    Correct: {analysis.answer_correct}")

            results.append(analysis)

        return self._summarize_results(results, is_moe, has_moe)

    def _detect_moe(self, model) -> bool:
        """Detect if model has MoE layers."""
        layers = self._get_model_layers(model)
        for layer in layers:
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "router") or hasattr(mlp, "gate"):
                    return True
                if hasattr(mlp, "experts"):
                    return True
        return False

    def _get_model_layers(self, model) -> list:
        """Get transformer layers from model."""
        for attr in ["model", "transformer", "decoder"]:
            submodel = getattr(model, attr, None)
            if submodel is not None:
                layers = getattr(submodel, "layers", None)
                if layers is not None:
                    return list(layers)
        return list(getattr(model, "layers", []))

    def _get_embed_weight(self, model) -> mx.array | None:
        """Get embedding weights for logit lens."""
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
                # Navigate through wrapper layers to find the actual weight array
                # Could be: obj.weight (direct), obj.weight.weight (wrapped), etc.
                weight = self._extract_weight_array(obj)
                if weight is not None:
                    return weight

        # Direct embedding
        embed = getattr(model, "embed_tokens", None)
        if embed is not None:
            weight = self._extract_weight_array(embed)
            if weight is not None:
                return weight

        return None

    def _extract_weight_array(self, obj) -> mx.array | None:
        """Extract the actual weight array from an embedding module."""
        # Try direct weight attribute
        if hasattr(obj, "weight"):
            weight = obj.weight
            if isinstance(weight, mx.array):
                return weight
            # Weight might be another module (e.g., TokenEmbedding.weight is Embedding)
            if hasattr(weight, "weight"):
                inner = weight.weight
                if isinstance(inner, mx.array):
                    return inner
        # Try parameters dict
        try:
            params = obj.parameters()
            if isinstance(params, dict):
                # Look for weight in nested structure
                if "weight" in params:
                    w = params["weight"]
                    if isinstance(w, mx.array):
                        return w
                    if isinstance(w, dict) and "weight" in w:
                        if isinstance(w["weight"], mx.array):
                            return w["weight"]
        except Exception:
            pass
        return None

    def _get_norm(self, model):
        """Get the final layer norm."""
        for path in [("model", "norm"), ("transformer", "ln_f"), ("decoder", "norm")]:
            obj = model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def _resolve_task_tokens(self, tokenizer, task_tokens: dict) -> dict[str, dict[str, int]]:
        """Resolve task token strings to IDs."""
        result = {}
        for task, tokens in task_tokens.items():
            result[task] = {}
            for token in tokens:
                try:
                    # Try encoding with and without space prefix
                    for variant in [token, f" {token}", f"▁{token}"]:
                        encoded = tokenizer.encode(variant, add_special_tokens=False)
                        if encoded:
                            result[task][variant] = encoded[0]
                except Exception:
                    pass
        return result

    def _run_logit_lens(
        self,
        model,
        tokenizer,
        embed_weight: mx.array,
        prompt: str,
        layers: list[int],
        task_token_ids: dict[str, dict[str, int]],
        capture_routing: bool = False,
    ) -> tuple[dict[int, dict[str, float]], dict]:
        """Run logit lens and optionally capture routing."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Get model components
        model_layers = self._get_model_layers(model)
        norm = self._get_norm(model)

        # Forward through embedding
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            h = model.model.embed_tokens(input_ids)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            h = model.transformer.wte(input_ids)
        else:
            h = model.embed_tokens(input_ids)

        vocab_probs: dict[int, dict[str, float]] = {}
        routing: dict[int, list] = {}

        # All task token IDs we care about
        all_task_ids = set()
        for task_tokens in task_token_ids.values():
            all_task_ids.update(task_tokens.values())

        # Forward through layers
        for i, layer in enumerate(model_layers):
            # Run layer
            layer_out = layer(h, mask=None, cache=None)

            # Handle different output formats
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

            if i in layers:
                # Project to vocabulary via logit lens
                if norm is not None:
                    h_normed = norm(h)
                else:
                    h_normed = h

                logits = h_normed @ embed_weight.T
                probs = mx.softmax(logits[0, -1, :], axis=-1)
                mx.eval(probs)

                # Get top-k tokens
                top_k = self.params.get("top_k_vocab", 20)
                top_indices = mx.argsort(probs)[-top_k:][::-1].tolist()

                layer_probs = {}
                for idx in top_indices:
                    token_str = tokenizer.decode([idx])
                    layer_probs[token_str] = float(probs[idx])

                # Also get probabilities for all task tokens
                for task_id in all_task_ids:
                    if task_id < probs.shape[0]:
                        token_str = tokenizer.decode([task_id])
                        layer_probs[token_str] = float(probs[task_id])

                vocab_probs[i] = layer_probs

                # Capture routing if MoE
                if capture_routing:
                    routing[i] = self._capture_layer_routing(layer, h)

        return vocab_probs, routing

    def _capture_layer_routing(
        self, layer, hidden_states: mx.array
    ) -> list[tuple[int, list[int], list[float]]]:
        """Capture expert routing for a layer."""
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return []

        router = getattr(mlp, "router", None) or getattr(mlp, "gate", None)
        if router is None:
            return []

        # Compute router logits
        batch_size, seq_len, hidden_size = hidden_states.shape
        h_flat = hidden_states.reshape(-1, hidden_size)

        try:
            # Get router weights
            if hasattr(router, "weight"):
                router_logits = h_flat @ router.weight.T
                if hasattr(router, "bias") and router.bias is not None:
                    router_logits = router_logits + router.bias
            else:
                # Try calling router directly
                router_out = router(h_flat)
                if isinstance(router_out, tuple):
                    router_logits = router_out[0]
                else:
                    router_logits = router_out

            # Get top-k experts
            k = getattr(router, "num_experts_per_tok", 2)
            if hasattr(router, "top_k"):
                k = router.top_k

            top_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
            top_logits = mx.take_along_axis(router_logits, top_indices, axis=-1)
            weights = mx.softmax(top_logits, axis=-1)

            mx.eval(top_indices, weights)

            # Build result for last position (most relevant for next token)
            result = []
            for pos in range(seq_len):
                pos_indices = top_indices[pos].tolist()
                pos_weights = weights[pos].tolist()
                result.append((pos, pos_indices, pos_weights))

            return result

        except Exception as e:
            self.log(f"    Warning: Could not capture routing: {e}")
            return []

    def _generate(self, model, tokenizer, prompt: str) -> str:
        """Generate output for a prompt."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        max_tokens = 20

        generated_ids = []
        for _ in range(max_tokens):
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            token_id = int(next_token[0])
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return tokenizer.decode(generated_ids)

    def _summarize_results(
        self, results: list[PromptAnalysis], is_moe: bool, has_moe: bool
    ) -> dict:
        """Summarize results for a model."""
        if not results:
            return {"error": "No results"}

        # Aggregate by task
        by_task: dict[str, list[PromptAnalysis]] = defaultdict(list)
        for r in results:
            by_task[r.task].append(r)

        # Compute average task token probability per layer
        all_layers = set()
        for r in results:
            all_layers.update(r.max_task_prob.keys())

        layer_avg_probs = {}
        for layer in sorted(all_layers):
            probs = [r.max_task_prob.get(layer, 0) for r in results]
            layer_avg_probs[layer] = sum(probs) / len(probs) if probs else 0

        # Find peak layer
        peak_layer = max(layer_avg_probs.items(), key=lambda x: x[1]) if layer_avg_probs else (0, 0)

        # Compute accuracy
        correct = sum(1 for r in results if r.answer_correct)
        accuracy = correct / len(results)

        # Build task summary
        task_summary = {}
        for task, task_results in by_task.items():
            task_layer_probs = {}
            for layer in sorted(all_layers):
                probs = [r.max_task_prob.get(layer, 0) for r in task_results]
                task_layer_probs[f"L{layer}"] = sum(probs) / len(probs) if probs else 0

            task_correct = sum(1 for r in task_results if r.answer_correct)
            task_summary[task] = {
                "count": len(task_results),
                "accuracy": task_correct / len(task_results),
                "layer_probs": task_layer_probs,
            }

        # Expert correlation (if MoE)
        expert_correlation = {}
        if is_moe and has_moe:
            expert_correlation = self._compute_expert_task_correlation(results)

        summary = {
            "is_moe": is_moe,
            "has_moe_detected": has_moe,
            "num_prompts": len(results),
            "accuracy": accuracy,
            "layer_avg_task_prob": {f"L{k}": v for k, v in layer_avg_probs.items()},
            "peak_layer": peak_layer[0],
            "peak_prob": peak_layer[1],
            "by_task": task_summary,
            "expert_task_correlation": expert_correlation,
        }

        # Log summary
        self.log("\n--- Summary ---")
        self.log(f"Accuracy: {accuracy:.1%}")
        self.log(f"Peak vocab alignment: L{peak_layer[0]} = {peak_layer[1]:.1%}")
        for layer, prob in sorted(layer_avg_probs.items()):
            bar = "█" * int(prob * 50)
            self.log(f"  L{layer:2d}: {prob:5.1%} {bar}")

        return summary

    def _compute_expert_task_correlation(self, results: list[PromptAnalysis]) -> dict:
        """Compute correlation between experts and task types."""
        # Count expert activations by task
        expert_task_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        task_counts: dict[str, int] = defaultdict(int)

        for r in results:
            task_counts[r.task] += 1
            for layer_idx, routing in r.routing.items():
                for pos, expert_indices, weights in routing:
                    # Only count last position (most relevant)
                    if pos == len(routing) - 1:
                        for exp_idx in expert_indices:
                            expert_task_counts[r.task][exp_idx] += 1

        # Find experts that specialize in each task
        task_specialists: dict[str, list[tuple[int, float]]] = {}
        for task, expert_counts in expert_task_counts.items():
            total = task_counts[task]
            specialists = []
            for exp_idx, count in expert_counts.items():
                rate = count / total if total > 0 else 0
                if rate > 0.3:  # More than 30% of prompts use this expert
                    specialists.append((exp_idx, rate))
            specialists.sort(key=lambda x: -x[1])
            task_specialists[task] = specialists[:5]

        return {
            "task_specialists": {
                task: [{"expert": e, "rate": r} for e, r in specs]
                for task, specs in task_specialists.items()
            },
        }

    def _build_comparison(self, moe_results: dict, baseline_results: dict) -> dict:
        """Build final comparison between MoE and baseline."""
        self.log("\n" + "=" * 70)
        self.log("COMPARISON: MoE vs Dense")
        self.log("=" * 70)

        # Extract key metrics
        moe_peak = moe_results.get("peak_prob", 0)
        baseline_peak = baseline_results.get("peak_prob", 0)

        moe_layers = moe_results.get("layer_avg_task_prob", {})
        baseline_layers = baseline_results.get("layer_avg_task_prob", {})

        # Compute delta
        delta_by_layer = {}
        for layer in set(moe_layers.keys()) | set(baseline_layers.keys()):
            moe_val = moe_layers.get(layer, 0)
            baseline_val = baseline_layers.get(layer, 0)
            delta_by_layer[layer] = moe_val - baseline_val

        avg_delta = sum(delta_by_layer.values()) / len(delta_by_layer) if delta_by_layer else 0

        # Log comparison
        self.log(f"\nMoE peak vocab alignment: {moe_peak:.1%}")
        self.log(f"Dense peak vocab alignment: {baseline_peak:.1%}")
        self.log(f"Delta (MoE - Dense): {moe_peak - baseline_peak:+.1%}")

        self.log("\nLayer-by-layer comparison:")
        self.log("  Layer | MoE    | Dense  | Delta")
        self.log("  ------|--------|--------|-------")
        for layer in sorted(delta_by_layer.keys()):
            moe_val = moe_layers.get(layer, 0)
            baseline_val = baseline_layers.get(layer, 0)
            delta = delta_by_layer[layer]
            self.log(f"  {layer:5} | {moe_val:5.1%} | {baseline_val:5.1%} | {delta:+.1%}")

        # Interpret results
        self.log("\n" + "=" * 70)
        self.log("CONCLUSION")
        self.log("=" * 70)

        significant_threshold = self.params.get("significant_threshold", 0.10)

        if moe_peak > significant_threshold and baseline_peak < significant_threshold:
            conclusion = "HYPOTHESIS SUPPORTED"
            interpretation = (
                "MoE shows significant vocab alignment while dense does not. "
                "MoE architecture DOES create pressure for vocabulary-aligned representations."
            )
        elif moe_peak > baseline_peak + 0.05:
            conclusion = "PARTIAL SUPPORT"
            interpretation = (
                "MoE shows higher vocab alignment than dense, but both are low. "
                "MoE may have weak effect, or scale matters."
            )
        elif abs(moe_peak - baseline_peak) < 0.02:
            conclusion = "NO DIFFERENCE"
            interpretation = (
                "MoE and dense show similar (low) vocab alignment. "
                "MoE architecture alone does NOT create vocab-aligned classifiers."
            )
        else:
            conclusion = "UNEXPECTED"
            interpretation = (
                f"Dense shows higher vocab alignment than MoE ({baseline_peak:.1%} vs {moe_peak:.1%}). "
                "This contradicts the hypothesis - needs investigation."
            )

        self.log(f"\n>>> {conclusion}")
        self.log(f">>> {interpretation}")

        return {
            "moe_results": moe_results,
            "baseline_results": baseline_results,
            "comparison": {
                "moe_peak_vocab_alignment": moe_peak,
                "dense_peak_vocab_alignment": baseline_peak,
                "delta": moe_peak - baseline_peak,
                "avg_delta_by_layer": avg_delta,
                "delta_by_layer": delta_by_layer,
            },
            "conclusion": conclusion,
            "interpretation": interpretation,
        }

    def evaluate(self) -> dict:
        """Return summary metrics."""
        latest = self.load_latest_results("results")
        if not latest:
            return {"error": "No results"}

        return {
            "conclusion": latest.get("conclusion", "Unknown"),
            "moe_peak": latest.get("comparison", {}).get("moe_peak_vocab_alignment", 0),
            "dense_peak": latest.get("comparison", {}).get("dense_peak_vocab_alignment", 0),
            "delta": latest.get("comparison", {}).get("delta", 0),
        }

    def cleanup(self) -> None:
        """Cleanup."""
        self.results = []
        self.baseline_results = []
