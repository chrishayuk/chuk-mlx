"""
CoT Correlation Experiment

Tests: Does L13 vocabulary signal predict/gate CoT generation?

Hypothesis:
- L4 probe → task type (non-vocab-aligned) → routing
- L13 vocab → generation strategy (vocab-aligned?) → CoT vs direct

This would explain:
1. Why dual-reward breaks computation (training L8 interferes with L4)
2. Why GPT-OSS has vocab alignment at L13 (gates verbalization)
"""

import logging
import re
from dataclasses import dataclass, field

import mlx.core as mx

from chuk_lazarus.experiments import ExperimentBase

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Results for a single prompt."""

    input: str
    task: str
    expected: str
    format: str
    generated: str
    is_cot: bool
    answer_correct: bool
    vocab_probs: dict[int, float] = field(default_factory=dict)  # layer -> max prob


class CoTCorrelationExperiment(ExperimentBase):
    """Correlate L13 vocab alignment with CoT generation."""

    def setup(self) -> None:
        """Initialize experiment."""
        self.log("Setting up CoT correlation experiment...")
        self.params = self.config.parameters
        self.test_prompts = self.params.get("test_prompts", [])
        self.results: list[PromptResult] = []

    def run(self) -> dict:
        """Run the experiment."""
        from chuk_lazarus.models_v2.loader import load_model

        self.log("=" * 60)
        self.log("COT CORRELATION EXPERIMENT")
        self.log("Does L13 vocabulary signal predict CoT generation?")
        self.log("=" * 60)

        # Load model
        self.log(f"\nLoading {self.config.model}...")
        loaded = load_model(self.config.model)
        model = loaded.model
        tokenizer = loaded.tokenizer

        num_layers = loaded.config.num_hidden_layers
        self.log(f"Model layers: {num_layers}")

        # Get embed tokens weight for logit lens
        embed_weight = model.model.embed_tokens.weight.parameters()["weight"]

        # Get layers to measure
        measure_layers = self.params.get("measure_layers", [13])
        self.log(f"Measuring layers: {measure_layers}")

        task_tokens = self.params.get("task_tokens", {})
        cot_indicators = self.params.get("cot_indicators", [])

        self.log(f"\nAnalyzing {len(self.test_prompts)} prompts...")

        for prompt_info in self.test_prompts:
            input_text = prompt_info["input"]
            task = prompt_info["task"]
            expected = prompt_info["expected"]
            fmt = prompt_info["format"]

            self.log(f"\n  [{fmt}] {input_text}")

            # 1. Generate output
            generated = self._generate(model, tokenizer, input_text)
            self.log(f"    Output: {generated[:60]}...")

            # 2. Check if output is CoT or direct
            is_cot = self._is_cot_output(generated, cot_indicators)
            self.log(f"    Is CoT: {is_cot}")

            # 3. Check if answer is correct
            answer_correct = self._check_answer(generated, expected)
            self.log(f"    Correct: {answer_correct}")

            # 4. Measure vocab alignment at each layer
            vocab_probs = self._measure_vocab_alignment(
                model, tokenizer, embed_weight, input_text, task, task_tokens, measure_layers
            )
            for layer, prob in vocab_probs.items():
                self.log(f"    L{layer} vocab: {prob:.1%}")

            self.results.append(
                PromptResult(
                    input=input_text,
                    task=task,
                    expected=expected,
                    format=fmt,
                    generated=generated,
                    is_cot=is_cot,
                    answer_correct=answer_correct,
                    vocab_probs=vocab_probs,
                )
            )

        return self._build_results()

    def _generate(self, model, tokenizer, prompt: str) -> str:
        """Generate output for a prompt."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        max_tokens = self.params.get("max_tokens", 50)

        # Simple greedy generation (without cache for simplicity)
        generated_ids = []

        for _ in range(max_tokens):
            output = model(input_ids)
            # Handle ModelOutput wrapper from framework
            logits = output.logits if hasattr(output, "logits") else output

            # Get next token
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)

            token_id = int(next_token[0])
            if token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

        return tokenizer.decode(generated_ids)

    def _is_cot_output(self, output: str, indicators: list[str]) -> bool:
        """Check if output contains CoT indicators."""
        output_lower = output.lower()
        for indicator in indicators:
            if indicator.lower() in output_lower:
                return True
        return False

    def _check_answer(self, output: str, expected: str) -> bool:
        """Check if output contains the expected answer."""
        # Extract numbers from output
        numbers = re.findall(r"-?\d+", output)
        return expected in numbers

    def _measure_vocab_alignment(
        self,
        model,
        tokenizer,
        embed_weight,
        prompt: str,
        task: str,
        task_tokens: dict,
        layers: list[int],
    ) -> dict[int, float]:
        """Measure task token probability at each layer via logit lens."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]

        # Forward through embedding and layers
        h = model.model.embed_tokens(input_ids)

        layer_probs = {}
        for i, layer in enumerate(model.model.layers):
            layer_out = layer(h, mask=None, cache=None)
            h = (
                layer_out.hidden_states
                if hasattr(layer_out, "hidden_states")
                else (layer_out[0] if isinstance(layer_out, tuple) else layer_out)
            )

            if i in layers:
                # Project to vocabulary via logit lens
                h_normed = model.model.norm(h)
                logits = h_normed @ embed_weight.T
                probs = mx.softmax(logits[0, -1, :], axis=-1)
                mx.eval(probs)

                # Find max prob for task tokens
                max_prob = 0.0
                tokens_for_task = task_tokens.get(task, [])
                for token_word in tokens_for_task:
                    token_ids = tokenizer.encode(token_word)
                    for tid in token_ids:
                        if tid < probs.shape[0]:
                            prob = float(probs[tid])
                            max_prob = max(max_prob, prob)

                layer_probs[i] = max_prob

        return layer_probs

    def _build_results(self) -> dict:
        """Build results with correlation analysis."""
        # Organize by format
        by_format = {}
        for r in self.results:
            if r.format not in by_format:
                by_format[r.format] = []
            by_format[r.format].append(r)

        # Calculate correlations
        results = {
            "model": self.config.model,
            "by_format": {},
            "correlation": {},
        }

        all_l13_probs = []
        all_is_cot = []

        for fmt, prompts in by_format.items():
            cot_count = sum(1 for p in prompts if p.is_cot)
            correct_count = sum(1 for p in prompts if p.answer_correct)
            avg_l13 = sum(p.vocab_probs.get(13, 0) for p in prompts) / len(prompts)

            results["by_format"][fmt] = {
                "count": len(prompts),
                "cot_rate": cot_count / len(prompts),
                "accuracy": correct_count / len(prompts),
                "avg_L13_vocab": avg_l13,
            }

            for p in prompts:
                all_l13_probs.append(p.vocab_probs.get(13, 0))
                all_is_cot.append(1.0 if p.is_cot else 0.0)

            self.log(f"\n{fmt.upper()} format:")
            self.log(f"  CoT rate: {cot_count}/{len(prompts)} = {cot_count / len(prompts):.1%}")
            self.log(
                f"  Accuracy: {correct_count}/{len(prompts)} = {correct_count / len(prompts):.1%}"
            )
            self.log(f"  Avg L13 vocab: {avg_l13:.1%}")

        # Calculate correlation coefficient
        if len(all_l13_probs) > 1:
            # Simple Pearson correlation
            mean_vocab = sum(all_l13_probs) / len(all_l13_probs)
            mean_cot = sum(all_is_cot) / len(all_is_cot)

            numerator = sum(
                (v - mean_vocab) * (c - mean_cot) for v, c in zip(all_l13_probs, all_is_cot)
            )
            denom_vocab = sum((v - mean_vocab) ** 2 for v in all_l13_probs) ** 0.5
            denom_cot = sum((c - mean_cot) ** 2 for c in all_is_cot) ** 0.5

            if denom_vocab > 0 and denom_cot > 0:
                correlation = numerator / (denom_vocab * denom_cot)
            else:
                correlation = 0.0

            results["correlation"] = {
                "L13_vocab_vs_cot": correlation,
                "interpretation": self._interpret_correlation(correlation),
            }

            self.log("\n--- CORRELATION ANALYSIS ---")
            self.log(f"L13 vocab ↔ CoT generation: r = {correlation:.3f}")
            self.log(f"Interpretation: {results['correlation']['interpretation']}")

        # Summary
        self.log("\n" + "=" * 60)
        self.log("CONCLUSION")
        self.log("=" * 60)

        if results.get("correlation", {}).get("L13_vocab_vs_cot", 0) > 0.5:
            self.log(">>> HIGH correlation between L13 vocab and CoT!")
            self.log(">>> L13 vocabulary alignment DOES predict CoT generation.")
            self.log(">>> Two-layer routing hypothesis SUPPORTED.")
        elif results.get("correlation", {}).get("L13_vocab_vs_cot", 0) > 0.2:
            self.log(">>> MODERATE correlation between L13 vocab and CoT.")
            self.log(">>> Some relationship exists but may not be causal.")
        else:
            self.log(">>> LOW/NO correlation between L13 vocab and CoT.")
            self.log(">>> L13 vocabulary signal does NOT predict CoT generation.")

        results["per_prompt"] = [
            {
                "input": r.input,
                "format": r.format,
                "task": r.task,
                "generated": r.generated[:100],
                "is_cot": r.is_cot,
                "correct": r.answer_correct,
                "vocab_probs": {f"L{k}": v for k, v in r.vocab_probs.items()},
            }
            for r in self.results
        ]

        return results

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        if r > 0.7:
            return "Strong positive correlation"
        elif r > 0.4:
            return "Moderate positive correlation"
        elif r > 0.2:
            return "Weak positive correlation"
        elif r > -0.2:
            return "No correlation"
        elif r > -0.4:
            return "Weak negative correlation"
        elif r > -0.7:
            return "Moderate negative correlation"
        else:
            return "Strong negative correlation"

    def evaluate(self) -> dict:
        """Return summary metrics."""
        if self.results:
            cot_count = sum(1 for r in self.results if r.is_cot)
            correct_count = sum(1 for r in self.results if r.answer_correct)
            return {
                "cot_rate": cot_count / len(self.results),
                "accuracy": correct_count / len(self.results),
            }
        return {"error": "No results"}

    def cleanup(self) -> None:
        """Cleanup."""
        self.results = []
