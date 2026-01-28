#!/usr/bin/env python3
"""
Causal Intervention Experiment for Virtual Expert Tool Calling

Tests whether we can causally influence tool calling behavior through activation steering:
1. Extract tool direction from trained probes
2. Apply steering during forward pass
3. Measure effect on tool call probability

Key questions:
- Does amplifying tool direction increase tool call generation?
- Can we steer toward specific tools?
- Can we inject novel tool schemas?
"""

import json
import sys
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from sklearn.linear_model import LogisticRegression

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class CausalInterventionExperiment:
    def __init__(self, model_path: str = "openai/gpt-oss-20b"):
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

        # Training data for extracting steering vectors
        self.tool_prompts = [
            "What is 127 * 89?",
            "Calculate 15% of 340",
            "What's the weather in Tokyo?",
            "Is it raining in London?",
            "What is the current price of Bitcoin?",
            "Who won the 2024 Super Bowl?",
            "Run this Python: print(sum(range(100)))",
            "Execute: import math; print(math.factorial(10))",
            "What's the temperature in New York?",
            "Calculate the area of a circle with radius 5",
        ]

        self.direct_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms",
            "Who wrote Romeo and Juliet?",
            "What color is the sky?",
            "Name three planets in our solar system",
            "What is the chemical symbol for water?",
            "Who painted the Mona Lisa?",
            "What continent is Egypt in?",
            "What is the opposite of 'hot'?",
            "What sound does a cat make?",
        ]

        # Test prompts (ambiguous - could go either way)
        # Use simple completion style that GPT-OSS understands
        self.test_prompts = [
            "The number 42 is",  # Could be trivia or calculation
            "The weather today is",  # Ambiguous weather query
            "To solve this math problem,",  # Could need tool or explanation
            "Looking up information about",  # Very ambiguous
            "Paris is known for",  # Could be weather or general knowledge
        ]

        # Tool-specific prompts for steering vectors
        self.tool_specific = {
            "calculator": [
                "What is 127 * 89?",
                "Calculate 15% of 340",
                "What's the square root of 144?",
                "Compute 2^10",
            ],
            "get_weather": [
                "What's the weather in Tokyo?",
                "Is it raining in London?",
                "What's the temperature in New York?",
                "Will it snow in Chicago?",
            ],
            "search": [
                "What is the current price of Bitcoin?",
                "Who won the 2024 Super Bowl?",
                "What's the latest news on AI?",
                "What movies are playing?",
            ],
            "code_exec": [
                "Run this Python: print(sum(range(100)))",
                "Execute: import math; print(math.factorial(10))",
                "Run Python code to list primes under 50",
                "Execute this: [x**2 for x in range(10)]",
            ],
        }

    def get_model_components(self):
        """Get embedding and layer modules."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def extract_hidden_state(self, prompt: str, layer: int) -> np.ndarray:
        """Extract hidden state at specified layer for a prompt."""
        embed_tokens, model_layers = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        seq_len = h.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        for i, layer_module in enumerate(model_layers):
            h = layer_module(h, mask=mask)
            if i == layer:
                break

        mx.eval(h)
        return np.array(h[0, -1, :].astype(mx.float32))

    def extract_tool_direction(self, layer: int = 8) -> tuple:
        """Extract the tool vs direct direction from training data."""
        print(f"Extracting tool direction at layer {layer}...")

        # Collect hidden states
        tool_states = [self.extract_hidden_state(p, layer) for p in self.tool_prompts]
        direct_states = [self.extract_hidden_state(p, layer) for p in self.direct_prompts]

        X = np.vstack([tool_states, direct_states])
        y = np.array([1] * len(tool_states) + [0] * len(direct_states))

        # Train logistic regression to get direction
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        # The coefficient vector is the "tool direction"
        tool_direction = clf.coef_[0]
        tool_direction = tool_direction / np.linalg.norm(tool_direction)  # Normalize

        # Compute separation statistics
        tool_mean = np.mean(tool_states, axis=0)
        direct_mean = np.mean(direct_states, axis=0)
        direction_diff = tool_mean - direct_mean
        direction_diff = direction_diff / np.linalg.norm(direction_diff)

        return tool_direction, clf, direction_diff

    def extract_tool_specific_directions(self, layer: int = 8) -> dict:
        """Extract directions for each specific tool."""
        print(f"Extracting tool-specific directions at layer {layer}...")

        directions = {}

        for tool_name, prompts in self.tool_specific.items():
            # Get states for this tool
            tool_states = [self.extract_hidden_state(p, layer) for p in prompts]

            # Get states for other tools (negative examples)
            other_states = []
            for other_tool, other_prompts in self.tool_specific.items():
                if other_tool != tool_name:
                    other_states.extend([self.extract_hidden_state(p, layer) for p in other_prompts[:2]])

            X = np.vstack([tool_states, other_states])
            y = np.array([1] * len(tool_states) + [0] * len(other_states))

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)

            direction = clf.coef_[0]
            direction = direction / np.linalg.norm(direction)

            directions[tool_name] = {
                "direction": direction,
                "accuracy": clf.score(X, y),
                "n_samples": len(X)
            }

        return directions

    def generate_with_steering(
        self,
        prompt: str,
        steering_direction: np.ndarray,
        steering_layer: int,
        steering_strength: float,
        max_tokens: int = 50
    ) -> str:
        """Generate text with activation steering applied at specified layer."""
        embed_tokens, model_layers = self.get_model_components()

        tokens = self.tokenizer.encode(prompt)
        generated = []

        for _ in range(max_tokens):
            input_ids = mx.array([tokens])
            h = embed_tokens(input_ids)
            seq_len = h.shape[1]
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

            # Forward through layers with steering
            for i, layer_module in enumerate(model_layers):
                h = layer_module(h, mask=mask)

                # Apply steering at specified layer
                if i == steering_layer:
                    steering_vec = mx.array(steering_direction * steering_strength)
                    # Add steering to last position
                    h_steered = h.astype(mx.float32)
                    # Create steering tensor
                    steering_tensor = mx.zeros_like(h_steered)
                    # We need to add to the last position
                    last_pos_steering = mx.expand_dims(mx.expand_dims(steering_vec, 0), 0)
                    # Add steering to hidden state at last position
                    h = h + mx.concatenate([
                        mx.zeros((1, seq_len - 1, h.shape[-1])),
                        last_pos_steering
                    ], axis=1).astype(h.dtype)

            # Get logits
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(h[:, -1, :])
            else:
                logits = self.model.model.lm_head(h[:, -1, :])

            mx.eval(logits)

            # Sample next token (greedy)
            next_token = int(mx.argmax(logits, axis=-1)[0])

            if next_token == self.tokenizer.eos_token_id:
                break

            tokens.append(next_token)
            generated.append(next_token)

        return self.tokenizer.decode(generated)

    def measure_tool_probability(self, prompt: str, layer: int, tool_direction: np.ndarray) -> float:
        """Measure how much a prompt activates the tool direction."""
        state = self.extract_hidden_state(prompt, layer)
        projection = np.dot(state, tool_direction)
        return float(projection)

    def run_steering_experiment(self, layer: int = 8, strengths: list = [-2, -1, 0, 1, 2, 4, 8]):
        """Run the main steering experiment."""
        print("\n=== Causal Steering Experiment ===")

        # Extract tool direction
        tool_direction, clf, direction_diff = self.extract_tool_direction(layer)

        results = {
            "layer": layer,
            "direction_accuracy": float(clf.score(
                np.vstack([
                    [self.extract_hidden_state(p, layer) for p in self.tool_prompts],
                    [self.extract_hidden_state(p, layer) for p in self.direct_prompts]
                ]),
                [1] * len(self.tool_prompts) + [0] * len(self.direct_prompts)
            )),
            "steering_results": []
        }

        print(f"Tool direction accuracy: {results['direction_accuracy']:.1%}")

        # Test steering on ambiguous prompts
        for test_prompt in self.test_prompts:
            prompt_result = {
                "prompt": test_prompt,
                "baseline_projection": self.measure_tool_probability(test_prompt, layer, tool_direction),
                "generations": {}
            }

            print(f"\n  Testing: '{test_prompt}'")
            print(f"    Baseline projection: {prompt_result['baseline_projection']:.3f}")

            for strength in strengths:
                try:
                    output = self.generate_with_steering(
                        test_prompt,
                        tool_direction,
                        layer,
                        strength,
                        max_tokens=30
                    )
                    prompt_result["generations"][str(strength)] = output[:100]
                    print(f"    Strength {strength:+.1f}: {output[:50]}...")
                except Exception as e:
                    prompt_result["generations"][str(strength)] = f"ERROR: {str(e)}"
                    print(f"    Strength {strength:+.1f}: ERROR - {e}")

            results["steering_results"].append(prompt_result)

        return results

    def run_tool_specific_steering(self, layer: int = 8):
        """Test steering toward specific tools."""
        print("\n=== Tool-Specific Steering Experiment ===")

        # Extract tool-specific directions
        tool_directions = self.extract_tool_specific_directions(layer)

        results = {"layer": layer, "tool_directions": {}, "steering_tests": []}

        for tool_name, data in tool_directions.items():
            results["tool_directions"][tool_name] = {
                "accuracy": data["accuracy"],
                "n_samples": data["n_samples"]
            }
            print(f"  {tool_name}: {data['accuracy']:.1%} accuracy")

        # Test: Can we steer an ambiguous prompt toward specific tools?
        test_prompt = "I need to"

        print(f"\n  Steering '{test_prompt}' toward each tool:")

        for tool_name, data in tool_directions.items():
            direction = data["direction"]

            try:
                output = self.generate_with_steering(
                    test_prompt,
                    direction,
                    layer,
                    steering_strength=4.0,
                    max_tokens=30
                )
                results["steering_tests"].append({
                    "tool": tool_name,
                    "prompt": test_prompt,
                    "output": output[:100]
                })
                print(f"    → {tool_name}: {output[:50]}...")
            except Exception as e:
                print(f"    → {tool_name}: ERROR - {e}")

        return results

    def run_novel_schema_injection(self, layer: int = 8):
        """Test if we can inject a novel tool schema."""
        print("\n=== Novel Schema Injection Experiment ===")

        # Create a "novel tool" direction by combining existing directions
        # This simulates what we'd do for a new tool
        tool_directions = self.extract_tool_specific_directions(layer)

        # Combine calculator + search directions for a hypothetical "research_calculator" tool
        calc_dir = tool_directions["calculator"]["direction"]
        search_dir = tool_directions["search"]["direction"]

        # Novel direction: weighted combination
        novel_direction = 0.6 * calc_dir + 0.4 * search_dir
        novel_direction = novel_direction / np.linalg.norm(novel_direction)

        results = {"layer": layer, "tests": []}

        # Test prompts that might benefit from a "research + calculate" tool
        test_prompts = [
            "To analyze this financial data, I would",
            "The statistics show that",
            "The trend in this data indicates",
        ]

        print("  Testing novel 'research_calculator' direction:")

        for prompt in test_prompts:
            try:
                # Without steering
                baseline = self.generate_with_steering(
                    prompt,
                    novel_direction,
                    layer,
                    steering_strength=0,
                    max_tokens=30
                )

                # With steering
                steered = self.generate_with_steering(
                    prompt,
                    novel_direction,
                    layer,
                    steering_strength=4.0,
                    max_tokens=30
                )

                results["tests"].append({
                    "prompt": prompt,
                    "baseline": baseline[:80],
                    "steered": steered[:80]
                })

                print(f"\n    Prompt: '{prompt}'")
                print(f"      Baseline: {baseline[:50]}...")
                print(f"      Steered:  {steered[:50]}...")

            except Exception as e:
                print(f"    ERROR for '{prompt}': {e}")

        return results

    def measure_steering_effect_quantitatively(self, layer: int = 8):
        """Quantitatively measure the effect of steering on tool probability."""
        print("\n=== Quantitative Steering Effect ===")

        tool_direction, clf, _ = self.extract_tool_direction(layer)

        results = {"layer": layer, "measurements": []}

        # Use direct prompts (should be low tool probability)
        # Measure how steering changes their projection onto tool direction
        for prompt in self.direct_prompts[:5]:
            baseline_proj = self.measure_tool_probability(prompt, layer, tool_direction)

            # We can't easily measure post-steering projection without running the model
            # But we can measure how the baseline relates to steering potential

            results["measurements"].append({
                "prompt": prompt,
                "baseline_projection": baseline_proj,
                "is_tool_prompt": False
            })

        for prompt in self.tool_prompts[:5]:
            baseline_proj = self.measure_tool_probability(prompt, layer, tool_direction)

            results["measurements"].append({
                "prompt": prompt,
                "baseline_projection": baseline_proj,
                "is_tool_prompt": True
            })

        # Compute statistics
        tool_projs = [m["baseline_projection"] for m in results["measurements"] if m["is_tool_prompt"]]
        direct_projs = [m["baseline_projection"] for m in results["measurements"] if not m["is_tool_prompt"]]

        results["statistics"] = {
            "tool_mean_projection": float(np.mean(tool_projs)),
            "direct_mean_projection": float(np.mean(direct_projs)),
            "separation": float(np.mean(tool_projs) - np.mean(direct_projs)),
            "cohens_d": float(
                (np.mean(tool_projs) - np.mean(direct_projs)) /
                np.sqrt((np.var(tool_projs) + np.var(direct_projs)) / 2)
            )
        }

        print(f"  Tool prompts mean projection: {results['statistics']['tool_mean_projection']:.3f}")
        print(f"  Direct prompts mean projection: {results['statistics']['direct_mean_projection']:.3f}")
        print(f"  Separation: {results['statistics']['separation']:.3f}")
        print(f"  Cohen's d: {results['statistics']['cohens_d']:.2f}")

        return results

    def run_full_experiment(self):
        """Run all causal intervention experiments."""
        print("=" * 60)
        print("CAUSAL INTERVENTION EXPERIMENT FOR VIRTUAL EXPERT TOOL CALLING")
        print("=" * 60)

        results = {}

        # 1. Main steering experiment
        print("\n[1/4] Running steering experiment...")
        results["steering"] = self.run_steering_experiment(layer=8)

        # 2. Tool-specific steering
        print("\n[2/4] Running tool-specific steering...")
        results["tool_specific"] = self.run_tool_specific_steering(layer=8)

        # 3. Novel schema injection
        print("\n[3/4] Running novel schema injection...")
        results["novel_schema"] = self.run_novel_schema_injection(layer=8)

        # 4. Quantitative measurement
        print("\n[4/4] Measuring steering effect quantitatively...")
        results["quantitative"] = self.measure_steering_effect_quantitatively(layer=8)

        # Summary
        results["summary"] = self._generate_summary(results)

        # Save
        output_path = RESULTS_DIR / "causal_intervention_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for finding in results["summary"]["findings"]:
            print(f"  - {finding}")

        return results

    def _generate_summary(self, results):
        """Generate summary of findings."""
        findings = []

        # Steering accuracy
        steering = results.get("steering", {})
        if "direction_accuracy" in steering:
            findings.append(f"Tool direction accuracy: {steering['direction_accuracy']:.1%}")

        # Tool-specific directions
        tool_specific = results.get("tool_specific", {}).get("tool_directions", {})
        if tool_specific:
            accs = [d["accuracy"] for d in tool_specific.values()]
            findings.append(f"Tool-specific direction accuracy: {np.mean(accs):.1%} average")

        # Quantitative separation
        quant = results.get("quantitative", {}).get("statistics", {})
        if "cohens_d" in quant:
            findings.append(f"Cohen's d for tool direction: {quant['cohens_d']:.2f}")
            findings.append(f"Tool-direct projection separation: {quant['separation']:.3f}")

        return {
            "findings": findings,
            "interpretation": "Causal intervention experiment for virtual expert tool calling"
        }


if __name__ == "__main__":
    experiment = CausalInterventionExperiment()
    results = experiment.run_full_experiment()
