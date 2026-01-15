"""
Steering Test: Can We Flip Generation Mode?

Takes symbolic inputs like "5 * 5 = " and steers toward CoT generation.
If steering works, we can *control* generation mode, not just detect it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import yaml

logger = logging.getLogger(__name__)


class SteeringTest:
    """
    Test if adding the CoT direction vector to symbolic inputs
    causes the model to generate CoT instead of direct answers.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.cot_direction = None

    def run(self) -> dict:
        """Run the steering test."""
        logger.info("Starting Steering Test")

        # Load model
        self._load_model()

        # Compute CoT direction from training data
        self._compute_cot_direction()

        # Test steering on symbolic inputs
        results = self._test_steering()

        # Save and print results
        self._save_results(results)

        return results

    def _load_model(self):
        """Load the model and tokenizer."""
        from chuk_lazarus.models_v2.loader import load_model

        model_name = self.config["model"]
        logger.info(f"Loading model: {model_name}")

        loaded = load_model(model_name)
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mx.eval(self.model.parameters())
        self.num_layers = len(self.model.model.layers)
        logger.info(f"Model loaded: {self.num_layers} layers")

    def _get_hidden_state(self, prompt: str, layer: int) -> mx.array:
        """Get hidden state at specified layer for last token."""
        if "Instruct" in self.config["model"]:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(formatted, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])
        output = self.model(input_ids, output_hidden_states=True)
        hidden = output.hidden_states[layer]
        return hidden[0, -1, :]

    def _compute_cot_direction(self):
        """Compute the CoT direction vector from training data."""
        logger.info("Computing CoT direction vector")

        steering_layer = self.config["parameters"].get("steering_layer", 4)
        train_data = self.config["parameters"]["train_data"]

        # Separate symbolic and semantic prompts
        symbolic_prompts = [item["prompt"] for item in train_data if item["format"] == "symbolic"]
        semantic_prompts = [item["prompt"] for item in train_data if item["format"] == "semantic"]

        # Get hidden states
        symbolic_hiddens = [self._get_hidden_state(p, steering_layer) for p in symbolic_prompts]
        semantic_hiddens = [self._get_hidden_state(p, steering_layer) for p in semantic_prompts]

        # Compute means
        symbolic_mean = mx.mean(mx.stack(symbolic_hiddens), axis=0)
        semantic_mean = mx.mean(mx.stack(semantic_hiddens), axis=0)

        # Direction: semantic - symbolic (toward CoT)
        direction = semantic_mean - symbolic_mean
        self.cot_direction = direction / (mx.linalg.norm(direction) + 1e-8)
        self.cot_direction_unnorm = direction

        logger.info(f"CoT direction computed at layer {steering_layer}")
        logger.info(f"  Direction norm: {mx.linalg.norm(direction).item():.3f}")

    def _generate_with_steering(
        self,
        prompt: str,
        steering_strength: float = 0.0,
        steering_layer: int = 4,
        max_tokens: int = 150,
    ) -> str:
        """Generate with steering at specified layer.

        Uses layer-by-layer forward pass to inject steering vector.
        """
        # Format prompt
        if "Instruct" in self.config["model"]:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(formatted, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])
        steering_vector = self.cot_direction * steering_strength

        generated = []
        for step in range(max_tokens):
            batch_size, seq_len = input_ids.shape

            # Embedding
            hidden_states = self.model.model.embed_tokens(input_ids)

            # Create causal mask
            import mlx.nn as nn

            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)

            # Forward through layers with steering injection
            for layer_idx, layer in enumerate(self.model.model.layers):
                output = layer(hidden_states, mask=mask, cache=None)
                hidden_states = output.hidden_states

                # Inject steering after the specified layer
                if layer_idx == steering_layer - 1:
                    # Add steering to last token only
                    steering_add = mx.zeros_like(hidden_states)
                    steering_add = steering_add.at[0, -1, :].add(steering_vector)
                    hidden_states = hidden_states + steering_add

            # Final norm
            hidden_states = self.model.model.norm(hidden_states)

            # LM head
            head_output = self.model.lm_head(hidden_states)
            next_token_logits = head_output.logits[0, -1, :]

            next_token = mx.argmax(next_token_logits)
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _generate_simple(self, prompt: str, max_tokens: int = 150) -> str:
        """Simple generation without steering (baseline)."""
        if "Instruct" in self.config["model"]:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokens = self.tokenizer(formatted, return_tensors="np")
        else:
            tokens = self.tokenizer(prompt, return_tensors="np")

        input_ids = mx.array(tokens["input_ids"])

        generated = []
        for _ in range(max_tokens):
            output = self.model(input_ids)
            next_token = mx.argmax(output.logits[0, -1, :])
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        return self.tokenizer.decode(generated).strip()

    def _detect_format(self, text: str) -> str:
        """Detect if output is CoT or direct."""
        import re

        cot_patterns = [
            r"to find",
            r"we need to",
            r"let's",
            r"first,",
            r"step \d",
            r"therefore",
            r"so,",
            r"this means",
            r"calculate",
            r"the answer is",
        ]
        text_lower = text.lower()
        cot_matches = sum(1 for p in cot_patterns if re.search(p, text_lower))
        word_count = len(text.split())

        if cot_matches >= 2 or word_count > 20:
            return "cot"
        return "direct"

    def _test_steering(self) -> dict:
        """Test steering on symbolic inputs."""
        logger.info("Testing steering on symbolic inputs")

        test_prompts = [
            "5 * 5 = ",
            "7 + 3 = ",
            "20 - 8 = ",
            "12 * 4 = ",
            "100 / 5 = ",
        ]

        steering_strengths = [0.0, 1.0, 3.0, 5.0, 7.0]  # Sweet spot is 3-5
        steering_layer = self.config["parameters"].get("steering_layer", 4)

        results = {
            "model": self.config["model"],
            "steering_layer": steering_layer,
            "cot_direction_norm": mx.linalg.norm(self.cot_direction_unnorm).item(),
            "tests": [],
        }

        for prompt in test_prompts:
            prompt_results = {"prompt": prompt, "generations": []}

            for strength in steering_strengths:
                logger.info(f"  Testing '{prompt}' with strength {strength}")

                if strength == 0.0:
                    # Use simple generation for baseline
                    output = self._generate_simple(prompt)
                else:
                    # Use steering generation
                    try:
                        output = self._generate_with_steering(
                            prompt, steering_strength=strength, steering_layer=steering_layer
                        )
                    except Exception as e:
                        logger.warning(f"Steering failed: {e}, using baseline")
                        output = self._generate_simple(prompt)

                detected = self._detect_format(output)

                prompt_results["generations"].append(
                    {
                        "strength": strength,
                        "output": output[:200],
                        "format": detected,
                        "word_count": len(output.split()),
                    }
                )

            results["tests"].append(prompt_results)

        return results

    def _save_results(self, results: dict):
        """Save and print results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"steering_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("STEERING TEST RESULTS")
        print("=" * 70)
        print(f"\nModel: {results['model']}")
        print(f"Steering layer: {results['steering_layer']}")
        print(f"CoT direction norm: {results['cot_direction_norm']:.3f}")

        print("\n" + "-" * 70)
        print("RESULTS BY PROMPT AND STEERING STRENGTH")
        print("-" * 70)

        for test in results["tests"]:
            print(f"\nPrompt: {test['prompt']}")
            for gen in test["generations"]:
                strength = gen["strength"]
                fmt = gen["format"]
                words = gen["word_count"]
                output_preview = gen["output"][:50].replace("\n", " ")
                marker = "←" if fmt == "cot" else ""
                print(
                    f'  strength={strength:.1f}: {fmt:6s} ({words:2d} words) "{output_preview}..." {marker}'
                )

        # Summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)

        for strength in [0.0, 1.0, 2.0, 3.0, 5.0]:
            cot_count = sum(
                1
                for test in results["tests"]
                for gen in test["generations"]
                if gen["strength"] == strength and gen["format"] == "cot"
            )
            total = len(results["tests"])
            print(f"  strength={strength:.1f}: {cot_count}/{total} generated CoT")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test = SteeringTest()
    test.run()
