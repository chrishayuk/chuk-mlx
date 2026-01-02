"""
Core ActivationSteering class for manipulating model behavior.

Applies learned directions to modify activations during inference.
Generic - works with any direction extracted from circuit analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from .config import SteeringConfig
from .hook import SteeringHook

if TYPE_CHECKING:
    from ..circuit.directions import DirectionBundle


class ActivationSteering:
    """
    Activation steering for manipulating model behavior.

    Applies learned directions to model activations during inference.
    Generic - works with any direction (arithmetic, tool-calling, safety, etc.)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id

        # Directions by layer
        self.directions: dict[int, mx.array] = {}
        self.direction_info: dict[int, dict] = {}  # Metadata

        # Original layer forwards (for restoration)
        self._original_forwards: dict[int, Callable] = {}
        self._is_steering = False

        # Detect structure
        self._detect_structure()

    def _detect_structure(self):
        """Detect model structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
            self._backbone = self.model.model
        elif hasattr(self.model, "layers"):
            self._layers = self.model.layers
            self._backbone = self.model
        else:
            raise ValueError("Cannot detect model layer structure")

        self.num_layers = len(self._layers)

    @classmethod
    def from_pretrained(cls, model_id: str) -> ActivationSteering:
        """Load model for steering."""
        from ..ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def add_direction(
        self,
        layer: int,
        direction: mx.array,
        name: str = "custom",
        positive_label: str = "positive",
        negative_label: str = "negative",
    ) -> None:
        """
        Add a steering direction for a specific layer.

        Args:
            layer: Layer index to apply steering
            direction: Direction vector [hidden_size]
            name: Name for the direction
            positive_label: Label for positive class
            negative_label: Label for negative class
        """
        import numpy as np

        if isinstance(direction, np.ndarray):
            direction = mx.array(direction, dtype=mx.float32)

        self.directions[layer] = direction
        self.direction_info[layer] = {
            "name": name,
            "positive_label": positive_label,
            "negative_label": negative_label,
        }

    def add_directions(self, bundle: DirectionBundle) -> None:
        """Add all directions from a bundle."""
        for layer, direction in bundle.directions.items():
            self.add_direction(
                layer=layer,
                direction=direction.direction,
                name=direction.name,
                positive_label=direction.positive_label,
                negative_label=direction.negative_label,
            )

    def clear_directions(self) -> None:
        """Clear all steering directions."""
        self.directions.clear()
        self.direction_info.clear()

    def _wrap_layer(
        self,
        layer_idx: int,
        coefficient: float,
        position: int | None = None,
    ) -> None:
        """Wrap a layer's forward to apply steering."""
        if layer_idx not in self.directions:
            return

        layer = self._layers[layer_idx]
        direction = self.directions[layer_idx]

        # Store original layer
        if layer_idx not in self._original_forwards:
            self._original_forwards[layer_idx] = layer

        original_layer = self._original_forwards[layer_idx]
        # Use scale_by_norm=True so coefficient is relative to activation magnitude
        hook = SteeringHook(direction, coefficient, position, scale_by_norm=True)

        # Create a wrapper class that intercepts calls
        class SteeredLayerWrapper:
            """Wrapper that applies steering after the layer runs."""

            def __init__(self, layer, hook):
                self._wrapped = layer
                self._hook = hook
                # Copy attributes that might be accessed
                for attr in [
                    "mlp",
                    "attn",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                ]:
                    if hasattr(layer, attr):
                        setattr(self, attr, getattr(layer, attr))

            def __call__(self, h, **kwargs):
                out = self._wrapped(h, **kwargs)
                # Handle different return types
                if hasattr(out, "hidden_states"):
                    out.hidden_states = self._hook(out.hidden_states)
                    return out
                elif isinstance(out, tuple):
                    return (self._hook(out[0]),) + out[1:]
                else:
                    return self._hook(out)

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        # Replace the layer in the list
        self._layers[layer_idx] = SteeredLayerWrapper(original_layer, hook)

    def _unwrap_layers(self) -> None:
        """Restore original layers (unwrap the wrappers)."""
        for layer_idx, original in self._original_forwards.items():
            self._layers[layer_idx] = original
        self._original_forwards.clear()

    def generate(
        self,
        prompt: str,
        config: SteeringConfig | None = None,
        steering_layers: list[int] | None = None,
        coefficient: float | None = None,
    ) -> str:
        """
        Generate with activation steering.

        Args:
            prompt: Input prompt
            config: Steering configuration
            steering_layers: Override layers to steer
            coefficient: Override steering coefficient

        Returns:
            Generated text
        """
        if config is None:
            config = SteeringConfig()

        layers = steering_layers or config.layers
        coef = coefficient if coefficient is not None else config.coefficient

        # Wrap layers for steering
        try:
            for layer in layers:
                self._wrap_layer(layer, coef, config.position)
            self._is_steering = True

            # Generate
            output = self._generate_text(prompt, config)

        finally:
            # Always restore original layers
            self._unwrap_layers()
            self._is_steering = False

        return output

    def compare_steering(
        self,
        prompt: str,
        coefficients: list[float] | None = None,
        config: SteeringConfig | None = None,
    ) -> dict[float, str]:
        """
        Compare outputs with different steering coefficients.

        Returns dict mapping coefficient to output.
        """
        if coefficients is None:
            coefficients = [-1.0, 0.0, 1.0]
        if config is None:
            config = SteeringConfig()

        results = {}
        for coef in coefficients:
            results[coef] = self.generate(prompt, config, coefficient=coef)

        return results

    def sweep_layers(
        self,
        prompt: str,
        layers: list[int] | None = None,
        coefficient: float = 1.0,
        config: SteeringConfig | None = None,
    ) -> dict[int, str]:
        """
        Sweep steering across layers one at a time.

        Returns dict mapping layer to output.
        """
        if config is None:
            config = SteeringConfig()

        if layers is None:
            layers = list(self.directions.keys())

        results = {}
        for layer in layers:
            results[layer] = self.generate(
                prompt,
                config,
                steering_layers=[layer],
                coefficient=coefficient,
            )

        return results

    def _generate_text(self, prompt: str, config: SteeringConfig) -> str:
        """Internal generation with current steering state."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Generate tokens one by one
        generated = []
        current_ids = input_ids

        for _ in range(config.max_new_tokens):
            outputs = self.model(current_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Get next token
            if config.temperature == 0:
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                logits = logits[:, -1, :] / config.temperature
                next_token = mx.random.categorical(logits)

            generated.append(int(next_token[0]))

            # Check for EOS
            if hasattr(self.tokenizer, "eos_token_id"):
                if generated[-1] == self.tokenizer.eos_token_id:
                    break

            current_ids = mx.concatenate([current_ids, next_token[:, None]], axis=1)

        return self.tokenizer.decode(generated)

    def print_comparison(
        self,
        prompt: str,
        coefficients: list[float] | None = None,
        config: SteeringConfig | None = None,
    ) -> None:
        """Print formatted comparison of steering effects."""
        if coefficients is None:
            coefficients = [-1.0, 0.0, 1.0]
        results = self.compare_steering(prompt, coefficients, config)

        # Get direction info
        layers = config.layers if config else list(self.directions.keys())[:1]
        if layers and layers[0] in self.direction_info:
            info = self.direction_info[layers[0]]
            neg_label = info["negative_label"]
            pos_label = info["positive_label"]
        else:
            neg_label = "negative"
            pos_label = "positive"

        print("\n" + "=" * 70)
        print("STEERING COMPARISON")
        print(f"Prompt: {prompt}")
        print(f"Layers: {layers}")
        print(f"Direction: {neg_label} -> {pos_label}")
        print("=" * 70)

        for coef, output in sorted(results.items()):
            direction_label = (
                f"<- {neg_label}" if coef < 0 else f"-> {pos_label}" if coef > 0 else "no steering"
            )
            print(f"\nCoef {coef:+.1f} ({direction_label}):")
            print(f"  {output}")

        print("=" * 70)

    def get_layer_probabilities(
        self,
        prompt: str,
        track_token: str,
        config: SteeringConfig | None = None,
        steering_layers: list[int] | None = None,
        coefficient: float | None = None,
    ) -> dict[int, float]:
        """
        Get token probabilities at each layer WITH steering applied.

        Args:
            prompt: Input prompt
            track_token: Token to track
            config: Steering configuration
            steering_layers: Override layers to steer
            coefficient: Override steering coefficient

        Returns:
            Dict mapping layer -> probability of track_token
        """
        from ..hooks import CaptureConfig, LayerSelection, ModelHooks
        from ..logit_lens import LogitLens

        if config is None:
            config = SteeringConfig()

        layers = steering_layers or config.layers
        coef = coefficient if coefficient is not None else config.coefficient

        # Get model config for hooks
        model_config = None
        if hasattr(self, "_model_config"):
            model_config = self._model_config

        # Set up hooks to capture all hidden states
        hooks = ModelHooks(self.model, model_config=model_config)
        hooks.configure(
            CaptureConfig(
                layers=LayerSelection.ALL,
                capture_hidden_states=True,
            )
        )

        # Wrap layers for steering
        try:
            for layer in layers:
                self._wrap_layer(layer, coef, config.position)
            self._is_steering = True

            # Forward pass with hooks
            input_ids = self.tokenizer.encode(prompt, return_tensors="np")
            input_ids = mx.array(input_ids)
            hooks.forward(input_ids)

        finally:
            # Always restore original layers
            self._unwrap_layers()
            self._is_steering = False

        # Analyze with logit lens
        lens = LogitLens(hooks, self.tokenizer)
        evolution = lens.track_token(track_token)

        # Convert to dict
        return dict(zip(evolution.layers, evolution.probabilities))

    def compare_layer_dynamics(
        self,
        prompt: str,
        track_token: str,
        coefficients: list[float] | None = None,
        config: SteeringConfig | None = None,
    ) -> dict[float, dict[int, float]]:
        """
        Compare layer dynamics with different steering coefficients.

        Returns:
            Dict mapping coefficient -> (layer -> probability)
        """
        if coefficients is None:
            coefficients = [-2.0, 0.0, 2.0]
        results = {}
        for coef in coefficients:
            results[coef] = self.get_layer_probabilities(
                prompt, track_token, config, coefficient=coef
            )
        return results

    def print_layer_dynamics(
        self,
        prompt: str,
        track_token: str,
        coefficients: list[float] | None = None,
        config: SteeringConfig | None = None,
        key_layers: list[int] | None = None,
    ) -> None:
        """
        Print formatted comparison of layer dynamics with steering.
        """
        if coefficients is None:
            coefficients = [-2.0, 0.0, 2.0]
        dynamics = self.compare_layer_dynamics(prompt, track_token, coefficients, config)

        # Get all layers
        all_layers = set()
        for layer_probs in dynamics.values():
            all_layers.update(layer_probs.keys())
        all_layers = sorted(all_layers)

        # Use key layers or sample
        if key_layers is None:
            if len(all_layers) > 20:
                key_layers = all_layers[::2]
            else:
                key_layers = all_layers

        steer_layers = config.layers if config else list(self.directions.keys())[:1]

        print("\n" + "=" * 80)
        print("LAYER DYNAMICS WITH STEERING")
        print(f"Prompt: {prompt!r}")
        print(f"Track: {track_token!r}")
        print(f"Steering at: L{steer_layers}")
        print("=" * 80)

        # Header
        header = "Layer |"
        for coef in sorted(coefficients):
            header += f" coef={coef:+.1f} |"
        print(header)
        print("-" * len(header))

        # Rows
        for layer in key_layers:
            row = f"L{layer:2d}   |"
            for coef in sorted(coefficients):
                prob = dynamics.get(coef, {}).get(layer, 0.0)
                bar = "#" * int(prob * 10)
                row += f" {prob:5.1%} {bar:10s} |"

                if layer in steer_layers:
                    row = row[:-1] + "* |"
            print(row)

        print("=" * 80)
        print("* = steering applied at this layer")
