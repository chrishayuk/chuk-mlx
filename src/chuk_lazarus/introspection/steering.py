"""
Activation steering for manipulating model behavior.

Applies learned directions to modify activations during inference.
Generic - works with any direction extracted from circuit analysis.

Example use cases:
- Gemma arithmetic: Suppress the "suppression circuit" to restore computation
- Tool-calling: Steer toward/away from tool use
- Safety: Increase/decrease safety tendency
- Factual: Steer toward truth vs context

Example:
    >>> from chuk_lazarus.introspection.steering import ActivationSteering
    >>> from chuk_lazarus.introspection.circuit import DirectionBundle
    >>>
    >>> # Load pre-extracted directions
    >>> directions = DirectionBundle.load("arithmetic_directions")
    >>>
    >>> # Create steering instance
    >>> steerer = ActivationSteering.from_pretrained("gemma-3-4b-it")
    >>> steerer.add_directions(directions)
    >>>
    >>> # Generate with steering
    >>> output = steerer.generate(
    ...     prompt="6 * 7 =",
    ...     steering_layers=[24],
    ...     coefficient=1.0,  # Positive = toward positive class
    ... )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if TYPE_CHECKING:
    from .circuit.directions import DirectionBundle


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    # Which layers to steer
    layers: list[int] = field(default_factory=lambda: [24])

    # Steering coefficient (positive = toward positive class)
    coefficient: float = 1.0

    # Apply only at specific positions
    position: int | None = None  # None = all positions

    # Normalization
    normalize_direction: bool = True
    scale_by_activation_norm: bool = False

    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.0


class SteeringHook:
    """
    Hook that modifies activations during forward pass.

    Generic - applies any direction vector to steer activations.
    """

    def __init__(
        self,
        direction: mx.array,
        coefficient: float = 1.0,
        position: int | None = None,
        normalize: bool = True,
        scale_by_norm: bool = False,
    ):
        """
        Initialize steering hook.

        Args:
            direction: Direction vector to add [hidden_size]
            coefficient: Scaling factor (positive = toward direction)
            position: Position to steer (None = all)
            normalize: Normalize direction before applying
            scale_by_norm: Scale by activation norm
        """
        self.direction = direction
        self.coefficient = coefficient
        self.position = position
        self.normalize = normalize
        self.scale_by_norm = scale_by_norm

        if self.normalize:
            norm = mx.sqrt(mx.sum(direction * direction))
            self.direction = direction / (norm + 1e-8)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Apply steering to hidden states.

        Args:
            h: Hidden states [batch, seq, hidden]

        Returns:
            Steered hidden states
        """
        steering = self.direction * self.coefficient

        if self.scale_by_norm:
            # Scale by mean activation norm
            h_norm = mx.sqrt(mx.mean(h * h))
            steering = steering * h_norm

        if self.position is not None:
            # Only steer at specific position
            steered = h.at[:, self.position, :].add(steering)
            return steered
        else:
            # Steer all positions
            return h + steering


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
        from .ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    def add_direction(
        self,
        layer: int,
        direction: np.ndarray | mx.array,
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
        # We can't just patch __call__ on the instance because Python
        # uses type(obj).__call__(obj, ...) not obj.__call__(...)
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
                    # Create a new output with steered hidden states
                    out.hidden_states = self._hook(out.hidden_states)
                    return out
                elif isinstance(out, tuple):
                    return (self._hook(out[0]),) + out[1:]
                else:
                    return self._hook(out)

            def __getattr__(self, name):
                # Forward attribute access to wrapped layer
                return getattr(self._wrapped, name)

        # Replace the layer in the list
        self._layers[layer_idx] = SteeredLayerWrapper(original_layer, hook)

    def _unwrap_layers(self) -> None:
        """Restore original layers (unwrap the wrappers)."""
        for layer_idx, original in self._original_forwards.items():
            # Put the original layer back in the list
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
        print(f"Direction: {neg_label} → {pos_label}")
        print("=" * 70)

        for coef, output in sorted(results.items()):
            direction_label = (
                f"← {neg_label}" if coef < 0 else f"→ {pos_label}" if coef > 0 else "no steering"
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

        This is the key method for understanding how steering affects
        the layer dynamics (e.g., does it prevent the L24 destruction?).

        Args:
            prompt: Input prompt
            track_token: Token to track
            config: Steering configuration
            steering_layers: Override layers to steer
            coefficient: Override steering coefficient

        Returns:
            Dict mapping layer -> probability of track_token
        """
        from .hooks import CaptureConfig, LayerSelection, ModelHooks
        from .logit_lens import LogitLens

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

        This shows how steering affects the destruction/rebuild pattern
        at each layer.

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

        Shows how steering affects the probability at each layer.
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
            # Sample every 2 layers if too many
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
                bar = "█" * int(prob * 10)
                row += f" {prob:5.1%} {bar:10s} |"

                # Mark if this is a steering layer
                if layer in steer_layers:
                    row = row[:-1] + "* |"
            print(row)

        print("=" * 80)
        print("* = steering applied at this layer")


# =============================================================================
# Convenience functions
# =============================================================================


def steer_model(
    model_id: str,
    prompt: str,
    directions: DirectionBundle,
    layers: list[int] | None = None,
    coefficient: float = 1.0,
) -> str:
    """
    Convenience function to apply steering and generate.

    Args:
        model_id: Model to load
        prompt: Input prompt
        directions: Direction bundle to use
        layers: Layers to steer (default: all in bundle)
        coefficient: Steering strength

    Returns:
        Generated text
    """
    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    config = SteeringConfig(
        layers=layers or list(directions.directions.keys()),
        coefficient=coefficient,
    )

    return steerer.generate(prompt, config)


def compare_steering_effects(
    model_id: str,
    prompt: str,
    directions: DirectionBundle,
    layer: int,
    coefficients: list[float] | None = None,
) -> dict[float, str]:
    """
    Compare steering effects at different coefficients.

    Returns dict mapping coefficient to generated output.
    """
    if coefficients is None:
        coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]
    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    config = SteeringConfig(layers=[layer])
    return steerer.compare_steering(prompt, coefficients, config)


# =============================================================================
# Backwards compatibility: Tool-calling specific steering
# =============================================================================


class SteeringMode(Enum):
    """Steering modes for tool-calling control (backwards compatibility)."""

    NORMAL = "normal"
    FORCE_TOOL = "force_tool"
    PREVENT_TOOL = "prevent_tool"
    BOOST_TOOL = "boost_tool"
    SUPPRESS_TOOL = "suppress_tool"


@dataclass
class LegacySteeringConfig:
    """Configuration for legacy tool-calling steering."""

    mode: SteeringMode = SteeringMode.NORMAL
    steering_scale: float = 1.0
    neuron_boost_scale: float = 5000.0
    use_kill_switch: bool = False
    kill_switch_boost: float = 0.0
    tool_promoters: list = None
    tool_suppressors: list = None

    def __post_init__(self):
        if self.tool_promoters is None:
            self.tool_promoters = [803, 2036, 831]
        if self.tool_suppressors is None:
            self.tool_suppressors = [1237, 821, 1347]


class SteeredGemmaMLP(nn.Module):
    """
    A Gemma MLP wrapper that applies steering during forward pass.
    (Kept for backwards compatibility with FunctionGemma experiments)
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        config: LegacySteeringConfig,
        layer_idx: int,
        control_layer: int = 11,
        gate_layer: int = 12,
        kill_switch_neuron: int = 230,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.config = config
        self.layer_idx = layer_idx
        self.control_layer = control_layer
        self.gate_layer = gate_layer
        self.kill_switch_neuron = kill_switch_neuron

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.original_mlp.gate_proj(x)
        up = self.original_mlp.up_proj(x)
        mlp_hidden = nn.gelu_approx(gate) * up

        if self.layer_idx == self.control_layer:
            mlp_hidden = self._apply_control_steering(mlp_hidden)
        if self.layer_idx == self.gate_layer:
            mlp_hidden = self._apply_gate_steering(mlp_hidden)

        return self.original_mlp.down_proj(mlp_hidden)

    def _apply_control_steering(self, mlp_hidden: mx.array) -> mx.array:
        if self.config.mode == SteeringMode.NORMAL:
            return mlp_hidden

        batch_size, seq_len, hidden_size = mlp_hidden.shape
        modification = mx.zeros((hidden_size,))

        scale = self.config.neuron_boost_scale
        if self.config.mode in [SteeringMode.BOOST_TOOL, SteeringMode.SUPPRESS_TOOL]:
            scale *= 0.3

        if self.config.mode in [SteeringMode.FORCE_TOOL, SteeringMode.BOOST_TOOL]:
            for neuron in self.config.tool_promoters:
                if neuron < hidden_size:
                    modification = modification.at[neuron].add(scale)
            for neuron in self.config.tool_suppressors:
                if neuron < hidden_size:
                    modification = modification.at[neuron].add(-scale * 0.5)
        else:
            for neuron in self.config.tool_promoters:
                if neuron < hidden_size:
                    modification = modification.at[neuron].add(-scale)
            for neuron in self.config.tool_suppressors:
                if neuron < hidden_size:
                    modification = modification.at[neuron].add(scale * 0.5)

        position_mask = mx.zeros((seq_len,))
        position_mask = position_mask.at[-1].add(1.0)
        modification = modification.reshape(1, 1, hidden_size)
        position_mask = position_mask.reshape(1, seq_len, 1)

        return mlp_hidden + modification * position_mask

    def _apply_gate_steering(self, mlp_hidden: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = mlp_hidden.shape

        if self.config.use_kill_switch:
            mask = mx.ones((hidden_size,))
            mask = mask.at[self.kill_switch_neuron].add(-1.0)
            position_mask = mx.zeros((seq_len,))
            position_mask = position_mask.at[-1].add(1.0)
            mask_broadcast = mask.reshape(1, 1, hidden_size)
            position_broadcast = position_mask.reshape(1, seq_len, 1)
            mlp_hidden = mlp_hidden * (1 - position_broadcast + position_broadcast * mask_broadcast)

        if self.config.kill_switch_boost != 0:
            modification = mx.zeros((hidden_size,))
            modification = modification.at[self.kill_switch_neuron].add(
                self.config.kill_switch_boost
            )
            position_mask = mx.zeros((seq_len,))
            position_mask = position_mask.at[-1].add(1.0)
            modification = modification.reshape(1, 1, hidden_size)
            position_mask = position_mask.reshape(1, seq_len, 1)
            mlp_hidden = mlp_hidden + modification * position_mask

        return mlp_hidden


class ToolCallingSteering:
    """
    Tool-calling specific steering (for FunctionGemma).
    Kept for backwards compatibility.

    For new code, use ActivationSteering instead.
    """

    CONTROL_LAYER = 11
    GATE_LAYER = 12
    KILL_SWITCH_NEURON = 230
    TOOL_PROMOTERS = [803, 2036, 831, 436, 969]
    TOOL_SUPPRESSORS = [1347, 1237, 821, 217, 543]

    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = config
        self._original_mlps = {}

    @classmethod
    def from_pretrained(cls, model_id: str) -> ToolCallingSteering:
        from .ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(study.adapter.model, study.adapter.tokenizer, study.adapter.config)

    def _install_steering(self, config: LegacySteeringConfig):
        layers = self.model.model.layers
        for layer_idx in [self.CONTROL_LAYER, self.GATE_LAYER]:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                self._original_mlps[layer_idx] = layer.mlp
                layer.mlp = SteeredGemmaMLP(
                    layer.mlp,
                    config,
                    layer_idx,
                    self.CONTROL_LAYER,
                    self.GATE_LAYER,
                    self.KILL_SWITCH_NEURON,
                )

    def _uninstall_steering(self):
        layers = self.model.model.layers
        for layer_idx, original_mlp in self._original_mlps.items():
            if layer_idx < len(layers):
                layers[layer_idx].mlp = original_mlp
        self._original_mlps.clear()

    def generate(
        self,
        prompt: str,
        mode: str = "normal",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        config = LegacySteeringConfig(mode=SteeringMode(mode), **kwargs)
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        input_ids = mx.array([tokens])

        self._install_steering(config)
        try:
            stop_tokens = [self.tokenizer.eos_token_id]
            end_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn_id != self.tokenizer.unk_token_id:
                stop_tokens.append(end_turn_id)

            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens,
            )
            new_tokens = generated[0, len(tokens) :].tolist()
            return self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        finally:
            self._uninstall_steering()

    def predict(self, prompt: str, mode: str = "normal", **kwargs) -> dict:
        config = LegacySteeringConfig(mode=SteeringMode(mode), **kwargs)
        tokens = self.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        input_ids = mx.array([tokens])

        self._install_steering(config)
        try:
            output = self.model(input_ids)
            logits = output.logits[0, -1, :]
            probs = mx.softmax(logits, axis=-1)
            top_indices = mx.argsort(probs)[-5:][::-1].tolist()

            results = []
            for idx in top_indices:
                prob = float(probs[idx])
                try:
                    token = self.tokenizer.decode([idx])
                except Exception:
                    token = f"[{idx}]"
                results.append((token, prob))

            top_token = results[0][0]
            tool_indicators = ["[", "{", "<", "function", "tool", "call", "Function"]
            tool_likely = any(ind in top_token for ind in tool_indicators)

            return {
                "prompt": prompt,
                "mode": mode,
                "top_tokens": results,
                "tool_likely": tool_likely,
            }
        finally:
            self._uninstall_steering()

    def compare_modes(self, prompt: str) -> dict:
        results = {}
        for mode in SteeringMode:
            results[mode.value] = self.predict(prompt, mode=mode.value)
        return results


def format_functiongemma_prompt(user_query: str, tools: list[dict] = None) -> str:
    """Format a prompt for FunctionGemma using its expected template."""
    import json

    if tools is None:
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Email recipient"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
            {
                "name": "set_timer",
                "description": "Set a timer for a specified duration",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Duration in minutes",
                        }
                    },
                    "required": ["duration_minutes"],
                },
            },
        ]

    tools_json = json.dumps(tools)
    return f"""<start_of_turn>developer
You are a model that can do function calling with the following functions:
{tools_json}
<end_of_turn>
<start_of_turn>user
{user_query}
<end_of_turn>
<start_of_turn>model
"""
