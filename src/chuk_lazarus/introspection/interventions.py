"""
Counterfactual Intervention API for causal analysis of language models.

Provides a unified interface for:
- Activation patching (interchange experiments)
- Causal tracing (finding critical components)
- What-if analysis with surgical interventions
- Component-level ablation and steering

Example:
    >>> from chuk_lazarus.introspection.interventions import (
    ...     CounterfactualIntervention,
    ...     InterventionConfig,
    ...     InterventionResult,
    ...     patch_activations,
    ...     trace_causal_path,
    ... )
    >>>
    >>> ci = CounterfactualIntervention.from_pretrained("model_id")
    >>>
    >>> # What-if: replace subject in Rome fact lookup
    >>> result = ci.patch_run(
    ...     clean_prompt="The capital of France is",
    ...     corrupt_prompt="The capital of Germany is",
    ...     patch_layers=[10, 11, 12],
    ...     patch_positions=[-1],  # Last token only
    ... )
    >>> print(f"Effect: {result.effect_size:.2f}")
    >>>
    >>> # Causal tracing: find where "Paris" is recalled
    >>> trace = ci.trace_token(
    ...     prompt="The capital of France is",
    ...     target_token="Paris",
    ... )
    >>> print(f"Critical layers: {trace.critical_layers}")
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration
# =============================================================================


class InterventionType(str, Enum):
    """Type of intervention to apply."""

    ZERO = "zero"  # Zero out activations
    MEAN = "mean"  # Replace with mean activation
    PATCH = "patch"  # Patch from another run
    NOISE = "noise"  # Add noise
    STEER = "steer"  # Add steering direction
    SCALE = "scale"  # Scale activations


class ComponentTarget(str, Enum):
    """Target component for intervention."""

    HIDDEN = "hidden"  # Residual stream
    ATTENTION = "attention"  # Attention output
    MLP = "mlp"  # MLP output
    ATTENTION_HEAD = "attn_head"  # Individual attention head
    MLP_NEURON = "mlp_neuron"  # Individual MLP neuron


class InterventionConfig(BaseModel):
    """Configuration for an intervention experiment."""

    model_config = ConfigDict(frozen=True)

    intervention_type: InterventionType = Field(
        default=InterventionType.PATCH, description="Type of intervention"
    )
    target: ComponentTarget = Field(default=ComponentTarget.HIDDEN, description="Target component")
    layers: tuple[int, ...] = Field(default_factory=tuple, description="Layers to intervene on")
    positions: tuple[int, ...] = Field(
        default_factory=lambda: (-1,), description="Token positions to intervene on"
    )
    heads: tuple[int, ...] | None = Field(
        default=None, description="Specific attention heads (if target is attn_head)"
    )
    neurons: tuple[int, ...] | None = Field(
        default=None, description="Specific neurons (if target is mlp_neuron)"
    )
    noise_scale: float = Field(default=0.1, description="Scale for noise intervention")
    scale_factor: float = Field(default=0.0, description="Scale factor (0 = ablate, 1 = identity)")


# =============================================================================
# Result Models
# =============================================================================


class InterventionResult(BaseModel):
    """Result of an intervention experiment."""

    model_config = ConfigDict(frozen=True)

    clean_output: str = Field(description="Output from clean run")
    intervened_output: str = Field(description="Output after intervention")
    clean_logits: tuple[float, ...] | None = Field(
        default=None, description="Target token logits from clean run"
    )
    intervened_logits: tuple[float, ...] | None = Field(
        default=None, description="Target token logits after intervention"
    )
    effect_size: float = Field(default=0.0, description="Magnitude of intervention effect")
    kl_divergence: float | None = Field(
        default=None, description="KL divergence between clean/intervened distributions"
    )
    intervention_config: InterventionConfig | None = Field(
        default=None, description="Configuration used"
    )


class PatchingResult(BaseModel):
    """Result of activation patching experiment."""

    model_config = ConfigDict(frozen=True)

    clean_prompt: str = Field(description="Clean input prompt")
    corrupt_prompt: str = Field(description="Corrupted input prompt")
    clean_output: str = Field(description="Output from clean run")
    corrupt_output: str = Field(description="Output from corrupt run")
    patched_output: str = Field(description="Output with patching")
    recovery_rate: float = Field(
        ge=0, le=1, default=0.0, description="How much of clean behavior was recovered"
    )
    effect_size: float = Field(default=0.0, description="Magnitude of patching effect")
    patched_layers: tuple[int, ...] = Field(
        default_factory=tuple, description="Layers that were patched"
    )
    patched_positions: tuple[int, ...] = Field(
        default_factory=tuple, description="Positions that were patched"
    )


class CausalTraceResult(BaseModel):
    """Result of causal tracing experiment."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="Input prompt")
    target_token: str = Field(description="Token being traced")
    target_token_id: int = Field(ge=0, description="Token ID")
    layer_effects: tuple[tuple[int, float], ...] = Field(
        default_factory=tuple, description="(layer, effect) pairs"
    )
    critical_layers: tuple[int, ...] = Field(
        default_factory=tuple, description="Layers with highest effects"
    )
    peak_layer: int = Field(ge=0, default=0, description="Layer with maximum effect")
    peak_effect: float = Field(default=0.0, description="Effect at peak layer")
    baseline_prob: float = Field(
        ge=0, le=1, default=0.0, description="Baseline probability of target token"
    )


class FullCausalTrace(BaseModel):
    """Complete causal tracing result with position × layer grid."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="Input prompt")
    target_token: str = Field(description="Token being traced")
    tokens: tuple[str, ...] = Field(default_factory=tuple, description="All tokens in prompt")
    effects: tuple[tuple[float, ...], ...] = Field(
        default_factory=tuple, description="Effect grid [position × layer]"
    )
    critical_positions: tuple[int, ...] = Field(
        default_factory=tuple, description="Positions with highest effects"
    )
    critical_layers: tuple[int, ...] = Field(
        default_factory=tuple, description="Layers with highest effects"
    )


# =============================================================================
# Intervention Hook
# =============================================================================


class InterventionHook:
    """Hook that applies interventions during forward pass."""

    def __init__(
        self,
        config: InterventionConfig,
        patch_activations: mx.array | None = None,
        steering_direction: mx.array | None = None,
    ):
        self.config = config
        self.patch_activations = patch_activations
        self.steering_direction = steering_direction
        self.captured: dict[int, mx.array] = {}

    def __call__(self, h: mx.array, layer_idx: int) -> mx.array:
        """Apply intervention to hidden states."""
        if layer_idx not in self.config.layers:
            return h

        # Apply intervention at specified positions
        positions = list(self.config.positions)
        seq_len = h.shape[1]

        # Handle negative indices
        positions = [p if p >= 0 else seq_len + p for p in positions]
        positions = [p for p in positions if 0 <= p < seq_len]

        if not positions:
            return h

        if self.config.intervention_type == InterventionType.ZERO:
            for pos in positions:
                h = self._set_position(h, pos, mx.zeros_like(h[:, pos, :]))

        elif self.config.intervention_type == InterventionType.SCALE:
            for pos in positions:
                h = self._set_position(h, pos, h[:, pos, :] * self.config.scale_factor)

        elif self.config.intervention_type == InterventionType.NOISE:
            for pos in positions:
                noise = mx.random.normal(h[:, pos, :].shape) * self.config.noise_scale
                h = self._set_position(h, pos, h[:, pos, :] + noise)

        elif self.config.intervention_type == InterventionType.PATCH:
            if self.patch_activations is not None:
                for pos in positions:
                    if pos < self.patch_activations.shape[1]:
                        h = self._set_position(h, pos, self.patch_activations[:, pos, :])

        elif self.config.intervention_type == InterventionType.STEER:
            if self.steering_direction is not None:
                for pos in positions:
                    h = self._set_position(h, pos, h[:, pos, :] + self.steering_direction)

        return h

    @staticmethod
    def _set_position(h: mx.array, pos: int, value: mx.array) -> mx.array:
        """Set a specific position in the hidden states."""
        # MLX doesn't support item assignment, so we reconstruct
        before = h[:, :pos, :] if pos > 0 else None
        after = h[:, pos + 1 :, :] if pos < h.shape[1] - 1 else None
        value = value.reshape(h.shape[0], 1, h.shape[2])

        parts = [p for p in [before, value, after] if p is not None]
        return mx.concatenate(parts, axis=1)


# =============================================================================
# Counterfactual Intervention Class
# =============================================================================


class CounterfactualIntervention:
    """
    Counterfactual intervention for causal analysis of language models.

    Provides methods for:
    - Activation patching
    - Causal tracing
    - What-if interventions
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

        # Detect model structure
        self._detect_structure()

        # State for capturing activations
        self._captured_activations: dict[int, mx.array] = {}
        self._active_hooks: list[Any] = []
        self._original_layers: dict[int, Any] = {}

    def _detect_structure(self) -> None:
        """Detect model structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = list(self.model.model.layers)
            self._backbone = self.model.model
        elif hasattr(self.model, "layers"):
            self._layers = list(self.model.layers)
            self._backbone = self.model
        else:
            raise ValueError("Cannot detect model layer structure")

        self.num_layers = len(self._layers)

    @classmethod
    def from_pretrained(cls, model_id: str) -> CounterfactualIntervention:
        """Load model for intervention experiments."""
        from .ablation import AblationStudy

        study = AblationStudy.from_pretrained(model_id)
        return cls(
            model=study.adapter.model,
            tokenizer=study.adapter.tokenizer,
            model_id=model_id,
        )

    # =========================================================================
    # Core Methods
    # =========================================================================

    def capture_activations(
        self,
        prompt: str,
        layers: list[int] | None = None,
    ) -> dict[int, mx.array]:
        """
        Run forward pass and capture hidden states at specified layers.

        Args:
            prompt: Input prompt
            layers: Layers to capture (None = all)

        Returns:
            Dict mapping layer_idx -> hidden states [batch, seq, hidden]
        """
        if layers is None:
            layers = list(range(self.num_layers))

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        captured: dict[int, mx.array] = {}

        # Wrap layers to capture activations
        original_layers = {}
        for layer_idx in layers:
            original_layers[layer_idx] = self._layers[layer_idx]

            layer = self._layers[layer_idx]

            class CaptureWrapper:
                def __init__(wrapper_self, wrapped, idx, captured_dict):
                    wrapper_self._wrapped = wrapped
                    wrapper_self._idx = idx
                    wrapper_self._captured = captured_dict

                def __call__(wrapper_self, h, **kwargs):
                    out = wrapper_self._wrapped(h, **kwargs)
                    if hasattr(out, "hidden_states"):
                        wrapper_self._captured[wrapper_self._idx] = out.hidden_states
                    elif isinstance(out, tuple):
                        wrapper_self._captured[wrapper_self._idx] = out[0]
                    else:
                        wrapper_self._captured[wrapper_self._idx] = out
                    return out

                def __getattr__(wrapper_self, name):
                    return getattr(wrapper_self._wrapped, name)

            self._layers[layer_idx] = CaptureWrapper(layer, layer_idx, captured)

        try:
            # Forward pass
            self.model(input_ids)
        finally:
            # Restore original layers
            for layer_idx, original in original_layers.items():
                self._layers[layer_idx] = original

        return captured

    def intervened_forward(
        self,
        prompt: str,
        config: InterventionConfig,
        patch_from: dict[int, mx.array] | None = None,
        steering_direction: mx.array | None = None,
    ) -> tuple[str, mx.array]:
        """
        Run forward pass with intervention applied.

        Args:
            prompt: Input prompt
            config: Intervention configuration
            patch_from: Activations to patch from (for PATCH type)
            steering_direction: Direction to steer (for STEER type)

        Returns:
            Tuple of (generated text, final logits)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Create hook
        hook = InterventionHook(
            config=config,
            patch_activations=(
                patch_from.get(list(config.layers)[0]) if patch_from and config.layers else None
            ),
            steering_direction=steering_direction,
        )

        # Wrap layers for intervention
        original_layers = {}
        for layer_idx in config.layers:
            if layer_idx >= len(self._layers):
                continue
            original_layers[layer_idx] = self._layers[layer_idx]
            layer = self._layers[layer_idx]

            class InterventionWrapper:
                def __init__(wrapper_self, wrapped, idx, hook_fn):
                    wrapper_self._wrapped = wrapped
                    wrapper_self._idx = idx
                    wrapper_self._hook = hook_fn

                def __call__(wrapper_self, h, **kwargs):
                    out = wrapper_self._wrapped(h, **kwargs)
                    if hasattr(out, "hidden_states"):
                        out.hidden_states = wrapper_self._hook(out.hidden_states, wrapper_self._idx)
                        return out
                    elif isinstance(out, tuple):
                        return (wrapper_self._hook(out[0], wrapper_self._idx),) + out[1:]
                    else:
                        return wrapper_self._hook(out, wrapper_self._idx)

                def __getattr__(wrapper_self, name):
                    return getattr(wrapper_self._wrapped, name)

            self._layers[layer_idx] = InterventionWrapper(layer, layer_idx, hook)

        try:
            # Forward and generate
            output = self._generate(input_ids, max_tokens=50)
            final_logits = self._get_next_logits(input_ids)
        finally:
            # Restore original layers
            for layer_idx, original in original_layers.items():
                self._layers[layer_idx] = original

        return output, final_logits

    # =========================================================================
    # High-Level APIs
    # =========================================================================

    def patch_run(
        self,
        clean_prompt: str,
        corrupt_prompt: str,
        patch_layers: list[int],
        patch_positions: list[int] | None = None,
    ) -> PatchingResult:
        """
        Run activation patching experiment.

        Captures activations from clean run, then patches them into
        corrupt run to measure recovery.

        Args:
            clean_prompt: Clean input prompt
            corrupt_prompt: Corrupted input prompt
            patch_layers: Layers to patch
            patch_positions: Positions to patch (default: last token)

        Returns:
            PatchingResult with recovery metrics
        """
        if patch_positions is None:
            patch_positions = [-1]

        # Capture clean activations
        clean_acts = self.capture_activations(clean_prompt, patch_layers)

        # Get clean and corrupt outputs
        clean_output = self._generate_from_prompt(clean_prompt, max_tokens=50)
        corrupt_output = self._generate_from_prompt(corrupt_prompt, max_tokens=50)

        # Patch run
        config = InterventionConfig(
            intervention_type=InterventionType.PATCH,
            target=ComponentTarget.HIDDEN,
            layers=tuple(patch_layers),
            positions=tuple(patch_positions),
        )

        patched_output, _ = self.intervened_forward(
            corrupt_prompt,
            config,
            patch_from=clean_acts,
        )

        # Compute recovery rate
        # Simple heuristic: compare similarity to clean vs corrupt
        clean_set = set(clean_output.split())
        corrupt_set = set(corrupt_output.split())
        patched_set = set(patched_output.split())

        if clean_set != corrupt_set:
            clean_dist = len(patched_set & clean_set) / max(1, len(clean_set))
            corrupt_dist = len(patched_set & corrupt_set) / max(1, len(corrupt_set))
            recovery = max(0.0, (clean_dist - corrupt_dist) / 2 + 0.5)
        else:
            recovery = 0.5

        return PatchingResult(
            clean_prompt=clean_prompt,
            corrupt_prompt=corrupt_prompt,
            clean_output=clean_output,
            corrupt_output=corrupt_output,
            patched_output=patched_output,
            recovery_rate=min(1.0, max(0.0, recovery)),
            effect_size=recovery - 0.5,
            patched_layers=tuple(patch_layers),
            patched_positions=tuple(patch_positions),
        )

    def trace_token(
        self,
        prompt: str,
        target_token: str,
        layers: list[int] | None = None,
        effect_threshold: float = 0.1,
    ) -> CausalTraceResult:
        """
        Trace where a target token's prediction is formed.

        Ablates each layer and measures effect on target token probability.

        Args:
            prompt: Input prompt
            target_token: Token to trace
            layers: Layers to test (default: all)
            effect_threshold: Threshold for "critical" layers

        Returns:
            CausalTraceResult with layer effects
        """
        if layers is None:
            layers = list(range(self.num_layers))

        # Get target token ID
        target_id = self.tokenizer.encode(target_token)
        if isinstance(target_id, list):
            target_id = target_id[0] if target_id else 0
        elif hasattr(target_id, "tolist"):
            target_id = target_id.tolist()[0] if len(target_id) > 0 else 0

        # Get baseline probability
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        baseline_logits = self._get_next_logits(input_ids)
        baseline_probs = mx.softmax(baseline_logits, axis=-1)
        baseline_prob = float(baseline_probs[0, target_id])

        # Test each layer
        layer_effects = []
        for layer_idx in layers:
            config = InterventionConfig(
                intervention_type=InterventionType.ZERO,
                target=ComponentTarget.HIDDEN,
                layers=(layer_idx,),
                positions=(-1,),
            )

            _, ablated_logits = self.intervened_forward(prompt, config)
            ablated_probs = mx.softmax(ablated_logits, axis=-1)
            ablated_prob = float(ablated_probs[0, target_id])

            effect = baseline_prob - ablated_prob
            layer_effects.append((layer_idx, effect))

        # Find critical layers
        sorted_effects = sorted(layer_effects, key=lambda x: abs(x[1]), reverse=True)
        critical = [layer for layer, effect in sorted_effects if abs(effect) >= effect_threshold]

        peak_layer, peak_effect = sorted_effects[0] if sorted_effects else (0, 0.0)

        return CausalTraceResult(
            prompt=prompt,
            target_token=target_token,
            target_token_id=target_id,
            layer_effects=tuple(layer_effects),
            critical_layers=tuple(critical[:5]),  # Top 5
            peak_layer=peak_layer,
            peak_effect=peak_effect,
            baseline_prob=baseline_prob,
        )

    def full_causal_trace(
        self,
        prompt: str,
        target_token: str,
        corrupt_prompt: str | None = None,
        layers: list[int] | None = None,
    ) -> FullCausalTrace:
        """
        Full causal tracing with position × layer grid.

        For each (position, layer), patches clean activation into corrupt run
        and measures recovery of target token.

        Args:
            prompt: Clean input prompt
            target_token: Token to trace
            corrupt_prompt: Corrupted prompt (default: adds noise)
            layers: Layers to test (default: all)

        Returns:
            FullCausalTrace with complete effect grid
        """
        if layers is None:
            layers = list(range(self.num_layers))

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        seq_len = input_ids.shape[1]

        # Get tokens
        tokens = []
        for i in range(seq_len):
            tok = self.tokenizer.decode([int(input_ids[0, i])])
            tokens.append(tok)

        # Get target token ID
        target_id = self.tokenizer.encode(target_token)
        if isinstance(target_id, list):
            target_id = target_id[0] if target_id else 0

        # Capture clean activations
        clean_acts = self.capture_activations(prompt, layers)

        # Get baseline probability
        baseline_logits = self._get_next_logits(input_ids)
        baseline_probs = mx.softmax(baseline_logits, axis=-1)
        baseline_prob = float(baseline_probs[0, target_id])

        # Create corrupt prompt if not provided
        if corrupt_prompt is None:
            # Simple corruption: replace random word
            words = prompt.split()
            if len(words) > 1:
                corrupt_prompt = " ".join(["[MASK]"] + words[1:])
            else:
                corrupt_prompt = "[MASK]"

        # Get corrupt baseline
        corrupt_ids = self.tokenizer.encode(corrupt_prompt, return_tensors="np")
        corrupt_ids = mx.array(corrupt_ids)
        corrupt_logits = self._get_next_logits(corrupt_ids)
        corrupt_probs = mx.softmax(corrupt_logits, axis=-1)
        corrupt_prob = float(corrupt_probs[0, target_id])

        # Build effect grid
        effects = []
        for pos in range(seq_len):
            pos_effects = []
            for layer_idx in layers:
                # Patch this position/layer
                config = InterventionConfig(
                    intervention_type=InterventionType.PATCH,
                    target=ComponentTarget.HIDDEN,
                    layers=(layer_idx,),
                    positions=(pos,),
                )

                _, patched_logits = self.intervened_forward(
                    corrupt_prompt,
                    config,
                    patch_from={layer_idx: clean_acts.get(layer_idx, mx.zeros((1, seq_len, 1)))},
                )
                patched_probs = mx.softmax(patched_logits, axis=-1)
                patched_prob = float(patched_probs[0, target_id])

                # Effect = how much probability was recovered
                if baseline_prob > corrupt_prob:
                    effect = (patched_prob - corrupt_prob) / max(0.01, baseline_prob - corrupt_prob)
                else:
                    effect = 0.0

                pos_effects.append(min(1.0, max(-1.0, effect)))

            effects.append(tuple(pos_effects))

        # Find critical positions and layers
        max_effects = [max(abs(e) for e in pos_effects) for pos_effects in effects]
        critical_positions = sorted(
            range(len(max_effects)), key=lambda i: max_effects[i], reverse=True
        )[:5]

        layer_max = [0.0] * len(layers)
        for pos_effects in effects:
            for i, e in enumerate(pos_effects):
                layer_max[i] = max(layer_max[i], abs(e))
        critical_layers = sorted(range(len(layers)), key=lambda i: layer_max[i], reverse=True)[:5]
        critical_layers = [layers[i] for i in critical_layers]

        return FullCausalTrace(
            prompt=prompt,
            target_token=target_token,
            tokens=tuple(tokens),
            effects=tuple(effects),
            critical_positions=tuple(critical_positions),
            critical_layers=tuple(critical_layers),
        )

    def ablate_component(
        self,
        prompt: str,
        layers: list[int],
        component: ComponentTarget = ComponentTarget.HIDDEN,
        positions: list[int] | None = None,
    ) -> InterventionResult:
        """
        Ablate (zero out) a component and observe effect.

        Args:
            prompt: Input prompt
            layers: Layers to ablate
            component: Component to ablate
            positions: Positions to ablate (default: all)

        Returns:
            InterventionResult comparing clean vs ablated
        """
        if positions is None:
            positions = [-1]

        # Get clean output
        clean_output = self._generate_from_prompt(prompt, max_tokens=50)
        clean_logits = self._get_next_logits(
            mx.array(self.tokenizer.encode(prompt, return_tensors="np"))
        )

        # Run with ablation
        config = InterventionConfig(
            intervention_type=InterventionType.ZERO,
            target=component,
            layers=tuple(layers),
            positions=tuple(positions),
        )

        ablated_output, ablated_logits = self.intervened_forward(prompt, config)

        # Compute effect size (L2 distance of logit distributions)
        effect = float(mx.sqrt(mx.sum((clean_logits - ablated_logits) ** 2)))

        # Compute KL divergence
        clean_probs = mx.softmax(clean_logits, axis=-1)
        ablated_probs = mx.softmax(ablated_logits, axis=-1)
        kl = float(
            mx.sum(clean_probs * (mx.log(clean_probs + 1e-10) - mx.log(ablated_probs + 1e-10)))
        )

        return InterventionResult(
            clean_output=clean_output,
            intervened_output=ablated_output,
            clean_logits=tuple(clean_logits[0, :10].tolist()),  # Top 10
            intervened_logits=tuple(ablated_logits[0, :10].tolist()),
            effect_size=effect,
            kl_divergence=max(0.0, kl),
            intervention_config=config,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from input IDs."""
        generated = []
        current_ids = input_ids

        for _ in range(max_tokens):
            outputs = self.model(current_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            if temperature == 0:
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                logits = logits[:, -1, :] / temperature
                next_token = mx.random.categorical(logits)

            token_id = int(next_token[0])
            generated.append(token_id)

            if hasattr(self.tokenizer, "eos_token_id"):
                if token_id == self.tokenizer.eos_token_id:
                    break

            current_ids = mx.concatenate([current_ids, next_token[:, None]], axis=1)

        return self.tokenizer.decode(generated)

    def _generate_from_prompt(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate from prompt string."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        return self._generate(input_ids, max_tokens)

    def _get_next_logits(self, input_ids: mx.array) -> mx.array:
        """Get logits for next token prediction."""
        outputs = self.model(input_ids)
        if hasattr(outputs, "logits"):
            return outputs.logits[:, -1, :]
        return outputs[:, -1, :]

    # =========================================================================
    # Printing Utilities
    # =========================================================================

    def print_patch_result(self, result: PatchingResult) -> None:
        """Print patching result summary."""
        print("\n" + "=" * 70)
        print("ACTIVATION PATCHING RESULT")
        print("=" * 70)
        print(f"Clean prompt: {result.clean_prompt}")
        print(f"Corrupt prompt: {result.corrupt_prompt}")
        print(f"Patched layers: {result.patched_layers}")
        print(f"Patched positions: {result.patched_positions}")
        print("-" * 70)
        print(f"Clean output: {result.clean_output}")
        print(f"Corrupt output: {result.corrupt_output}")
        print(f"Patched output: {result.patched_output}")
        print("-" * 70)
        recovery_bar = "█" * int(result.recovery_rate * 20) + "░" * (
            20 - int(result.recovery_rate * 20)
        )
        print(f"Recovery rate: [{recovery_bar}] {result.recovery_rate:.1%}")
        print(f"Effect size: {result.effect_size:+.2f}")

    def print_trace_result(self, result: CausalTraceResult) -> None:
        """Print causal trace result."""
        print("\n" + "=" * 70)
        print("CAUSAL TRACE RESULT")
        print("=" * 70)
        print(f"Prompt: {result.prompt}")
        print(f"Target: {result.target_token!r} (id={result.target_token_id})")
        print(f"Baseline probability: {result.baseline_prob:.2%}")
        print("-" * 70)

        print("\nLayer Effects:")
        for layer, effect in result.layer_effects:
            bar_len = int(abs(effect) * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            sign = "+" if effect > 0 else "-" if effect < 0 else " "
            marker = " *" if layer in result.critical_layers else ""
            print(f"L{layer:2d}: [{bar}] {sign}{abs(effect):.3f}{marker}")

        print("-" * 70)
        print(f"Peak layer: L{result.peak_layer} (effect={result.peak_effect:.3f})")
        print(f"Critical layers: {result.critical_layers}")

    def print_full_trace(self, result: FullCausalTrace, max_positions: int = 15) -> None:
        """Print full causal trace heatmap."""
        print("\n" + "=" * 80)
        print("FULL CAUSAL TRACE")
        print("=" * 80)
        print(f"Prompt: {result.prompt}")
        print(f"Target: {result.target_token!r}")
        print(f"Tokens: {len(result.tokens)}")

        # Limit positions shown
        positions_to_show = min(len(result.tokens), max_positions)

        print("-" * 80)
        print("Effect Grid (position × layer):")
        print("Intensity: ░ < 25% < ▒ < 50% < ▓ < 75% < █")
        print()

        # Header
        header = "Pos Token      |"
        for i, layer in enumerate(result.critical_layers[:8]):
            header += f" L{layer:2d}"
        print(header)
        print("-" * len(header))

        # Rows
        chars = " ░▒▓█"
        for pos in range(positions_to_show):
            token = result.tokens[pos][:8].ljust(8)
            row = f"{pos:3d} {token}   |"

            if pos < len(result.effects):
                for layer_idx in result.critical_layers[:8]:
                    if layer_idx < len(result.effects[pos]):
                        effect = result.effects[pos][layer_idx]
                        char_idx = int(abs(effect) * (len(chars) - 1))
                        char_idx = min(char_idx, len(chars) - 1)
                        row += f"  {chars[char_idx]} "
                    else:
                        row += "    "

            print(row)

        print("-" * 80)
        print(f"Critical positions: {result.critical_positions}")
        print(f"Critical layers: {result.critical_layers}")


# =============================================================================
# Convenience Functions
# =============================================================================


def patch_activations(
    model: nn.Module,
    tokenizer: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    patch_layers: list[int],
    patch_positions: list[int] | None = None,
) -> PatchingResult:
    """
    Convenience function for activation patching.

    Args:
        model: The model
        tokenizer: Tokenizer
        clean_prompt: Clean input prompt
        corrupt_prompt: Corrupted input prompt
        patch_layers: Layers to patch
        patch_positions: Positions to patch

    Returns:
        PatchingResult
    """
    ci = CounterfactualIntervention(model, tokenizer)
    return ci.patch_run(clean_prompt, corrupt_prompt, patch_layers, patch_positions)


def trace_causal_path(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    target_token: str,
    layers: list[int] | None = None,
) -> CausalTraceResult:
    """
    Convenience function for causal tracing.

    Args:
        model: The model
        tokenizer: Tokenizer
        prompt: Input prompt
        target_token: Token to trace
        layers: Layers to test

    Returns:
        CausalTraceResult
    """
    ci = CounterfactualIntervention(model, tokenizer)
    return ci.trace_token(prompt, target_token, layers)
