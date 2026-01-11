"""Activation patching for causal intervention experiments.

Provides tools for patching activations from one prompt into another
to test causal relationships in neural network computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .accessor import AsyncModelAccessor
from .enums import PatchEffect
from .models.patching import PatchingLayerResult, PatchingResult

if TYPE_CHECKING:
    import mlx.core as mx

    from .models.patching import CommutativityResult


class LayerPatch(BaseModel):
    """A patch to apply at a specific layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    layer: int = Field(description="Layer index to patch")
    activation: Any = Field(description="Activation to patch (numpy or mlx array)")
    blend: float = Field(default=1.0, ge=0.0, le=1.0, description="Blend factor")
    position: int = Field(default=-1, description="Token position (-1 for last)")


class ActivationPatcher(BaseModel):
    """Activation patching for causal intervention experiments.

    Example:
        >>> patcher = ActivationPatcher(model=model, tokenizer=tokenizer, config=config)
        >>> # Capture source activation
        >>> source_activation = await patcher.capture_activation("7*8=", layer=22)
        >>> # Patch into target
        >>> result = await patcher.patch_and_generate(
        ...     target_prompt="7+8=",
        ...     source_activation=source_activation,
        ...     layer=22,
        ...     blend=1.0,
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(description="The neural network model")
    tokenizer: Any = Field(description="The tokenizer")
    config: Any = Field(default=None, description="Optional configuration")
    _accessor: AsyncModelAccessor = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes after model creation."""
        self._accessor = AsyncModelAccessor(model=self.model, config=self.config)

    async def capture_activation(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Capture activation at a specific layer and position.

        Args:
            prompt: The prompt to process
            layer: Layer index to capture
            position: Token position (-1 for last)

        Returns:
            Activation vector as numpy array
        """
        import mlx.core as mx

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        captured = await self._accessor.forward_through_layers(
            input_ids,
            layers=[layer],
            capture_hidden_states=True,
        )

        h = captured[layer]
        if position == -1:
            position = h.shape[1] - 1
        activation = h[0, position, :]
        return np.array(activation.astype(mx.float32), copy=False)

    def _create_patched_layer(
        self,
        original_layer: Any,
        source_activation: mx.array,
        blend: float,
        position: int = -1,
    ) -> Any:
        """Create a wrapper layer that patches activations."""
        import mlx.core as mx

        class PatchedLayerWrapper:
            def __init__(self, layer: Any, activation: mx.array, blend: float, pos: int):
                self._wrapped = layer
                self._activation = activation
                self._blend = blend
                self._position = pos
                # Copy attributes for compatibility
                for attr in [
                    "mlp",
                    "attn",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                ]:
                    if hasattr(layer, attr):
                        setattr(self, attr, getattr(layer, attr))

            def __call__(self, h: mx.array, **kwargs) -> Any:
                result = self._wrapped(h, **kwargs)

                # Extract hidden states
                if hasattr(result, "hidden_states"):
                    hs = result.hidden_states
                elif isinstance(result, tuple):
                    hs = result[0]
                else:
                    hs = result

                # Determine position to patch
                pos = self._position if self._position >= 0 else hs.shape[1] - 1

                # Patch: blend original with source activation
                original = hs[:, pos : pos + 1, :]
                patched = (1 - self._blend) * original + self._blend * self._activation.reshape(
                    1, 1, -1
                )
                new_hs = mx.concatenate([hs[:, :pos, :], patched, hs[:, pos + 1 :, :]], axis=1)

                if hasattr(result, "hidden_states"):
                    result.hidden_states = new_hs
                    return result
                elif isinstance(result, tuple):
                    return (new_hs,) + result[1:]
                return new_hs

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        return PatchedLayerWrapper(original_layer, source_activation, blend, position)

    async def patch_and_predict(
        self,
        target_prompt: str,
        source_activation: np.ndarray | mx.array,
        layer: int,
        blend: float = 1.0,
        position: int = -1,
    ) -> tuple[str, float]:
        """Patch activation and get top prediction.

        Args:
            target_prompt: Prompt to patch into
            source_activation: Activation to inject
            layer: Layer to patch at
            blend: Blend factor (0=original, 1=full replacement)
            position: Position to patch (-1 for last)

        Returns:
            Tuple of (top_token, probability)
        """
        import mlx.core as mx

        # Convert to mx.array if needed
        if isinstance(source_activation, np.ndarray):
            source_activation = mx.array(source_activation.astype(np.float32))

        # Save original layer
        original_layer = self._accessor.get_layer(layer)

        # Install patched layer
        patched_layer = self._create_patched_layer(
            original_layer, source_activation, blend, position
        )
        self._accessor.set_layer(layer, patched_layer)

        try:
            # Run forward pass
            input_ids = mx.array(self.tokenizer.encode(target_prompt))[None, :]
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Get top prediction
            probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_idx = int(mx.argmax(probs))
            top_prob = float(probs[top_idx])
            top_token = self.tokenizer.decode([top_idx])

            return top_token, top_prob
        finally:
            # Restore original layer
            self._accessor.set_layer(layer, original_layer)

    async def sweep_layers(
        self,
        target_prompt: str,
        source_prompt: str,
        layers: list[int] | None = None,
        blend: float = 1.0,
        source_answer: str | None = None,
        target_answer: str | None = None,
    ) -> PatchingResult:
        """Sweep patching across multiple layers.

        Args:
            target_prompt: Prompt to patch into
            source_prompt: Prompt to get source activation from
            layers: Layers to test (None = every 10th layer)
            blend: Blend factor
            source_answer: Expected answer from source (for transfer detection)
            target_answer: Expected answer from target

        Returns:
            Complete patching result
        """
        import mlx.core as mx

        if layers is None:
            num_layers = self._accessor.num_layers
            layers = list(range(0, num_layers, max(1, num_layers // 10)))

        # Get baseline prediction
        input_ids = mx.array(self.tokenizer.encode(target_prompt))[None, :]
        outputs = self.model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        baseline_idx = int(mx.argmax(probs))
        baseline_prob = float(probs[baseline_idx])
        baseline_token = self.tokenizer.decode([baseline_idx])

        # Capture source activations at all layers
        source_ids = mx.array(self.tokenizer.encode(source_prompt))[None, :]
        source_captured = await self._accessor.forward_through_layers(source_ids, layers=layers)

        # Test each layer
        layer_results = []
        for layer in layers:
            source_activation = source_captured[layer][0, -1, :]
            top_token, top_prob = await self.patch_and_predict(
                target_prompt,
                source_activation,
                layer,
                blend,
            )

            # Determine effect
            if top_token == baseline_token:
                effect = PatchEffect.NO_CHANGE
            elif source_answer and source_answer.startswith(top_token.strip()):
                effect = PatchEffect.TRANSFERRED
            elif target_answer and target_answer.startswith(top_token.strip()):
                effect = PatchEffect.STILL_TARGET
            else:
                effect = PatchEffect.CHANGED

            layer_results.append(
                PatchingLayerResult(
                    layer=layer,
                    top_token=top_token,
                    top_prob=top_prob,
                    baseline_token=baseline_token,
                    baseline_prob=baseline_prob,
                    effect=effect,
                )
            )

        return PatchingResult(
            model_id=getattr(self.config, "model_id", "unknown"),
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            source_answer=source_answer,
            target_answer=target_answer,
            blend=blend,
            layers=layers,
            baseline_token=baseline_token,
            baseline_prob=baseline_prob,
            layer_results=layer_results,
        )


class CommutativityAnalyzer(BaseModel):
    """Analyze whether representations respect commutativity (A*B = B*A).

    Example:
        >>> analyzer = CommutativityAnalyzer(model=model, tokenizer=tokenizer, config=config)
        >>> result = await analyzer.analyze(layer=22)
        >>> print(f"Mean similarity: {result.mean_similarity:.4f}")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(description="The neural network model")
    tokenizer: Any = Field(description="The tokenizer")
    config: Any = Field(default=None, description="Optional configuration")
    _accessor: AsyncModelAccessor = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes after model creation."""
        self._accessor = AsyncModelAccessor(model=self.model, config=self.config)

    async def get_activation(self, prompt: str, layer: int) -> np.ndarray:
        """Get last-token hidden state for a prompt at a given layer."""
        import mlx.core as mx

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        captured = await self._accessor.forward_through_layers(input_ids, layers=[layer])
        h = captured[layer][0, -1, :]
        return np.array(h.astype(mx.float32), copy=False)

    async def analyze(
        self,
        layer: int | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ) -> CommutativityResult:
        """Analyze commutativity at a specific layer.

        Args:
            layer: Layer to analyze (default: 60% through network)
            pairs: Explicit pairs to test (default: all single-digit multiplication)

        Returns:
            CommutativityResult with similarity statistics
        """
        from .models.patching import CommutativityPair, CommutativityResult

        if layer is None:
            layer = int(self._accessor.num_layers * 0.6)

        if pairs is None:
            pairs = []
            for a in range(2, 10):
                for b in range(a + 1, 10):
                    pairs.append((f"{a}*{b}=", f"{b}*{a}="))

        similarities = []
        pair_results = []

        for prompt_a, prompt_b in pairs:
            h_a = await self.get_activation(prompt_a, layer)
            h_b = await self.get_activation(prompt_b, layer)

            # Cosine similarity
            dot = np.dot(h_a, h_b)
            norm_a = np.linalg.norm(h_a)
            norm_b = np.linalg.norm(h_b)
            sim = float(dot / (norm_a * norm_b + 1e-8))

            similarities.append(sim)
            pair_results.append(
                CommutativityPair(
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    similarity=sim,
                )
            )

        return CommutativityResult(
            model_id=getattr(self.config, "model_id", "unknown"),
            layer=layer,
            num_pairs=len(pairs),
            mean_similarity=float(np.mean(similarities)),
            std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            pairs=pair_results,
        )
