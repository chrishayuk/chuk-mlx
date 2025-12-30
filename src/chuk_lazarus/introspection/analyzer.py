"""
Async-native model analyzer with pydantic models.

Provides a clean API for introspecting models:
- Load models from HuggingFace
- Run logit lens analysis
- Track token evolution across layers

Example:
    >>> from chuk_lazarus.introspection import ModelAnalyzer, AnalysisConfig
    >>>
    >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
    ...     result = await analyzer.analyze("The capital of France is")
    ...     print(result.final_prediction)
    ...     for layer in result.layer_predictions:
    ...         print(f"Layer {layer.layer_idx}: {layer.top_token}")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from .hooks import CaptureConfig, ModelHooks, PositionSelection
from .logit_lens import LogitLens


class LayerStrategy(str, Enum):
    """Strategy for selecting which layers to capture."""

    ALL = "all"
    EVENLY_SPACED = "evenly_spaced"
    FIRST_LAST = "first_last"
    CUSTOM = "custom"


class TrackStrategy(str, Enum):
    """Strategy for automatic token tracking."""

    MANUAL = "manual"  # Use track_tokens list explicitly
    TOP_K_FINAL = "top_k_final"  # Track top-k tokens from final layer
    EMERGENT = "emergent"  # Find tokens that spike mid-network
    TOOL_TOKENS = "tool_tokens"  # Track common tool-calling tokens


class AnalysisConfig(BaseModel):
    """Configuration for model analysis."""

    layer_strategy: LayerStrategy = Field(
        default=LayerStrategy.EVENLY_SPACED,
        description="How to select layers for capture",
    )
    layer_step: int = Field(
        default=4,
        ge=1,
        description="Step size for evenly spaced layer capture",
    )
    custom_layers: list[int] | None = Field(
        default=None,
        description="Specific layers to capture when using CUSTOM strategy",
    )
    position_strategy: PositionSelection = Field(
        default=PositionSelection.LAST,
        description="Which sequence positions to capture",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top predictions to return",
    )
    track_tokens: list[str] = Field(
        default_factory=list,
        description="Tokens to track across layers (when using MANUAL strategy)",
    )
    track_strategy: TrackStrategy = Field(
        default=TrackStrategy.MANUAL,
        description="Strategy for automatic token tracking",
    )
    compute_entropy: bool = Field(
        default=True,
        description="Compute entropy for each layer's distribution",
    )
    compute_transitions: bool = Field(
        default=True,
        description="Compute KL/JS divergence between consecutive layers",
    )
    compute_residual_decomposition: bool = Field(
        default=False,
        description="Decompose residual stream into attention vs FFN contributions (requires ALL layers)",
    )

    model_config = ConfigDict(use_enum_values=False)


class TokenPrediction(BaseModel):
    """A single token prediction with probability."""

    token: str = Field(description="Decoded token string")
    token_id: int = Field(description="Token ID in vocabulary")
    probability: float = Field(ge=0, le=1, description="Probability")
    rank: int = Field(ge=1, description="Rank among all tokens")

    model_config = ConfigDict(frozen=True)


class LayerPredictionResult(BaseModel):
    """Predictions at a single layer."""

    layer_idx: int = Field(ge=0, description="Layer index")
    predictions: list[TokenPrediction] = Field(description="Top-k predictions")
    entropy: float = Field(default=0.0, description="Shannon entropy of the full distribution")
    entropy_normalized: float = Field(
        default=0.0, description="Entropy normalized by max entropy (0=certain, 1=uniform)"
    )

    @property
    def top_token(self) -> str:
        """Get the top predicted token."""
        return self.predictions[0].token if self.predictions else ""

    @property
    def top_probability(self) -> float:
        """Get the probability of top token."""
        return self.predictions[0].probability if self.predictions else 0.0

    @property
    def is_confident(self) -> bool:
        """Check if the layer is confident (normalized entropy < 0.3)."""
        return self.entropy_normalized < 0.3

    model_config = ConfigDict(frozen=True)


class LayerTransition(BaseModel):
    """Metrics for the transition between two consecutive layers."""

    from_layer: int = Field(ge=0, description="Source layer index")
    to_layer: int = Field(ge=0, description="Target layer index")
    kl_divergence: float = Field(ge=0, description="KL divergence from source to target")
    js_divergence: float = Field(ge=0, description="Jensen-Shannon divergence (symmetric)")
    top_token_changed: bool = Field(description="Whether the top prediction changed")
    entropy_delta: float = Field(description="Change in entropy (negative = more confident)")

    @property
    def is_significant(self) -> bool:
        """Check if this transition shows significant computation (JS > 0.1)."""
        return self.js_divergence > 0.1

    model_config = ConfigDict(frozen=True)


class ResidualContribution(BaseModel):
    """Contribution of a component (attention or FFN) to the residual stream.

    Computed by measuring how much each component changes the probability
    distribution when its contribution is ablated.
    """

    layer_idx: int = Field(ge=0, description="Layer index")
    attention_norm: float = Field(ge=0, description="L2 norm of attention contribution")
    ffn_norm: float = Field(ge=0, description="L2 norm of FFN contribution")
    total_norm: float = Field(ge=0, description="L2 norm of total layer contribution")
    attention_fraction: float = Field(
        ge=0, le=1, description="Attention's fraction of total contribution"
    )
    ffn_fraction: float = Field(ge=0, le=1, description="FFN's fraction of total contribution")

    @property
    def dominant_component(self) -> str:
        """Which component contributes more: 'attention' or 'ffn'."""
        return "attention" if self.attention_fraction > self.ffn_fraction else "ffn"

    model_config = ConfigDict(frozen=True)


class TokenEvolutionResult(BaseModel):
    """Evolution of a specific token's probability across layers."""

    token: str = Field(description="The tracked token")
    token_id: int = Field(description="Token ID")
    layer_probabilities: dict[int, float] = Field(description="Probability at each captured layer")
    layer_ranks: dict[int, int | None] = Field(
        description="Rank at each layer (None if not in top-100)"
    )
    emergence_layer: int | None = Field(
        default=None,
        description="First layer where token becomes top-1",
    )

    model_config = ConfigDict(frozen=True)


class AnalysisResult(BaseModel):
    """Complete analysis result for a prompt."""

    prompt: str = Field(description="The analyzed prompt")
    tokens: list[str] = Field(description="Tokenized prompt")
    num_layers: int = Field(ge=1, description="Total layers in model")
    captured_layers: list[int] = Field(description="Layers that were captured")
    final_prediction: list[TokenPrediction] = Field(
        description="Top-k predictions from final layer"
    )
    layer_predictions: list[LayerPredictionResult] = Field(
        description="Predictions at each captured layer"
    )
    layer_transitions: list[LayerTransition] = Field(
        default_factory=list,
        description="Transition metrics between consecutive captured layers",
    )
    token_evolutions: list[TokenEvolutionResult] = Field(
        default_factory=list,
        description="Evolution of tracked tokens",
    )
    residual_contributions: list[ResidualContribution] = Field(
        default_factory=list,
        description="Residual stream decomposition (attention vs FFN) per layer",
    )

    @property
    def predicted_token(self) -> str:
        """Get the model's top prediction."""
        return self.final_prediction[0].token if self.final_prediction else ""

    @property
    def predicted_probability(self) -> float:
        """Get probability of top prediction."""
        return self.final_prediction[0].probability if self.final_prediction else 0.0

    @property
    def decision_layer(self) -> int | None:
        """Find the layer where the final prediction first becomes confident.

        Returns the first layer where:
        1. The top token matches the final prediction
        2. The layer is confident (normalized entropy < 0.3)

        Returns None if the model never becomes confident.
        """
        final_token = self.predicted_token
        for pred in self.layer_predictions:
            if pred.top_token == final_token and pred.is_confident:
                return pred.layer_idx
        return None

    @property
    def max_kl_transition(self) -> LayerTransition | None:
        """Find the transition with highest KL divergence (where computation happens)."""
        if not self.layer_transitions:
            return None
        return max(self.layer_transitions, key=lambda t: t.kl_divergence)

    @property
    def significant_transitions(self) -> list[LayerTransition]:
        """Get all transitions with JS divergence > 0.1."""
        return [t for t in self.layer_transitions if t.is_significant]

    @property
    def attention_dominant_layers(self) -> list[int]:
        """Layers where attention contributes more than FFN."""
        return [
            c.layer_idx for c in self.residual_contributions if c.dominant_component == "attention"
        ]

    @property
    def ffn_dominant_layers(self) -> list[int]:
        """Layers where FFN contributes more than attention."""
        return [c.layer_idx for c in self.residual_contributions if c.dominant_component == "ffn"]

    @property
    def max_attention_layer(self) -> int | None:
        """Layer with highest attention contribution."""
        if not self.residual_contributions:
            return None
        return max(self.residual_contributions, key=lambda c: c.attention_norm).layer_idx

    @property
    def max_ffn_layer(self) -> int | None:
        """Layer with highest FFN contribution."""
        if not self.residual_contributions:
            return None
        return max(self.residual_contributions, key=lambda c: c.ffn_norm).layer_idx

    model_config = ConfigDict(frozen=True)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str = Field(description="Model identifier")
    num_layers: int = Field(ge=1, description="Number of layers")
    hidden_size: int = Field(ge=1, description="Hidden dimension size")
    vocab_size: int = Field(ge=1, description="Vocabulary size")
    has_tied_embeddings: bool = Field(description="Whether embeddings are tied")

    model_config = ConfigDict(frozen=True)


class ModelAnalyzer:
    """
    Async-native model analyzer for introspection.

    Provides a clean interface for:
    - Loading models from HuggingFace
    - Running logit lens analysis
    - Tracking token evolution

    Example:
        >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
        ...     result = await analyzer.analyze("Hello world")
        ...     print(result.predicted_token)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
        embedding_scale: float | None = None,
        config: Any | None = None,
    ):
        """
        Initialize analyzer with a model and tokenizer.

        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            model_id: Identifier for the model
            embedding_scale: Optional scale factor for embeddings.
                Some models (e.g., Gemma) scale embeddings by sqrt(hidden_size).
                If not provided, will try to detect from config or model properties.
            config: Optional model config (provides embedding_scale, layer info, etc.)
        """
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._config = config
        self._hooks: ModelHooks | None = None

        # Get embedding scale from config if not explicitly provided
        if embedding_scale is not None:
            self._embedding_scale = embedding_scale
        elif config is not None and hasattr(config, "embedding_scale"):
            self._embedding_scale = config.embedding_scale
        else:
            self._embedding_scale = None

    @classmethod
    @asynccontextmanager
    async def from_pretrained(
        cls,
        model_id: str,
        embedding_scale: float | None = None,
    ) -> AsyncIterator[ModelAnalyzer]:
        """
        Load a model from HuggingFace and create an analyzer.

        Uses the model families registry to auto-detect the model type
        and load with the appropriate config. Embedding scale is
        automatically detected from the config for Gemma and other
        models that need it.

        Args:
            model_id: HuggingFace model ID or local path
            embedding_scale: Optional scale factor for embeddings.
                Usually auto-detected from config. Override if needed.

        Yields:
            ModelAnalyzer instance

        Example:
            >>> # Any model works - family auto-detected
            >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
            ...     result = await analyzer.analyze("Hello")

            >>> # Gemma models automatically get embedding_scale from config
            >>> async with ModelAnalyzer.from_pretrained("mlx-community/gemma-3-270m-it-bf16") as analyzer:
            ...     result = await analyzer.analyze("Hello")
        """
        # Load model in thread pool to not block
        loop = asyncio.get_event_loop()
        model, tokenizer, config = await loop.run_in_executor(
            None,
            lambda: _load_model_sync(model_id),
        )

        analyzer = cls(model, tokenizer, model_id, embedding_scale=embedding_scale, config=config)
        try:
            yield analyzer
        finally:
            # Cleanup if needed
            pass

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "custom",
        embedding_scale: float | None = None,
        config: Any | None = None,
    ) -> ModelAnalyzer:
        """
        Create analyzer from an existing model.

        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            model_id: Optional identifier
            embedding_scale: Optional scale factor for embeddings.
            config: Optional model config (provides embedding_scale, layer info)

        Returns:
            ModelAnalyzer instance
        """
        return cls(model, tokenizer, model_id, embedding_scale=embedding_scale, config=config)

    @property
    def config(self) -> Any | None:
        """Get the model config if available."""
        return self._config

    @property
    def model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        # Use config if available, fall back to introspection
        if self._config is not None:
            num_layers = self._config.num_hidden_layers
            hidden_size = self._config.hidden_size
            vocab_size = self._config.vocab_size
            has_tied = getattr(self._config, "tie_word_embeddings", False)
        else:
            num_layers = self._get_num_layers()
            hidden_size = self._get_hidden_size()
            vocab_size = self._get_vocab_size()
            has_tied = getattr(self._model, "tie_word_embeddings", False)

        return ModelInfo(
            model_id=self._model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            has_tied_embeddings=has_tied,
        )

    async def analyze(
        self,
        prompt: str,
        config: AnalysisConfig | None = None,
    ) -> AnalysisResult:
        """
        Analyze a prompt using logit lens.

        Args:
            prompt: Text prompt to analyze
            config: Analysis configuration

        Returns:
            AnalysisResult with predictions at each layer
        """
        if config is None:
            config = AnalysisConfig()

        # Run analysis in thread pool (MLX operations)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._analyze_sync(prompt, config),
        )

    async def analyze_batch(
        self,
        prompts: list[str],
        config: AnalysisConfig | None = None,
    ) -> list[AnalysisResult]:
        """
        Analyze multiple prompts.

        Args:
            prompts: List of prompts to analyze
            config: Analysis configuration (shared across all)

        Returns:
            List of AnalysisResult, one per prompt

        Note:
            Analyses run sequentially to avoid MLX Metal backend
            threading issues when multiple GPU operations run concurrently.
        """
        # Run analyses sequentially to avoid MLX Metal thread-safety issues
        # (concurrent GPU operations from multiple threads can cause segfaults)
        results = []
        for prompt in prompts:
            result = await self.analyze(prompt, config)
            results.append(result)
        return results

    def _analyze_sync(
        self,
        prompt: str,
        config: AnalysisConfig,
    ) -> AnalysisResult:
        """Synchronous analysis implementation."""
        import math

        # Tokenize
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        tokens = [self._tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Get layers to capture
        num_layers = self._get_num_layers()
        layers_to_capture = self._get_layers_to_capture(num_layers, config)

        # Setup hooks with proper enum-based config
        # Pass both embedding_scale and config for maximum compatibility
        hooks = ModelHooks(
            self._model, embedding_scale=self._embedding_scale, model_config=self._config
        )
        hooks.configure(
            CaptureConfig(
                layers=layers_to_capture,
                capture_hidden_states=True,
                positions=config.position_strategy,
            )
        )

        # Forward pass
        logits = hooks.forward(input_ids)

        # Get final predictions
        final_predictions = self._get_top_predictions(logits[0, -1, :], config.top_k)

        # Get layer predictions using logit lens with entropy
        lens = LogitLens(hooks, self._tokenizer)
        layer_predictions = []
        layer_probs_cache: dict[int, mx.array] = {}  # Cache probs for transition computation
        vocab_size = self._get_vocab_size()
        max_entropy = math.log(vocab_size)  # Maximum entropy for uniform distribution

        for pred in lens.get_layer_predictions(position=-1, top_k=config.top_k):
            predictions = [
                TokenPrediction(
                    token=tok,
                    token_id=tid,
                    probability=prob,
                    rank=i + 1,
                )
                for i, (tok, tid, prob) in enumerate(
                    zip(pred.top_tokens, pred.top_ids, pred.top_probs)
                )
            ]

            # Compute entropy if requested
            entropy = 0.0
            entropy_normalized = 0.0
            if config.compute_entropy:
                # Get layer logits via hooks and compute full distribution
                layer_logits = hooks.get_layer_logits(pred.layer_idx, normalize=True)
                if layer_logits is not None:
                    # Get logits for last position
                    if layer_logits.ndim == 3:
                        pos_logits = layer_logits[0, -1, :]
                    else:
                        pos_logits = layer_logits[-1, :]
                    probs = mx.softmax(pos_logits)
                    layer_probs_cache[pred.layer_idx] = probs
                    entropy = self._compute_entropy(probs)
                    entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

            layer_predictions.append(
                LayerPredictionResult(
                    layer_idx=pred.layer_idx,
                    predictions=predictions,
                    entropy=entropy,
                    entropy_normalized=entropy_normalized,
                )
            )

        # Compute layer transitions (KL/JS divergence)
        layer_transitions = []
        if config.compute_transitions and len(layer_predictions) > 1:
            for i in range(len(layer_predictions) - 1):
                from_pred = layer_predictions[i]
                to_pred = layer_predictions[i + 1]

                from_layer = from_pred.layer_idx
                to_layer = to_pred.layer_idx

                # Get probability distributions
                from_probs = layer_probs_cache.get(from_layer)
                to_probs = layer_probs_cache.get(to_layer)

                if from_probs is not None and to_probs is not None:
                    kl_div = self._compute_kl_divergence(from_probs, to_probs)
                    js_div = self._compute_js_divergence(from_probs, to_probs)
                else:
                    kl_div = 0.0
                    js_div = 0.0

                layer_transitions.append(
                    LayerTransition(
                        from_layer=from_layer,
                        to_layer=to_layer,
                        kl_divergence=kl_div,
                        js_divergence=js_div,
                        top_token_changed=from_pred.top_token != to_pred.top_token,
                        entropy_delta=to_pred.entropy - from_pred.entropy,
                    )
                )

        # Determine tokens to track based on strategy
        tokens_to_track = self._get_tokens_to_track(config, layer_predictions)

        # Track token evolutions
        token_evolutions = []
        for token in tokens_to_track:
            try:
                evolution = lens.track_token(token, position=-1)
                token_evolutions.append(
                    TokenEvolutionResult(
                        token=token,
                        token_id=evolution.token_id,
                        layer_probabilities=dict(zip(evolution.layers, evolution.probabilities)),
                        layer_ranks=dict(zip(evolution.layers, evolution.ranks)),
                        emergence_layer=evolution.emergence_layer,
                    )
                )
            except Exception:
                # Token not found in vocabulary
                pass

        # Compute residual stream decomposition if requested
        residual_contributions = []
        if config.compute_residual_decomposition:
            residual_contributions = self._compute_residual_decomposition(hooks, layers_to_capture)

        return AnalysisResult(
            prompt=prompt,
            tokens=tokens,
            num_layers=num_layers,
            captured_layers=layers_to_capture,
            final_prediction=final_predictions,
            layer_predictions=layer_predictions,
            layer_transitions=layer_transitions,
            token_evolutions=token_evolutions,
            residual_contributions=residual_contributions,
        )

    def _compute_entropy(self, probs: mx.array) -> float:
        """Compute Shannon entropy of a probability distribution."""
        # Avoid log(0) by clipping
        probs_clipped = mx.clip(probs, 1e-10, 1.0)
        entropy = -mx.sum(probs_clipped * mx.log(probs_clipped))
        return float(entropy)

    def _compute_kl_divergence(self, p: mx.array, q: mx.array) -> float:
        """Compute KL divergence D(P || Q)."""
        # Clip to avoid log(0) and division by zero
        p_clipped = mx.clip(p, 1e-10, 1.0)
        q_clipped = mx.clip(q, 1e-10, 1.0)
        kl = mx.sum(p_clipped * mx.log(p_clipped / q_clipped))
        return float(mx.maximum(kl, mx.array(0.0)))  # KL should be >= 0

    def _compute_js_divergence(self, p: mx.array, q: mx.array) -> float:
        """Compute Jensen-Shannon divergence (symmetric, bounded [0, ln(2)])."""
        m = (p + q) / 2
        js = 0.5 * self._compute_kl_divergence(p, m) + 0.5 * self._compute_kl_divergence(q, m)
        return js

    def _compute_residual_decomposition(
        self,
        hooks: ModelHooks,
        captured_layers: list[int],
    ) -> list[ResidualContribution]:
        """Compute residual stream decomposition using hidden state differences.

        This computes the contribution of each layer to the residual stream by
        measuring the L2 norm of the hidden state change. For a true attention vs FFN
        decomposition, we would need to capture intermediate states within each layer.

        Since standard transformer layers are: h' = h + attn(norm(h)) + ffn(norm(h + attn))
        We approximate using the layer-wise contribution norm.

        For models that expose attention/FFN outputs separately (via capture_attention_output
        and capture_ffn_output in CaptureConfig), we use those directly.
        """
        contributions = []

        # Get embeddings as the "layer -1" state
        embeddings = hooks.state.embeddings
        if embeddings is None:
            return contributions

        # Get last position hidden state from embeddings
        if embeddings.ndim == 3:
            prev_hidden = embeddings[0, -1, :]  # [hidden_size]
        else:
            prev_hidden = embeddings[-1, :]

        for layer_idx in captured_layers:
            hidden = hooks.state.hidden_states.get(layer_idx)
            if hidden is None:
                continue

            # Get last position hidden state
            if hidden.ndim == 3:
                curr_hidden = hidden[0, -1, :]
            else:
                curr_hidden = hidden[-1, :]

            # Compute total layer contribution (delta)
            delta = curr_hidden - prev_hidden
            total_norm = float(mx.sqrt(mx.sum(delta * delta)))

            # Try to get attention and FFN contributions
            attn_output = hooks.state.attention_outputs.get(layer_idx)
            ffn_output = hooks.state.ffn_outputs.get(layer_idx)

            if attn_output is not None and ffn_output is not None:
                # We have separate outputs - compute their norms
                if attn_output.ndim == 3:
                    attn_vec = attn_output[0, -1, :]
                else:
                    attn_vec = attn_output[-1, :]

                if ffn_output.ndim == 3:
                    ffn_vec = ffn_output[0, -1, :]
                else:
                    ffn_vec = ffn_output[-1, :]

                attn_norm = float(mx.sqrt(mx.sum(attn_vec * attn_vec)))
                ffn_norm = float(mx.sqrt(mx.sum(ffn_vec * ffn_vec)))
            else:
                # Approximate: split total contribution equally
                # In reality, attention and FFN contribute differently,
                # but without intermediate captures, we can only report total
                attn_norm = total_norm / 2.0
                ffn_norm = total_norm / 2.0

            # Compute fractions (handle zero case)
            total = attn_norm + ffn_norm
            if total > 0:
                attn_fraction = attn_norm / total
                ffn_fraction = ffn_norm / total
            else:
                attn_fraction = 0.5
                ffn_fraction = 0.5

            contributions.append(
                ResidualContribution(
                    layer_idx=layer_idx,
                    attention_norm=attn_norm,
                    ffn_norm=ffn_norm,
                    total_norm=total_norm,
                    attention_fraction=attn_fraction,
                    ffn_fraction=ffn_fraction,
                )
            )

            # Update prev_hidden for next iteration
            prev_hidden = curr_hidden

        return contributions

    def _get_tokens_to_track(
        self, config: AnalysisConfig, layer_predictions: list[LayerPredictionResult]
    ) -> list[str]:
        """Determine which tokens to track based on strategy."""
        if config.track_strategy == TrackStrategy.MANUAL:
            return config.track_tokens

        if config.track_strategy == TrackStrategy.TOP_K_FINAL:
            # Track top-k tokens from final layer prediction
            if layer_predictions:
                final_pred = layer_predictions[-1]
                return [p.token for p in final_pred.predictions]
            return []

        if config.track_strategy == TrackStrategy.TOOL_TOKENS:
            # Common tool-calling tokens to track
            return [
                "{",
                "get_",
                "create_",
                "delete_",
                "update_",
                "function",
                "tool",
                "<tool_call>",
                "```json",
            ]

        if config.track_strategy == TrackStrategy.EMERGENT:
            # Find tokens that appear in middle layers but not early
            # This requires more sophisticated analysis
            # For now, combine top-k final with tool tokens
            tokens = set()
            if layer_predictions:
                final_pred = layer_predictions[-1]
                tokens.update(p.token for p in final_pred.predictions)
            tokens.update(["{", "get_", "function"])
            return list(tokens)

        return config.track_tokens

    def _get_top_predictions(
        self,
        logits: mx.array,
        top_k: int,
    ) -> list[TokenPrediction]:
        """Get top-k predictions from logits."""
        probs = mx.softmax(logits)
        top_idx = mx.argsort(probs)[::-1][:top_k].tolist()
        predictions = []
        for rank, idx in enumerate(top_idx, 1):
            predictions.append(
                TokenPrediction(
                    token=self._tokenizer.decode([idx]),
                    token_id=idx,
                    probability=float(probs[idx]),
                    rank=rank,
                )
            )
        return predictions

    def _get_layers_to_capture(
        self,
        num_layers: int,
        config: AnalysisConfig,
    ) -> list[int]:
        """Determine which layers to capture based on config."""
        strategy = config.layer_strategy

        if strategy == LayerStrategy.ALL:
            return list(range(num_layers))

        if strategy == LayerStrategy.FIRST_LAST:
            return [0, num_layers - 1]

        if strategy == LayerStrategy.CUSTOM:
            if config.custom_layers:
                return sorted(set(config.custom_layers))
            return [0, num_layers - 1]

        # EVENLY_SPACED
        layers = list(range(0, num_layers, config.layer_step))
        if (num_layers - 1) not in layers:
            layers.append(num_layers - 1)
        return sorted(set(layers))

    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return len(self._model.model.layers)
        if hasattr(self._model, "layers"):
            return len(self._model.layers)
        return 32  # Fallback

    def _get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        if hasattr(self._model, "args") and hasattr(self._model.args, "hidden_size"):
            return self._model.args.hidden_size
        if hasattr(self._model, "model") and hasattr(self._model.model, "hidden_size"):
            return self._model.model.hidden_size
        return 4096  # Fallback

    def _get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return len(self._tokenizer)


def _is_quantized_model(config: dict, model_id: str) -> bool:
    """Check if a model is quantized based on config or model ID."""
    # Check config for quantization markers
    if "quantization_config" in config:
        return True

    # Check model ID for common quantization markers
    model_id_lower = model_id.lower()
    quant_markers = ["-4bit", "-8bit", "-q4", "-q8", "gguf", "gptq", "awq"]
    return any(marker in model_id_lower for marker in quant_markers)


def _load_model_sync(model_id: str) -> tuple[nn.Module, Any, Any]:
    """
    Load model synchronously using the model families registry.

    Only uses our native loader via the family registry.
    Unsupported models will raise an error.

    Returns:
        (model, tokenizer, config) tuple
    """
    import json

    from ..inference.loader import DType, HFLoader
    from ..models_v2.families.registry import detect_model_family, get_family_info

    # Download model
    result = HFLoader.download(model_id)
    model_path = result.model_path

    # Load config to detect model family
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Detect model family from config
    family_type = detect_model_family(config_data)

    if family_type is None:
        model_type = config_data.get("model_type", "unknown")
        archs = config_data.get("architectures", [])
        raise ValueError(
            f"Unsupported model family. model_type={model_type}, architectures={archs}. "
            f"Model must be registered in the families registry."
        )

    family_info = get_family_info(family_type)
    if family_info is None:
        raise ValueError(f"No family info registered for {family_type}")

    print(f"Using native loader for {family_type.value}")

    # Get config and model classes from registry
    config_class = family_info.config_class
    model_class = family_info.model_class

    # Create config from HuggingFace config dict
    config = config_class.from_hf_config(config_data)

    # Create model
    model = model_class(config)

    # Apply weights using unified loader
    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)

    # Load tokenizer
    tokenizer = HFLoader.load_tokenizer(model_path)

    return model, tokenizer, config


# Convenience function for simple usage
async def analyze_prompt(
    model_id: str,
    prompt: str,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """
    Convenience function to analyze a single prompt.

    Args:
        model_id: HuggingFace model ID
        prompt: Text to analyze
        config: Optional analysis configuration

    Returns:
        AnalysisResult

    Example:
        >>> result = await analyze_prompt(
        ...     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ...     "The capital of France is",
        ... )
        >>> print(result.predicted_token)
    """
    async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
        return await analyzer.analyze(prompt, config)
