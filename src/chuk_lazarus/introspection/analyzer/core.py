"""
Core ModelAnalyzer class for async-native model introspection.

This module provides the main ModelAnalyzer class for analyzing
model behavior through logit lens and token tracking.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..hooks import CaptureConfig, ModelHooks
from ..logit_lens import LogitLens
from .config import AnalysisConfig, LayerStrategy, TrackStrategy
from .loader import _load_model_sync
from .models import (
    AnalysisResult,
    LayerPredictionResult,
    LayerTransition,
    ModelInfo,
    ResidualContribution,
    TokenEvolutionResult,
    TokenPrediction,
)
from .utils import compute_entropy, compute_js_divergence, compute_kl_divergence


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
        adapter_path: str | None = None,
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
            adapter_path: Optional path to LoRA adapter weights

        Yields:
            ModelAnalyzer instance

        Example:
            >>> async with ModelAnalyzer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") as analyzer:
            ...     result = await analyzer.analyze("Hello")
        """
        # Load model in thread pool to not block
        loop = asyncio.get_event_loop()
        model, tokenizer, config = await loop.run_in_executor(
            None,
            lambda: _load_model_sync(model_id, adapter_path=adapter_path),
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
        # Tokenize
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        tokens = [self._tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        # Get layers to capture
        num_layers = self._get_num_layers()
        layers_to_capture = self._get_layers_to_capture(num_layers, config)

        # Setup hooks with proper enum-based config
        hooks = ModelHooks(
            self._model,
            embedding_scale=self._embedding_scale,
            model_config=self._config,
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
                layer_logits = hooks.get_layer_logits(pred.layer_idx, normalize=True)
                if layer_logits is not None:
                    if layer_logits.ndim == 3:
                        pos_logits = layer_logits[0, -1, :]
                    else:
                        pos_logits = layer_logits[-1, :]
                    probs = mx.softmax(pos_logits)
                    layer_probs_cache[pred.layer_idx] = probs
                    entropy = compute_entropy(probs)
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

                from_probs = layer_probs_cache.get(from_layer)
                to_probs = layer_probs_cache.get(to_layer)

                if from_probs is not None and to_probs is not None:
                    kl_div = compute_kl_divergence(from_probs, to_probs)
                    js_div = compute_js_divergence(from_probs, to_probs)
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

    def _compute_residual_decomposition(
        self,
        hooks: ModelHooks,
        captured_layers: list[int],
    ) -> list[ResidualContribution]:
        """Compute residual stream decomposition using hidden state differences."""
        contributions = []

        # Get embeddings as the "layer -1" state
        embeddings = hooks.state.embeddings
        if embeddings is None:
            return contributions

        # Get last position hidden state from embeddings
        if embeddings.ndim == 3:
            prev_hidden = embeddings[0, -1, :]
        else:
            prev_hidden = embeddings[-1, :]

        for layer_idx in captured_layers:
            hidden = hooks.state.hidden_states.get(layer_idx)
            if hidden is None:
                continue

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
                attn_norm = total_norm / 2.0
                ffn_norm = total_norm / 2.0

            # Compute fractions
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

            prev_hidden = curr_hidden

        return contributions

    def _get_tokens_to_track(
        self, config: AnalysisConfig, layer_predictions: list[LayerPredictionResult]
    ) -> list[str]:
        """Determine which tokens to track based on strategy."""
        if config.track_strategy == TrackStrategy.MANUAL:
            return config.track_tokens

        if config.track_strategy == TrackStrategy.TOP_K_FINAL:
            if layer_predictions:
                final_pred = layer_predictions[-1]
                return [p.token for p in final_pred.predictions]
            return []

        if config.track_strategy == TrackStrategy.TOOL_TOKENS:
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
