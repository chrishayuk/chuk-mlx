"""Async-native ExpertRouter for MoE expert manipulation.

Provides utilities for forcing, ablating, and analyzing expert routing
in Mixture of Experts models.

Example:
    >>> from chuk_lazarus.introspection.moe import ExpertRouter
    >>>
    >>> async with await ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
    ...     result = await router.generate_with_forced_expert(
    ...         prompt="127 * 89 = ",
    ...         expert_idx=6,
    ...         max_tokens=20,
    ...     )
    ...     print(result.response)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from .enums import MoEArchitecture, MoEImplementationType
from .models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertPair,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
    TopKVariationResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ExpertRouter:
    """Async-native utility for manipulating expert routing.

    This class provides methods to:
    - Force all routing to a specific expert
    - Ablate (remove) experts from routing
    - Vary top-k expert selection
    - Capture and analyze router weights
    - Analyze expert co-activation patterns

    Example:
        >>> async with await ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
        ...     # Chat with a specific expert
        ...     result = await router.chat_with_expert("127 * 89 = ", expert_idx=6)
        ...     print(result.response)
        ...
        ...     # Compare multiple experts
        ...     comparison = await router.compare_experts(
        ...         "def fibonacci(n):",
        ...         expert_indices=[6, 7, 20],
        ...     )
        ...     for r in comparison.expert_results:
        ...         print(f"Expert {r.expert_idx}: {r.response[:50]}...")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_info: MoEModelInfo,
    ):
        """Initialize ExpertRouter.

        Args:
            model: The loaded MLX model.
            tokenizer: The tokenizer for the model.
            model_info: Information about the MoE architecture.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._info = model_info
        self._moe_type = self._detect_moe_type()

        if not self._info.moe_layers:
            raise ValueError("Model has no MoE layers")

    @classmethod
    async def from_pretrained(cls, model_id: str) -> ExpertRouter:
        """Load model and create ExpertRouter.

        Args:
            model_id: HuggingFace model ID or local path.

        Returns:
            Configured ExpertRouter instance.

        Example:
            >>> router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        """
        # Run model loading in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model, tokenizer, model_info = await loop.run_in_executor(
            None, cls._load_model_sync, model_id
        )
        return cls(model, tokenizer, model_info)

    @staticmethod
    def _load_model_sync(model_id: str) -> tuple[nn.Module, Any, MoEModelInfo]:
        """Synchronously load model (called in thread pool)."""
        from ...inference.loader import DType, HFLoader
        from ...models_v2.families.registry import detect_model_family, get_family_info

        logger.info(f"Loading model: {model_id}")

        result = HFLoader.download(model_id)
        model_path = result.model_path

        with open(model_path / "config.json") as f:
            config_data = json.load(f)

        family_type = detect_model_family(config_data)
        if family_type is None:
            raise ValueError(f"Unsupported model: {model_id}")

        family_info = get_family_info(family_type)
        config = family_info.config_class.from_hf_config(config_data)
        model = family_info.model_class(config)

        HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
        tokenizer = HFLoader.load_tokenizer(model_path)

        # Extract MoE info
        model_info = ExpertRouter._extract_moe_info(model)

        return model, tokenizer, model_info

    @staticmethod
    def _extract_moe_info(model: nn.Module) -> MoEModelInfo:
        """Extract MoE information from a model."""
        layers = list(model.model.layers)
        moe_layers: list[int] = []
        num_experts = 0
        num_experts_per_tok = 0
        has_shared_expert = False
        architecture = MoEArchitecture.GENERIC

        for i, layer in enumerate(layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_layers.append(i)
                router = layer.mlp.router
                num_experts = getattr(router, "num_experts", 0)
                num_experts_per_tok = getattr(router, "num_experts_per_tok", 0)

                # Detect architecture
                if hasattr(layer.mlp, "shared_expert"):
                    has_shared_expert = True
                    architecture = MoEArchitecture.LLAMA4

        # Detect GPT-OSS style
        if num_experts == 32 and num_experts_per_tok == 4:
            architecture = MoEArchitecture.GPT_OSS
        elif num_experts == 8 and num_experts_per_tok == 2:
            architecture = MoEArchitecture.MIXTRAL

        return MoEModelInfo(
            moe_layers=tuple(moe_layers),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            total_layers=len(layers),
            architecture=architecture,
            has_shared_expert=has_shared_expert,
        )

    def _detect_moe_type(self) -> MoEImplementationType:
        """Detect the MoE type based on model structure."""
        if not self._info.moe_layers:
            return MoEImplementationType.NONE

        layer_idx = self._info.moe_layers[0]
        layer = self._model.model.layers[layer_idx]
        mlp = layer.mlp

        # Check for GPT-OSS batched experts style
        if hasattr(mlp, "experts") and hasattr(mlp.experts, "gate_up_proj"):
            return MoEImplementationType.GPT_OSS_BATCHED

        return MoEImplementationType.STANDARD

    async def __aenter__(self) -> ExpertRouter:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        pass

    @property
    def info(self) -> MoEModelInfo:
        """Get MoE model information."""
        return self._info

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    # =========================================================================
    # Generation Methods
    # =========================================================================

    async def chat_with_expert(
        self,
        prompt: str,
        expert_idx: int,
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
    ) -> ExpertChatResult:
        """Generate response with routing forced to a specific expert.

        Args:
            prompt: The input prompt.
            expert_idx: Expert index to force routing to.
            max_tokens: Maximum tokens to generate.
            layers: Specific layers to modify (None = all MoE layers).
            temperature: Sampling temperature.
            apply_chat_template: Whether to apply chat template.

        Returns:
            ExpertChatResult with response and statistics.
        """
        loop = asyncio.get_event_loop()
        response, stats = await loop.run_in_executor(
            None,
            self._generate_with_forced_expert_sync,
            prompt,
            expert_idx,
            max_tokens,
            layers,
            temperature,
            apply_chat_template,
        )

        return ExpertChatResult(
            prompt=prompt,
            response=response,
            expert_idx=expert_idx,
            stats=stats,
        )

    def _generate_with_forced_expert_sync(
        self,
        prompt: str,
        expert_idx: int,
        max_tokens: int,
        layers: list[int] | None,
        temperature: float,
        apply_chat_template: bool,
    ) -> tuple[str, GenerationStats]:
        """Synchronous implementation of forced expert generation."""
        # Apply chat template if requested
        if apply_chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            if self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers = layers if layers else list(self._info.moe_layers)

        # Store original forward functions
        original_forwards: dict[int, Any] = {}

        try:
            # Patch MoE layers to force expert
            for layer_idx in target_layers:
                layer = self._model.model.layers[layer_idx]
                original_forwards[layer_idx] = layer.mlp.__call__

                if self._moe_type == MoEImplementationType.GPT_OSS_BATCHED:
                    layer.mlp.__call__ = self._make_forced_expert_forward_gpt_oss(
                        layer.mlp, expert_idx
                    )
                else:
                    layer.mlp.__call__ = self._make_forced_expert_forward_standard(
                        layer.mlp, expert_idx
                    )

            # Generate
            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                logits, cache = self._model(input_ids, cache=cache)
                next_token = self._sample_token(logits, temperature)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            # Restore original forwards
            for layer_idx, original in original_forwards.items():
                self._model.model.layers[layer_idx].mlp.__call__ = original

        stats = GenerationStats(
            expert_idx=expert_idx,
            tokens_generated=len(generated),
            layers_modified=len(target_layers),
            moe_type=self._moe_type,
            prompt_tokens=input_ids.shape[-1],
        )

        return text, stats

    def _make_forced_expert_forward_gpt_oss(self, mlp: nn.Module, expert_idx: int) -> Any:
        """Create a forward function that forces routing to one expert (GPT-OSS style)."""

        def forced_forward(x: mx.array) -> mx.array:
            # Get the expert weights
            experts = mlp.experts
            gate_up = experts.gate_up_proj  # Shape: [num_experts, hidden, 2*intermediate]
            down = experts.down_proj  # Shape: [num_experts, intermediate, hidden]

            # Extract single expert weights
            expert_gate_up = gate_up[expert_idx]  # [hidden, 2*intermediate]
            expert_down = down[expert_idx]  # [intermediate, hidden]

            # Apply SwiGLU
            intermediate_size = expert_gate_up.shape[-1] // 2
            gate_up_out = x @ expert_gate_up
            gate = gate_up_out[..., :intermediate_size]
            up = gate_up_out[..., intermediate_size:]

            # SwiGLU activation
            hidden = nn.silu(gate) * up

            # Down projection
            output = hidden @ expert_down.T

            return output

        return forced_forward

    def _make_forced_expert_forward_standard(self, mlp: nn.Module, expert_idx: int) -> Any:
        """Create a forward function that forces routing to one expert (standard MoE)."""

        def forced_forward(x: mx.array) -> mx.array:
            expert = mlp.experts[expert_idx]
            return expert(x)

        return forced_forward

    def _sample_token(self, logits: mx.array, temperature: float) -> int:
        """Sample a token from logits."""
        logits = logits[:, -1, :]  # Get last position

        if temperature == 0.0:
            return int(mx.argmax(logits, axis=-1).item())

        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        return int(mx.random.categorical(mx.log(probs)).item())

    async def compare_experts(
        self,
        prompt: str,
        expert_indices: list[int],
        *,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> ExpertComparisonResult:
        """Compare multiple experts on the same prompt.

        Args:
            prompt: The input prompt.
            expert_indices: List of expert indices to compare.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ExpertComparisonResult with results from each expert.
        """
        results: list[ExpertChatResult] = []

        for expert_idx in expert_indices:
            result = await self.chat_with_expert(
                prompt,
                expert_idx,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)

        return ExpertComparisonResult(
            prompt=prompt,
            expert_results=tuple(results),
        )

    async def generate_with_ablation(
        self,
        prompt: str,
        expert_indices: list[int],
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate with specific experts ablated (removed from routing).

        Args:
            prompt: The input prompt.
            expert_indices: Expert indices to ablate.
            max_tokens: Maximum tokens to generate.
            layers: Specific layers to modify (None = all MoE layers).

        Returns:
            Tuple of (response text, generation stats).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_with_ablation_sync,
            prompt,
            expert_indices,
            max_tokens,
            layers,
        )

    def _generate_with_ablation_sync(
        self,
        prompt: str,
        expert_indices: list[int],
        max_tokens: int,
        layers: list[int] | None,
    ) -> tuple[str, GenerationStats]:
        """Synchronous implementation of ablated generation."""
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers = layers if layers else list(self._info.moe_layers)
        ablate_set = set(expert_indices)

        original_forwards: dict[int, Any] = {}

        try:
            for layer_idx in target_layers:
                layer = self._model.model.layers[layer_idx]
                original_forwards[layer_idx] = layer.mlp.__call__
                layer.mlp.__call__ = self._make_ablated_forward(layer.mlp, ablate_set)

            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                logits, cache = self._model(input_ids, cache=cache)
                next_token = self._sample_token(logits, 0.0)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            for layer_idx, original in original_forwards.items():
                self._model.model.layers[layer_idx].mlp.__call__ = original

        stats = GenerationStats(
            expert_idx=-1,  # No specific expert forced
            tokens_generated=len(generated),
            layers_modified=len(target_layers),
            moe_type=self._moe_type,
            prompt_tokens=input_ids.shape[-1],
        )

        return text, stats

    def _make_ablated_forward(self, mlp: nn.Module, ablate_set: set[int]) -> Any:
        """Create a forward function that ablates specific experts."""
        original_call = mlp.__call__

        def ablated_forward(x: mx.array) -> mx.array:
            # For now, fall back to original - proper ablation requires
            # modifying router weights before selection
            # This is a placeholder for the full implementation
            return original_call(x)

        return ablated_forward

    async def generate_with_topk(
        self,
        prompt: str,
        k: int,
        *,
        max_tokens: int = 100,
    ) -> TopKVariationResult:
        """Generate with modified top-k expert selection.

        Args:
            prompt: The input prompt.
            k: Number of experts to use (instead of default).
            max_tokens: Maximum tokens to generate.

        Returns:
            TopKVariationResult with both normal and modified responses.
        """
        loop = asyncio.get_event_loop()

        # Get normal response
        normal_text = await loop.run_in_executor(
            None, self._generate_normal_sync, prompt, max_tokens
        )

        # Get modified top-k response
        topk_text = await loop.run_in_executor(
            None, self._generate_with_topk_sync, prompt, k, max_tokens
        )

        return TopKVariationResult(
            prompt=prompt,
            k_value=k,
            default_k=self._info.num_experts_per_tok,
            response=topk_text,
            normal_response=normal_text,
        )

    def _generate_normal_sync(self, prompt: str, max_tokens: int) -> str:
        """Synchronous normal generation."""
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        generated: list[int] = []
        cache = None

        for _ in range(max_tokens):
            logits, cache = self._model(input_ids, cache=cache)
            next_token = self._sample_token(logits, 0.0)
            generated.append(next_token)

            if next_token == self._tokenizer.eos_token_id:
                break

            input_ids = mx.array([[next_token]])

        return self._tokenizer.decode(generated)

    def _generate_with_topk_sync(self, prompt: str, k: int, max_tokens: int) -> str:
        """Synchronous top-k modified generation.

        Modifies the router to use a different k value during generation.
        This allows testing how model behavior changes with more or fewer experts.

        Args:
            prompt: The input prompt.
            k: Number of experts to select per token.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text with modified top-k routing.
        """
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers = list(self._info.moe_layers)

        # Store original router forwards and num_experts_per_tok
        original_router_calls: dict[int, Any] = {}
        original_k_values: dict[int, int] = {}

        def make_modified_topk_router(router: Any, new_k: int) -> Any:
            """Create a router forward that uses a different k value."""

            def modified_router_call(x: mx.array) -> tuple[mx.array, mx.array]:
                # Compute router logits
                logits = router.gate(x)

                # Clamp k to valid range
                effective_k = min(new_k, router.num_experts)
                effective_k = max(1, effective_k)

                # Get top-k experts with modified k
                indices = mx.argsort(logits, axis=-1)[:, :, -effective_k:]
                indices = indices[:, :, ::-1]  # Descending order

                # Get corresponding weights
                weights = mx.softmax(logits, axis=-1)

                # Gather top-k weights
                top_k_weights = mx.take_along_axis(weights, indices, axis=-1)

                # Renormalize weights to sum to 1
                top_k_weights = top_k_weights / (
                    mx.sum(top_k_weights, axis=-1, keepdims=True) + 1e-6
                )

                return top_k_weights, indices

            return modified_router_call

        def make_modified_moe_forward(mlp: nn.Module, new_k: int, original_forward: Any) -> Any:
            """Create an MoE forward that uses modified router k."""

            def modified_forward(x: mx.array) -> mx.array:
                batch_size, seq_len, hidden_size = x.shape

                # Get routing with modified k
                weights, indices = make_modified_topk_router(mlp.router, new_k)(x)

                # Effective k may differ from requested k
                effective_k = indices.shape[-1]

                # Flatten for processing
                x_flat = x.reshape(-1, hidden_size)
                indices_flat = indices.reshape(-1, effective_k)
                weights_flat = weights.reshape(-1, effective_k)

                # Compute expert outputs
                output = mx.zeros_like(x_flat)

                # Handle different MoE architectures
                if self._moe_type == MoEImplementationType.GPT_OSS_BATCHED:
                    # GPT-OSS style with batched experts
                    experts = mlp.experts
                    gate_up = experts.gate_up_proj
                    down = experts.down_proj

                    for k_idx in range(effective_k):
                        expert_indices = indices_flat[:, k_idx]
                        expert_weights = weights_flat[:, k_idx]

                        # Process each unique expert
                        for exp_idx in range(mlp.router.num_experts):
                            mask = expert_indices == exp_idx
                            if not mx.any(mask):
                                continue

                            # Extract expert weights
                            expert_gate_up = gate_up[exp_idx]
                            expert_down = down[exp_idx]

                            # Apply SwiGLU
                            intermediate_size = expert_gate_up.shape[-1] // 2
                            gate_up_out = x_flat @ expert_gate_up
                            gate = gate_up_out[..., :intermediate_size]
                            up = gate_up_out[..., intermediate_size:]
                            hidden = nn.silu(gate) * up
                            expert_out = hidden @ expert_down.T

                            # Weight and accumulate
                            weight_masked = mx.where(
                                mask[:, None],
                                expert_weights[:, None],
                                mx.zeros_like(expert_weights[:, None]),
                            )
                            output = output + expert_out * weight_masked

                else:
                    # Standard MoE with separate expert modules
                    experts = getattr(mlp, "experts", [])

                    for expert_idx, expert in enumerate(experts):
                        # Find tokens routed to this expert
                        expert_mask = indices_flat == expert_idx
                        expert_weights = mx.sum(
                            weights_flat * expert_mask.astype(weights_flat.dtype),
                            axis=-1,
                        )

                        if mx.any(expert_weights > 0):
                            expert_out = expert(x_flat)
                            output = output + expert_out * expert_weights[:, None]

                # Reshape back
                output = output.reshape(batch_size, seq_len, hidden_size)
                return output

            return modified_forward

        try:
            # Patch MoE layers with modified top-k
            for layer_idx in target_layers:
                layer = self._model.model.layers[layer_idx]
                mlp = layer.mlp

                # Store originals
                original_router_calls[layer_idx] = mlp.__call__
                if hasattr(mlp, "router"):
                    original_k_values[layer_idx] = mlp.router.num_experts_per_tok

                # Apply modified forward
                mlp.__call__ = make_modified_moe_forward(mlp, k, mlp.__call__)

            # Generate with modified routing
            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                logits, cache = self._model(input_ids, cache=cache)
                next_token = self._sample_token(logits, 0.0)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            # Restore original forwards
            for layer_idx, original in original_router_calls.items():
                self._model.model.layers[layer_idx].mlp.__call__ = original

        return text

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    async def capture_router_weights(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
    ) -> list[LayerRouterWeights]:
        """Capture router weights for each token position.

        Args:
            prompt: The input prompt.
            layers: Specific layers to capture (None = all MoE layers).

        Returns:
            List of LayerRouterWeights for each layer.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_router_weights_sync, prompt, layers)

    def _capture_router_weights_sync(
        self,
        prompt: str,
        layers: list[int] | None,
    ) -> list[LayerRouterWeights]:
        """Synchronous router weight capture."""
        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        tokens = [self._tokenizer.decode([t]) for t in input_ids[0].tolist()]
        target_layers = layers if layers else list(self._info.moe_layers)

        results: list[LayerRouterWeights] = []
        captured_weights: dict[int, list[tuple[list[int], list[float]]]] = {
            layer_idx: [] for layer_idx in target_layers
        }

        # Get the MLP class for class-level patching (instance patching doesn't work)
        mlp_class = type(self._model.model.layers[0].mlp)
        original_call = mlp_class.__call__

        def patched_call(mlp_self: Any, x: mx.array) -> mx.array:
            # Find which layer this MLP belongs to
            layer_idx = -1
            for i, layer in enumerate(self._model.model.layers):
                if layer.mlp is mlp_self:
                    layer_idx = i
                    break

            # Only capture for target layers
            if layer_idx in target_layers:
                router = mlp_self.router
                k = self._info.num_experts_per_tok

                # Router may return (weights, indices) tuple or just logits
                router_result = router(x)

                if isinstance(router_result, tuple):
                    # Router already computed weights and indices
                    weights, indices = router_result
                    # weights: (batch * seq_len, k), indices: (batch * seq_len, k)
                    seq_len = x.shape[1] if x.ndim == 3 else 1
                    for pos in range(seq_len):
                        pos_indices = indices[pos].tolist()
                        pos_weights = [float(weights[pos, i]) for i in range(k)]
                        captured_weights[layer_idx].append((pos_indices, pos_weights))
                else:
                    # Router returned logits, compute weights ourselves
                    router_logits = router_result
                    weights = mx.softmax(router_logits, axis=-1)
                    for pos in range(x.shape[1]):
                        pos_weights = weights[0, pos]
                        top_indices = mx.argsort(pos_weights)[-k:][::-1].tolist()
                        top_weights = [float(pos_weights[i]) for i in top_indices]
                        captured_weights[layer_idx].append((top_indices, top_weights))

            return original_call(mlp_self, x)

        try:
            # Patch at class level
            mlp_class.__call__ = patched_call

            # Run forward pass
            self._model(input_ids)

        finally:
            # Restore original
            mlp_class.__call__ = original_call

        # Convert to structured results
        for layer_idx in target_layers:
            positions: list[RouterWeightCapture] = []
            for pos_idx, (exp_indices, weights) in enumerate(captured_weights[layer_idx]):
                token = tokens[pos_idx] if pos_idx < len(tokens) else ""
                positions.append(
                    RouterWeightCapture(
                        layer_idx=layer_idx,
                        position_idx=pos_idx,
                        token=token,
                        expert_indices=tuple(exp_indices),
                        weights=tuple(weights),
                    )
                )
            results.append(LayerRouterWeights(layer_idx=layer_idx, positions=tuple(positions)))

        return results

    async def analyze_coactivation(
        self,
        prompts: list[str],
        *,
        layer_idx: int | None = None,
    ) -> CoactivationAnalysis:
        """Analyze expert co-activation patterns across prompts.

        Args:
            prompts: List of prompts to analyze.
            layer_idx: Specific layer to analyze (None = first MoE layer).

        Returns:
            CoactivationAnalysis with co-activation statistics.
        """
        loop = asyncio.get_event_loop()
        target_layer = layer_idx if layer_idx is not None else self._info.moe_layers[0]

        return await loop.run_in_executor(
            None, self._analyze_coactivation_sync, prompts, target_layer
        )

    def _analyze_coactivation_sync(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> CoactivationAnalysis:
        """Synchronous co-activation analysis."""
        from collections import Counter

        expert_counts: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        total_activations = 0

        for prompt in prompts:
            weights_list = self._capture_router_weights_sync(prompt, [layer_idx])
            if not weights_list:
                continue

            layer_weights = weights_list[0]
            for pos in layer_weights.positions:
                experts = pos.expert_indices
                total_activations += 1

                for exp in experts:
                    expert_counts[exp] += 1

                # Count pairs
                for i, exp_a in enumerate(experts):
                    for exp_b in experts[i + 1 :]:
                        pair = (min(exp_a, exp_b), max(exp_a, exp_b))
                        pair_counts[pair] += 1

        # Build top pairs
        top_pairs: list[ExpertPair] = []
        for (exp_a, exp_b), count in pair_counts.most_common(20):
            rate = count / total_activations if total_activations > 0 else 0.0
            top_pairs.append(
                ExpertPair(
                    expert_a=exp_a,
                    expert_b=exp_b,
                    coactivation_count=count,
                    coactivation_rate=rate,
                )
            )

        # Find generalist experts (high activation rate)
        threshold = total_activations / self._info.num_experts * 1.5
        generalists = tuple(exp for exp, count in expert_counts.items() if count > threshold)

        return CoactivationAnalysis(
            layer_idx=layer_idx,
            total_activations=total_activations,
            top_pairs=tuple(top_pairs),
            specialist_pairs=(),  # Would need additional analysis
            generalist_experts=generalists,
        )
