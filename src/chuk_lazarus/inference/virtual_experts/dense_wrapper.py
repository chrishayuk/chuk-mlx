"""
Virtual Router Wrapper for Non-MoE (Dense) Models.

This module allows any dense model to use virtual expert plugins by
creating a synthetic routing mechanism in the hidden state space.

Unlike MoE models where we intercept actual router decisions, dense models
get a "virtual" routing layer that:
1. Analyzes hidden states at specified layers
2. Computes routing scores to virtual experts
3. Decides whether to use a plugin or continue with model generation

With CoT rewriting enabled:
1. User query is rewritten to normalized VirtualExpertAction JSON
2. Routing is based on the action JSON hidden state (not raw query)
3. Expert receives the structured action (not raw query)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import (
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertResult,
)
from .cot_rewriter import CoTRewriter, VirtualExpertAction
from .plugins.math import MathExpertPlugin
from .registry import VirtualExpertRegistry, get_default_registry


class VirtualDenseRouter(nn.Module):
    """
    Virtual router for dense (non-MoE) models.

    Creates routing decisions based on learned directions in activation space,
    without requiring an actual MoE architecture.
    """

    def __init__(self, hidden_size: int, num_virtual_experts: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_virtual_experts = num_virtual_experts

        # Learned parameters for each virtual expert
        self.directions: list[mx.array] = [
            mx.zeros((hidden_size,)) for _ in range(num_virtual_experts)
        ]
        self.scales: list[float] = [1.0] * num_virtual_experts
        self.biases: list[float] = [0.0] * num_virtual_experts
        self.thresholds: list[float] = [0.0] * num_virtual_experts

        self._calibrated: list[bool] = [False] * num_virtual_experts

    def calibrate_expert(
        self,
        expert_idx: int,
        positive_activations: list[mx.array],
        negative_activations: list[mx.array],
    ) -> None:
        """Calibrate a virtual expert using positive/negative examples."""
        if expert_idx >= self.num_virtual_experts:
            raise ValueError(f"Expert index {expert_idx} >= {self.num_virtual_experts}")

        pos_stack = mx.stack(positive_activations)
        neg_stack = mx.stack(negative_activations)

        pos_mean = mx.mean(pos_stack, axis=0)
        neg_mean = mx.mean(neg_stack, axis=0)

        direction = pos_mean - neg_mean
        norm = mx.linalg.norm(direction)
        direction = direction / (norm + 1e-10)

        mx.eval(direction)
        self.directions[expert_idx] = direction

        pos_projs = [float(mx.sum(h * direction)) for h in positive_activations]
        neg_projs = [float(mx.sum(h * direction)) for h in negative_activations]

        self.thresholds[expert_idx] = (np.mean(pos_projs) + np.mean(neg_projs)) / 2

        avg_pos_proj = np.mean(pos_projs)
        threshold = self.thresholds[expert_idx]
        if abs(avg_pos_proj - threshold) > 0.01:
            self.scales[expert_idx] = 5.0 / (avg_pos_proj - threshold)
        else:
            self.scales[expert_idx] = 1.0

        self.biases[expert_idx] = -threshold * self.scales[expert_idx]
        self._calibrated[expert_idx] = True

    def get_routing_score(self, x: mx.array, expert_idx: int = 0) -> float:
        """Get routing score for a virtual expert."""
        if not self._calibrated[expert_idx]:
            return 0.0

        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        x_last = x[-1]
        proj = float(mx.sum(x_last * self.directions[expert_idx]))

        threshold = self.thresholds[expert_idx]
        score = (proj - threshold) / (abs(threshold) + 1.0)
        score = max(0.0, min(1.0, (score + 1) / 2))

        return score

    def should_route_to_expert(
        self,
        x: mx.array,
        expert_idx: int = 0,
        threshold: float = 0.5,
    ) -> bool:
        """Determine if input should route to virtual expert."""
        return self.get_routing_score(x, expert_idx) > threshold


class VirtualDenseWrapper:
    """
    Main interface for adding virtual experts to dense (non-MoE) models.

    This wrapper:
    1. Hooks into the model's forward pass to extract hidden states
    2. Uses VirtualDenseRouter to decide when to use plugins
    3. Intercepts generation to delegate to plugins when appropriate

    Example:
        >>> wrapper = VirtualDenseWrapper(model, tokenizer, "llama-3.2-1b")
        >>> wrapper.register_plugin(MathExpertPlugin())
        >>> wrapper.calibrate()
        >>>
        >>> result = wrapper.solve("127 * 89 = ")
        >>> print(result.answer)  # "11303"
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
        registry: VirtualExpertRegistry | None = None,
        target_layer: int | None = None,
        routing_threshold: float = 0.5,
        cot_rewriter: CoTRewriter | None = None,
    ):
        """
        Initialize the wrapper.

        Args:
            model: The dense model to wrap
            tokenizer: The tokenizer
            model_id: Model identifier for logging
            registry: Plugin registry (uses default if None)
            target_layer: Which layer to extract hidden states from (default: middle)
            routing_threshold: Score threshold for routing to virtual expert
            cot_rewriter: Optional CoT rewriter for query normalization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.registry = registry or get_default_registry()
        self.routing_threshold = routing_threshold
        self.cot_rewriter = cot_rewriter

        # Detect model structure
        self._detect_structure()

        # Use middle layer by default
        if target_layer is None:
            target_layer = self.num_layers // 2
        self.target_layer = target_layer

        # Create virtual router
        num_plugins = max(1, len(self.registry))
        self.router = VirtualDenseRouter(self.hidden_size, num_plugins)

        self._calibrated = False
        self._use_cot_calibration = False  # Set True when calibrated with action JSONs

        # Populate rewriter with expert examples if provided
        if self.cot_rewriter:
            self._populate_rewriter_examples()

    def _detect_structure(self):
        """Detect model backbone structure and hidden size."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._backbone = self.model.model
            self._layers = list(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            self._backbone = self.model
            self._layers = list(self.model.layers)
        else:
            raise ValueError("Cannot detect model structure")

        self.num_layers = len(self._layers)
        self._embed = getattr(self._backbone, "embed_tokens", None)
        self._norm = getattr(self._backbone, "norm", None)
        self._lm_head = getattr(self.model, "lm_head", None)

        if hasattr(self.model, "config"):
            self._embed_scale = getattr(self.model.config, "embedding_scale", None)
            self.hidden_size = getattr(self.model.config, "hidden_size", None)
        else:
            self._embed_scale = None
            self.hidden_size = None

        # Try to infer hidden size from embedding
        if self.hidden_size is None and self._embed is not None:
            self.hidden_size = self._embed.weight.shape[-1]

        if self.hidden_size is None:
            raise ValueError("Could not determine hidden size")

    def register_plugin(self, plugin: VirtualExpertPlugin) -> None:
        """Register a new virtual expert plugin."""
        self.registry.register(plugin)
        # Rebuild router
        num_plugins = len(self.registry)
        self.router = VirtualDenseRouter(self.hidden_size, num_plugins)
        self._calibrated = False

    def set_cot_rewriter(self, rewriter: CoTRewriter) -> None:
        """
        Set the CoT rewriter for query normalization.

        When set, queries are rewritten to VirtualExpertAction JSON
        before routing, and calibration uses action JSONs.

        Automatically populates the rewriter with examples from registered experts.
        """
        self.cot_rewriter = rewriter
        self._calibrated = False  # Need to recalibrate with CoT

        # Populate rewriter with expert examples
        self._populate_rewriter_examples()

    def _populate_rewriter_examples(self) -> None:
        """Populate the CoT rewriter with examples from registered experts."""
        if not self.cot_rewriter:
            return

        # Check if rewriter supports set_expert_info (FewShotCoTRewriter does)
        if not hasattr(self.cot_rewriter, 'set_expert_info'):
            return

        for plugin in self.registry.get_all():
            # Get CoT examples from the expert
            if hasattr(plugin, 'get_cot_examples'):
                cot_examples = plugin.get_cot_examples()
                examples = [
                    {"query": ex.query, "action": ex.action.model_dump()}
                    for ex in cot_examples.examples
                ]
                self.cot_rewriter.set_expert_info(
                    expert_name=plugin.name,
                    description=plugin.description,
                    examples=examples,
                )

    def _get_hidden_state(self, prompt: str) -> mx.array:
        """Get hidden state at target layer for last position."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        h = self._embed(input_ids)
        if self._embed_scale:
            h = h * self._embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for idx, layer in enumerate(self._layers):
            if idx == self.target_layer:
                break
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

        mx.eval(h)
        return h[0, -1, :]

    def calibrate(self, use_cot: bool | None = None) -> None:
        """
        Calibrate all registered plugins.

        Args:
            use_cot: If True, calibrate on action JSONs (requires cot_rewriter or
                     plugin with get_calibration_actions). If None, auto-detect.
        """
        plugins = self.registry.get_all()

        # Auto-detect CoT calibration
        if use_cot is None:
            use_cot = self.cot_rewriter is not None or any(
                hasattr(p, "get_calibration_actions") for p in plugins
            )

        self._use_cot_calibration = use_cot

        for plugin_idx, plugin in enumerate(plugins):
            if use_cot and hasattr(plugin, "get_calibration_actions"):
                # New CoT-based calibration using action JSONs
                pos_actions, neg_actions = plugin.get_calibration_actions()
                pos_activations = [self._get_hidden_state(a) for a in pos_actions]
                neg_activations = [self._get_hidden_state(a) for a in neg_actions]
            else:
                # Legacy calibration using raw prompts
                pos_prompts, neg_prompts = plugin.get_calibration_prompts()
                pos_activations = [self._get_hidden_state(p) for p in pos_prompts]
                neg_activations = [self._get_hidden_state(p) for p in neg_prompts]

            self.router.calibrate_expert(plugin_idx, pos_activations, neg_activations)

        self._calibrated = True

    def _generate_direct(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate directly without virtual experts."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if generated and not any(c.isdigit() for c in token_str):
                break

        return self.tokenizer.decode(generated).strip()

    def solve(self, prompt: str, max_tokens: int = 20) -> VirtualExpertResult:
        """
        Solve a problem, using virtual experts when appropriate.

        With CoT rewriter:
        1. Query is rewritten to VirtualExpertAction JSON
        2. Routing is based on action JSON hidden state
        3. Expert receives the action (not raw query)

        Without CoT rewriter (legacy):
        1. Routing is based on raw query hidden state
        2. Expert receives raw query

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            VirtualExpertResult with the answer and metadata
        """
        if not self._calibrated:
            self.calibrate()

        # CoT rewrite if rewriter is available
        action: VirtualExpertAction | None = None
        routing_input = prompt
        skip_expert_routing = False

        if self.cot_rewriter:
            expert_names = [p.name for p in self.registry.get_all()]
            action = self.cot_rewriter.rewrite(prompt, expert_names)

            # If CoT rewriter says "none", skip expert routing entirely
            if action.expert == "none":
                skip_expert_routing = True
            else:
                # Handle both Pydantic (model_dump_json) and dataclass (to_json)
                if hasattr(action, 'model_dump_json'):
                    routing_input = action.model_dump_json()
                else:
                    routing_input = action.to_json()

        # Find best matching plugin (skip if CoT said "none")
        plugins = self.registry.get_all()
        best_plugin: VirtualExpertPlugin | None = None
        best_score = 0.0
        best_idx = -1

        if not skip_expert_routing:
            # Get hidden state for routing decision
            hidden = self._get_hidden_state(routing_input)

            for plugin_idx, plugin in enumerate(plugins):
                score = self.router.get_routing_score(hidden[None, None, :], plugin_idx)
                if score > self.routing_threshold and score > best_score:
                    # With CoT: check if action targets this expert
                    # Without CoT: use legacy can_handle
                    if action:
                        if action.expert == plugin.name:
                            best_plugin = plugin
                            best_score = score
                            best_idx = plugin_idx
                    elif plugin.can_handle(prompt):
                        best_plugin = plugin
                        best_score = score
                        best_idx = plugin_idx

        # Check if we should use plugin
        correct_answer = None
        if best_plugin and isinstance(best_plugin, MathExpertPlugin):
            _, correct_answer = best_plugin.extract_and_evaluate(prompt)

        if best_plugin and best_score > self.routing_threshold:
            # Execute with action or prompt
            if action:
                # New interface: pass structured action to execute()
                # MathExpert.execute() handles VirtualExpertAction objects
                from chuk_virtual_expert import VirtualExpertAction as VEAction
                ve_action = VEAction(
                    expert=action.expert,
                    operation=action.operation,
                    parameters=action.parameters,
                    confidence=action.confidence,
                    reasoning=action.reasoning,
                )
                result = best_plugin.execute(ve_action)
            else:
                # Legacy interface: pass raw prompt
                result = best_plugin.execute(prompt)

            # Check if result is successful
            result_success = True
            if hasattr(result, 'success'):
                result_success = result.success

            if result and result_success:
                # Extract answer from result
                if isinstance(result, str):
                    answer = result
                elif hasattr(result, 'data') and result.data:
                    # VirtualExpertResult from chuk_virtual_expert
                    data = result.data
                    if isinstance(data, dict):
                        answer = str(data.get('formatted', data.get('result', data)))
                    else:
                        answer = str(data)
                else:
                    answer = str(result)

                return VirtualExpertResult(
                    prompt=prompt,
                    answer=answer,
                    correct_answer=correct_answer,
                    approach=VirtualExpertApproach.VIRTUAL_EXPERT,
                    used_virtual_expert=True,
                    plugin_name=best_plugin.name,
                    routing_score=best_score,
                    virtual_expert_selected_count=1,
                    total_tokens=1,
                )

        # Fall back to model generation
        answer = self._generate_direct(prompt, max_tokens)

        return VirtualExpertResult(
            prompt=prompt,
            answer=answer,
            correct_answer=correct_answer,
            approach=VirtualExpertApproach.MODEL_DIRECT,
            used_virtual_expert=False,
            plugin_name=None,
            routing_score=best_score if best_idx >= 0 else 0.0,
            virtual_expert_selected_count=0,
            total_tokens=1,
        )

    def compare(self, prompt: str) -> None:
        """Compare model-only vs virtual expert on a single prompt."""
        plugin = self.registry.find_handler(prompt)
        correct = None
        if plugin and isinstance(plugin, MathExpertPlugin):
            _, correct = plugin.extract_and_evaluate(prompt)

        model_answer = self._generate_direct(prompt)
        result = self.solve(prompt)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"Correct answer: {correct}")
        print("-" * 60)
        print(f"Model alone:      {model_answer}")
        print(f"Virtual expert:   {result.answer}")
        if result.plugin_name:
            print(f"Plugin used:      {result.plugin_name}")
        print(f"Routing score:    {result.routing_score:.3f}")
        print(f"Correct:          {result.is_correct}")
        print(f"{'=' * 60}")

    def benchmark(self, problems: list[str]) -> VirtualExpertAnalysis:
        """Run benchmark on a list of problems."""
        import re

        if not self._calibrated:
            self.calibrate()

        results = []
        correct_with = 0
        correct_without = 0
        times_used = 0
        routing_scores = []
        plugins_used: dict[str, int] = {}

        for prompt in problems:
            plugin = self.registry.find_handler(prompt)
            correct = None
            if plugin and isinstance(plugin, MathExpertPlugin):
                _, correct = plugin.extract_and_evaluate(prompt)

            # Model alone
            model_answer = self._generate_direct(prompt)
            model_correct = False
            if correct is not None:
                try:
                    match = re.search(r"-?\d+(?:\.\d+)?", model_answer)
                    if match:
                        model_correct = abs(float(match.group()) - correct) < 0.01
                except (ValueError, TypeError):
                    pass

            if model_correct:
                correct_without += 1

            # With virtual expert
            result = self.solve(prompt)

            if result.is_correct:
                correct_with += 1

            if result.used_virtual_expert:
                times_used += 1
                if result.plugin_name:
                    plugins_used[result.plugin_name] = plugins_used.get(result.plugin_name, 0) + 1

            if result.routing_score is not None:
                routing_scores.append(result.routing_score)

            results.append(result)

        return VirtualExpertAnalysis(
            model_name=self.model_id,
            total_problems=len(problems),
            correct_with_virtual=correct_with,
            correct_without_virtual=correct_without,
            times_virtual_used=times_used,
            avg_routing_score=np.mean(routing_scores) if routing_scores else 0,
            plugins_used=plugins_used,
            results=results,
        )


def create_virtual_dense_wrapper(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    plugins: list[VirtualExpertPlugin] | None = None,
    **kwargs,
) -> VirtualDenseWrapper:
    """
    Factory function to create a virtual expert wrapper for dense models.

    Args:
        model: The dense model
        tokenizer: The tokenizer
        model_id: Model identifier
        plugins: Additional plugins to register
        **kwargs: Additional arguments for VirtualDenseWrapper

    Returns:
        Configured VirtualDenseWrapper
    """
    wrapper = VirtualDenseWrapper(model, tokenizer, model_id, **kwargs)

    if plugins:
        for plugin in plugins:
            wrapper.register_plugin(plugin)

    return wrapper
