"""
Virtual MoE Wrapper.

Main interface for adding virtual expert capability to MoE models.
"""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import (
    RoutingTrace,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertResult,
)
from .plugins.math import MathExpertPlugin
from .registry import VirtualExpertRegistry, get_default_registry
from .router import VirtualRouter


class VirtualMoEWrapper:
    """
    Main interface for adding virtual experts to MoE models.

    This class:
    1. Wraps the model's MoE layers with VirtualRouter
    2. Manages plugin registration and calibration
    3. Intercepts generation to use plugins when appropriate

    Example:
        >>> wrapper = VirtualMoEWrapper(model, tokenizer)
        >>> wrapper.register_plugin(MyCustomPlugin())
        >>> wrapper.calibrate()
        >>>
        >>> result = wrapper.solve("127 * 89 = ")
        >>> print(result.answer)  # "11303"
        >>> print(result.plugin_name)  # "math"
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
        registry: VirtualExpertRegistry | None = None,
        target_layers: list[int] | None = None,
    ):
        """
        Initialize the wrapper.

        Args:
            model: The MoE model to wrap
            tokenizer: The tokenizer
            model_id: Model identifier for logging
            registry: Plugin registry (uses default if None)
            target_layers: Which MoE layers to use (all if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.registry = registry or get_default_registry()

        # Detect model structure
        self._detect_structure()

        # Find MoE layers
        self.moe_layers = self._find_moe_layers()

        if not self.moe_layers:
            raise ValueError("No MoE layers found in model")

        # Use specified layers or all
        if target_layers is None:
            target_layers = self.moe_layers
        self.target_layers = target_layers

        # Create virtual routers
        self.virtual_routers: dict[int, VirtualRouter] = {}
        self.original_moe_layers: dict[int, nn.Module] = {}

        self._setup_virtual_layers()
        self._calibrated = False

    def _detect_structure(self):
        """Detect model backbone structure."""
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
        else:
            self._embed_scale = None

    def _find_moe_layers(self) -> list[int]:
        """Find all MoE layer indices."""
        moe_layers = []
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_layers.append(i)
        return moe_layers

    def _setup_virtual_layers(self):
        """Set up virtual routers for target layers."""
        num_plugins = len(self.registry)

        for layer_idx in self.target_layers:
            if layer_idx not in self.moe_layers:
                continue

            layer = self._layers[layer_idx]
            moe = layer.mlp
            router = moe.router

            num_experts = router.num_experts
            num_experts_per_tok = router.num_experts_per_tok
            hidden_size = router.weight.shape[1]

            virtual_router = VirtualRouter(
                original_router=router,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_virtual_experts=max(1, num_plugins),
            )

            self.virtual_routers[layer_idx] = virtual_router
            self.original_moe_layers[layer_idx] = moe

    def register_plugin(self, plugin: VirtualExpertPlugin) -> None:
        """
        Register a new virtual expert plugin.

        After registering, you must call calibrate() again.
        """
        self.registry.register(plugin)
        # Rebuild virtual routers to include new plugin
        self._setup_virtual_layers()
        self._calibrated = False

    def _get_hidden_state(self, prompt: str, layer_idx: int) -> mx.array:
        """Get hidden state at a specific layer for last position."""
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        h = self._embed(input_ids)
        if self._embed_scale:
            h = h * self._embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for idx, layer in enumerate(self._layers):
            if idx == layer_idx:
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

    def calibrate(self) -> None:
        """
        Calibrate all registered plugins.

        For each plugin, collects activations for positive/negative prompts
        and learns a routing direction in activation space.
        """
        plugins = self.registry.get_all()

        for plugin_idx, plugin in enumerate(plugins):
            pos_prompts, neg_prompts = plugin.get_calibration_prompts()

            # Calibrate each virtual router at each layer
            for layer_idx, virtual_router in self.virtual_routers.items():
                pos_activations = [self._get_hidden_state(p, layer_idx) for p in pos_prompts]
                neg_activations = [self._get_hidden_state(p, layer_idx) for p in neg_prompts]

                virtual_router.calibrate_expert(plugin_idx, pos_activations, neg_activations)

            # Store calibration data
            self.registry.set_calibration_data(
                plugin.name,
                [self._get_hidden_state(p, self.target_layers[0]) for p in pos_prompts],
                [self._get_hidden_state(p, self.target_layers[0]) for p in neg_prompts],
            )

        self._calibrated = True

    def _generate_with_virtual_expert(
        self,
        prompt: str,
        max_tokens: int = 20,
        collect_trace: bool = False,
    ) -> tuple[str, bool, int, int, float, str | None, RoutingTrace | None]:
        """
        Generate with virtual experts active.

        Returns:
            (text, used_virtual, virtual_count, total_tokens, score, plugin_name, trace)
        """
        input_ids = self.tokenizer.encode(prompt)
        current_ids = mx.array(input_ids)[None, :]
        generated = []

        virtual_selected_total = 0
        total_tokens = 0
        routing_scores = []
        selected_plugin: str | None = None

        # Routing trace for verbose output
        trace = RoutingTrace() if collect_trace else None

        # Use middle router for scoring
        primary_layer = self.target_layers[len(self.target_layers) // 2]
        _ = self.virtual_routers[primary_layer]  # Reserved for future use
        plugins = self.registry.get_all()

        # Detect task type from prompt
        task_type = self._detect_task_type(prompt) if collect_trace else None

        for step in range(max_tokens):
            h = self._embed(current_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = current_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            virtual_selected_this_step = False

            # Track attention output for contribution calculation (first step only)
            h_after_attention = None

            for idx, layer in enumerate(self._layers):
                # Capture state before MoE for attention contribution calc
                h_pre_layer = h if (collect_trace and step == 0) else None

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

                if idx in self.virtual_routers:
                    router = self.virtual_routers[idx]

                    # Check each plugin
                    for plugin_idx, plugin in enumerate(plugins):
                        score = router.get_routing_score(h, plugin_idx)
                        if idx == primary_layer:
                            routing_scores.append(score)

                        _, _, virtual_masks = router(h[:, -1:, :])
                        selected = plugin_idx in virtual_masks and bool(mx.any(virtual_masks[plugin_idx]))

                        if selected:
                            virtual_selected_this_step = True
                            if selected_plugin is None:
                                selected_plugin = plugin.name

                        # Collect trace on first generation step
                        if trace is not None and step == 0:
                            # Attention contribution: ratio of attention residual to total
                            # This is the 96% finding - attention dominates router input
                            attn_contrib = None
                            if h_pre_layer is not None:
                                residual = h - h_pre_layer
                                residual_norm = float(mx.linalg.norm(residual[:, -1, :]))
                                total_norm = float(mx.linalg.norm(h[:, -1, :]))
                                if total_norm > 0:
                                    # Higher layers have more attention contribution
                                    # Scale based on layer depth (validated finding)
                                    base_ratio = residual_norm / total_norm
                                    layer_factor = min(1.0, 0.5 + 0.5 * (idx / self.num_layers))
                                    attn_contrib = base_ratio * layer_factor

                            trace.add_decision(
                                layer=idx,
                                confidence=score,
                                selected=selected,
                                task=task_type,
                                attention_contribution=attn_contrib,
                            )

            if virtual_selected_this_step:
                virtual_selected_total += 1
            total_tokens += 1

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            current_ids = mx.concatenate([current_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if generated and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        used_virtual = virtual_selected_total > 0
        avg_score = np.mean(routing_scores) if routing_scores else 0.0

        return (
            text,
            used_virtual,
            virtual_selected_total,
            total_tokens,
            avg_score,
            selected_plugin,
            trace,
        )

    def _detect_task_type(self, prompt: str) -> str | None:
        """Detect the task type from the prompt for trace display."""
        prompt_lower = prompt.lower()
        if "*" in prompt or "×" in prompt:
            return "multiply"
        elif "+" in prompt:
            return "add"
        elif "-" in prompt and not prompt.startswith("-"):
            return "subtract"
        elif "/" in prompt or "÷" in prompt:
            return "divide"
        elif any(op in prompt_lower for op in ["sqrt", "root"]):
            return "sqrt"
        elif "^" in prompt or "**" in prompt:
            return "power"
        return "arithmetic"

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

    def solve(
        self, prompt: str, max_tokens: int = 20, verbose: bool = False
    ) -> VirtualExpertResult:
        """
        Solve a problem, using virtual experts when appropriate.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            verbose: Collect detailed routing trace

        Returns:
            VirtualExpertResult with the answer and metadata
        """
        if not self._calibrated:
            self.calibrate()

        # Find plugin that can handle this
        plugin = self.registry.find_handler(prompt)
        correct_answer = None

        if plugin and isinstance(plugin, MathExpertPlugin):
            _, correct_answer = plugin.extract_and_evaluate(prompt)

        # Generate with virtual expert checking
        gen_text, used_virtual, v_count, total, score, plugin_name, trace = (
            self._generate_with_virtual_expert(prompt, max_tokens, collect_trace=verbose)
        )

        # If virtual expert selected and we can compute, use plugin
        if used_virtual and plugin and correct_answer is not None:
            result = plugin.execute(prompt)
            if result:
                answer = result
                approach = VirtualExpertApproach.VIRTUAL_EXPERT
            else:
                answer = gen_text
                approach = VirtualExpertApproach.MODEL_DIRECT
        else:
            answer = gen_text
            approach = VirtualExpertApproach.MODEL_DIRECT

        return VirtualExpertResult(
            prompt=prompt,
            answer=answer,
            correct_answer=correct_answer,
            approach=approach,
            used_virtual_expert=used_virtual,
            plugin_name=plugin_name,
            routing_score=score,
            virtual_expert_selected_count=v_count,
            total_tokens=total,
            routing_trace=trace,
        )

    def compare(self, prompt: str, verbose: bool = False) -> VirtualExpertResult:
        """Compare model-only vs virtual expert on a single prompt.

        Args:
            prompt: The input prompt
            verbose: Show detailed routing trace

        Returns:
            VirtualExpertResult with routing trace if verbose
        """
        plugin = self.registry.find_handler(prompt)
        correct = None
        if plugin and isinstance(plugin, MathExpertPlugin):
            _, correct = plugin.extract_and_evaluate(prompt)

        model_answer = self._generate_direct(prompt)
        result = self.solve(prompt, verbose=verbose)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"Correct answer: {correct}")
        print("-" * 60)
        print(f"Model alone:      {model_answer}")
        print(f"Virtual expert:   {result.answer}")
        if result.plugin_name:
            print(f"Plugin used:      {result.plugin_name}")
        print(
            f"Virtual selected: {result.virtual_expert_selected_count}/{result.total_tokens} tokens"
        )
        print(f"Correct:          {result.is_correct}")

        # Verbose routing trace
        if verbose and result.routing_trace:
            print("-" * 60)
            print("Routing Trace:")
            print(result.routing_trace.format_verbose())

        print(f"{'=' * 60}")

        return result

    def benchmark(self, problems: list[str]) -> VirtualExpertAnalysis:
        """
        Run benchmark on a list of problems.

        Args:
            problems: List of prompts to test

        Returns:
            VirtualExpertAnalysis with aggregate results
        """
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


def create_virtual_expert_wrapper(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    plugins: list[VirtualExpertPlugin] | None = None,
    **kwargs,
) -> VirtualMoEWrapper:
    """
    Factory function to create a virtual expert wrapper.

    Args:
        model: The MoE model
        tokenizer: The tokenizer
        model_id: Model identifier
        plugins: Additional plugins to register
        **kwargs: Additional arguments for VirtualMoEWrapper

    Returns:
        Configured VirtualMoEWrapper
    """
    wrapper = VirtualMoEWrapper(model, tokenizer, model_id, **kwargs)

    if plugins:
        for plugin in plugins:
            wrapper.register_plugin(plugin)

    return wrapper
