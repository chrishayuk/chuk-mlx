"""Service layer for ablation CLI commands.

This module provides the AblationService class that wraps AblationStudy
to provide a simple interface for CLI commands.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from .config import AblationConfig, ComponentType
from .study import AblationStudy


class AblationCriterionFunctions:
    """Pre-defined criterion functions for ablation studies."""

    @staticmethod
    def function_call(output: str) -> bool:
        """Check if output contains function call markers."""
        markers = (
            "<start_function_call>",
            "<function_call>",
            "get_weather(",
            '{"name":',
        )
        return any(m in output for m in markers)

    @staticmethod
    def sorry(output: str) -> bool:
        """Check if output contains apology."""
        return "sorry" in output.lower() or "apologize" in output.lower()

    @staticmethod
    def positive(output: str) -> bool:
        """Check if output contains positive sentiment."""
        markers = ("great", "good", "excellent", "wonderful", "love")
        return any(w in output.lower() for w in markers)

    @staticmethod
    def negative(output: str) -> bool:
        """Check if output contains negative sentiment."""
        markers = ("bad", "terrible", "awful", "hate", "poor")
        return any(w in output.lower() for w in markers)

    @staticmethod
    def refusal(output: str) -> bool:
        """Check if output contains refusal markers."""
        markers = ("cannot", "can't", "won't", "unable", "decline")
        return any(m in output.lower() for m in markers)

    @staticmethod
    def contains(output: str, substring: str) -> bool:
        """Check if output contains a specific substring."""
        return substring in output


class AblationServiceConfig(BaseModel):
    """Configuration for AblationService."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    layers: list[int] | None = Field(default=None, description="Layers to ablate")
    component: ComponentType = Field(default=ComponentType.MLP, description="Component to ablate")
    max_tokens: int = Field(default=50, description="Max tokens to generate")
    multi_mode: bool = Field(default=False, description="Ablate all layers together")
    use_raw: bool = Field(default=False, description="Use raw mode (no chat template)")


class SingleAblationResult(BaseModel):
    """Result of a single ablation test."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    expected: str = Field(..., description="Expected output/criterion")
    ablation_name: str = Field(..., description="Ablation description")
    output: str = Field(..., description="Generated output")
    passes_criterion: bool = Field(..., description="Whether output passes criterion")


class MultiPromptAblationResult(BaseModel):
    """Result of multi-prompt ablation."""

    model_config = ConfigDict(frozen=True)

    ablation_name: str = Field(..., description="Ablation description")
    results: list[SingleAblationResult] = Field(
        default_factory=list, description="Results per prompt"
    )


class AblationSweepResult(BaseModel):
    """Result of an ablation sweep."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Input prompt")
    criterion: str = Field(..., description="Criterion used")
    layers: list[int] = Field(..., description="Layers swept")
    baseline_passes: bool = Field(..., description="Whether baseline passes")
    results_by_layer: dict[int, bool] = Field(
        default_factory=dict, description="Pass/fail by layer"
    )
    causal_layers: list[int] = Field(
        default_factory=list, description="Layers that break the criterion when ablated"
    )


class AblationService:
    """Service class for ablation operations.

    Provides a high-level interface for CLI commands to run ablation studies
    without needing to understand the internal architecture.
    """

    Config = AblationServiceConfig

    @classmethod
    def get_criterion_function(cls, criterion_name: str) -> Callable[[str], bool]:
        """Get a criterion function by name.

        Args:
            criterion_name: Name of the criterion or substring to check.

        Returns:
            Criterion function.
        """
        criterion_map = {
            "function_call": AblationCriterionFunctions.function_call,
            "sorry": AblationCriterionFunctions.sorry,
            "positive": AblationCriterionFunctions.positive,
            "negative": AblationCriterionFunctions.negative,
            "refusal": AblationCriterionFunctions.refusal,
        }

        if criterion_name in criterion_map:
            func = criterion_map[criterion_name]
            func.__name__ = criterion_name
            return func
        else:
            # Treat as substring check
            def substring_criterion(x: str, s: str = criterion_name) -> bool:
                return s in x

            substring_criterion.__name__ = f"contains_{criterion_name}"
            return substring_criterion

    @classmethod
    async def run_ablation_sweep(
        cls,
        model: str,
        prompt: str,
        criterion: str | Callable[[str], bool],
        layers: list[int] | None = None,
        component: ComponentType = ComponentType.MLP,
        max_tokens: int = 50,
    ) -> AblationSweepResult:
        """Run ablation sweep across layers.

        Args:
            model: Model path or name.
            prompt: Prompt to test.
            criterion: Criterion name or function.
            layers: Layers to sweep (default: all).
            component: Component to ablate.
            max_tokens: Max tokens to generate.

        Returns:
            AblationSweepResult with results for each layer.
        """
        study = AblationStudy.from_pretrained(model)

        if layers is None:
            layers = list(range(study.adapter.num_layers))

        # Get criterion function
        if isinstance(criterion, str):
            criterion_fn = cls.get_criterion_function(criterion)
            criterion_name = criterion
        else:
            criterion_fn = criterion
            criterion_name = getattr(criterion, "__name__", "custom")

        config = AblationConfig(component=component, max_new_tokens=max_tokens)

        # Run sweep using framework
        result = study.run_layer_sweep(
            prompt=prompt,
            criterion=criterion_fn,
            layers=layers,
            component=component,
            task_name="ablation_study",
            config=config,
        )

        # Convert to service result
        results_by_layer = {}
        causal_layers = []

        for r in result.results:
            passes = criterion_fn(r.ablated_output)
            results_by_layer[r.layer] = passes
            if result.baseline_passes and not passes:
                causal_layers.append(r.layer)

        return AblationSweepResult(
            prompt=prompt,
            criterion=criterion_name,
            layers=layers,
            baseline_passes=result.baseline_passes,
            results_by_layer=results_by_layer,
            causal_layers=causal_layers,
        )

    @classmethod
    async def run_multi_ablation(
        cls,
        model: str,
        prompt: str,
        layers: list[int],
        criterion: str | Callable[[str], bool],
        component: ComponentType = ComponentType.MLP,
        max_tokens: int = 50,
    ) -> tuple[SingleAblationResult, SingleAblationResult]:
        """Run multi-layer ablation (all layers together).

        Args:
            model: Model path or name.
            prompt: Prompt to test.
            layers: Layers to ablate together.
            criterion: Criterion name or function.
            component: Component to ablate.
            max_tokens: Max tokens to generate.

        Returns:
            Tuple of (baseline_result, ablated_result).
        """
        study = AblationStudy.from_pretrained(model)

        # Get criterion function
        if isinstance(criterion, str):
            criterion_fn = cls.get_criterion_function(criterion)
            criterion_name = criterion
        else:
            criterion_fn = criterion
            criterion_name = getattr(criterion, "__name__", "custom")

        config = AblationConfig(component=component, max_new_tokens=max_tokens)

        # Get baseline
        baseline_output = study.ablate_and_generate(prompt, layers=[], config=config)
        baseline_passes = criterion_fn(baseline_output)

        baseline_result = SingleAblationResult(
            prompt=prompt,
            expected=criterion_name,
            ablation_name="baseline",
            output=baseline_output,
            passes_criterion=baseline_passes,
        )

        # Get ablated
        ablated_output = study.ablate_and_generate(prompt, layers=layers, config=config)
        ablated_passes = criterion_fn(ablated_output)

        layer_str = ",".join(str(layer) for layer in layers)
        ablated_result = SingleAblationResult(
            prompt=prompt,
            expected=criterion_name,
            ablation_name=f"L{layer_str}",
            output=ablated_output,
            passes_criterion=ablated_passes,
        )

        return baseline_result, ablated_result

    @classmethod
    async def run_multi_prompt_ablation(
        cls,
        model: str,
        prompt_pairs: list[tuple[str, str]],
        layers: list[int] | None = None,
        component: ComponentType = ComponentType.MLP,
        max_tokens: int = 50,
        multi_mode: bool = False,
    ) -> list[MultiPromptAblationResult]:
        """Run ablation test on multiple prompts.

        Args:
            model: Model path or name.
            prompt_pairs: List of (prompt, expected) pairs.
            layers: Layers to test.
            component: Component to ablate.
            max_tokens: Max tokens to generate.
            multi_mode: If True, ablate all layers together. If False, sweep each.

        Returns:
            List of results per ablation setting.
        """
        study = AblationStudy.from_pretrained(model)

        if layers is None:
            layers = list(range(study.adapter.num_layers))

        config = AblationConfig(component=component, max_new_tokens=max_tokens)

        results: list[MultiPromptAblationResult] = []

        # Baseline
        baseline_results = []
        for prompt, expected in prompt_pairs:
            output = study.ablate_and_generate(prompt, layers=[], config=config)
            baseline_results.append(
                SingleAblationResult(
                    prompt=prompt,
                    expected=expected,
                    ablation_name="baseline",
                    output=output,
                    passes_criterion=expected in output,
                )
            )
        results.append(
            MultiPromptAblationResult(ablation_name="baseline", results=baseline_results)
        )

        if multi_mode:
            # Single test with all layers together
            layer_str = ",".join(str(layer) for layer in layers)
            layer_results = []
            for prompt, expected in prompt_pairs:
                output = study.ablate_and_generate(prompt, layers=layers, config=config)
                layer_results.append(
                    SingleAblationResult(
                        prompt=prompt,
                        expected=expected,
                        ablation_name=f"L{layer_str}",
                        output=output,
                        passes_criterion=expected in output,
                    )
                )
            results.append(
                MultiPromptAblationResult(ablation_name=f"L{layer_str}", results=layer_results)
            )
        else:
            # Sweep each layer
            for layer in layers:
                layer_results = []
                for prompt, expected in prompt_pairs:
                    output = study.ablate_and_generate(prompt, layers=[layer], config=config)
                    layer_results.append(
                        SingleAblationResult(
                            prompt=prompt,
                            expected=expected,
                            ablation_name=f"L{layer}",
                            output=output,
                            passes_criterion=expected in output,
                        )
                    )
                results.append(
                    MultiPromptAblationResult(ablation_name=f"L{layer}", results=layer_results)
                )

        return results

    @classmethod
    def parse_layers_string(cls, layers_str: str | None, num_layers: int) -> list[int]:
        """Parse comma-separated layer list with support for ranges.

        Args:
            layers_str: Layer specification string.
            num_layers: Total number of layers in model.

        Returns:
            List of layer indices.
        """
        if not layers_str:
            return list(range(num_layers))

        layers = []
        for part in layers_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))
        return layers

    @classmethod
    def parse_prompt_pairs(cls, prompts_str: str) -> list[tuple[str, str]]:
        """Parse prompts string into (prompt, expected) pairs.

        Args:
            prompts_str: Format: "prompt1:expected1|prompt2:expected2".

        Returns:
            List of (prompt, expected) tuples.
        """
        pairs = []
        for item in prompts_str.split("|"):
            item = item.strip()
            if ":" in item:
                prompt, expected = item.rsplit(":", 1)
                pairs.append((prompt.strip(), expected.strip()))
            else:
                pairs.append((item, ""))
        return pairs


__all__ = [
    "AblationService",
    "AblationServiceConfig",
    "AblationCriterionFunctions",
    "SingleAblationResult",
    "MultiPromptAblationResult",
    "AblationSweepResult",
]
