"""Generation service for CLI commands.

This module provides services for token generation with analysis.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Configuration for generation with analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompt: str = Field(..., description="Prompt to generate from")
    max_tokens: int = Field(default=30, description="Max tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_k: int = Field(default=10, description="Top-k for analysis")
    layer_step: int = Field(default=4, description="Layer step for analysis")
    track_tokens: list[str] = Field(default_factory=list, description="Tokens to track")
    chat_template_file: str | None = Field(default=None, description="Chat template file")
    use_raw: bool = Field(default=False, description="Use raw mode")
    expected_answer: str | None = Field(default=None, description="Expected answer")
    find_answer: str | None = Field(default=None, description="Pattern to find")
    no_find_answer: bool = Field(default=False, description="Disable answer finding")
    compare_format: bool = Field(default=False, description="Compare with/without trailing space")
    show_tokens: bool = Field(default=False, description="Show token breakdown")


class GenerationResult(BaseModel):
    """Result of generation with analysis."""

    model_config = ConfigDict(frozen=True)

    generated_text: str = Field(default="")
    tokens: list[str] = Field(default_factory=list)
    prompt: str = Field(default="")
    expected_answer: str | None = Field(default=None)
    answer_found: bool = Field(default=False)
    answer_onset: int | None = Field(default=None)
    is_answer_first: bool = Field(default=False)
    has_trailing_space: bool = Field(default=False)
    comparison_results: list[dict[str, Any]] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "GENERATION WITH ANALYSIS",
            f"{'=' * 70}",
        ]

        if self.comparison_results:
            # Format comparison mode
            lines.append("\n=== Format Comparison ===")
            for r in self.comparison_results:
                marker = "[space]" if r.get("has_trailing_space") else "[no-space]"
                lines.append(f"{marker} {r.get('prompt', '')!r}")
                lines.append(f"  -> {r.get('output', '')!r}")
                if r.get("expected"):
                    if r.get("answer_found"):
                        onset = r.get("onset_index", "?")
                        first = " (answer-first)" if r.get("is_answer_first") else " (delayed)"
                        lines.append(f"  Expected: {r.get('expected')}, onset={onset}{first}")
                    else:
                        lines.append(f"  Expected: {r.get('expected')}, NOT FOUND")
                lines.append("")
        else:
            # Single generation mode
            marker = "[space]" if self.has_trailing_space else "[no-space]"
            lines.append(f"\n{marker} {self.prompt!r}")
            lines.append(f"  -> {self.generated_text!r}")

            if self.expected_answer:
                if self.answer_found:
                    first = " (answer-first)" if self.is_answer_first else " (delayed)"
                    lines.append(f"  Expected: {self.expected_answer}, onset={self.answer_onset}{first}")
                else:
                    lines.append(f"  Expected: {self.expected_answer}, NOT FOUND")

            if self.tokens:
                lines.append(f"\n  Tokens: {' '.join(repr(t) for t in self.tokens[:10])}")
                if len(self.tokens) > 10:
                    lines.append("  ...")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save results to file."""
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class GenerationService:
    """Service for generation with analysis."""

    @classmethod
    async def generate(cls, config: GenerationConfig) -> GenerationResult:
        """Generate with logit lens analysis.

        Tests format issues and answer onset timing.
        """
        from mlx_lm import generate, load

        from ..utils import apply_chat_template, extract_expected_answer

        model, tokenizer = load(config.model)

        # Check chat template
        use_raw = config.use_raw
        has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

        # Handle format comparison mode
        if config.compare_format:
            prompts = []
            base = config.prompt.rstrip()
            prompts.append(base)  # without trailing space
            prompts.append(base + " ")  # with trailing space

            comparison_results = []
            for prompt in prompts:
                formatted_prompt = prompt
                if not use_raw and has_chat_template:
                    formatted_prompt = apply_chat_template(tokenizer, prompt)

                if config.temperature == 0:
                    output = generate(
                        model, tokenizer,
                        prompt=formatted_prompt,
                        max_tokens=config.max_tokens,
                        verbose=False,
                    )
                else:
                    output = generate(
                        model, tokenizer,
                        prompt=formatted_prompt,
                        max_tokens=config.max_tokens,
                        temp=config.temperature,
                        verbose=False,
                    )

                expected = config.expected_answer or extract_expected_answer(prompt)
                onset_info = cls._find_answer_onset(output, expected, tokenizer)

                comparison_results.append({
                    "prompt": prompt,
                    "has_trailing_space": prompt.endswith(" "),
                    "output": output,
                    "expected": expected,
                    **onset_info,
                })

            return GenerationResult(
                generated_text="",
                prompt=config.prompt,
                comparison_results=comparison_results,
            )

        # Single generation mode
        formatted_prompt = config.prompt
        if not use_raw and has_chat_template:
            formatted_prompt = apply_chat_template(tokenizer, config.prompt)

        if config.temperature == 0:
            output = generate(
                model, tokenizer,
                prompt=formatted_prompt,
                max_tokens=config.max_tokens,
                verbose=False,
            )
        else:
            output = generate(
                model, tokenizer,
                prompt=formatted_prompt,
                max_tokens=config.max_tokens,
                temp=config.temperature,
                verbose=False,
            )

        # Get tokens
        prompt_ids = tokenizer.encode(formatted_prompt)
        output_ids = tokenizer.encode(formatted_prompt + output)
        gen_ids = output_ids[len(prompt_ids):]
        tokens = [tokenizer.decode([tid]) for tid in gen_ids]

        # Find answer onset
        expected = config.expected_answer or extract_expected_answer(config.prompt)
        onset_info = cls._find_answer_onset(output, expected, tokenizer)

        return GenerationResult(
            generated_text=output,
            tokens=tokens,
            prompt=config.prompt,
            expected_answer=expected,
            answer_found=onset_info.get("answer_found", False),
            answer_onset=onset_info.get("onset_index"),
            is_answer_first=onset_info.get("is_answer_first", False),
            has_trailing_space=config.prompt.endswith(" "),
        )

    @staticmethod
    def _normalize_number(s: str) -> str:
        """Normalize a number string."""
        return re.sub(r"[\s,\u202f\u00a0]+", "", s)

    @classmethod
    def _find_answer_onset(
        cls,
        output: str,
        expected_answer: str | None,
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Find where the expected answer appears in output."""
        if not expected_answer:
            return {"answer_found": False, "onset_index": None, "is_answer_first": False}

        # Normalize expected answer
        normalized_expected = cls._normalize_number(expected_answer)

        # Tokenize output
        output_ids = tokenizer.encode(output)
        tokens = [tokenizer.decode([tid]) for tid in output_ids]

        # Look for answer in tokens
        for i, token in enumerate(tokens):
            normalized_token = cls._normalize_number(token.strip())
            if normalized_token and normalized_expected.startswith(normalized_token):
                return {
                    "answer_found": True,
                    "onset_index": i,
                    "is_answer_first": i == 0,
                }

        # Check if answer appears in full output
        if normalized_expected in cls._normalize_number(output):
            return {"answer_found": True, "onset_index": None, "is_answer_first": False}

        return {"answer_found": False, "onset_index": None, "is_answer_first": False}


class LogitEvolutionConfig(BaseModel):
    """Configuration for logit evolution analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompt: str = Field(..., description="Prompt to analyze")
    track_tokens: list[str] = Field(default_factory=list, description="Tokens to track")
    layer_step: int = Field(default=4, description="Layer step")
    top_k: int = Field(default=10, description="Top-k predictions")


class LogitEvolutionResult(BaseModel):
    """Result of logit evolution analysis."""

    model_config = ConfigDict(frozen=True)

    evolutions: list[dict[str, Any]] = Field(default_factory=list)
    model_id: str = Field(default="")
    prompt: str = Field(default="")
    tracked_tokens: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "LOGIT EVOLUTION ANALYSIS",
            f"{'=' * 70}",
            f"Model: {self.model_id}",
            f"Prompt: {self.prompt[:50]}...",
            f"Tracking: {', '.join(self.tracked_tokens)}",
            "",
        ]

        for evo in self.evolutions:
            layer = evo.get("layer", "?")
            top_token = evo.get("top_token", "?")
            top_prob = evo.get("top_prob", 0)
            lines.append(f"Layer {layer:>3}: {top_token!r:<15} ({top_prob:.1%})")

            # Show tracked token probabilities
            if "tracked" in evo:
                for token, prob in evo["tracked"].items():
                    lines.append(f"           {token!r:<15} ({prob:.1%})")

        return "\n".join(lines)


class LogitEvolutionService:
    """Service for logit evolution analysis."""

    @classmethod
    async def analyze(cls, config: LogitEvolutionConfig) -> LogitEvolutionResult:
        """Analyze logit evolution across layers.

        Shows how token predictions change layer by layer.
        """
        from ..analyzer import AnalysisConfig, LayerStrategy, ModelAnalyzer

        async with ModelAnalyzer.from_pretrained(config.model) as analyzer:
            info = analyzer.model_info

            # Configure to capture at regular intervals
            layers_to_capture = list(range(0, info.num_layers, config.layer_step))
            if info.num_layers - 1 not in layers_to_capture:
                layers_to_capture.append(info.num_layers - 1)

            analysis_config = AnalysisConfig(
                layer_strategy=LayerStrategy.SPECIFIC,
                capture_layers=layers_to_capture,
                top_k=config.top_k,
                track_tokens=config.track_tokens,
            )

            result = await analyzer.analyze(config.prompt, analysis_config)

            evolutions = []
            for lp in result.layer_predictions:
                evo = {
                    "layer": lp.layer_idx,
                    "top_token": lp.top_token,
                    "top_prob": lp.probability,
                }

                # Add tracked token info if available
                if config.track_tokens and result.token_evolutions:
                    tracked = {}
                    for te in result.token_evolutions:
                        if te.token in config.track_tokens:
                            # Find probability at this layer
                            for layer_info in te.layer_probabilities:
                                if layer_info.get("layer") == lp.layer_idx:
                                    tracked[te.token] = layer_info.get("probability", 0)
                    if tracked:
                        evo["tracked"] = tracked

                evolutions.append(evo)

            return LogitEvolutionResult(
                evolutions=evolutions,
                model_id=config.model,
                prompt=config.prompt,
                tracked_tokens=config.track_tokens,
            )


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "GenerationService",
    "LogitEvolutionConfig",
    "LogitEvolutionResult",
    "LogitEvolutionService",
]
