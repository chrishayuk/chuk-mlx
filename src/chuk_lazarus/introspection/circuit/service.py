"""Circuit service for CLI commands.

This module provides the CircuitService class that wraps circuit
functionality for CLI commands.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CircuitCaptureConfig(BaseModel):
    """Configuration for circuit capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    prompts: list[str] = Field(..., description="Prompts to capture")
    layer: int = Field(..., description="Layer to capture at")
    results: list[int] | None = Field(default=None, description="Expected results")
    extract_direction: bool = Field(default=False, description="Extract direction")
    output_path: str | None = Field(default=None, description="Output path")


class CircuitCaptureResult(BaseModel):
    """Result of circuit capture."""

    model_config = ConfigDict(frozen=True)

    num_prompts: int = Field(default=0)
    layer: int = Field(default=0)
    output_path: str | None = Field(default=None)
    direction_norm: float | None = Field(default=None)
    activations_shape: list[int] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT CAPTURE",
            f"{'=' * 70}",
            f"\nCaptured {self.num_prompts} prompts at layer {self.layer}",
        ]
        if self.activations_shape:
            lines.append(f"Activations shape: {self.activations_shape}")
        if self.direction_norm is not None:
            lines.append(f"Direction norm: {self.direction_norm:.4f}")
        if self.output_path:
            lines.append(f"Saved to: {self.output_path}")
        return "\n".join(lines)


class CircuitInvokeConfig(BaseModel):
    """Configuration for circuit invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    prompts: list[str] = Field(..., description="Prompts to invoke on")
    method: str = Field(default="project", description="Invocation method")
    coefficient: float | None = Field(default=None, description="Coefficient")
    layer: int | None = Field(default=None, description="Target layer")
    top_k: int = Field(default=10, description="Top-k predictions")


class CircuitInvokeResult(BaseModel):
    """Result of circuit invocation."""

    model_config = ConfigDict(frozen=True)

    results: list[dict[str, Any]] = Field(default_factory=list)
    method: str = Field(default="")

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT INVOCATION",
            f"{'=' * 70}",
            f"Method: {self.method}",
            "",
        ]
        for r in self.results:
            prompt = r.get("prompt", "")[:30]
            prediction = r.get("prediction", "?")
            score = r.get("score", 0)
            lines.append(f"  {prompt:<30} -> {prediction} (score={score:.3f})")
        return "\n".join(lines)


class CircuitTestConfig(BaseModel):
    """Configuration for circuit testing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    prompts: list[str] = Field(..., description="Test prompts")
    expected_results: list[int] | None = Field(default=None, description="Expected results")
    threshold: float = Field(default=0.1, description="Threshold")


class CircuitTestResult(BaseModel):
    """Result of circuit testing."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(default=0.0)
    results: list[dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    correct: int = Field(default=0)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT TEST",
            f"{'=' * 70}",
            f"\nAccuracy: {self.accuracy:.1%} ({self.correct}/{self.total})",
        ]
        return "\n".join(lines)


class CircuitViewConfig(BaseModel):
    """Configuration for circuit viewing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file: str = Field(..., description="Circuit file path")
    show_activations: bool = Field(default=False, description="Show activations")
    show_direction: bool = Field(default=True, description="Show direction")


class CircuitViewResult(BaseModel):
    """Result of circuit viewing."""

    model_config = ConfigDict(frozen=True)

    info: dict[str, Any] = Field(default_factory=dict)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT VIEW",
            f"{'=' * 70}",
        ]
        for key, value in self.info.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class CircuitCompareConfig(BaseModel):
    """Configuration for circuit comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file_a: str = Field(..., description="First circuit file")
    circuit_file_b: str = Field(..., description="Second circuit file")


class CircuitCompareResult(BaseModel):
    """Result of circuit comparison."""

    model_config = ConfigDict(frozen=True)

    similarity: float = Field(default=0.0)
    differences: list[str] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT COMPARISON",
            f"{'=' * 70}",
            f"\nCosine similarity: {self.similarity:.4f}",
        ]
        if self.differences:
            lines.append("\nDifferences:")
            for diff in self.differences:
                lines.append(f"  - {diff}")
        return "\n".join(lines)


class CircuitDecodeConfig(BaseModel):
    """Configuration for circuit decoding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model path or name")
    circuit_file: str = Field(..., description="Circuit file path")
    top_k: int = Field(default=20, description="Top-k tokens")


class CircuitDecodeResult(BaseModel):
    """Result of circuit decoding."""

    model_config = ConfigDict(frozen=True)

    top_tokens: list[dict[str, Any]] = Field(default_factory=list)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            "CIRCUIT DECODE",
            f"{'=' * 70}",
        ]
        for token in self.top_tokens[:20]:
            lines.append(f"  {token.get('token', '?')!r}: {token.get('score', 0):.4f}")
        return "\n".join(lines)


class CircuitExportConfig(BaseModel):
    """Configuration for circuit export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit_file: str = Field(..., description="Circuit file path")
    output_path: str | None = Field(default=None, description="Output path")
    output_format: str = Field(default="json", description="Output format")
    direction: str = Field(default="TB", description="Graph direction")


class CircuitExportResult(BaseModel):
    """Result of circuit export."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(default="")
    format: str = Field(default="")
    output_path: str | None = Field(default=None)

    def to_display(self) -> str:
        """Format result for display."""
        if self.output_path:
            return f"Exported to: {self.output_path}"
        return self.content


class CircuitService:
    """Service class for circuit operations."""

    @classmethod
    async def capture(cls, config: CircuitCaptureConfig) -> CircuitCaptureResult:
        """Capture circuit activations."""
        import mlx.core as mx

        from ...models_v2 import load_model
        from ..accessor import ModelAccessor

        # Load model using framework loader
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer
        model_config = load_result.config

        accessor = ModelAccessor(model=model, config=model_config)

        def get_hidden_at_layer(prompt: str, layer: int) -> np.ndarray:
            """Get hidden state at specific layer."""
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            h = accessor.embed(input_ids)

            seq_len = input_ids.shape[1]
            mask = accessor.create_causal_mask(seq_len, h.dtype)

            for idx, lyr in enumerate(accessor.layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                if idx == layer:
                    return np.array(h[0, -1, :].tolist())

            return np.array(h[0, -1, :].tolist())

        # Collect activations
        activations = []
        for prompt in config.prompts:
            h = get_hidden_at_layer(prompt, config.layer)
            activations.append(h)

        activations = np.array(activations)

        # Extract direction if requested
        direction = None
        direction_norm = None
        if config.extract_direction and config.results:
            # Use Ridge regression to find direction
            from sklearn.linear_model import Ridge

            y = np.array(config.results)
            ridge = Ridge(alpha=1.0)
            ridge.fit(activations, y)
            direction = ridge.coef_
            direction_norm = float(np.linalg.norm(direction))

        # Save if output path specified
        if config.output_path:
            output_data = {
                "model": config.model,
                "layer": config.layer,
                "num_prompts": len(config.prompts),
                "prompts": config.prompts,
            }
            if config.results:
                output_data["results"] = config.results
            if direction is not None:
                output_data["direction"] = direction.tolist()

            with open(config.output_path, "w") as f:
                json.dump(output_data, f, indent=2)

        return CircuitCaptureResult(
            num_prompts=len(config.prompts),
            layer=config.layer,
            output_path=config.output_path,
            direction_norm=direction_norm,
            activations_shape=list(activations.shape),
        )

    @classmethod
    async def invoke(cls, config: CircuitInvokeConfig) -> CircuitInvokeResult:
        """Invoke a captured circuit."""
        import mlx.core as mx

        from ...models_v2 import load_model
        from ..accessor import ModelAccessor

        # Load circuit
        with open(config.circuit_file) as f:
            circuit_data = json.load(f)

        direction = np.array(circuit_data.get("direction", []))
        layer = config.layer or circuit_data.get("layer", 0)

        # Load model using framework loader
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer
        model_config = load_result.config

        accessor = ModelAccessor(model=model, config=model_config)

        def get_hidden_at_layer(prompt: str, layer: int) -> np.ndarray:
            input_ids = mx.array(tokenizer.encode(prompt))[None, :]
            h = accessor.embed(input_ids)

            seq_len = input_ids.shape[1]
            mask = accessor.create_causal_mask(seq_len, h.dtype)

            for idx, lyr in enumerate(accessor.layers):
                try:
                    out = lyr(h, mask=mask)
                except TypeError:
                    out = lyr(h)
                h = (
                    out.hidden_states
                    if hasattr(out, "hidden_states")
                    else (out[0] if isinstance(out, tuple) else out)
                )
                if idx == layer:
                    return np.array(h[0, -1, :].tolist())

            return np.array(h[0, -1, :].tolist())

        results = []
        for prompt in config.prompts:
            h = get_hidden_at_layer(prompt, layer)

            # Project onto direction
            if len(direction) > 0:
                score = float(np.dot(h, direction) / (np.linalg.norm(direction) + 1e-8))
                prediction = "positive" if score > 0 else "negative"
            else:
                score = 0.0
                prediction = "unknown"

            results.append(
                {
                    "prompt": prompt,
                    "score": score,
                    "prediction": prediction,
                }
            )

        return CircuitInvokeResult(
            results=results,
            method=config.method,
        )

    @classmethod
    async def test(cls, config: CircuitTestConfig) -> CircuitTestResult:
        """Test circuit predictions."""
        # Use invoke to get predictions
        invoke_config = CircuitInvokeConfig(
            model=config.model,
            circuit_file=config.circuit_file,
            prompts=config.prompts,
        )
        invoke_result = await cls.invoke(invoke_config)

        # Compare with expected
        correct = 0
        total = len(config.prompts)
        results = []

        expected = config.expected_results or [0] * total

        for i, (r, exp) in enumerate(zip(invoke_result.results, expected)):
            pred = 1 if r["score"] > config.threshold else 0
            is_correct = pred == exp
            if is_correct:
                correct += 1

            results.append(
                {
                    **r,
                    "expected": exp,
                    "predicted": pred,
                    "correct": is_correct,
                }
            )

        return CircuitTestResult(
            accuracy=correct / total if total > 0 else 0.0,
            results=results,
            total=total,
            correct=correct,
        )

    @classmethod
    async def view(cls, config: CircuitViewConfig) -> CircuitViewResult:
        """View circuit contents."""
        with open(config.circuit_file) as f:
            circuit_data = json.load(f)

        info = {
            "model": circuit_data.get("model", "unknown"),
            "layer": circuit_data.get("layer", "unknown"),
            "num_prompts": circuit_data.get("num_prompts", 0),
        }

        if config.show_direction and "direction" in circuit_data:
            direction = np.array(circuit_data["direction"])
            info["direction_dim"] = len(direction)
            info["direction_norm"] = float(np.linalg.norm(direction))

        return CircuitViewResult(info=info)

    @classmethod
    async def compare(cls, config: CircuitCompareConfig) -> CircuitCompareResult:
        """Compare two circuits."""
        with open(config.circuit_file_a) as f:
            circuit_a = json.load(f)
        with open(config.circuit_file_b) as f:
            circuit_b = json.load(f)

        differences = []

        # Compare metadata
        if circuit_a.get("model") != circuit_b.get("model"):
            differences.append(
                f"Different models: {circuit_a.get('model')} vs {circuit_b.get('model')}"
            )
        if circuit_a.get("layer") != circuit_b.get("layer"):
            differences.append(
                f"Different layers: {circuit_a.get('layer')} vs {circuit_b.get('layer')}"
            )

        # Compare directions
        similarity = 0.0
        dir_a = np.array(circuit_a.get("direction", []))
        dir_b = np.array(circuit_b.get("direction", []))

        if len(dir_a) > 0 and len(dir_b) > 0 and len(dir_a) == len(dir_b):
            similarity = float(
                np.dot(dir_a, dir_b) / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b) + 1e-8)
            )
        elif len(dir_a) != len(dir_b):
            differences.append(f"Different direction dimensions: {len(dir_a)} vs {len(dir_b)}")

        return CircuitCompareResult(
            similarity=similarity,
            differences=differences,
        )

    @classmethod
    async def decode(cls, config: CircuitDecodeConfig) -> CircuitDecodeResult:
        """Decode circuit through vocabulary."""
        from ...models_v2 import load_model

        # Load circuit
        with open(config.circuit_file) as f:
            circuit_data = json.load(f)

        direction = np.array(circuit_data.get("direction", []))
        if len(direction) == 0:
            return CircuitDecodeResult(top_tokens=[])

        # Load model using framework loader
        load_result = load_model(config.model)
        model = load_result.model
        tokenizer = load_result.tokenizer

        # Get unembedding matrix
        if hasattr(model, "lm_head"):
            unembed = np.array(model.lm_head.weight.tolist())
        elif hasattr(model, "output"):
            unembed = np.array(model.output.weight.tolist())
        else:
            return CircuitDecodeResult(top_tokens=[])

        # Project direction through vocabulary
        scores = np.dot(unembed, direction)
        top_indices = np.argsort(scores)[-config.top_k :][::-1]

        top_tokens = []
        for idx in top_indices:
            token = tokenizer.decode([int(idx)])
            top_tokens.append(
                {
                    "token": token,
                    "token_id": int(idx),
                    "score": float(scores[idx]),
                }
            )

        return CircuitDecodeResult(top_tokens=top_tokens)

    @classmethod
    async def export(cls, config: CircuitExportConfig) -> CircuitExportResult:
        """Export circuit in various formats."""
        with open(config.circuit_file) as f:
            circuit_data = json.load(f)

        if config.output_format == "json":
            content = json.dumps(circuit_data, indent=2)
        elif config.output_format == "dot":
            # Simple DOT format
            content = cls._to_dot(circuit_data, config.direction)
        elif config.output_format == "mermaid":
            content = cls._to_mermaid(circuit_data, config.direction)
        else:
            content = json.dumps(circuit_data, indent=2)

        if config.output_path:
            with open(config.output_path, "w") as f:
                f.write(content)

        return CircuitExportResult(
            content=content,
            format=config.output_format,
            output_path=config.output_path,
        )

    @staticmethod
    def _to_dot(circuit_data: dict, direction: str = "TB") -> str:
        """Convert circuit to DOT format."""
        lines = [
            "digraph Circuit {",
            f"  rankdir={direction};",
            f'  label="Circuit at layer {circuit_data.get("layer", "?")}";',
            "  node [shape=box];",
        ]

        # Add nodes for prompts
        for i, prompt in enumerate(circuit_data.get("prompts", [])[:10]):
            short = prompt[:20].replace('"', '\\"')
            lines.append(f'  p{i} [label="{short}..."];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _to_mermaid(circuit_data: dict, direction: str = "TB") -> str:
        """Convert circuit to Mermaid format."""
        lines = [
            f"graph {direction}",
            f"  subgraph Layer{circuit_data.get('layer', '?')}",
        ]

        for i, prompt in enumerate(circuit_data.get("prompts", [])[:10]):
            short = prompt[:20].replace('"', "'")
            lines.append(f'    p{i}["{short}..."]')

        lines.append("  end")
        return "\n".join(lines)


__all__ = [
    "CircuitCaptureConfig",
    "CircuitCaptureResult",
    "CircuitCompareConfig",
    "CircuitCompareResult",
    "CircuitDecodeConfig",
    "CircuitDecodeResult",
    "CircuitExportConfig",
    "CircuitExportResult",
    "CircuitInvokeConfig",
    "CircuitInvokeResult",
    "CircuitService",
    "CircuitTestConfig",
    "CircuitTestResult",
    "CircuitViewConfig",
    "CircuitViewResult",
]
