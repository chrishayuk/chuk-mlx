"""
Core AblationStudy class for running ablation experiments.

Provides infrastructure for running ablation experiments across different
model families to identify causal circuits.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from .adapter import ModelAdapter
from .config import AblationConfig, ComponentType
from .models import AblationResult, LayerSweepResult


class AblationStudy:
    """
    Run ablation studies on a model to identify causal circuits.

    Example:
        >>> study = AblationStudy.from_pretrained("model_id")
        >>> results = study.run_mlp_sweep(
        ...     prompt="What is the weather?",
        ...     criterion=lambda x: "function_call" in x,
        ... )
    """

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_family: str | None = None,
    ) -> AblationStudy:
        """
        Load a model for ablation study.

        Args:
            model_id: HuggingFace model ID or local path
            model_family: Optional model family hint (gemma, llama, etc.)

        Returns:
            AblationStudy instance
        """
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        # Download model
        model_path = snapshot_download(
            model_id,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"],
        )

        # Detect model family from config if not specified
        if model_family is None:
            model_family = cls._detect_family(model_path)

        # Load model based on family
        model, config = cls._load_model(model_path, model_family)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        adapter = ModelAdapter(model, tokenizer, config)
        return cls(adapter)

    @staticmethod
    def _detect_family(model_path: str) -> str:
        """Detect model family from config.json."""
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", "").lower()
        architectures = config.get("architectures", [])

        # Map to our families
        if "gemma" in model_type or any("gemma" in a.lower() for a in architectures):
            return "gemma"
        elif "qwen" in model_type or any("qwen" in a.lower() for a in architectures):
            return "qwen3"
        elif "gpt_oss" in model_type or any("gptoss" in a.lower() for a in architectures):
            return "gpt_oss"
        elif "llama" in model_type or any("llama" in a.lower() for a in architectures):
            if "4" in model_type:
                return "llama4"
            return "llama"
        elif "mamba" in model_type:
            return "mamba"
        elif "jamba" in model_type:
            return "jamba"
        elif "granite" in model_type:
            return "granite"
        elif "starcoder" in model_type:
            return "starcoder2"
        else:
            # Default to llama-style (most common)
            return "llama"

    @staticmethod
    def _load_model(model_path: str, family: str) -> tuple[Any, Any]:
        """Load model based on family using existing from_pretrained methods."""
        import asyncio

        from mlx.utils import tree_unflatten

        if family == "gemma":
            from ...models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
            from ...models_v2.families.gemma.convert import load_hf_config, load_weights

            hf_config = load_hf_config(model_path)
            config = GemmaConfig.from_hf_config(hf_config)
            model = GemmaForCausalLM(config)
            raw_weights = load_weights(model_path)
            sanitized_weights = model.sanitize(raw_weights)
            nested_weights = tree_unflatten(list(sanitized_weights.items()))
            model.update(nested_weights)
            mx.eval(model.parameters())
            return model, config

        elif family == "llama":
            from ...inference.loader import DType, HFLoader, StandardWeightConverter
            from ...models_v2.families.llama import LlamaConfig, LlamaForCausalLM

            # Load config
            with open(Path(model_path) / "config.json") as f:
                config_data = json.load(f)
            config = LlamaConfig.from_hf_config(config_data)

            # Create model
            model = LlamaForCausalLM(config)

            # Load weights using HFLoader
            converter = StandardWeightConverter(tie_word_embeddings=config.tie_word_embeddings)
            loaded = HFLoader.load_weights(Path(model_path), DType.BFLOAT16, converter)
            nested = HFLoader.build_nested_weights(loaded)
            model.update(nested)
            mx.eval(model.parameters())
            return model, config

        elif family == "granite":
            from ...models_v2.families.granite import GraniteConfig, GraniteForCausalLM
            from ...models_v2.families.granite.convert import load_hf_config, load_weights

            hf_config = load_hf_config(model_path)
            config = GraniteConfig.from_hf_config(hf_config)
            model = GraniteForCausalLM(config)
            raw_weights = load_weights(model_path)
            if hasattr(model, "sanitize"):
                raw_weights = model.sanitize(raw_weights)
            nested_weights = tree_unflatten(list(raw_weights.items()))
            model.update(nested_weights)
            mx.eval(model.parameters())
            return model, config

        elif family == "jamba":
            from ...models_v2.families.jamba import JambaForCausalLM

            model = asyncio.run(JambaForCausalLM.from_pretrained_async(model_path))
            mx.eval(model.parameters())
            return model, model.config

        elif family == "starcoder2":
            from ...models_v2.families.starcoder2 import StarCoder2ForCausalLM

            model = asyncio.run(StarCoder2ForCausalLM.from_pretrained_async(model_path))
            mx.eval(model.parameters())
            return model, model.config

        elif family == "qwen3":
            from mlx.utils import tree_unflatten

            from ...inference.loader import DType, HFLoader, StandardWeightConverter
            from ...models_v2.families.qwen3 import Qwen3Config, Qwen3ForCausalLM

            with open(Path(model_path) / "config.json") as f:
                config_data = json.load(f)
            config = Qwen3Config.from_hf_config(config_data)
            model = Qwen3ForCausalLM(config)
            converter = StandardWeightConverter(tie_word_embeddings=config.tie_word_embeddings)
            loaded = HFLoader.load_weights(Path(model_path), DType.BFLOAT16, converter)
            sanitized = model.sanitize(
                loaded.weights, tie_word_embeddings=config.tie_word_embeddings
            )
            nested = tree_unflatten(list(sanitized.items()))
            model.update(nested)
            mx.eval(model.parameters())
            return model, config

        elif family == "gpt_oss":
            from mlx.utils import tree_unflatten

            from ...inference.loader import HFLoader
            from ...models_v2.families.gpt_oss import GptOssConfig, GptOssForCausalLM

            with open(Path(model_path) / "config.json") as f:
                config_data = json.load(f)
            config = GptOssConfig.from_hf_config(config_data)
            model = GptOssForCausalLM(config)
            raw_weights = HFLoader.load_raw_weights(Path(model_path))
            sanitized = model.sanitize(raw_weights, tie_word_embeddings=config.tie_word_embeddings)
            nested = tree_unflatten(list(sanitized.items()))
            model.update(nested)
            mx.eval(model.parameters())
            return model, config

        else:
            raise ValueError(
                f"Unsupported model family: {family}. "
                f"Supported: gemma, llama, granite, jamba, starcoder2, qwen3, gpt_oss"
            )

    def ablate_and_generate(
        self,
        prompt: str,
        layers: list[int],
        component: ComponentType = ComponentType.MLP,
        config: AblationConfig | None = None,
    ) -> str:
        """
        Generate with specified layers ablated.

        Args:
            prompt: Input prompt
            layers: Layer indices to ablate
            component: Which component to ablate
            config: Ablation configuration

        Returns:
            Generated text
        """
        if config is None:
            config = AblationConfig()

        # Tokenize
        input_ids = self.adapter.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Store original weights
        original_weights = {}

        for layer_idx in layers:
            original_weights[layer_idx] = {}

            if component in [ComponentType.MLP, ComponentType.BOTH, ComponentType.MLP_DOWN]:
                orig = self.adapter.get_mlp_down_weight(layer_idx)
                original_weights[layer_idx]["mlp_down"] = mx.array(orig)
                self.adapter.set_mlp_down_weight(layer_idx, mx.zeros_like(orig))

            if component in [ComponentType.ATTENTION, ComponentType.BOTH, ComponentType.ATTN_O]:
                orig = self.adapter.get_attn_o_weight(layer_idx)
                original_weights[layer_idx]["attn_o"] = mx.array(orig)
                self.adapter.set_attn_o_weight(layer_idx, mx.zeros_like(orig))

        # Generate
        output = self.adapter.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        # Restore weights
        for layer_idx, weights in original_weights.items():
            if "mlp_down" in weights:
                self.adapter.set_mlp_down_weight(layer_idx, weights["mlp_down"])
            if "attn_o" in weights:
                self.adapter.set_attn_o_weight(layer_idx, weights["attn_o"])

        return output

    def run_layer_sweep(
        self,
        prompt: str,
        criterion: Callable[[str], bool],
        layers: list[int] | None = None,
        component: ComponentType = ComponentType.MLP,
        task_name: str = "task",
        config: AblationConfig | None = None,
    ) -> LayerSweepResult:
        """
        Sweep through layers and test which ones are causal for a criterion.

        Args:
            prompt: Input prompt
            criterion: Function that returns True if output matches criterion
            layers: Layers to test (default: all)
            component: Component to ablate
            task_name: Name for this task
            config: Ablation configuration

        Returns:
            LayerSweepResult with per-layer results
        """
        if config is None:
            config = AblationConfig(component=component)

        if layers is None:
            layers = list(range(self.adapter.num_layers))

        # Get original output
        input_ids = self.adapter.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        original_output = self.adapter.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )
        original_criterion = criterion(original_output)

        results = []
        for layer_idx in layers:
            ablated_output = self.ablate_and_generate(prompt, [layer_idx], component, config)
            ablated_criterion = criterion(ablated_output)

            results.append(
                AblationResult(
                    layer=layer_idx,
                    component=component.value,
                    original_output=original_output,
                    ablated_output=ablated_output,
                    original_criterion=original_criterion,
                    ablated_criterion=ablated_criterion,
                    criterion_changed=original_criterion != ablated_criterion,
                    output_coherent=self._is_coherent(ablated_output),
                )
            )

        return LayerSweepResult(
            task_name=task_name,
            criterion_name=criterion.__name__ if hasattr(criterion, "__name__") else "criterion",
            results=results,
        )

    def run_multi_task_sweep(
        self,
        tasks: list[tuple[str, str, Callable[[str], bool]]],
        layers: list[int] | None = None,
        component: ComponentType = ComponentType.MLP,
        config: AblationConfig | None = None,
    ) -> dict[str, LayerSweepResult]:
        """
        Run ablation sweep for multiple tasks.

        Args:
            tasks: List of (task_name, prompt, criterion) tuples
            layers: Layers to test
            component: Component to ablate
            config: Ablation configuration

        Returns:
            Dict mapping task names to LayerSweepResults
        """
        results = {}
        for task_name, prompt, criterion in tasks:
            print(f"Running task: {task_name}")
            results[task_name] = self.run_layer_sweep(
                prompt, criterion, layers, component, task_name, config
            )
        return results

    @staticmethod
    def _is_coherent(text: str) -> bool:
        """Check if output is coherent (not gibberish)."""
        if text.count("<escape>") > 5:
            return False
        if text.count("\n") > 20 and len(text) < 100:
            return False
        if len(set(text)) < 10 and len(text) > 50:
            return False
        return True

    def print_sweep_summary(self, result: LayerSweepResult):
        """Print summary of a layer sweep."""
        print(f"\n{'=' * 60}")
        print(f"Task: {result.task_name}")
        print(f"Criterion: {result.criterion_name}")
        print(f"{'=' * 60}")

        print(f"\n{'Layer':<8} {'Original':<10} {'Ablated':<10} {'Changed':<10} {'Coherent'}")
        print("-" * 50)

        for r in result.results:
            changed_str = "YES ***" if r.criterion_changed else "no"
            coherent_str = "yes" if r.output_coherent else "NO"
            print(
                f"{r.layer:<8} {str(r.original_criterion):<10} "
                f"{str(r.ablated_criterion):<10} {changed_str:<10} {coherent_str}"
            )

        print(f"\nCausal layers: {result.causal_layers or 'None'}")

    def print_multi_task_matrix(self, results: dict[str, LayerSweepResult]):
        """Print a matrix of layer causality across tasks."""
        if not results:
            return

        first_result = next(iter(results.values()))
        layers = [r.layer for r in first_result.results]
        task_names = list(results.keys())

        print(f"\n{'=' * 80}")
        print("CAUSALITY MATRIX")
        print(f"{'=' * 80}")

        # Header
        print(f"\n{'Layer':<8}", end="")
        for name in task_names:
            short_name = name[:12]
            print(f" {short_name:>12}", end="")
        print("  | Count")
        print("-" * (10 + 14 * len(task_names)))

        # Per-layer
        for layer in layers:
            print(f"{layer:<8}", end="")
            count = 0
            for name in task_names:
                result = results[name]
                layer_result = next((r for r in result.results if r.layer == layer), None)
                if layer_result and layer_result.criterion_changed:
                    print(f" {'***CAUSAL':>12}", end="")
                    count += 1
                else:
                    print(f" {'-':>12}", end="")
            print(f"  | {count}/{len(task_names)}")

        # Find universal layers
        universal = []
        for layer in layers:
            count = sum(
                1
                for name in task_names
                if any(r.layer == layer and r.criterion_changed for r in results[name].results)
            )
            if count >= len(task_names) - 1:
                universal.append(layer)

        print(
            f"\nUniversal decision layers (affect {len(task_names) - 1}+ tasks): "
            f"{universal or 'None'}"
        )

    def save_results(self, results: dict[str, LayerSweepResult], path: str | Path):
        """Save results to JSON."""
        path = Path(path)

        data = {
            name: {
                "task_name": result.task_name,
                "criterion_name": result.criterion_name,
                "causal_layers": result.causal_layers,
                "results": [
                    {
                        "layer": r.layer,
                        "component": r.component,
                        "criterion_changed": r.criterion_changed,
                        "output_coherent": r.output_coherent,
                        "original_output": r.original_output[:200],
                        "ablated_output": r.ablated_output[:200],
                    }
                    for r in result.results
                ],
            }
            for name, result in results.items()
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {path}")
