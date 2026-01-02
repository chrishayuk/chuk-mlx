#!/usr/bin/env python3
"""
Logit lens analysis with async-native, pydantic-native API.

This example demonstrates the recommended way to use the introspection module:
- Async context manager for model loading
- Pydantic models for configuration and results
- Proper enums instead of magic strings

Usage:
    uv run python examples/introspection/logit_lens.py

    # With custom model
    uv run python examples/introspection/logit_lens.py --model "mlx-community/Llama-3.2-1B-Instruct-4bit"

    # With custom prompt and token tracking
    uv run python examples/introspection/logit_lens.py --prompt "2 + 2 =" --track "4" --track " 4"
"""

import argparse
import asyncio

from chuk_lazarus.introspection import (
    AnalysisConfig,
    AnalysisResult,
    LayerStrategy,
    ModelAnalyzer,
    PositionSelection,
)


def display_result(result: AnalysisResult) -> None:
    """Display analysis result in a readable format."""
    print(f"\n{'='*60}")
    print(f"Prompt: {result.prompt}")
    print(f"{'='*60}")

    # Show tokens
    if len(result.tokens) <= 10:
        print(f"\nTokens ({len(result.tokens)}): {result.tokens}")
    else:
        print(f"\nTokens ({len(result.tokens)}): {result.tokens[:5]}...{result.tokens[-3:]}")

    print(f"Model has {result.num_layers} layers, captured: {result.captured_layers}")

    # Final predictions
    print(f"\nFinal prediction (top {len(result.final_prediction)}):")
    for pred in result.final_prediction:
        bar = "#" * int(pred.probability * 50)
        print(f"  {pred.probability:.4f} {bar} {repr(pred.token)}")

    # Layer predictions
    print("\n--- Logit Lens Analysis ---")
    print("\nTop prediction at each captured layer:")
    for layer_pred in result.layer_predictions:
        top_tok = repr(layer_pred.top_token)
        print(f"  Layer {layer_pred.layer_idx:2d}: {top_tok:20s} ({layer_pred.top_probability:.4f})")

    # Token evolutions
    if result.token_evolutions:
        print("\n--- Token Evolution ---")
        for evolution in result.token_evolutions:
            print(f"\nToken {repr(evolution.token)}:")
            for layer_idx in sorted(evolution.layer_probabilities.keys()):
                prob = evolution.layer_probabilities[layer_idx]
                rank = evolution.layer_ranks.get(layer_idx)
                rank_str = f"rank {rank}" if rank else "not in top-100"
                bar = "#" * int(prob * 100)
                print(f"  Layer {layer_idx:2d}: {prob:.4f} {bar} ({rank_str})")

            if evolution.emergence_layer is not None:
                print(f"  --> Becomes top-1 at layer {evolution.emergence_layer}")


async def run_analysis(
    model_id: str,
    prompts: list[tuple[str, list[str]]],
    layer_strategy: LayerStrategy,
    layer_step: int,
) -> None:
    """Run analysis on all prompts."""
    print("=" * 60)
    print("Logit Lens Analysis")
    print("=" * 60)

    print(f"\nLoading model: {model_id}")

    async with ModelAnalyzer.from_pretrained(model_id) as analyzer:
        # Show model info (auto-detected from model families registry)
        info = analyzer.model_info
        config = analyzer.config

        print(f"Model: {info.model_id}")
        if config is not None:
            print(f"  Family: {config.model_type}")
        print(f"  Layers: {info.num_layers}")
        print(f"  Hidden size: {info.hidden_size}")
        print(f"  Vocab size: {info.vocab_size}")
        print(f"  Tied embeddings: {info.has_tied_embeddings}")

        # Show embedding scale if model uses one (e.g., Gemma)
        if config is not None and config.embedding_scale is not None:
            print(f"  Embedding scale: {config.embedding_scale:.2f} (auto-detected)")

        # Analyze each prompt
        for prompt, track_tokens in prompts:
            # Use proper enums - no magic strings
            config = AnalysisConfig(
                layer_strategy=layer_strategy,
                layer_step=layer_step,
                position_strategy=PositionSelection.LAST,
                track_tokens=track_tokens,
            )
            result = await analyzer.analyze(prompt, config)
            display_result(result)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze TinyLlama with default prompts
  python examples/introspection/logit_lens.py

  # Use a different model
  python examples/introspection/logit_lens.py --model "mlx-community/Llama-3.2-1B-Instruct-4bit"

  # Custom prompt with token tracking
  python examples/introspection/logit_lens.py --prompt "The answer to 2+2 is" --track "4" --track " 4"

  # Capture all layers
  python examples/introspection/logit_lens.py --all-layers
        """,
    )
    parser.add_argument(
        "--model", "-m",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt", "-p",
        action="append",
        help="Prompt to analyze (can specify multiple)",
    )
    parser.add_argument(
        "--track", "-t",
        action="append",
        help="Token to track across layers (can specify multiple)",
    )
    parser.add_argument(
        "--layer-step", "-s",
        type=int,
        default=4,
        help="Capture every Nth layer (default: 4)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Capture all layers instead of evenly spaced",
    )
    args = parser.parse_args()

    # Default prompts if none specified
    if args.prompt is None:
        prompts = [
            ("The capital of France is", ["Paris", " Paris"]),
            ("def hello_world():\n    print(", ['"', "'"]),
            ("The quick brown fox jumps over the", ["lazy", " lazy"]),
        ]
    else:
        prompts = [(p, args.track or []) for p in args.prompt]

    # Use proper enum for layer strategy
    layer_strategy = LayerStrategy.ALL if args.all_layers else LayerStrategy.EVENLY_SPACED

    # Run async analysis
    asyncio.run(run_analysis(args.model, prompts, layer_strategy, args.layer_step))


if __name__ == "__main__":
    main()
