#!/usr/bin/env python3
"""
Gemma Inference Example

Usage:
    uv run python examples/inference/gemma_inference.py
    uv run python examples/inference/gemma_inference.py --model gemma3-270m
    uv run python examples/inference/gemma_inference.py --prompt "Write a haiku"
    uv run python examples/inference/gemma_inference.py --chat
    uv run python examples/inference/gemma_inference.py --list
"""

from enum import Enum

from _common import chat_loop, create_parser, list_models, print_result

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class GemmaModel(str, Enum):
    GEMMA3_270M = "mlx-community/gemma-3-270m-it-bf16"
    FUNCTIONGEMMA_270M = "mlx-community/functiongemma-270m-it-bf16"
    GEMMA3_1B = "mlx-community/gemma-3-1b-it-bf16"
    GEMMA3_4B = "mlx-community/gemma-3-4b-it-bf16"
    GEMMA3_12B = "mlx-community/gemma-3-12b-it-bf16"
    GEMMA3_27B = "mlx-community/gemma-3-27b-it-bf16"


MODEL_ALIASES = {
    "gemma3-270m": GemmaModel.GEMMA3_270M,
    "functiongemma": GemmaModel.FUNCTIONGEMMA_270M,
    "gemma3-1b": GemmaModel.GEMMA3_1B,
    "gemma3-4b": GemmaModel.GEMMA3_4B,
    "gemma3-12b": GemmaModel.GEMMA3_12B,
    "gemma3-27b": GemmaModel.GEMMA3_27B,
}


def main():
    parser = create_parser("Gemma Inference", MODEL_ALIASES, default_model="gemma3-1b")
    args = parser.parse_args()

    if args.list:
        list_models(MODEL_ALIASES)
        return

    model_id = args.model_id or MODEL_ALIASES[args.model].value
    config = UnifiedPipelineConfig(
        default_max_tokens=args.max_tokens, default_temperature=args.temperature
    )
    pipeline = UnifiedPipeline.from_pretrained(model_id, pipeline_config=config)

    if args.chat:
        chat_loop(pipeline, "Gemma")
    else:
        print_result(pipeline, args.prompt)


if __name__ == "__main__":
    main()
