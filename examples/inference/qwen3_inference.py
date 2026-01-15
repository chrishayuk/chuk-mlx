#!/usr/bin/env python3
"""
Qwen3 Inference Example

Usage:
    uv run python examples/inference/qwen3_inference.py
    uv run python examples/inference/qwen3_inference.py --model qwen3-1.7b
    uv run python examples/inference/qwen3_inference.py --prompt "Write a haiku"
    uv run python examples/inference/qwen3_inference.py --chat
    uv run python examples/inference/qwen3_inference.py --list
"""

from enum import Enum

from _common import chat_loop, create_parser, list_models, print_result

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class Qwen3Model(str, Enum):
    QWEN3_0_6B = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN3_4B = "Qwen/Qwen3-4B"
    QWEN3_8B = "Qwen/Qwen3-8B"


MODEL_ALIASES = {
    "qwen3-0.6b": Qwen3Model.QWEN3_0_6B,
    "qwen3-1.7b": Qwen3Model.QWEN3_1_7B,
    "qwen3-4b": Qwen3Model.QWEN3_4B,
    "qwen3-8b": Qwen3Model.QWEN3_8B,
}


def main():
    parser = create_parser("Qwen3 Inference", MODEL_ALIASES)
    args = parser.parse_args()

    if args.list:
        list_models(MODEL_ALIASES, width=12)
        return

    model_id = args.model_id or MODEL_ALIASES[args.model].value
    config = UnifiedPipelineConfig(
        default_max_tokens=args.max_tokens, default_temperature=args.temperature
    )
    pipeline = UnifiedPipeline.from_pretrained(model_id, pipeline_config=config)

    if args.chat:
        chat_loop(pipeline, "Qwen3")
    else:
        print_result(pipeline, args.prompt)


if __name__ == "__main__":
    main()
