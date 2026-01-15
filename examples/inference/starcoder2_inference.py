#!/usr/bin/env python3
"""
StarCoder2 Inference Example

Code generation using StarCoder2 models.

Usage:
    uv run python examples/inference/starcoder2_inference.py
    uv run python examples/inference/starcoder2_inference.py --model starcoder2-7b
    uv run python examples/inference/starcoder2_inference.py --prompt "def quicksort(arr):"
    uv run python examples/inference/starcoder2_inference.py --list
"""

from enum import Enum

from _common import create_parser, list_models

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class StarCoder2Model(str, Enum):
    STARCODER2_3B = "bigcode/starcoder2-3b"
    STARCODER2_7B = "bigcode/starcoder2-7b"
    STARCODER2_15B = "bigcode/starcoder2-15b"


MODEL_ALIASES = {
    "starcoder2-3b": StarCoder2Model.STARCODER2_3B,
    "starcoder2-7b": StarCoder2Model.STARCODER2_7B,
    "starcoder2-15b": StarCoder2Model.STARCODER2_15B,
    "3b": StarCoder2Model.STARCODER2_3B,
    "7b": StarCoder2Model.STARCODER2_7B,
    "15b": StarCoder2Model.STARCODER2_15B,
}


def main():
    parser = create_parser(
        "StarCoder2 Code Completion",
        MODEL_ALIASES,
        include_chat=False,
        temperature=0.2,
        prompt="def fibonacci(n):",
    )
    args = parser.parse_args()

    if args.list:
        list_models(MODEL_ALIASES)
        return

    model_id = args.model_id or MODEL_ALIASES[args.model].value
    config = UnifiedPipelineConfig(
        default_max_tokens=args.max_tokens, default_temperature=args.temperature
    )
    pipeline = UnifiedPipeline.from_pretrained(model_id, pipeline_config=config)

    # Use raw generate for code completion (not chat)
    result = pipeline.generate(args.prompt)
    print(f"\n{args.prompt}{result.text}")
    print(f"\n[{result.stats.tokens_per_second:.1f} tok/s]")


if __name__ == "__main__":
    main()
