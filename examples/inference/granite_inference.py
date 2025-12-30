#!/usr/bin/env python3
"""
IBM Granite Inference Example

Supports: Granite 3.x (dense), Granite 4.x (hybrid Mamba/Transformer)

Usage:
    uv run python examples/inference/granite_inference.py
    uv run python examples/inference/granite_inference.py --model granite-4.0-micro
    uv run python examples/inference/granite_inference.py --prompt "Explain MLX"
    uv run python examples/inference/granite_inference.py --chat
    uv run python examples/inference/granite_inference.py --list
"""

from enum import Enum

from _common import chat_loop, create_parser, list_models, print_result

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class GraniteModel(str, Enum):
    GRANITE_3_1_2B = "ibm-granite/granite-3.1-2b-instruct"
    GRANITE_3_1_8B = "ibm-granite/granite-3.1-8b-instruct"
    GRANITE_3_3_2B = "ibm-granite/granite-3.3-2b-instruct"
    GRANITE_3_3_8B = "ibm-granite/granite-3.3-8b-instruct"
    GRANITE_4_0_MICRO = "ibm-granite/granite-4.0-micro"
    GRANITE_4_0_TINY = "ibm-granite/granite-4.0-tiny-preview"


MODEL_ALIASES = {
    "granite-3.1-2b": GraniteModel.GRANITE_3_1_2B,
    "granite-3.1-8b": GraniteModel.GRANITE_3_1_8B,
    "granite-3.3-2b": GraniteModel.GRANITE_3_3_2B,
    "granite-3.3-8b": GraniteModel.GRANITE_3_3_8B,
    "granite-4.0-micro": GraniteModel.GRANITE_4_0_MICRO,
    "granite-4.0-tiny": GraniteModel.GRANITE_4_0_TINY,
}


def main():
    parser = create_parser("Granite Inference", MODEL_ALIASES, max_tokens=100)
    args = parser.parse_args()

    if args.list:
        list_models(MODEL_ALIASES, width=20)
        return

    model_id = args.model_id or MODEL_ALIASES[args.model].value
    config = UnifiedPipelineConfig(default_max_tokens=args.max_tokens, default_temperature=args.temperature)
    pipeline = UnifiedPipeline.from_pretrained(model_id, pipeline_config=config)

    if args.chat:
        chat_loop(pipeline, pipeline.family_type.value)
    else:
        print_result(pipeline, args.prompt)


if __name__ == "__main__":
    main()
