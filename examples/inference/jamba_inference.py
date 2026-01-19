#!/usr/bin/env python3
"""
Jamba Inference Example

AI21 Labs' Jamba: hybrid Mamba-Transformer with MoE.

Usage:
    uv run python examples/inference/jamba_inference.py
    uv run python examples/inference/jamba_inference.py --prompt "Explain quantum computing"
    uv run python examples/inference/jamba_inference.py --chat
    uv run python examples/inference/jamba_inference.py --list

Note: Large models (52B+) require significant memory.
"""

from enum import Enum

from _common import chat_loop, create_parser, list_models, print_result

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class JambaModel(str, Enum):
    JAMBA_REASONING_3B = "ai21labs/AI21-Jamba-Reasoning-3B"
    JAMBA_V0_1 = "ai21labs/Jamba-v0.1"
    JAMBA_1_5_MINI = "ai21labs/AI21-Jamba-1.5-Mini"
    JAMBA_1_5_LARGE = "ai21labs/AI21-Jamba-1.5-Large"


MODEL_ALIASES = {
    "jamba": JambaModel.JAMBA_REASONING_3B,
    "jamba-3b": JambaModel.JAMBA_REASONING_3B,
    "jamba-v0.1": JambaModel.JAMBA_V0_1,
    "jamba-mini": JambaModel.JAMBA_1_5_MINI,
    "jamba-large": JambaModel.JAMBA_1_5_LARGE,
}


def main():
    parser = create_parser("Jamba Inference", MODEL_ALIASES)
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
        chat_loop(pipeline, "Jamba")
    else:
        print_result(pipeline, args.prompt)


if __name__ == "__main__":
    main()
