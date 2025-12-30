#!/usr/bin/env python3
"""
Llama Family Inference Example

Supports: TinyLlama, SmolLM2, Llama 2/3, Mistral

Usage:
    uv run python examples/inference/llama_inference.py
    uv run python examples/inference/llama_inference.py --model smollm2-360m
    uv run python examples/inference/llama_inference.py --prompt "Write a haiku"
    uv run python examples/inference/llama_inference.py --chat
    uv run python examples/inference/llama_inference.py --list
"""

from enum import Enum

from _common import chat_loop, create_parser, list_models, print_result

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig


class LlamaModel(str, Enum):
    TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
    SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"
    SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"


MODEL_ALIASES = {
    "tinyllama": LlamaModel.TINYLLAMA,
    "smollm2-135m": LlamaModel.SMOLLM2_135M,
    "smollm2-360m": LlamaModel.SMOLLM2_360M,
    "smollm2-1.7b": LlamaModel.SMOLLM2_1_7B,
    "llama2-7b": LlamaModel.LLAMA2_7B,
    "llama3.2-1b": LlamaModel.LLAMA3_2_1B,
    "llama3.2-3b": LlamaModel.LLAMA3_2_3B,
    "llama3.1-8b": LlamaModel.LLAMA3_1_8B,
    "mistral-7b": LlamaModel.MISTRAL_7B,
}


def main():
    parser = create_parser("Llama Family Inference", MODEL_ALIASES, max_tokens=100)
    args = parser.parse_args()

    if args.list:
        list_models(MODEL_ALIASES)
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
