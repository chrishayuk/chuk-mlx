"""
CHUK-MLX: A unified MLX-based LLM training framework.

This package provides:
- Model loading with LoRA support
- Multiple training paradigms (SFT, DPO, GRPO, PPO)
- Data generation and preprocessing
- Hybrid LLM + RNN expert architecture
- MCP tool integration

Quick Start:
    from chuk_lazarus.models import load_model
    from chuk_lazarus.training import SFTTrainer
    from chuk_lazarus.data import SFTDataset

    model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dataset = SFTDataset("./data/train.jsonl", tokenizer)
    trainer = SFTTrainer(model, tokenizer)
    trainer.train(dataset)
"""

__version__ = "0.2.0"
