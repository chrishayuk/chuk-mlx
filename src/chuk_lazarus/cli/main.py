"""
Main CLI entry point for chuk-lazarus.

Usage:
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data ./data/train.jsonl
    lazarus train dpo --model ./checkpoints/sft/final --data ./data/preferences.jsonl
    lazarus train grpo --model ./checkpoints/sft/final --reward-script ./reward.py
    lazarus generate --type math --output ./data/lazarus_math
    lazarus infer --model ./checkpoints/dpo/final --prompt "Calculate 2+2"

Data Commands:
    lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl
    lazarus data lengths stats -c lengths.jsonl
    lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/
    lazarus data batchplan info -p batch_plan/
    lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4  # sharded view
    lazarus data batchplan verify -p batch_plan/ -l lengths.jsonl
    lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/  # distributed sharding

Batching Analysis Commands:
    lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512
    lazarus data batching histogram -c lengths.jsonl --bins 20
    lazarus data batching suggest -c lengths.jsonl --goal waste --num-buckets 4
    lazarus data batch generate -p batch_plan/ -d train.jsonl -t gpt2 -o batches/

Tokenizer Commands:
    lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
    lazarus tokenizer decode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ids "1,2,3"
    lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"
    lazarus tokenizer compare -t1 model1 -t2 model2 --text "Test" --verbose
    lazarus tokenizer doctor -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer fingerprint -t model --save fingerprint.json
    lazarus tokenizer fingerprint -t model --verify fingerprint.json --strict
    lazarus tokenizer benchmark -t TinyLlama/TinyLlama-1.1B-Chat-v1.0
    lazarus tokenizer benchmark -t model --samples 5000 --compare
    lazarus tokenizer analyze coverage -t model --file corpus.txt
    lazarus tokenizer analyze entropy -t model --file corpus.txt
    lazarus tokenizer analyze fit-score -t model --file corpus.txt
    lazarus tokenizer analyze efficiency -t model --file corpus.txt
    lazarus tokenizer analyze vocab-suggest -t model --file corpus.txt
    lazarus tokenizer curriculum length-buckets -t model --file corpus.txt
    lazarus tokenizer curriculum reasoning-density -t model --file corpus.txt
    lazarus tokenizer training throughput -t model --file corpus.txt
    lazarus tokenizer training pack -t model --file corpus.txt --max-length 512
    lazarus tokenizer regression run -t model --tests tests.yaml
    lazarus tokenizer research soft-tokens -n 10 -d 768 --prefix task
    lazarus tokenizer research analyze-embeddings -f embeddings.json --cluster
    lazarus tokenizer research morph -f embeddings.json -s 0 -t 1 --method spherical
    lazarus tokenizer instrument histogram -t model --file corpus.txt
    lazarus tokenizer instrument oov -t model --file corpus.txt
    lazarus tokenizer instrument waste -t model --file corpus.txt --max-length 512
    lazarus tokenizer instrument vocab-diff -t1 model1 -t2 model2 --file corpus.txt

Gym Commands (Online Learning):
    lazarus gym run -t gpt2 --mock --num-episodes 10  # Test with mock stream
    lazarus gym run -t gpt2 --host localhost --port 8023  # Connect to puzzle arcade
    lazarus gym run -t gpt2 --mock --output buffer.json  # Save samples to buffer
    lazarus gym info  # Display gym stream configuration

Benchmark Commands:
    lazarus bench  # Run benchmark with synthetic data
    lazarus bench -d train.jsonl -t gpt2  # Benchmark with real dataset
    lazarus bench --bucket-edges 128,256,512 --token-budget 4096  # Custom config

Introspection Commands (Model Analysis):
    lazarus introspect analyze -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p "The capital of France is"
    lazarus introspect analyze -m model -p "Hello" --track "world,there" --layer-strategy all
    lazarus introspect compare -m1 model1 -m2 model2 -p "The answer is" --track "42"
    lazarus introspect hooks -m model -p "Test" --layers 0,4,8 --capture-attention

Ablation Study Commands (Causal Circuit Discovery):
    lazarus introspect ablate -m model -p "prompt" -c function_call --component mlp
    lazarus introspect ablate -m model -p "prompt" -c sorry --layers 5,8,10,11,12
    lazarus introspect weight-diff -b base_model -f finetuned_model -o diff.json
    lazarus introspect activation-diff -b base -f finetuned -p "prompt1,prompt2"

Activation Steering Commands:
    lazarus introspect steer -m model --extract --positive "good prompt" --negative "bad prompt" -o direction.npz
    lazarus introspect steer -m model -d direction.npz -p "test prompt" -c 1.0
    lazarus introspect steer -m model -d direction.npz -p "test prompt" --compare "-2,-1,0,1,2"

Circuit Analysis Commands:
    lazarus introspect arithmetic -m model --quick  # Test arithmetic emergence across layers
    lazarus introspect uncertainty -m model -p "100 - 37 = |100 - 37 ="  # Predict confidence

Training with Batching:
    lazarus train sft --model model --data train.jsonl --batchplan batch_plan/
    lazarus train sft --model model --data train.jsonl --bucket-edges 128,256,512 --token-budget 4096
    lazarus train sft --model model --data train.jsonl --pack --pack-max-len 2048
    lazarus train sft --model model --data train.jsonl --online --gym-host localhost --gym-port 8023
"""

import argparse
import asyncio
import logging
import sys

# Import command handlers from modules
from .commands.data import (
    data_batch_generate,
    data_batching_analyze,
    data_batching_histogram,
    data_batching_suggest,
    data_batchplan_build,
    data_batchplan_info,
    data_batchplan_shard,
    data_batchplan_verify,
    data_lengths_build,
    data_lengths_stats,
)
from .commands.experiment import (
    experiment_info,
    experiment_list,
    experiment_run,
    experiment_status,
)
from .commands.gym import bench_pipeline, gym_info, gym_run
from .commands.infer import run_inference
from .commands.introspect import (
    introspect_ablate,
    introspect_activation_cluster,
    introspect_activation_diff,
    introspect_analyze,
    introspect_arithmetic,
    introspect_circuit_capture,
    introspect_circuit_compare,
    introspect_circuit_decode,
    introspect_circuit_export,
    introspect_circuit_invoke,
    introspect_circuit_test,
    introspect_circuit_view,
    introspect_classifier,
    introspect_commutativity,
    introspect_compare,
    introspect_directions,
    introspect_early_layers,
    introspect_embedding,
    introspect_format_sensitivity,
    introspect_generate,
    introspect_hooks,
    introspect_layer,
    introspect_logit_lens,
    introspect_memory,
    introspect_memory_inject,
    introspect_metacognitive,
    introspect_moe_expert,
    introspect_neurons,
    introspect_operand_directions,
    introspect_patch,
    introspect_probe,
    introspect_steer,
    introspect_uncertainty,
    introspect_virtual_expert,
    introspect_weight_diff,
)
from .commands.tokenizer import (
    analyze_coverage,
    analyze_diff,
    analyze_efficiency,
    analyze_entropy,
    analyze_fit_score,
    analyze_vocab_suggest,
    curriculum_length_buckets,
    curriculum_reasoning_density,
    instrument_histogram,
    instrument_oov,
    instrument_vocab_diff,
    instrument_waste,
    regression_run,
    research_analyze_embeddings,
    research_morph,
    research_soft_tokens,
    runtime_registry,
    tokenizer_benchmark,
    tokenizer_compare,
    tokenizer_decode,
    tokenizer_doctor,
    tokenizer_encode,
    tokenizer_fingerprint,
    tokenizer_vocab,
    training_pack,
    training_throughput,
)
from .commands.train import (
    generate_data_cmd,
    train_dpo_cmd,
    train_grpo_cmd,
    train_sft_cmd,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def app():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="chuk-lazarus: MLX-based LLM training framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train SFT
    lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl

    # Train DPO
    lazarus train dpo --model ./checkpoints/sft/final --data preferences.jsonl

    # Generate training data
    lazarus generate --type math --output ./data/lazarus

    # Run inference
    lazarus infer --model ./checkpoints/dpo/final --prompt "What is 2+2?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_subparsers = train_parser.add_subparsers(dest="train_type", help="Training type")

    # SFT training
    sft_parser = train_subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", required=True, help="Model name or path")
    sft_parser.add_argument("--data", required=True, help="Training data path (JSONL)")
    sft_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    sft_parser.add_argument("--output", default="./checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--max-steps", type=int, help="Max training steps (overrides epochs)")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    sft_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    sft_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    sft_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    sft_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj). "
        "Options: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    sft_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3'). Frozen layers are not trained.",
    )
    sft_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    sft_parser.add_argument("--mask-prompt", action="store_true", help="Mask prompt in loss")
    sft_parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    # Batching options
    sft_parser.add_argument("--batchplan", help="Use pre-computed batch plan directory")
    sft_parser.add_argument(
        "--bucket-edges",
        help="Bucket edges for length-based batching (e.g., 128,256,512)",
    )
    sft_parser.add_argument(
        "--token-budget",
        type=int,
        help="Token budget for dynamic batching (replaces --batch-size)",
    )
    sft_parser.add_argument("--pack", action="store_true", help="Enable sequence packing")
    sft_parser.add_argument("--pack-max-len", type=int, help="Max length for packed sequences")
    sft_parser.add_argument(
        "--pack-mode",
        choices=["first_fit", "best_fit", "greedy"],
        default="first_fit",
        help="Packing algorithm",
    )
    # Online training options
    sft_parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online training with gym stream",
    )
    sft_parser.add_argument(
        "--gym-host",
        default="localhost",
        help="Gym server host for online training",
    )
    sft_parser.add_argument(
        "--gym-port",
        type=int,
        default=8023,
        help="Gym server port for online training",
    )
    sft_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size for online training",
    )
    sft_parser.set_defaults(func=lambda args: asyncio.run(train_sft_cmd(args)))

    # DPO training
    dpo_parser = train_subparsers.add_parser("dpo", help="Direct Preference Optimization")
    dpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    dpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    dpo_parser.add_argument("--data", required=True, help="Preference data path (JSONL)")
    dpo_parser.add_argument("--eval-data", help="Evaluation data path (JSONL)")
    dpo_parser.add_argument("--output", default="./checkpoints/dpo", help="Output directory")
    dpo_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    dpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    dpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    dpo_parser.set_defaults(func=lambda args: asyncio.run(train_dpo_cmd(args)))

    # GRPO training
    grpo_parser = train_subparsers.add_parser(
        "grpo", help="Group Relative Policy Optimization (RL with verifiable rewards)"
    )
    grpo_parser.add_argument("--model", required=True, help="Policy model name or path")
    grpo_parser.add_argument("--ref-model", help="Reference model (default: same as --model)")
    grpo_parser.add_argument("--output", default="./checkpoints/grpo", help="Output directory")
    grpo_parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    grpo_parser.add_argument(
        "--prompts-per-iteration", type=int, default=16, help="Prompts per iteration"
    )
    grpo_parser.add_argument("--group-size", type=int, default=4, help="Responses per prompt")
    grpo_parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    grpo_parser.add_argument("--kl-coef", type=float, default=0.1, help="KL penalty coefficient")
    grpo_parser.add_argument(
        "--max-response-length", type=int, default=256, help="Max response tokens"
    )
    grpo_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    grpo_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    grpo_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    grpo_parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules (default: q_proj,v_proj)",
    )
    grpo_parser.add_argument(
        "--freeze-layers",
        help="Layers to freeze (e.g., '0-12' or '0,1,2,3')",
    )
    grpo_parser.add_argument(
        "--reward-script",
        required=True,
        help="Python script defining reward_fn(prompt, response) -> float and get_prompts() -> list[str]",
    )
    grpo_parser.add_argument(
        "--config",
        help="YAML config file (overrides other arguments)",
    )
    grpo_parser.set_defaults(func=lambda args: asyncio.run(train_grpo_cmd(args)))

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--type", required=True, choices=["math"], help="Data type")
    gen_parser.add_argument("--output", default="./data/generated", help="Output directory")
    gen_parser.add_argument("--sft-samples", type=int, default=10000, help="SFT samples")
    gen_parser.add_argument("--dpo-samples", type=int, default=5000, help="DPO samples")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.set_defaults(func=lambda args: asyncio.run(generate_data_cmd(args)))

    # Infer subcommand
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", required=True, help="Model name or path")
    infer_parser.add_argument("--adapter", help="LoRA adapter path")
    infer_parser.add_argument("--prompt", help="Single prompt")
    infer_parser.add_argument("--prompt-file", help="File with prompts (one per line)")
    infer_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    infer_parser.add_argument("--chat", action="store_true", help="Use chat template (for chat models)")
    infer_parser.add_argument("--system", help="System prompt (only used with --chat)")
    infer_parser.set_defaults(func=lambda args: asyncio.run(run_inference(args)))

    # Tokenizer subcommand
    tok_parser = subparsers.add_parser("tokenizer", help="Tokenizer utilities")
    tok_subparsers = tok_parser.add_subparsers(dest="tok_command", help="Tokenizer commands")

    # Encode command
    encode_parser = tok_subparsers.add_parser("encode", help="Encode text to tokens")
    encode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    encode_parser.add_argument("--text", help="Text to encode")
    encode_parser.add_argument("--file", "-f", help="File to encode")
    encode_parser.add_argument("--special-tokens", action="store_true", help="Add special tokens")
    encode_parser.set_defaults(func=tokenizer_encode)

    # Decode command
    decode_parser = tok_subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    decode_parser.add_argument("--ids", required=True, help="Token IDs (comma or space separated)")
    decode_parser.set_defaults(func=tokenizer_decode)

    # Vocab command
    vocab_parser = tok_subparsers.add_parser("vocab", help="Display vocabulary info")
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--show-all", action="store_true", help="Show full vocabulary")
    vocab_parser.add_argument("--search", "-s", help="Search for tokens containing string")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Max results for search")
    vocab_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for display")
    vocab_parser.add_argument("--pause", action="store_true", help="Pause between chunks")
    vocab_parser.set_defaults(func=tokenizer_vocab)

    # Compare command
    compare_parser = tok_subparsers.add_parser("compare", help="Compare two tokenizers")
    compare_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    compare_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    compare_parser.add_argument("--text", required=True, help="Text to compare")
    compare_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full tokenization"
    )
    compare_parser.set_defaults(func=tokenizer_compare)

    # Doctor command
    doctor_parser = tok_subparsers.add_parser("doctor", help="Run tokenizer health check")
    doctor_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    doctor_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix detected issues (patches chat template if missing)",
    )
    doctor_parser.add_argument(
        "--format",
        choices=["chatml", "llama", "phi", "gemma", "zephyr", "vicuna", "alpaca"],
        help="Specify chat template format when using --fix (auto-detects if not set)",
    )
    doctor_parser.add_argument(
        "--output",
        "-o",
        help="Save patched tokenizer to directory (requires --fix)",
    )
    doctor_parser.set_defaults(func=tokenizer_doctor)

    # Fingerprint command
    fingerprint_parser = tok_subparsers.add_parser(
        "fingerprint", help="Generate or verify tokenizer fingerprint"
    )
    fingerprint_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    fingerprint_parser.add_argument("--save", "-s", help="Save fingerprint to JSON file")
    fingerprint_parser.add_argument("--verify", help="Verify against fingerprint (file or string)")
    fingerprint_parser.add_argument(
        "--strict", action="store_true", help="Strict verification (merges must match)"
    )
    fingerprint_parser.set_defaults(func=tokenizer_fingerprint)

    # Benchmark command
    bench_parser = tok_subparsers.add_parser("benchmark", help="Benchmark tokenizer throughput")
    bench_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    bench_parser.add_argument("--file", "-f", help="Corpus file (one text per line)")
    bench_parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)",
    )
    bench_parser.add_argument(
        "--avg-length",
        type=int,
        default=100,
        help="Avg words per sample for synthetic corpus",
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for synthetic corpus"
    )
    bench_parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of parallel workers"
    )
    bench_parser.add_argument("--warmup", type=int, default=10, help="Warmup samples before timing")
    bench_parser.add_argument(
        "--special-tokens",
        action="store_true",
        help="Add special tokens during encoding",
    )
    bench_parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare HuggingFace vs Fast (MLX) backend",
    )
    bench_parser.set_defaults(func=tokenizer_benchmark)

    # === Analyze subcommands ===
    analyze_parser = tok_subparsers.add_parser("analyze", help="Token analysis tools")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", help="Analysis type")

    # Coverage analysis
    cov_parser = analyze_subparsers.add_parser("coverage", help="Analyze token coverage")
    cov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    cov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    cov_parser.add_argument("--fragments", action="store_true", help="Include fragment analysis")
    cov_parser.set_defaults(func=analyze_coverage)

    # Entropy analysis
    ent_parser = analyze_subparsers.add_parser("entropy", help="Analyze token entropy")
    ent_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    ent_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    ent_parser.add_argument("--top-n", type=int, default=100, help="Top N tokens to show")
    ent_parser.set_defaults(func=analyze_entropy)

    # Fit score
    fit_parser = analyze_subparsers.add_parser("fit-score", help="Calculate tokenizer-dataset fit")
    fit_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    fit_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    fit_parser.set_defaults(func=analyze_fit_score)

    # Diff analysis
    diff_parser = analyze_subparsers.add_parser("diff", help="Compare tokenizers on corpus")
    diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    diff_parser.set_defaults(func=analyze_diff)

    # Efficiency analysis
    eff_parser = analyze_subparsers.add_parser(
        "efficiency", help="Analyze token efficiency metrics"
    )
    eff_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    eff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    eff_parser.set_defaults(func=analyze_efficiency)

    # Vocab suggestion
    vocab_parser = analyze_subparsers.add_parser(
        "vocab-suggest", help="Suggest vocabulary additions"
    )
    vocab_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    vocab_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_parser.add_argument("--min-freq", type=int, default=5, help="Minimum frequency")
    vocab_parser.add_argument("--min-frag", type=int, default=3, help="Minimum fragmentation")
    vocab_parser.add_argument("--limit", type=int, default=50, help="Maximum candidates")
    vocab_parser.add_argument("--show", type=int, default=20, help="Number to display")
    vocab_parser.set_defaults(func=analyze_vocab_suggest)

    # === Curriculum subcommands ===
    curr_parser = tok_subparsers.add_parser("curriculum", help="Curriculum learning tools")
    curr_subparsers = curr_parser.add_subparsers(dest="curriculum_command", help="Curriculum type")

    # Length buckets
    len_parser = curr_subparsers.add_parser("length-buckets", help="Create length-based curriculum")
    len_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    len_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    len_parser.add_argument("--num-buckets", type=int, default=5, help="Number of buckets")
    len_parser.add_argument("--schedule", action="store_true", help="Show curriculum schedule")
    len_parser.set_defaults(func=curriculum_length_buckets)

    # Reasoning density
    reason_parser = curr_subparsers.add_parser(
        "reasoning-density", help="Score by reasoning density"
    )
    reason_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    reason_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    reason_parser.add_argument(
        "--descending", action="store_true", help="Sort descending (hardest first)"
    )
    reason_parser.set_defaults(func=curriculum_reasoning_density)

    # === Training subcommands ===
    train_tok_parser = tok_subparsers.add_parser("training", help="Training utilities")
    train_tok_subparsers = train_tok_parser.add_subparsers(
        dest="training_command", help="Training tool"
    )

    # Throughput profiling
    thru_parser = train_tok_subparsers.add_parser(
        "throughput", help="Profile tokenization throughput"
    )
    thru_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    thru_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    thru_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    thru_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    thru_parser.set_defaults(func=training_throughput)

    # Sequence packing
    pack_parser = train_tok_subparsers.add_parser("pack", help="Pack sequences for training")
    pack_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    pack_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    pack_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    pack_parser.add_argument("--output", "-o", help="Output file (JSONL)")
    pack_parser.set_defaults(func=training_pack)

    # === Regression subcommands ===
    reg_parser = tok_subparsers.add_parser("regression", help="Token regression testing")
    reg_subparsers = reg_parser.add_subparsers(dest="regression_command", help="Regression tool")

    # Run tests
    run_parser = reg_subparsers.add_parser("run", help="Run regression tests")
    run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    run_parser.add_argument("--tests", required=True, help="Test suite YAML file")
    run_parser.set_defaults(func=regression_run)

    # === Runtime subcommands ===
    runtime_parser = tok_subparsers.add_parser("runtime", help="Runtime token utilities")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", help="Runtime tool")

    # Registry
    registry_parser = runtime_subparsers.add_parser(
        "registry", help="Display special token registry"
    )
    registry_parser.add_argument("--tokenizer", "-t", help="Tokenizer name or path")
    registry_parser.add_argument("--standard", action="store_true", help="Show standard registry")
    registry_parser.set_defaults(func=runtime_registry)

    # === Research subcommands ===
    research_parser = tok_subparsers.add_parser("research", help="Research playground tools")
    research_subparsers = research_parser.add_subparsers(
        dest="research_command", help="Research tool"
    )

    # Soft tokens
    soft_parser = research_subparsers.add_parser(
        "soft-tokens", help="Create soft token bank for prompt tuning"
    )
    soft_parser.add_argument(
        "--num-tokens", "-n", type=int, default=10, help="Number of soft tokens"
    )
    soft_parser.add_argument(
        "--embedding-dim", "-d", type=int, default=768, help="Embedding dimension"
    )
    soft_parser.add_argument("--prefix", "-p", default="prompt", help="Token name prefix")
    soft_parser.add_argument(
        "--init-method",
        choices=["random_normal", "random_uniform", "zeros"],
        default="random_normal",
        help="Initialization method",
    )
    soft_parser.add_argument("--init-std", type=float, default=0.02, help="Std dev for random init")
    soft_parser.add_argument("--output", "-o", help="Save bank to JSON file")
    soft_parser.set_defaults(func=research_soft_tokens)

    # Analyze embeddings
    emb_parser = research_subparsers.add_parser(
        "analyze-embeddings", help="Analyze embedding space"
    )
    emb_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    emb_parser.add_argument("--num-clusters", "-k", type=int, default=10, help="Number of clusters")
    emb_parser.add_argument("--cluster", action="store_true", help="Show cluster analysis")
    emb_parser.add_argument("--project", action="store_true", help="Show 2D projection stats")
    emb_parser.set_defaults(func=research_analyze_embeddings)

    # Morph
    morph_parser = research_subparsers.add_parser("morph", help="Morph between token embeddings")
    morph_parser.add_argument("--file", "-f", required=True, help="JSON file with embeddings")
    morph_parser.add_argument("--source", "-s", type=int, required=True, help="Source token index")
    morph_parser.add_argument("--target", "-t", type=int, required=True, help="Target token index")
    morph_parser.add_argument(
        "--method",
        "-m",
        choices=["linear", "spherical", "bezier", "cubic"],
        default="linear",
        help="Morph method",
    )
    morph_parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    morph_parser.add_argument("--normalize", action="store_true", help="Normalize output")
    morph_parser.add_argument("--output", "-o", help="Save trajectory to JSON")
    morph_parser.set_defaults(func=research_morph)

    # === Instrumentation subcommands ===
    instrument_parser = tok_subparsers.add_parser("instrument", help="Tokenizer instrumentation")
    instrument_subparsers = instrument_parser.add_subparsers(
        dest="instrument_command", help="Instrumentation tool"
    )

    # Histogram
    hist_parser = instrument_subparsers.add_parser(
        "histogram", help="Display token length histogram"
    )
    hist_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    hist_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    hist_parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    hist_parser.add_argument("--width", type=int, default=50, help="Chart width")
    hist_parser.add_argument("--quick", action="store_true", help="Quick stats only")
    hist_parser.set_defaults(func=instrument_histogram)

    # OOV analysis
    oov_parser = instrument_subparsers.add_parser("oov", help="Analyze OOV and rare tokens")
    oov_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    oov_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    oov_parser.add_argument("--vocab-size", type=int, default=50000, help="Expected vocab size")
    oov_parser.add_argument("--show-rare", action="store_true", help="Show rare tokens")
    oov_parser.add_argument("--max-freq", type=int, default=5, help="Max frequency for rare")
    oov_parser.add_argument("--top-k", type=int, default=20, help="Number of rare tokens to show")
    oov_parser.set_defaults(func=instrument_oov)

    # Waste analysis
    waste_parser = instrument_subparsers.add_parser(
        "waste", help="Analyze padding and truncation waste"
    )
    waste_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    waste_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    waste_parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    waste_parser.set_defaults(func=instrument_waste)

    # Vocab diff
    vocab_diff_parser = instrument_subparsers.add_parser(
        "vocab-diff", help="Compare two tokenizers on a corpus"
    )
    vocab_diff_parser.add_argument("--tokenizer1", "-t1", required=True, help="First tokenizer")
    vocab_diff_parser.add_argument("--tokenizer2", "-t2", required=True, help="Second tokenizer")
    vocab_diff_parser.add_argument("--file", "-f", help="Input file (one text per line)")
    vocab_diff_parser.add_argument("--examples", type=int, default=5, help="Max examples to show")
    vocab_diff_parser.add_argument(
        "--cost", action="store_true", help="Show retokenization cost estimate"
    )
    vocab_diff_parser.set_defaults(func=instrument_vocab_diff)

    # === Data subcommand ===
    data_parser = subparsers.add_parser("data", help="Data processing utilities")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")

    # === Lengths subcommands ===
    lengths_parser = data_subparsers.add_parser("lengths", help="Length cache utilities")
    lengths_subparsers = lengths_parser.add_subparsers(
        dest="lengths_command", help="Lengths commands"
    )

    # Build length cache
    lengths_build_parser = lengths_subparsers.add_parser(
        "build", help="Build length cache from dataset"
    )
    lengths_build_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    lengths_build_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    lengths_build_parser.add_argument(
        "--output", "-o", required=True, help="Output cache file path"
    )
    lengths_build_parser.set_defaults(func=data_lengths_build)

    # Length cache stats
    lengths_stats_parser = lengths_subparsers.add_parser(
        "stats", help="Show length cache statistics"
    )
    lengths_stats_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    lengths_stats_parser.set_defaults(func=data_lengths_stats)

    # === BatchPlan subcommands ===
    batchplan_parser = data_subparsers.add_parser("batchplan", help="Batch plan utilities")
    batchplan_subparsers = batchplan_parser.add_subparsers(
        dest="batchplan_command", help="BatchPlan commands"
    )

    # Build batch plan
    batchplan_build_parser = batchplan_subparsers.add_parser(
        "build", help="Build batch plan from length cache"
    )
    batchplan_build_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_build_parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="Number of epochs (default: 1)"
    )
    batchplan_build_parser.add_argument(
        "--token-budget",
        "-b",
        type=int,
        default=4096,
        help="Token budget per batch (default: 4096)",
    )
    batchplan_build_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges (comma-separated, default: 128,256,512)",
    )
    batchplan_build_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batchplan_build_parser.add_argument(
        "--predictable",
        "-p",
        action="store_true",
        help="Use predictable mode (deterministic batching)",
    )
    batchplan_build_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for predictable mode (default: 42)",
    )
    batchplan_build_parser.add_argument("--dataset-hash", help="Dataset hash for fingerprinting")
    batchplan_build_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for batch plan"
    )
    batchplan_build_parser.set_defaults(func=data_batchplan_build)

    # Batch plan info
    batchplan_info_parser = batchplan_subparsers.add_parser(
        "info", help="Show batch plan information"
    )
    batchplan_info_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_info_parser.add_argument(
        "--show-batches",
        "-n",
        type=int,
        default=0,
        help="Number of sample batches to show",
    )
    batchplan_info_parser.add_argument(
        "--rank",
        "-r",
        type=int,
        default=None,
        help="Worker rank for sharded view (0-indexed)",
    )
    batchplan_info_parser.add_argument(
        "--world-size", "-w", type=int, default=None, help="Total number of workers"
    )
    batchplan_info_parser.set_defaults(func=data_batchplan_info)

    # Batch plan verify
    batchplan_verify_parser = batchplan_subparsers.add_parser(
        "verify", help="Verify batch plan reproducibility"
    )
    batchplan_verify_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batchplan_verify_parser.add_argument("--lengths", "-l", required=True, help="Length cache file")
    batchplan_verify_parser.set_defaults(func=data_batchplan_verify)

    # Batch plan shard
    batchplan_shard_parser = batchplan_subparsers.add_parser(
        "shard", help="Create sharded batch plans for distributed training"
    )
    batchplan_shard_parser.add_argument(
        "--plan", "-p", required=True, help="Source batch plan directory"
    )
    batchplan_shard_parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        required=True,
        help="Number of distributed workers",
    )
    batchplan_shard_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for sharded plans"
    )
    batchplan_shard_parser.set_defaults(func=data_batchplan_shard)

    # === Batching analysis subcommands ===
    batching_parser = data_subparsers.add_parser("batching", help="Batching analysis utilities")
    batching_subparsers = batching_parser.add_subparsers(
        dest="batching_command", help="Batching commands"
    )

    # Analyze batching efficiency
    batching_analyze_parser = batching_subparsers.add_parser(
        "analyze", help="Analyze batching efficiency"
    )
    batching_analyze_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_analyze_parser.add_argument(
        "--bucket-edges",
        default="128,256,512",
        help="Bucket edges to analyze (comma-separated, default: 128,256,512)",
    )
    batching_analyze_parser.add_argument(
        "--overflow-max",
        type=int,
        default=2048,
        help="Max length for overflow bucket (default: 2048)",
    )
    batching_analyze_parser.add_argument("--output", "-o", help="Save JSON report to file")
    batching_analyze_parser.set_defaults(func=data_batching_analyze)

    # Length histogram
    batching_histogram_parser = batching_subparsers.add_parser(
        "histogram", help="Display length histogram"
    )
    batching_histogram_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_histogram_parser.add_argument(
        "--bins", type=int, default=15, help="Number of histogram bins (default: 15)"
    )
    batching_histogram_parser.add_argument(
        "--width", type=int, default=50, help="Chart width (default: 50)"
    )
    batching_histogram_parser.set_defaults(func=data_batching_histogram)

    # Suggest bucket edges
    batching_suggest_parser = batching_subparsers.add_parser(
        "suggest", help="Suggest optimal bucket edges"
    )
    batching_suggest_parser.add_argument("--cache", "-c", required=True, help="Length cache file")
    batching_suggest_parser.add_argument(
        "--num-buckets",
        "-n",
        type=int,
        default=4,
        help="Number of buckets (default: 4)",
    )
    batching_suggest_parser.add_argument(
        "--goal",
        "-g",
        choices=["waste", "balance", "memory"],
        default="waste",
        help="Optimization goal: waste (minimize padding), balance (even bucket sizes), memory (power-of-2 edges)",
    )
    batching_suggest_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    batching_suggest_parser.set_defaults(func=data_batching_suggest)

    # === Batch generation subcommands ===
    batch_parser = data_subparsers.add_parser("batch", help="Batch file generation")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", help="Batch commands")

    # Generate NPZ batch files
    batch_generate_parser = batch_subparsers.add_parser(
        "generate", help="Generate NPZ batch files from BatchPlan"
    )
    batch_generate_parser.add_argument("--plan", "-p", required=True, help="Batch plan directory")
    batch_generate_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset file (JSONL or JSON)"
    )
    batch_generate_parser.add_argument(
        "--tokenizer", "-t", required=True, help="Tokenizer name or path"
    )
    batch_generate_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for NPZ files"
    )
    batch_generate_parser.set_defaults(func=data_batch_generate)

    # === Gym subcommand ===
    gym_parser = subparsers.add_parser("gym", help="Gym streaming utilities")
    gym_subparsers = gym_parser.add_subparsers(dest="gym_command", help="Gym commands")

    # Gym run command
    gym_run_parser = gym_subparsers.add_parser(
        "run", help="Run gym episode streaming and collect samples"
    )
    gym_run_parser.add_argument("--tokenizer", "-t", required=True, help="Tokenizer name or path")
    gym_run_parser.add_argument("--host", default="localhost", help="Gym server host")
    gym_run_parser.add_argument("--port", type=int, default=8023, help="Gym server port")
    gym_run_parser.add_argument(
        "--transport",
        choices=["telnet", "websocket", "http"],
        default="telnet",
        help="Transport protocol",
    )
    gym_run_parser.add_argument(
        "--output-mode",
        choices=["json", "text", "binary"],
        default="json",
        help="Output format from gym",
    )
    gym_run_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size",
    )
    gym_run_parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout")
    gym_run_parser.add_argument("--retries", type=int, default=3, help="Max connection retries")
    gym_run_parser.add_argument(
        "--difficulty-min",
        type=float,
        default=0.0,
        help="Minimum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--difficulty-max",
        type=float,
        default=1.0,
        help="Maximum puzzle difficulty",
    )
    gym_run_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to collect (infinite if not set)",
    )
    gym_run_parser.add_argument("--output", "-o", help="Output file for buffer (JSON)")
    gym_run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Mock mode for testing
    gym_run_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock gym stream for testing",
    )
    gym_run_parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes (mock mode)",
    )
    gym_run_parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=5,
        help="Steps per episode (mock mode)",
    )
    gym_run_parser.add_argument(
        "--success-rate",
        type=float,
        default=0.7,
        help="Success rate (mock mode)",
    )
    gym_run_parser.set_defaults(func=gym_run)

    # Gym info command
    gym_info_parser = gym_subparsers.add_parser(
        "info", help="Display gym stream configuration info"
    )
    gym_info_parser.set_defaults(func=gym_info)

    # =========================================================================
    # Experiment command - run experiments from experiments/ directory
    # =========================================================================
    exp_parser = subparsers.add_parser(
        "experiment",
        help="Discover and run experiments",
        description="Run experiments from the experiments/ directory using the experiments framework.",
    )
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command", help="Experiment commands")

    # Experiment list command
    exp_list_parser = exp_subparsers.add_parser("list", help="List all discovered experiments")
    exp_list_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory (default: auto-detect)",
    )
    exp_list_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_list_parser.set_defaults(
        func=lambda args: experiment_list(
            experiments_dir=args.dir,
            json_output=args.json,
        )
    )

    # Experiment info command
    exp_info_parser = exp_subparsers.add_parser("info", help="Show detailed experiment information")
    exp_info_parser.add_argument("name", help="Experiment name")
    exp_info_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_info_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_info_parser.set_defaults(
        func=lambda args: experiment_info(
            name=args.name,
            experiments_dir=args.dir,
            json_output=args.json,
        )
    )

    # Experiment run command
    exp_run_parser = exp_subparsers.add_parser("run", help="Run an experiment")
    exp_run_parser.add_argument("name", help="Experiment name")
    exp_run_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_run_parser.add_argument(
        "--config",
        "-c",
        help="Path to custom config YAML file",
    )
    exp_run_parser.add_argument(
        "--param",
        "-p",
        action="append",
        dest="params",
        help="Parameter override (key=value), can specify multiple",
    )
    exp_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without running",
    )
    exp_run_parser.set_defaults(
        func=lambda args: experiment_run(
            name=args.name,
            experiments_dir=args.dir,
            config_file=args.config,
            params=args.params,
            dry_run=args.dry_run,
        )
    )

    # Experiment status command
    exp_status_parser = exp_subparsers.add_parser(
        "status", help="Show experiment status and results"
    )
    exp_status_parser.add_argument("name", help="Experiment name")
    exp_status_parser.add_argument(
        "--dir",
        "-d",
        help="Path to experiments directory",
    )
    exp_status_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        dest="show_all",
        help="Show all runs, not just latest",
    )
    exp_status_parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    exp_status_parser.set_defaults(
        func=lambda args: experiment_status(
            name=args.name,
            experiments_dir=args.dir,
            show_all=args.show_all,
            json_output=args.json,
        )
    )

    # =========================================================================
    # Bench command - comprehensive pipeline benchmark
    # =========================================================================
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark the batching pipeline",
        description="Run comprehensive benchmarks on tokenization, batching, packing, and efficiency.",
    )
    bench_parser.add_argument(
        "-d",
        "--dataset",
        help="JSONL dataset file (optional - uses synthetic data if not provided)",
    )
    bench_parser.add_argument(
        "-t",
        "--tokenizer",
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    bench_parser.add_argument(
        "--bucket-edges",
        default="128,256,512,1024",
        help="Bucket edge lengths (comma-separated, default: 128,256,512,1024)",
    )
    bench_parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
        help="Token budget per microbatch (default: 4096)",
    )
    bench_parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    bench_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process from dataset",
    )
    bench_parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (when no dataset, default: 1000)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    bench_parser.set_defaults(func=bench_pipeline)

    # =========================================================================
    # Introspect command - model introspection and logit lens
    # =========================================================================
    introspect_parser = subparsers.add_parser(
        "introspect",
        help="Model introspection and logit lens analysis",
        description="Analyze model behavior using logit lens and attention visualization.",
    )
    introspect_subparsers = introspect_parser.add_subparsers(
        dest="introspect_command", help="Introspection commands"
    )

    # Analyze command - main logit lens analysis
    analyze_parser = introspect_subparsers.add_parser(
        "analyze",
        help="Run logit lens analysis on a prompt",
        description="Analyze how model predictions evolve across layers using logit lens.",
    )
    analyze_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    analyze_parser.add_argument(
        "--adapter",
        "-a",
        help="Path to LoRA adapter weights (for analyzing fine-tuned models)",
    )
    analyze_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to analyze (required unless --prefix is used)",
    )
    analyze_parser.add_argument(
        "--layer-strategy",
        choices=["all", "evenly_spaced", "first_last", "custom"],
        default="evenly_spaced",
        help="Layer selection strategy (default: evenly_spaced)",
    )
    analyze_parser.add_argument(
        "--layer-step",
        "-s",
        type=int,
        default=4,
        help="Step size for evenly_spaced strategy (default: 4)",
    )
    analyze_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    analyze_parser.add_argument(
        "--track",
        "-t",
        action="append",
        help="Token to track evolution (can specify multiple, e.g., --track Paris --track ' Paris')",
    )
    analyze_parser.add_argument(
        "--embedding-scale",
        type=float,
        help="Embedding scale factor (e.g., 33.94 for Gemma with hidden_size=1152)",
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    analyze_parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Capture all layers (overrides --layer-strategy)",
    )
    analyze_parser.add_argument(
        "--layers",
        "-l",
        help="Specific layers to analyze (comma-separated, e.g., '13,14,15,16,17'). Overrides --layer-strategy",
    )
    analyze_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template (for non-chat models or direct testing)",
    )
    analyze_parser.add_argument(
        "--find-answer",
        action="store_true",
        help="Generate tokens first to find where the answer starts, then analyze at that position (default for chat mode)",
    )
    analyze_parser.add_argument(
        "--no-find-answer",
        action="store_true",
        help="Disable answer position detection (analyze immediate next token)",
    )
    analyze_parser.add_argument(
        "--expected",
        help="Expected answer token(s) to find in generated output (used with --find-answer)",
    )
    analyze_parser.add_argument(
        "--gen-tokens",
        "-n",
        type=int,
        default=30,
        help="Number of tokens to generate when using --find-answer (default: 30)",
    )
    analyze_parser.add_argument(
        "--prefix",
        help="Analyze at a specific prefix (bypasses --prompt, --raw, --find-answer). Useful for testing specific positions in generated output.",
    )
    analyze_parser.add_argument(
        "--steer",
        help="Apply steering during analysis. Either 'direction.npz:coefficient' or just 'direction.npz' (use --strength for coefficient)",
    )
    analyze_parser.add_argument(
        "--steer-neuron",
        type=int,
        help="Single neuron index to steer (alternative to --steer with a direction file)",
    )
    analyze_parser.add_argument(
        "--steer-layer",
        type=int,
        help="Layer to apply neuron steering (required with --steer-neuron)",
    )
    analyze_parser.add_argument(
        "--strength",
        type=float,
        help="Steering strength/coefficient when using --steer or --steer-neuron (default: 1.0)",
    )
    analyze_parser.add_argument(
        "--inject-layer",
        type=int,
        help="Layer at which to inject a token embedding (use with --inject-token)",
    )
    analyze_parser.add_argument(
        "--inject-token",
        help="Token to inject at --inject-layer (e.g., '2491' to force that answer)",
    )
    analyze_parser.add_argument(
        "--inject-blend",
        type=float,
        default=1.0,
        help="Blend factor for injection: 0=original, 1=full replacement (default: 1.0)",
    )
    analyze_parser.add_argument(
        "--compute-override",
        choices=["arithmetic", "none"],
        default="none",
        help="Override model computation with Python at layer boundary. "
        "'arithmetic' detects A*B=, A+B=, etc and injects correct answer at --compute-layer",
    )
    analyze_parser.add_argument(
        "--compute-layer",
        type=int,
        help="Layer at which to inject computed answer (default: 80%% of model depth)",
    )
    analyze_parser.set_defaults(func=lambda args: asyncio.run(introspect_analyze(args)))

    # Compare command - compare two models
    compare_introspect_parser = introspect_subparsers.add_parser(
        "compare",
        help="Compare two models' predictions using logit lens",
        description="Compare how predictions evolve in two different models.",
    )
    compare_introspect_parser.add_argument(
        "--model1",
        "-m1",
        required=True,
        help="First model name or HuggingFace ID",
    )
    compare_introspect_parser.add_argument(
        "--model2",
        "-m2",
        required=True,
        help="Second model name or HuggingFace ID",
    )
    compare_introspect_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Prompt to analyze",
    )
    compare_introspect_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    compare_introspect_parser.add_argument(
        "--track",
        help="Tokens to track evolution (comma-separated)",
    )
    compare_introspect_parser.set_defaults(func=lambda args: asyncio.run(introspect_compare(args)))

    # Hooks command - low-level hook demonstration
    hooks_parser = introspect_subparsers.add_parser(
        "hooks",
        help="Low-level hook demonstration",
        description="Demonstrate low-level hook API for capturing intermediate states.",
    )
    hooks_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    hooks_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Prompt to analyze",
    )
    hooks_parser.add_argument(
        "--layers",
        help="Layers to capture (comma-separated, e.g., '0,4,8,12')",
    )
    hooks_parser.add_argument(
        "--capture-attention",
        action="store_true",
        help="Also capture attention weights",
    )
    hooks_parser.add_argument(
        "--last-only",
        action="store_true",
        help="Only capture last sequence position (more memory efficient)",
    )
    hooks_parser.add_argument(
        "--no-logit-lens",
        action="store_true",
        help="Skip logit lens analysis",
    )
    hooks_parser.set_defaults(func=lambda args: asyncio.run(introspect_hooks(args)))

    # Ablation command - run ablation studies
    ablation_parser = introspect_subparsers.add_parser(
        "ablate",
        help="Run ablation studies to identify causal circuits",
        description="Ablate model components to identify which layers/heads are causal for specific behaviors.",
    )
    ablation_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    ablation_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to test (required unless using --prompts)",
    )
    ablation_parser.add_argument(
        "--criterion",
        "-c",
        help="Criterion to check (e.g., 'function_call', 'sorry', 'positive', or expected text)",
    )
    ablation_parser.add_argument(
        "--component",
        choices=["mlp", "attention", "both"],
        default="mlp",
        help="Component to ablate (default: mlp)",
    )
    ablation_parser.add_argument(
        "--layers",
        help="Layers to test (comma-separated, e.g., '5,8,10,11,12'). Default: all",
    )
    ablation_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    ablation_parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens to generate (default: 60)",
    )
    ablation_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show actual generated outputs (original and ablated)",
    )
    ablation_parser.add_argument(
        "--multi",
        action="store_true",
        help="Ablate all specified layers together (default: sweep each layer separately)",
    )
    ablation_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    ablation_parser.add_argument(
        "--prompts",
        help="Multiple prompts to test (pipe-separated, e.g., '10*10=|45*45=|47*47=')",
    )
    ablation_parser.set_defaults(func=introspect_ablate)

    # Weight divergence command - compare weights between two models
    weight_div_parser = introspect_subparsers.add_parser(
        "weight-diff",
        help="Compare weight divergence between two models",
        description="Compute per-layer, per-component weight differences between base and fine-tuned models.",
    )
    weight_div_parser.add_argument(
        "--base",
        "-b",
        required=True,
        help="Base model (HuggingFace ID or path)",
    )
    weight_div_parser.add_argument(
        "--finetuned",
        "-f",
        required=True,
        help="Fine-tuned model (HuggingFace ID or path)",
    )
    weight_div_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    weight_div_parser.set_defaults(func=introspect_weight_diff)

    # Activation divergence command - compare activations on same prompt
    activation_div_parser = introspect_subparsers.add_parser(
        "activation-diff",
        help="Compare activation divergence between two models",
        description="Run same prompts through two models and compare hidden state representations.",
    )
    activation_div_parser.add_argument(
        "--base",
        "-b",
        required=True,
        help="Base model (HuggingFace ID or path)",
    )
    activation_div_parser.add_argument(
        "--finetuned",
        "-f",
        required=True,
        help="Fine-tuned model (HuggingFace ID or path)",
    )
    activation_div_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (comma-separated or @file.txt)",
    )
    activation_div_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    activation_div_parser.set_defaults(func=introspect_activation_diff)

    # Layer analysis command - representation similarity and clustering
    layer_parser = introspect_subparsers.add_parser(
        "layer",
        help="Analyze what specific layers do with representation similarity",
        description="Analyze representations at specific layers to understand what layers 'see'.",
    )
    layer_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    layer_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt). Example: 'prompt1|prompt2|prompt3'",
    )
    layer_parser.add_argument(
        "--labels",
        "-l",
        help="Labels for prompts (comma-separated, same order as prompts). Example: 'working,broken,working,broken'",
    )
    layer_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, e.g., '2,4,6,8'). Default: auto (key layers)",
    )
    layer_parser.add_argument(
        "--attention",
        "-a",
        action="store_true",
        help="Also analyze attention patterns",
    )
    layer_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    layer_parser.set_defaults(func=introspect_layer)

    # Format sensitivity command - quick test for trailing space effects
    format_parser = introspect_subparsers.add_parser(
        "format-sensitivity",
        help="Quick format sensitivity check (trailing space vs no space)",
        description="Automatically test prompts with and without trailing space.",
    )
    format_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    format_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Base prompts (pipe-separated or @file.txt). Trailing space will be added/removed automatically.",
    )
    format_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated). Default: auto",
    )
    format_parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Only show summary (skip detailed output)",
    )
    format_parser.set_defaults(func=introspect_format_sensitivity)

    # Generate command - multi-token generation to test next-token lock hypothesis
    generate_parser = introspect_subparsers.add_parser(
        "generate",
        help="Generate multiple tokens to test next-token lock hypothesis",
        description="Test whether format issues cause simple next-token lock or complex gates.",
    )
    generate_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    generate_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (pipe-separated or @file.txt)",
    )
    generate_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)",
    )
    generate_parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Temperature (0=greedy, default: 0)",
    )
    generate_parser.add_argument(
        "--compare-format",
        "-c",
        action="store_true",
        help="Auto-create with/without trailing space variants and compare",
    )
    generate_parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Show individual generated tokens",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    generate_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template (for non-chat models or direct testing)",
    )
    generate_parser.set_defaults(func=lambda args: asyncio.run(introspect_generate(args)))

    # Metacognitive command - detect strategy switch
    metacog_parser = introspect_subparsers.add_parser(
        "metacognitive",
        help="Detect metacognitive strategy switch (direct vs chain-of-thought)",
        description="""Probe the model's decision layer to detect strategy selection.

At approximately 70% depth, the model's token prediction reveals its strategy:
- DIRECT: Predicts a digit → will output answer immediately
- CoT: Predicts space/word → will use chain-of-thought reasoning

This is the "metacognitive switch" - the model deciding HOW to solve, not WHAT the answer is.
        """,
    )
    metacog_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    metacog_parser.add_argument(
        "--problems",
        "-p",
        default="5 + 5 =",
        help="Arithmetic problems (pipe-separated or @file.txt). Default: '5 + 5 ='",
    )
    metacog_parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Auto-generate random arithmetic problems",
    )
    metacog_parser.add_argument(
        "--num-problems",
        "-n",
        type=int,
        default=20,
        help="Number of problems to generate (with --generate, default: 20)",
    )
    metacog_parser.add_argument(
        "--decision-layer",
        "-l",
        type=int,
        help="Layer to probe for strategy (default: ~70%% of model depth)",
    )
    metacog_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem generation (default: 42)",
    )
    metacog_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    metacog_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    metacog_parser.set_defaults(func=lambda args: asyncio.run(introspect_metacognitive(args)))

    # Steer command - activation steering
    steer_parser = introspect_subparsers.add_parser(
        "steer",
        help="Apply activation steering to manipulate model behavior",
        description="""Activation steering: modify model behavior by adding learned directions.

Three modes of operation:
1. Extract direction: --extract --positive "good" --negative "bad" -o direction.npz
2. Apply direction: --direction direction.npz -p "prompt"
3. Compare coefficients: --direction direction.npz -p "prompt" --compare "-1,0,1"
        """,
    )
    steer_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    steer_parser.add_argument(
        "--prompts",
        "-p",
        help="Prompts to steer (pipe-separated or @file.txt)",
    )
    steer_parser.add_argument(
        "--direction",
        "-d",
        help="Path to direction file (.npz or .json)",
    )
    steer_parser.add_argument(
        "--neuron",
        type=int,
        help="Single neuron index to steer (creates a one-hot direction)",
    )
    steer_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to apply steering (default: auto from direction or middle)",
    )
    steer_parser.add_argument(
        "--coefficient",
        "-c",
        "--strength",
        type=float,
        default=1.0,
        help="Steering coefficient/strength (default: 1.0, negative = toward negative class)",
    )
    steer_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract direction from contrastive prompts (requires --positive and --negative)",
    )
    steer_parser.add_argument(
        "--positive",
        help="Positive class prompt (for direction extraction or on-the-fly steering)",
    )
    steer_parser.add_argument(
        "--negative",
        help="Negative class prompt (for direction extraction or on-the-fly steering)",
    )
    steer_parser.add_argument(
        "--compare",
        help="Compare outputs at multiple coefficients (comma-separated, e.g., '-2,-1,0,1,2')",
    )
    steer_parser.add_argument(
        "--name",
        help="Name for the direction (for logging)",
    )
    steer_parser.add_argument(
        "--positive-label",
        help="Label for positive class (default: 'positive')",
    )
    steer_parser.add_argument(
        "--negative-label",
        help="Label for negative class (default: 'negative')",
    )
    steer_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    steer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0 = greedy)",
    )
    steer_parser.add_argument(
        "--output",
        "-o",
        help="Save results/direction to file",
    )
    steer_parser.set_defaults(func=introspect_steer)

    # Arithmetic command - systematic arithmetic circuit study
    arithmetic_parser = introspect_subparsers.add_parser(
        "arithmetic",
        help="Run systematic arithmetic study to find emergence layers",
        description="""Test arithmetic problems of varying difficulty and track when
answers first emerge as top predictions across layers.

This reveals where computation happens in the model:
- Easy problems may emerge early
- Hard problems may require more layers
- Multiplication may use different circuits than addition
        """,
    )
    arithmetic_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    arithmetic_parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick mode (subset of tests)",
    )
    arithmetic_parser.add_argument(
        "--easy-only",
        action="store_true",
        help="Only run easy problems (1-digit)",
    )
    arithmetic_parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only run hard problems (3-digit)",
    )
    arithmetic_parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw prompt without chat template",
    )
    arithmetic_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    arithmetic_parser.set_defaults(func=introspect_arithmetic)

    # Uncertainty command - detect model confidence before generation
    uncertainty_parser = introspect_subparsers.add_parser(
        "uncertainty",
        help="Detect model uncertainty using hidden state geometry",
        description="""Predict model confidence before generation by analyzing
hidden state geometry at a key layer.

Uses distance to "compute center" vs "refusal center" to predict
whether the model will produce a confident answer or show uncertainty.

Key insight: Working prompts cluster in one region of hidden space,
broken/uncertain prompts cluster in another. The distance ratio
predicts output behavior.
        """,
    )
    uncertainty_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    uncertainty_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to test (pipe-separated or @file.txt)",
    )
    uncertainty_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Detection layer (default: ~70%% of model depth)",
    )
    uncertainty_parser.add_argument(
        "--working",
        "-w",
        help="Comma-separated working examples for calibration (default: arithmetic with trailing space)",
    )
    uncertainty_parser.add_argument(
        "--broken",
        "-b",
        help="Comma-separated broken examples for calibration (default: arithmetic without trailing space)",
    )
    uncertainty_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    uncertainty_parser.set_defaults(func=lambda args: asyncio.run(introspect_uncertainty(args)))

    # Probe command - train linear probe to find task classification layers
    probe_parser = introspect_subparsers.add_parser(
        "probe",
        help="Train linear probe to find task classification layers",
        description="""Train logistic regression probes at each layer to find where
the model classifies different types of prompts.

This reveals task classification in ACTIVATION SPACE (not logit space).

Example:
    lazarus introspect probe -m model \\
        --class-a "2+2=|45*45=|100-37=" --label-a math \\
        --class-b "Capital of France?|Write a poem" --label-b other
        """,
    )
    probe_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    probe_parser.add_argument(
        "--class-a",
        required=True,
        help="Class A prompts (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--class-b",
        required=True,
        help="Class B prompts (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--label-a",
        default="class_a",
        help="Label for class A (default: 'class_a')",
    )
    probe_parser.add_argument(
        "--label-b",
        default="class_b",
        help="Label for class B (default: 'class_b')",
    )
    probe_parser.add_argument(
        "--test",
        "-t",
        help="Test prompts to classify after training (pipe-separated or @file.txt)",
    )
    probe_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    probe_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Specific layer to probe (default: find best layer)",
    )
    probe_parser.add_argument(
        "--save-direction",
        help="Save extracted direction vector to .npz file",
    )
    probe_parser.add_argument(
        "--method",
        choices=["logistic", "difference"],
        default="logistic",
        help="Direction extraction method: 'logistic' (probe weights) or 'difference' (mean difference)",
    )
    probe_parser.set_defaults(func=lambda args: asyncio.run(introspect_probe(args)))

    # Neurons command - analyze individual neuron activations
    neurons_parser = introspect_subparsers.add_parser(
        "neurons",
        help="Analyze individual neuron activations across prompts",
        description="""Show how specific neurons fire across different prompts.

Useful for understanding what individual neurons encode after running a probe.

Examples:
    # Analyze top neurons from probe across prompts
    lazarus introspect neurons -m model -l 15 \\
        --prompts "2+2=|45*45=|47*47=|67*83=" \\
        --neurons 808,1190,1168,891

    # Load neurons from saved direction
    lazarus introspect neurons -m model -l 15 \\
        --prompts "2+2=|47*47=" \\
        --from-direction difficulty.npz --top-k 6

    # Track neuron across multiple layers
    lazarus introspect neurons -m model --layers 15,19,20,21 \\
        --prompts "10*10=|25*25=|100*100=|17*19=|23*29=|47*53=" \\
        --neurons 1930 \\
        --labels "trivial|easy|memorized|medium|hard|hardest"
        """,
    )
    neurons_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    neurons_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Single layer to analyze (use --layers for multiple)",
    )
    neurons_parser.add_argument(
        "--layers",
        help="Multiple layers to analyze (comma-separated, e.g., '15,19,20,21')",
    )
    neurons_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt)",
    )
    neurons_parser.add_argument(
        "--neurons",
        "-n",
        help="Neuron indices to analyze (comma-separated, e.g., '808,1190,1168')",
    )
    neurons_parser.add_argument(
        "--neuron-names",
        help="Names for neurons (pipe-separated, same order as --neurons, e.g., 'Confidence|Computation|Effort')",
    )
    neurons_parser.add_argument(
        "--from-direction",
        help="Load top neurons from saved direction .npz file",
    )
    neurons_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show when using --from-direction or --auto-discover (default: 10)",
    )
    neurons_parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Automatically discover discriminative neurons by variance/separation across prompts",
    )
    neurons_parser.add_argument(
        "--labels",
        help="Labels for prompts (pipe-separated, same order as prompts). Required for --auto-discover",
    )
    neurons_parser.add_argument(
        "--steer",
        help="Apply steering during analysis. Either 'direction.npz:coefficient' or just 'direction.npz' (use --strength for coefficient)",
    )
    neurons_parser.add_argument(
        "--strength",
        type=float,
        help="Steering strength/coefficient when using --steer (default: 1.0)",
    )
    neurons_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    neurons_parser.set_defaults(func=introspect_neurons)

    # Cluster command - visualize activation clusters
    cluster_parser = introspect_subparsers.add_parser(
        "cluster",
        help="Visualize activation clusters using PCA",
        description="""Project hidden states to 2D to see if different prompt types
cluster separately in activation space.

Shows ASCII scatter plot and cluster statistics.

Supports two syntaxes:
1. Legacy two-class: --class-a "prompts" --class-b "prompts" --label-a X --label-b Y
2. Multi-class: --prompts "p1|p2|p3" --label L1 --prompts "p4|p5" --label L2 ...

Multi-class example:
    lazarus introspect cluster -m model \\
        --prompts "45*45=|25*25=|15*15=" --label mult \\
        --prompts "123+456=|100+37=|50+50=" --label add \\
        --prompts "The capital of France is|The opposite of hot is" --label language \\
        --layer 19 --save-plot L19_cluster.png
        """,
    )
    cluster_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    # Legacy two-class syntax
    cluster_parser.add_argument(
        "--class-a",
        help="Class A prompts (pipe-separated or @file.txt) [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--class-b",
        help="Class B prompts (pipe-separated or @file.txt) [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--label-a",
        default="class_a",
        help="Label for class A (default: 'class_a') [legacy syntax]",
    )
    cluster_parser.add_argument(
        "--label-b",
        default="class_b",
        help="Label for class B (default: 'class_b') [legacy syntax]",
    )
    # New multi-class syntax
    cluster_parser.add_argument(
        "--prompts",
        action="append",
        dest="prompt_groups",
        help="Prompts for a class (pipe-separated or @file.txt). Use multiple times with --label.",
    )
    cluster_parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Label for the preceding --prompts group. Must match number of --prompts.",
    )
    cluster_parser.add_argument(
        "--layer",
        "-l",
        help="Layer(s) to analyze - single int or comma-separated (e.g., '19' or '19,20,21'). Default: ~50%% of model depth",
    )
    cluster_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    cluster_parser.add_argument(
        "--save-plot",
        help="Save matplotlib scatter plot to file (e.g., cluster.png)",
    )
    cluster_parser.set_defaults(func=lambda args: asyncio.run(introspect_activation_cluster(args)))

    # Memory command - extract memory organization structure
    memory_parser = introspect_subparsers.add_parser(
        "memory",
        help="Extract memory organization structure for facts",
        description="""Extract how facts are organized in model memory by analyzing
neighborhood activation patterns.

For each query, captures what other facts co-activate, revealing:
- Memory organization (row vs column, clusters)
- Asymmetry (does A->B activate same as B->A?)
- Attractor nodes (frequently co-activated facts)
- Difficulty patterns (which facts are hardest to retrieve)

Built-in fact types:
- multiplication: Single-digit times tables (2-9)
- addition: Single-digit addition
- capitals: Country capitals
- elements: Periodic table elements

Custom facts via CSV/JSON file.

Examples:
    # Extract times table memory structure
    lazarus introspect memory -m model --facts multiplication --layer 20

    # Extract capital city memory
    lazarus introspect memory -m model --facts capitals --layer 15

    # Custom facts from file
    lazarus introspect memory -m model --facts @my_facts.json --layer 20
        """,
    )
    memory_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    memory_parser.add_argument(
        "--facts",
        "-f",
        required=True,
        help="Fact type: 'multiplication', 'addition', 'capitals', 'elements', or @file.json",
    )
    memory_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: ~80%% of model depth)",
    )
    memory_parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top predictions to capture per query (default: 30)",
    )
    memory_parser.add_argument(
        "--output",
        "-o",
        help="Save detailed results to JSON file",
    )
    memory_parser.add_argument(
        "--save-plot",
        help="Save visualization to file (e.g., memory_structure.png)",
    )
    memory_parser.add_argument(
        "--classify",
        action="store_true",
        help="Show memorization classification (memorized/partial/weak/not memorized)",
    )
    memory_parser.set_defaults(func=lambda args: asyncio.run(introspect_memory(args)))

    # Memory-inject command - external memory injection
    memory_inject_parser = introspect_subparsers.add_parser(
        "memory-inject",
        help="External memory injection for fact retrieval",
        description="""External memory injection: Inject correct answers from an external store.

This provides circuit-guided memory externalization by:
1. Building a store of (query, value) vector pairs from known facts
2. Matching input queries to stored entries by similarity
3. Injecting retrieved values into the residual stream

Use cases:
- Override incorrect model answers
- Rescue out-of-distribution query formats
- Add new facts without fine-tuning

Examples:
    # Test on multiplication with standard query
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --query "7*8="

    # Rescue non-standard format (force injection even if below threshold)
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --query "seven times eight equals" --force

    # Multiple queries
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --queries "7*8=|6*7=|9*9="

    # Evaluate on all facts
    lazarus introspect memory-inject -m openai/gpt-oss-20b \\
        --facts multiplication --evaluate
        """,
    )
    memory_inject_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    memory_inject_parser.add_argument(
        "--facts",
        "-f",
        required=True,
        help="Fact type: 'multiplication', 'addition', or @file.json",
    )
    memory_inject_parser.add_argument(
        "--query",
        "-q",
        help="Single query to test",
    )
    memory_inject_parser.add_argument(
        "--queries",
        help="Multiple queries separated by | (e.g., '7*8=|6*7=')",
    )
    memory_inject_parser.add_argument(
        "--query-layer",
        type=int,
        help="Layer for query matching (default: ~92%% of model depth)",
    )
    memory_inject_parser.add_argument(
        "--inject-layer",
        type=int,
        help="Layer to inject values (default: ~88%% of model depth)",
    )
    memory_inject_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor: 0=no injection, 1=full replacement (default: 1.0)",
    )
    memory_inject_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum similarity to use injection (default: 0.7)",
    )
    memory_inject_parser.add_argument(
        "--force",
        action="store_true",
        help="Force injection even if similarity is below threshold",
    )
    memory_inject_parser.add_argument(
        "--save-store",
        help="Save memory store to file (e.g., memory.npz)",
    )
    memory_inject_parser.add_argument(
        "--load-store",
        help="Load memory store from file",
    )
    memory_inject_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate baseline vs injected accuracy on all facts",
    )
    memory_inject_parser.set_defaults(func=lambda args: asyncio.run(introspect_memory_inject(args)))

    # Directions command - compare multiple direction vectors for orthogonality
    directions_parser = introspect_subparsers.add_parser(
        "directions",
        help="Compare direction vectors for orthogonality",
        description="""Compare multiple saved direction vectors to check if they are orthogonal.

This confirms whether extracted dimensions (e.g., difficulty, operation, format)
represent truly independent features in activation space.

Example:
    lazarus introspect directions \\
        difficulty.npz uncertainty.npz format.npz operation.npz
        """,
    )
    directions_parser.add_argument(
        "files",
        nargs="+",
        help="Direction files to compare (.npz format from 'introspect probe --save-direction')",
    )
    directions_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Cosine similarity threshold for 'orthogonal' (default: 0.1)",
    )
    directions_parser.add_argument(
        "--output",
        "-o",
        help="Save similarity matrix to JSON file",
    )
    directions_parser.set_defaults(func=introspect_directions)

    # Operand-directions command - analyze how operands are encoded
    operand_directions_parser = introspect_subparsers.add_parser(
        "operand-directions",
        help="Analyze how operands A and B are encoded in activation space",
        description="""Extract operand directions (A_d and B_d) to analyze encoding structure.

This is useful for understanding if a model uses:
- Compositional encoding (like GPT-OSS): A and B in separate orthogonal subspaces
- Holistic encoding (like Gemma): entire expression encoded together

Examples:
    # Analyze multiplication operand encoding
    lazarus introspect operand-directions -m model \\
        --digits 2,3,4,5,6,7,8,9 --operation "*" --layers 8,16,20,24

    # Save directions for later analysis
    lazarus introspect operand-directions -m model \\
        --output operand_dirs.npz
        """,
    )
    operand_directions_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    operand_directions_parser.add_argument(
        "--digits",
        help="Digits to use (comma-separated, default: 2,3,4,5,6,7,8,9)",
    )
    operand_directions_parser.add_argument(
        "--operation",
        default="*",
        help="Operation to test (default: '*')",
    )
    operand_directions_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, default: auto key layers)",
    )
    operand_directions_parser.add_argument(
        "--output",
        "-o",
        help="Save results to file (.json or .npz)",
    )
    operand_directions_parser.set_defaults(func=introspect_operand_directions)

    # Embedding command - analyze what's encoded at embedding level
    embedding_parser = introspect_subparsers.add_parser(
        "embedding",
        help="Analyze what information is encoded at embedding level vs after layers",
        description="""Test the RLVF backprop hypothesis: does task information exist in raw embeddings?

Tests:
1. Task type detection (arithmetic vs language) from embeddings
2. Operation type detection (mult vs add) from embeddings
3. Answer correlation with embeddings vs after layers

If task type is 100% detectable from embeddings, this suggests RLVF gradients
backpropagate all the way to the embedding layer.

Examples:
    # Test embedding analysis
    lazarus introspect embedding -m model

    # Test with specific operation
    lazarus introspect embedding -m model --operation mult

    # Analyze specific layers
    lazarus introspect embedding -m model --layers 0,1,2,4
        """,
    )
    embedding_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    embedding_parser.add_argument(
        "--operation",
        choices=["mult", "add", "all", "*", "+"],
        help="Operation type to test (default: all)",
    )
    embedding_parser.add_argument(
        "--layers",
        help="Layers to compare against embeddings (comma-separated, default: 0,1,2)",
    )
    embedding_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    embedding_parser.set_defaults(func=introspect_embedding)

    # Commutativity command - test if representations respect A*B = B*A
    commutativity_parser = introspect_subparsers.add_parser(
        "commutativity",
        help="Test if internal representations respect commutativity (A*B = B*A)",
        description="""Test commutativity in internal representations.

For multiplication, A*B and B*A should produce the same answer. This test checks
whether the internal representations for commutative pairs are similar, which
would indicate a lookup table structure rather than an algorithm.

High commutativity similarity (>0.99) suggests the model memorizes individual facts
rather than computing them algorithmically.

Examples:
    # Test all commutative pairs (2-9)
    lazarus introspect commutativity -m model

    # Test specific pairs
    lazarus introspect commutativity -m model \\
        --pairs "2*3,3*2|7*8,8*7|4*5,5*4"

    # Analyze at specific layer
    lazarus introspect commutativity -m model --layer 20
        """,
    )
    commutativity_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    commutativity_parser.add_argument(
        "--pairs",
        help="Explicit commutative pairs to test (e.g., '2*3,3*2|7*8,8*7')",
    )
    commutativity_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: ~60%% of model depth)",
    )
    commutativity_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    commutativity_parser.set_defaults(func=introspect_commutativity)

    # Patch command - activation patching between prompts
    patch_parser = introspect_subparsers.add_parser(
        "patch",
        help="Perform activation patching between source and target prompts",
        description="""Activation patching: transfer activations from source to target prompt.

This is a causal intervention technique that tests whether activations from
one prompt can transfer computation to another prompt.

For example, patching activations from "7*8=" into "7+8=" at the right layer
should cause the model to output "56" instead of "15".

Examples:
    # Patch multiplication into addition
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8="

    # Patch at specific layer
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8=" --layer 20

    # Patch with partial blend
    lazarus introspect patch -m model \\
        --source "7*8=" --target "7+8=" --blend 0.5
        """,
    )
    patch_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    patch_parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Source prompt to patch FROM",
    )
    patch_parser.add_argument(
        "--target",
        "-t",
        required=True,
        help="Target prompt to patch INTO",
    )
    patch_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Single layer to patch at",
    )
    patch_parser.add_argument(
        "--layers",
        help="Multiple layers to sweep (comma-separated, default: all key layers)",
    )
    patch_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor: 0=no change, 1=full replacement (default: 1.0)",
    )
    patch_parser.add_argument(
        "--max-tokens",
        "-n",
        type=int,
        default=10,
        help="Max tokens to generate (default: 10)",
    )
    patch_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    patch_parser.set_defaults(func=introspect_patch)

    # Early layers command - analyze what information is encoded in early layers
    early_layers_parser = introspect_subparsers.add_parser(
        "early-layers",
        help="Analyze what information is encoded in early layers",
        description="""Analyze early layer information encoding using linear probes.

This command reveals how information is organized in early transformer layers:
- Cross-expression similarity at the '=' position
- Linear probe extraction of operation type, operands, and answer
- The "orthogonal subspaces paradox": high similarity but separable information

Key insight: Even when cosine similarity is high (0.997), information can be
linearly extracted because it's encoded in orthogonal directions.

Examples:
    # Basic analysis with default settings
    lazarus introspect early-layers -m model

    # Analyze specific layers
    lazarus introspect early-layers -m model --layers 0,1,2,4,8

    # Include position-wise analysis
    lazarus introspect early-layers -m model --analyze-positions

    # Test specific operations
    lazarus introspect early-layers -m model --operations "*,+,-"
        """,
    )
    early_layers_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    early_layers_parser.add_argument(
        "--layers",
        help="Layers to analyze (comma-separated, default: 0,1,2,4,8,12)",
    )
    early_layers_parser.add_argument(
        "--operations",
        help="Operations to test (comma-separated, default: *,+)",
    )
    early_layers_parser.add_argument(
        "--digits",
        help="Digit range for operands (e.g., 2-8, default: 2-8)",
    )
    early_layers_parser.add_argument(
        "--analyze-positions",
        action="store_true",
        help="Include position-wise analysis (slower but more detailed)",
    )
    early_layers_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    early_layers_parser.set_defaults(func=introspect_early_layers)

    # Circuit command - direct circuit invocation and manipulation
    circuit_parser = introspect_subparsers.add_parser(
        "circuit",
        help="Direct circuit capture, interpolation, and invocation",
        description="""Experimental: Capture and manipulate computation circuits directly.

Subcommands:
  capture   - Capture activations for a computation (e.g., "7 * 4 = 28")
  invoke    - Interpolate/combine captured circuits to compute new values
  decode    - Decode activations back to tokens/answers

Examples:
    # Capture multiplication examples
    lazarus introspect circuit capture -m model \\
        --prompts "7*4=28|6*8=48|9*3=27" --layer 19 --save mult_circuit.npz

    # Invoke circuit with new operands (interpolate)
    lazarus introspect circuit invoke -m model \\
        --circuit mult_circuit.npz --operands "5,6" --layer 19

    # Decode what answer the circuit produces
    lazarus introspect circuit decode -m model \\
        --activations circuit_state.npz --layer 19
        """,
    )
    circuit_subparsers = circuit_parser.add_subparsers(dest="circuit_command")

    # Circuit capture
    capture_parser = circuit_subparsers.add_parser(
        "capture",
        help="Capture circuit activations for known computations",
    )
    capture_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    capture_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Computation prompts (pipe-separated, e.g., '7*4=|6*8=' or '7*4=28|6*8=48')",
    )
    capture_parser.add_argument(
        "--results",
        "-r",
        help="Expected results (pipe-separated, e.g., '28|48') - use with prompts like '7*4=|6*8='",
    )
    capture_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        required=True,
        help="Layer to capture activations from",
    )
    capture_parser.add_argument(
        "--save",
        "-o",
        required=True,
        help="Save captured circuit to .npz file",
    )
    capture_parser.add_argument(
        "--extract-direction",
        action="store_true",
        help="Extract and save the direction that encodes the result value",
    )
    capture_parser.add_argument(
        "--position",
        choices=["last", "answer", "operator"],
        default="last",
        help="Position to capture: last token, answer position, or operator position",
    )
    capture_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_capture(args)))

    # Circuit invoke
    invoke_parser = circuit_subparsers.add_parser(
        "invoke",
        help="Invoke circuit with new operands via interpolation",
    )
    invoke_parser.add_argument(
        "--model",
        "-m",
        help="Model name or HuggingFace ID (required for 'steer' method)",
    )
    invoke_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Captured circuit file (.npz)",
    )
    invoke_parser.add_argument(
        "--operands",
        help="New operands to compute (pipe-separated pairs, e.g., '5,6|8,9')",
    )
    invoke_parser.add_argument(
        "--prompts",
        "-p",
        dest="invoke_prompts",
        help="Prompts to run through circuit (for 'steer' method, e.g., '5*6=|8*9=')",
    )
    invoke_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer for circuit (default: from circuit file)",
    )
    invoke_parser.add_argument(
        "--method",
        choices=["steer", "interpolate", "extrapolate", "linear"],
        default="linear",
        help="How to invoke circuit: steer (uses direction), linear/interpolate/extrapolate (uses activations)",
    )
    invoke_parser.add_argument(
        "--output",
        "-o",
        help="Save result to file",
    )
    invoke_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_invoke(args)))

    # Circuit decode
    decode_parser = circuit_subparsers.add_parser(
        "decode",
        help="Decode circuit activations to see what answer they produce",
    )
    decode_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    decode_parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="Base prompt to inject activations into",
    )
    decode_parser.add_argument(
        "--inject",
        "-i",
        required=True,
        help="Activations to inject (.npz file)",
    )
    decode_parser.add_argument(
        "--inject-idx",
        type=int,
        default=0,
        help="Index of activation to inject (default: 0)",
    )
    decode_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to inject at",
    )
    decode_parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Blend factor (0=original, 1=full injection)",
    )
    decode_parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate",
    )
    decode_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_decode(args)))

    # Circuit test - apply trained direction to new activations (proper OOD testing)
    test_parser = circuit_subparsers.add_parser(
        "test",
        help="Test if a circuit generalizes to new inputs",
    )
    test_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Trained circuit file (.npz from 'circuit capture --extract-direction')",
    )
    # Option 1: Provide pre-captured activations
    test_parser.add_argument(
        "--test-activations",
        "-t",
        help="Pre-captured test activations (.npz file)",
    )
    # Option 2: Capture on the fly with model + prompts
    test_parser.add_argument(
        "--model",
        "-m",
        help="Model to use for capturing test activations",
    )
    test_parser.add_argument(
        "--prompts",
        "-p",
        help="Test prompts (e.g., '1*1=|11*11=|10*5=')",
    )
    test_parser.add_argument(
        "--results",
        "-r",
        help="Expected results (e.g., '1|121|50')",
    )
    test_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    test_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_test(args)))

    # Circuit compare - compare multiple circuits (directions)
    compare_circuit_parser = circuit_subparsers.add_parser(
        "compare",
        help="Compare multiple circuits (e.g., add vs mult vs div)",
        description="""Compare the directions/circuits extracted for different operations.

Shows:
- Cosine similarity between circuit directions
- Angle between circuits (orthogonal = independent computations)
- Top neurons for each circuit

Example:
    lazarus introspect circuit compare \\
        -c mult_circuit.npz add_circuit.npz sub_circuit.npz div_circuit.npz
        """,
    )
    compare_circuit_parser.add_argument(
        "--circuits",
        "-c",
        nargs="+",
        required=True,
        help="Circuit files to compare (.npz files from 'circuit capture --extract-direction')",
    )
    compare_circuit_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show per circuit (default: 10)",
    )
    compare_circuit_parser.add_argument(
        "--output",
        "-o",
        help="Save comparison results to JSON file",
    )
    compare_circuit_parser.set_defaults(
        func=lambda args: asyncio.run(introspect_circuit_compare(args))
    )

    # Circuit view - display circuit contents
    view_parser = circuit_subparsers.add_parser(
        "view",
        help="View the contents of a captured circuit file",
        description="""Display circuit metadata, entries, and optionally as a formatted table.

Examples:
    # Basic view (shows first 20 entries)
    lazarus introspect circuit view -c mult_complete_table.npz

    # Show as multiplication/addition table grid
    lazarus introspect circuit view -c mult_complete_table.npz --table

    # Show with direction statistics and top neurons
    lazarus introspect circuit view -c mult_complete_table.npz --stats

    # Show all entries
    lazarus introspect circuit view -c mult_complete_table.npz --limit 0
        """,
    )
    view_parser.add_argument(
        "--circuit",
        "-c",
        required=True,
        help="Circuit file to view (.npz)",
    )
    view_parser.add_argument(
        "--table",
        "-t",
        action="store_true",
        help="Display as a formatted grid (for arithmetic circuits)",
    )
    view_parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show direction statistics and top neurons",
    )
    view_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Max entries to show in list view (0 for all, default: 20)",
    )
    view_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to show with --stats (default: 10)",
    )
    view_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_view(args)))

    # Circuit export - export circuit to various formats
    export_parser = circuit_subparsers.add_parser(
        "export",
        help="Export circuit graph to DOT, JSON, Mermaid, or HTML format",
        description="""Export ablation or direction results as a circuit graph.

Supports multiple output formats:
- DOT (Graphviz): For rendering with graphviz tools
- JSON: For programmatic processing
- Mermaid: For embedding in documentation
- HTML: Interactive visualization using vis.js

Examples:
    # Export ablation results to DOT
    lazarus introspect circuit export -i ablation_results.json -o circuit.dot --format dot

    # Export to interactive HTML
    lazarus introspect circuit export -i ablation_results.json -o circuit.html --format html

    # Export directions to Mermaid diagram
    lazarus introspect circuit export -i directions.json -o circuit.md --format mermaid
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file (ablation results JSON or directions JSON)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["dot", "json", "mermaid", "html"],
        default="json",
        help="Output format (default: json)",
    )
    export_parser.add_argument(
        "--type",
        choices=["ablation", "directions"],
        default="ablation",
        help="Input data type: ablation results or extracted directions (default: ablation)",
    )
    export_parser.add_argument(
        "--name",
        help="Circuit name (default: derived from input file)",
    )
    export_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum effect threshold for ablation circuits (default: 0.1)",
    )
    export_parser.add_argument(
        "--direction",
        choices=["TB", "LR", "BT", "RL"],
        default="TB",
        help="Graph direction: TB (top-bottom), LR (left-right), etc. (default: TB)",
    )
    export_parser.set_defaults(func=lambda args: asyncio.run(introspect_circuit_export(args)))

    # Virtual Expert command - add virtual experts to models
    virtual_expert_parser = introspect_subparsers.add_parser(
        "virtual-expert",
        help="Add virtual expert (tool) capabilities to models",
        description="""Virtual Expert System - route to external tools via MoE routing.

For MoE models (like GPT-OSS), intercepts actual router decisions.
For dense models (like LLaMA), creates virtual routing in activation space.

Actions:
  analyze   - Analyze which experts activate for different prompt categories (MoE only)
  solve     - Solve a single problem with virtual expert
  benchmark - Run benchmark comparing model vs virtual expert
  compare   - Compare model-only vs virtual expert on a prompt
  interactive - Interactive REPL mode

Examples:
    # Analyze expert routing (MoE models)
    lazarus introspect virtual-expert analyze -m openai/gpt-oss-20b

    # Solve with virtual expert
    lazarus introspect virtual-expert solve -m model -p "127 * 89 = "

    # Run benchmark
    lazarus introspect virtual-expert benchmark -m model

    # Compare approaches
    lazarus introspect virtual-expert compare -m model -p "127 * 89 = "

    # Interactive mode
    lazarus introspect virtual-expert interactive -m model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    virtual_expert_parser.add_argument(
        "action",
        nargs="?",
        choices=["analyze", "solve", "benchmark", "compare", "interactive"],
        default="solve",
        help="Action to perform (default: solve)",
    )
    virtual_expert_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    virtual_expert_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to solve/compare (required for solve/compare)",
    )
    virtual_expert_parser.add_argument(
        "--problems",
        help="Problems for benchmark (pipe-separated or @file.txt)",
    )
    virtual_expert_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    virtual_expert_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed routing decisions (layer-by-layer trace)",
    )
    virtual_expert_parser.set_defaults(
        func=lambda args: asyncio.run(introspect_virtual_expert(args))
    )

    # MoE Expert command - direct expert manipulation
    moe_expert_parser = introspect_subparsers.add_parser(
        "moe-expert",
        help="Direct manipulation of MoE expert routing",
        description="""MoE Expert Explorer - Analyze how MoE models route tokens to experts.

Actions:
  explore       - Interactive REPL for real-time expert analysis (default)
  analyze       - Identify expert specializations across all categories
  chat          - Force all routing through a single expert
  compare       - Compare outputs from multiple specific experts
  ablate        - Remove an expert from routing (see what breaks)
  weights       - Show router weights for a prompt
  trace         - Trace expert assignments across layers
  heatmap       - Generate routing heatmap visualization
  full-taxonomy  - Semantic trigram pattern analysis across categories
  domain-test    - Demonstrate that domain experts don't exist
  token-routing  - Demonstrate that single token routing is context-dependent
  context-test   - Test context independence of routing
  context-window   - Test how much context the router uses (trigram vs attention)
  attention-routing - Analyze how attention patterns drive expert routing
  attention-pattern - Show attention weights for a specific position

MoE Type Detection & Compression:
  moe-type-analyze   - Detect pseudo vs native MoE (is it compressible?)
  moe-type-compare   - Compare MoE types between two models
  moe-overlay-compute  - Compute overlay representation (base + low-rank deltas)
  moe-overlay-verify   - Verify reconstruction accuracy (<1% error)
  moe-overlay-estimate - Estimate storage savings for full model
  moe-overlay-compress - Compress model to overlay format (saves to disk)

Quick Start:
    # Interactive explorer (recommended starting point)
    lazarus introspect moe-expert explore -m openai/gpt-oss-20b

Examples:
    # Prove domain experts don't exist (7+ experts handle ALL domains)
    lazarus introspect moe-expert domain-test -m openai/gpt-oss-20b

    # Show same token routes to different experts based on context
    lazarus introspect moe-expert token-routing -m openai/gpt-oss-20b

    # Full semantic trigram taxonomy analysis
    lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b

    # Generate routing heatmap visualization
    lazarus introspect moe-expert heatmap -m openai/gpt-oss-20b -p "def fibonacci(n):"

    # Chat with Expert 6 (force all tokens through it)
    lazarus introspect moe-expert chat -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = "

    # Kill an expert and see what breaks
    lazarus introspect moe-expert ablate -m openai/gpt-oss-20b --expert 6 -p "127 * 89 = " --benchmark

MoE Compression Examples:
    # Detect if model is compressible (pseudo-MoE vs native-MoE)
    lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b

    # Compare two models
    lazarus introspect moe-expert moe-type-compare -m openai/gpt-oss-20b -c allenai/OLMoE-1B-7B-0924

    # Full compression pipeline
    lazarus introspect moe-expert moe-overlay-compute -m openai/gpt-oss-20b
    lazarus introspect moe-expert moe-overlay-verify -m openai/gpt-oss-20b
    lazarus introspect moe-expert moe-overlay-estimate -m openai/gpt-oss-20b

    # Actually compress model to disk (36GB -> ~7GB)
    lazarus introspect moe-expert moe-overlay-compress -m openai/gpt-oss-20b
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    moe_expert_parser.add_argument(
        "action",
        nargs="?",
        choices=[
            # Interactive
            "explore",
            # Core analysis
            "analyze",
            "chat",
            "compare",
            "ablate",
            # Routing visualization
            "weights",
            "trace",
            "heatmap",
            # Semantic trigram methodology
            "full-taxonomy",
            "domain-test",
            "token-routing",
            "context-test",
            "context-window",
            "attention-routing",
            "attention-pattern",
            # MoE type detection
            "moe-type-analyze",
            "moe-type-compare",
            # MoE compression
            "moe-overlay-compute",
            "moe-overlay-verify",
            "moe-overlay-estimate",
            "moe-overlay-compress",
            # Expert dynamics analysis
            "cold-experts",
            "generation-dynamics",
            "expert-circuits",
            "expert-interference",
            "expert-merging",
            "attention-prediction",
            "task-prediction",
            "routing-manipulation",
            "context-attention-routing",
        ],
        default="explore",
        help="Action to perform (default: explore)",
    )
    moe_expert_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID (must be MoE model)",
    )
    moe_expert_parser.add_argument(
        "--expert",
        "-e",
        type=int,
        help="Expert index for chat/ablate (0-based)",
    )
    moe_expert_parser.add_argument(
        "--experts",
        help="Expert indices for compare (comma-separated, e.g., '6,7,20')",
    )
    moe_expert_parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt to test",
    )
    moe_expert_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run ablation benchmark on multiple problems",
    )
    moe_expert_parser.add_argument(
        "--layer",
        type=int,
        help="Target MoE layer for analysis (default: middle layer)",
    )
    moe_expert_parser.add_argument(
        "--layers",
        help="Layers to analyze for trace (comma-separated or 'all')",
    )
    moe_expert_parser.add_argument(
        "--examples",
        type=int,
        default=4,
        help="Number of example prompts to show per pattern (default: 4)",
    )
    moe_expert_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    moe_expert_parser.add_argument(
        "--token",
        "-t",
        help="Target token for context-test (e.g., 'the', 'def', '127')",
    )
    moe_expert_parser.add_argument(
        "--contexts",
        help="Comma-separated contexts to test (e.g., 'the cat,the dog,under the bridge')",
    )
    # Arguments for heatmap action
    moe_expert_parser.add_argument(
        "--prompts",
        nargs="+",
        help="Multiple prompts for heatmap (e.g., --prompts 'Hello' 'World')",
    )
    moe_expert_parser.add_argument(
        "--show-weights",
        action="store_true",
        help="For heatmap: show raw weight values in addition to expert indices",
    )
    # Arguments for full-taxonomy action
    moe_expert_parser.add_argument(
        "--categories",
        help="Comma-separated categories for full-taxonomy (e.g., 'arithmetic,code,analogy')",
    )
    moe_expert_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output (e.g., expert specializations for full-taxonomy)",
    )
    # Arguments for moe-type-compare action
    moe_expert_parser.add_argument(
        "--compare-model",
        "-c",
        help="Second model for moe-type-compare action",
    )
    # Arguments for moe-type-analyze action
    moe_expert_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show expert orthogonality heatmap visualization (for moe-type-analyze)",
    )
    # Arguments for moe-overlay-* actions (compression)
    moe_expert_parser.add_argument(
        "--quality",
        "-q",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Compression quality preset: fast (~12x), balanced (~8x, default), quality (~5x)",
    )
    moe_expert_parser.add_argument(
        "--gate-rank",
        type=int,
        help="Override gate projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--up-rank",
        type=int,
        help="Override up projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--down-rank",
        type=int,
        help="Override down projection rank (advanced)",
    )
    moe_expert_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Start fresh instead of resuming from checkpoint (for moe-overlay-compress)",
    )
    moe_expert_parser.set_defaults(func=introspect_moe_expert)

    # =========================================================================
    # Classifier Emergence Commands
    # =========================================================================

    # Multi-class classifier probe
    classifier_parser = introspect_subparsers.add_parser(
        "classifier",
        help="Train multi-class linear probe for operation classification",
        description="""Train logistic regression probes at each layer to find where
the model distinguishes between multiple operation types (e.g., multiply, add, subtract, divide).

Example:
  lazarus introspect classifier -m meta-llama/Llama-3.2-1B \\
    --classes "multiply:7*8=|12*5=" \\
    --classes "add:23+45=|17+38=" \\
    --classes "subtract:50-23=|89-34=" \\
    --classes "divide:48/6=|81/9=" \\
    --test "11*12=|11+12=|15-6=|12/4="
        """,
    )
    classifier_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    classifier_parser.add_argument(
        "--classes",
        "-c",
        action="append",
        required=True,
        help="Class definition in format 'label:prompt1|prompt2|...' (can specify multiple)",
    )
    classifier_parser.add_argument(
        "--test",
        "-t",
        help="Test prompts to classify (pipe-separated or @file.txt)",
    )
    classifier_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    classifier_parser.set_defaults(func=lambda args: asyncio.run(introspect_classifier(args)))

    # Logit lens analysis
    logit_lens_parser = introspect_subparsers.add_parser(
        "logit-lens",
        help="Apply logit lens to check vocabulary-mappable classifiers",
        description="""Project hidden states at specified layer through the unembedding
matrix to see which vocabulary tokens emerge. Useful for checking if classifiers
project to specific tokens (e.g., 'multiply', 'add').

Example:
  lazarus introspect logit-lens -m meta-llama/Llama-3.2-1B \\
    --prompts "7*8=|23+45=|50-23=|48/6=" \\
    --layer 8 \\
    --targets "multiply|add|subtract|divide"
        """,
    )
    logit_lens_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name or HuggingFace ID",
    )
    logit_lens_parser.add_argument(
        "--adapter",
        "-a",
        help="Path to LoRA adapter directory (for analyzing fine-tuned models)",
    )
    logit_lens_parser.add_argument(
        "--prompts",
        "-p",
        required=True,
        help="Prompts to analyze (pipe-separated or @file.txt)",
    )
    logit_lens_parser.add_argument(
        "--layer",
        "-l",
        type=int,
        help="Layer to analyze (default: 55%% depth)",
    )
    logit_lens_parser.add_argument(
        "--targets",
        "-t",
        action="append",
        help="Target tokens to track probability (can specify multiple)",
    )
    logit_lens_parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    logit_lens_parser.set_defaults(func=lambda args: asyncio.run(introspect_logit_lens(args)))

    return parser


def main():
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    elif args.command == "train" and args.train_type is None:
        parser.parse_args(["train", "--help"])
    elif args.command == "tokenizer" and args.tok_command is None:
        parser.parse_args(["tokenizer", "--help"])
    elif args.command == "gym" and getattr(args, "gym_command", None) is None:
        parser.parse_args(["gym", "--help"])
    elif args.command == "introspect" and getattr(args, "introspect_command", None) is None:
        parser.parse_args(["introspect", "--help"])
    elif args.command == "experiment" and getattr(args, "exp_command", None) is None:
        parser.parse_args(["experiment", "--help"])


if __name__ == "__main__":
    main()
