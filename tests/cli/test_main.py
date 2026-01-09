"""Tests for main CLI entry point."""

import argparse
from unittest.mock import Mock, patch

import pytest

from chuk_lazarus.cli.main import app, main


class TestAppParser:
    """Tests for the app() function that creates the argument parser."""

    def test_app_returns_parser(self):
        """Test that app() returns an ArgumentParser."""
        parser = app()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_with_no_args(self):
        """Test that parser fails with no args."""
        parser = app()
        # Parse with no args should succeed (command is optional)
        args = parser.parse_args([])
        assert args.command is None

    def test_train_sft_parser(self):
        """Test train sft command parser."""
        parser = app()
        args = parser.parse_args(["train", "sft", "--model", "test-model", "--data", "test.jsonl"])
        assert args.command == "train"
        assert args.train_type == "sft"
        assert args.model == "test-model"
        assert args.data == "test.jsonl"
        assert args.epochs == 3
        assert args.batch_size == 4
        assert args.learning_rate == 1e-5
        assert args.max_length == 512
        assert args.use_lora is False
        assert args.lora_rank == 8
        assert args.mask_prompt is False
        assert args.log_interval == 10

    def test_train_sft_with_all_options(self):
        """Test train sft with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "train",
                "sft",
                "--model",
                "model",
                "--data",
                "data.jsonl",
                "--eval-data",
                "eval.jsonl",
                "--output",
                "./out",
                "--epochs",
                "5",
                "--batch-size",
                "8",
                "--learning-rate",
                "2e-5",
                "--max-length",
                "1024",
                "--use-lora",
                "--lora-rank",
                "16",
                "--mask-prompt",
                "--log-interval",
                "20",
                "--batchplan",
                "plan/",
                "--bucket-edges",
                "128,256,512",
                "--token-budget",
                "8192",
                "--pack",
                "--pack-max-len",
                "2048",
                "--pack-mode",
                "best_fit",
                "--online",
                "--gym-host",
                "example.com",
                "--gym-port",
                "9000",
                "--buffer-size",
                "50000",
            ]
        )
        assert args.eval_data == "eval.jsonl"
        assert args.output == "./out"
        assert args.epochs == 5
        assert args.batch_size == 8
        assert args.learning_rate == 2e-5
        assert args.max_length == 1024
        assert args.use_lora is True
        assert args.lora_rank == 16
        assert args.mask_prompt is True
        assert args.log_interval == 20
        assert args.batchplan == "plan/"
        assert args.bucket_edges == "128,256,512"
        assert args.token_budget == 8192
        assert args.pack is True
        assert args.pack_max_len == 2048
        assert args.pack_mode == "best_fit"
        assert args.online is True
        assert args.gym_host == "example.com"
        assert args.gym_port == 9000
        assert args.buffer_size == 50000

    def test_train_dpo_parser(self):
        """Test train dpo command parser."""
        parser = app()
        args = parser.parse_args(
            ["train", "dpo", "--model", "policy-model", "--data", "prefs.jsonl"]
        )
        assert args.command == "train"
        assert args.train_type == "dpo"
        assert args.model == "policy-model"
        assert args.data == "prefs.jsonl"
        assert args.epochs == 3
        assert args.batch_size == 4
        assert args.learning_rate == 1e-6
        assert args.beta == 0.1
        assert args.max_length == 512
        assert args.use_lora is False
        assert args.lora_rank == 8

    def test_train_dpo_with_all_options(self):
        """Test train dpo with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "train",
                "dpo",
                "--model",
                "policy",
                "--ref-model",
                "reference",
                "--data",
                "data.jsonl",
                "--eval-data",
                "eval.jsonl",
                "--output",
                "./dpo_out",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--learning-rate",
                "5e-7",
                "--beta",
                "0.2",
                "--max-length",
                "768",
                "--use-lora",
                "--lora-rank",
                "32",
            ]
        )
        assert args.ref_model == "reference"
        assert args.eval_data == "eval.jsonl"
        assert args.output == "./dpo_out"
        assert args.epochs == 2
        assert args.batch_size == 2
        assert args.learning_rate == 5e-7
        assert args.beta == 0.2
        assert args.max_length == 768
        assert args.use_lora is True
        assert args.lora_rank == 32

    def test_generate_parser(self):
        """Test generate command parser."""
        parser = app()
        args = parser.parse_args(["generate", "--type", "math"])
        assert args.command == "generate"
        assert args.type == "math"
        assert args.output == "./data/generated"
        assert args.sft_samples == 10000
        assert args.dpo_samples == 5000
        assert args.seed == 42

    def test_generate_with_options(self):
        """Test generate with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "generate",
                "--type",
                "math",
                "--output",
                "./custom",
                "--sft-samples",
                "5000",
                "--dpo-samples",
                "2500",
                "--seed",
                "123",
            ]
        )
        assert args.output == "./custom"
        assert args.sft_samples == 5000
        assert args.dpo_samples == 2500
        assert args.seed == 123

    def test_infer_parser(self):
        """Test infer command parser."""
        parser = app()
        args = parser.parse_args(["infer", "--model", "test-model", "--prompt", "Hello"])
        assert args.command == "infer"
        assert args.model == "test-model"
        assert args.prompt == "Hello"
        assert args.max_tokens == 256
        assert args.temperature == 0.7

    def test_infer_with_options(self):
        """Test infer with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "infer",
                "--model",
                "model",
                "--adapter",
                "lora",
                "--prompt-file",
                "prompts.txt",
                "--max-tokens",
                "512",
                "--temperature",
                "0.9",
            ]
        )
        assert args.adapter == "lora"
        assert args.prompt_file == "prompts.txt"
        assert args.max_tokens == 512
        assert args.temperature == 0.9

    def test_tokenizer_encode_parser(self):
        """Test tokenizer encode command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "encode", "-t", "gpt2", "--text", "hello"])
        assert args.command == "tokenizer"
        assert args.tok_command == "encode"
        assert args.tokenizer == "gpt2"
        assert args.text == "hello"
        assert args.special_tokens is False

    def test_tokenizer_encode_with_file(self):
        """Test tokenizer encode with file."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "encode",
                "-t",
                "gpt2",
                "-f",
                "input.txt",
                "--special-tokens",
            ]
        )
        assert args.file == "input.txt"
        assert args.special_tokens is True

    def test_tokenizer_decode_parser(self):
        """Test tokenizer decode command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "decode", "-t", "gpt2", "--ids", "1,2,3"])
        assert args.command == "tokenizer"
        assert args.tok_command == "decode"
        assert args.tokenizer == "gpt2"
        assert args.ids == "1,2,3"

    def test_tokenizer_vocab_parser(self):
        """Test tokenizer vocab command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "vocab", "-t", "gpt2"])
        assert args.tok_command == "vocab"
        assert args.tokenizer == "gpt2"
        assert args.show_all is False
        assert args.limit == 50
        assert args.chunk_size == 1000
        assert args.pause is False

    def test_tokenizer_vocab_with_search(self):
        """Test tokenizer vocab with search."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "vocab",
                "-t",
                "gpt2",
                "--show-all",
                "-s",
                "hello",
                "--limit",
                "100",
                "--chunk-size",
                "500",
                "--pause",
            ]
        )
        assert args.show_all is True
        assert args.search == "hello"
        assert args.limit == 100
        assert args.chunk_size == 500
        assert args.pause is True

    def test_tokenizer_compare_parser(self):
        """Test tokenizer compare command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "compare",
                "-t1",
                "gpt2",
                "-t2",
                "llama",
                "--text",
                "test",
            ]
        )
        assert args.tok_command == "compare"
        assert args.tokenizer1 == "gpt2"
        assert args.tokenizer2 == "llama"
        assert args.text == "test"
        assert args.verbose is False

    def test_tokenizer_doctor_parser(self):
        """Test tokenizer doctor command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "doctor", "-t", "gpt2"])
        assert args.tok_command == "doctor"
        assert args.tokenizer == "gpt2"
        assert args.verbose is False
        assert args.fix is False

    def test_tokenizer_doctor_with_fix(self):
        """Test tokenizer doctor with fix options."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "doctor",
                "-t",
                "gpt2",
                "-v",
                "--fix",
                "--format",
                "chatml",
                "-o",
                "patched/",
            ]
        )
        assert args.verbose is True
        assert args.fix is True
        assert args.format == "chatml"
        assert args.output == "patched/"

    def test_tokenizer_fingerprint_parser(self):
        """Test tokenizer fingerprint command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "fingerprint", "-t", "gpt2"])
        assert args.tok_command == "fingerprint"
        assert args.tokenizer == "gpt2"

    def test_tokenizer_fingerprint_save_verify(self):
        """Test tokenizer fingerprint with save and verify."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "fingerprint",
                "-t",
                "gpt2",
                "-s",
                "fp.json",
                "--verify",
                "fp2.json",
                "--strict",
            ]
        )
        assert args.save == "fp.json"
        assert args.verify == "fp2.json"
        assert args.strict is True

    def test_tokenizer_benchmark_parser(self):
        """Test tokenizer benchmark command."""
        parser = app()
        args = parser.parse_args(["tokenizer", "benchmark", "-t", "gpt2"])
        assert args.tok_command == "benchmark"
        assert args.tokenizer == "gpt2"
        assert args.samples == 1000
        assert args.avg_length == 100
        assert args.seed == 42
        assert args.workers == 1
        assert args.warmup == 10
        assert args.special_tokens is False
        assert args.compare is False

    def test_tokenizer_benchmark_with_options(self):
        """Test tokenizer benchmark with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "benchmark",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "-n",
                "5000",
                "--avg-length",
                "200",
                "--seed",
                "999",
                "-w",
                "4",
                "--warmup",
                "50",
                "--special-tokens",
                "-c",
            ]
        )
        assert args.file == "corpus.txt"
        assert args.samples == 5000
        assert args.avg_length == 200
        assert args.seed == 999
        assert args.workers == 4
        assert args.warmup == 50
        assert args.special_tokens is True
        assert args.compare is True

    def test_tokenizer_analyze_coverage(self):
        """Test tokenizer analyze coverage command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "coverage",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--fragments",
            ]
        )
        assert args.tok_command == "analyze"
        assert args.analyze_command == "coverage"
        assert args.tokenizer == "gpt2"
        assert args.file == "corpus.txt"
        assert args.fragments is True

    def test_tokenizer_analyze_entropy(self):
        """Test tokenizer analyze entropy command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "entropy",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--top-n",
                "50",
            ]
        )
        assert args.analyze_command == "entropy"
        assert args.top_n == 50

    def test_tokenizer_analyze_fit_score(self):
        """Test tokenizer analyze fit-score command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "fit-score",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
            ]
        )
        assert args.analyze_command == "fit-score"

    def test_tokenizer_analyze_diff(self):
        """Test tokenizer analyze diff command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "diff",
                "-t1",
                "gpt2",
                "-t2",
                "llama",
                "-f",
                "corpus.txt",
            ]
        )
        assert args.analyze_command == "diff"
        assert args.tokenizer1 == "gpt2"
        assert args.tokenizer2 == "llama"

    def test_tokenizer_analyze_efficiency(self):
        """Test tokenizer analyze efficiency command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "efficiency",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
            ]
        )
        assert args.analyze_command == "efficiency"

    def test_tokenizer_analyze_vocab_suggest(self):
        """Test tokenizer analyze vocab-suggest command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "analyze",
                "vocab-suggest",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--min-freq",
                "10",
                "--min-frag",
                "5",
                "--limit",
                "100",
                "--show",
                "30",
            ]
        )
        assert args.analyze_command == "vocab-suggest"
        assert args.min_freq == 10
        assert args.min_frag == 5
        assert args.limit == 100
        assert args.show == 30

    def test_tokenizer_curriculum_length_buckets(self):
        """Test tokenizer curriculum length-buckets command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "curriculum",
                "length-buckets",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--num-buckets",
                "10",
                "--schedule",
            ]
        )
        assert args.curriculum_command == "length-buckets"
        assert args.num_buckets == 10
        assert args.schedule is True

    def test_tokenizer_curriculum_reasoning_density(self):
        """Test tokenizer curriculum reasoning-density command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "curriculum",
                "reasoning-density",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--descending",
            ]
        )
        assert args.curriculum_command == "reasoning-density"
        assert args.descending is True

    def test_tokenizer_training_throughput(self):
        """Test tokenizer training throughput command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "training",
                "throughput",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
            ]
        )
        assert args.training_command == "throughput"

    def test_tokenizer_training_pack(self):
        """Test tokenizer training pack command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "training",
                "pack",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--max-length",
                "1024",
            ]
        )
        assert args.training_command == "pack"
        assert args.max_length == 1024

    def test_tokenizer_regression_run(self):
        """Test tokenizer regression run command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "regression",
                "run",
                "-t",
                "gpt2",
                "--tests",
                "tests.yaml",
            ]
        )
        assert args.regression_command == "run"
        assert args.tests == "tests.yaml"

    def test_tokenizer_runtime_registry(self):
        """Test tokenizer runtime registry command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "runtime",
                "registry",
                "-t",
                "gpt2",
                "--standard",
            ]
        )
        assert args.runtime_command == "registry"
        assert args.standard is True

    def test_tokenizer_research_soft_tokens(self):
        """Test tokenizer research soft-tokens command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "research",
                "soft-tokens",
                "-n",
                "20",
                "-d",
                "1024",
                "-p",
                "task",
                "--init-method",
                "random_uniform",
                "--init-std",
                "0.01",
                "-o",
                "soft.json",
            ]
        )
        assert args.research_command == "soft-tokens"
        assert args.num_tokens == 20
        assert args.embedding_dim == 1024
        assert args.prefix == "task"
        assert args.init_method == "random_uniform"
        assert args.init_std == 0.01
        assert args.output == "soft.json"

    def test_tokenizer_research_analyze_embeddings(self):
        """Test tokenizer research analyze-embeddings command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "research",
                "analyze-embeddings",
                "-f",
                "emb.json",
                "-k",
                "5",
                "--cluster",
                "--project",
            ]
        )
        assert args.research_command == "analyze-embeddings"
        assert args.file == "emb.json"
        assert args.num_clusters == 5
        assert args.cluster is True
        assert args.project is True

    def test_tokenizer_research_morph(self):
        """Test tokenizer research morph command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "research",
                "morph",
                "-f",
                "emb.json",
                "-s",
                "0",
                "-t",
                "1",
                "-m",
                "spherical",
                "--steps",
                "20",
                "--normalize",
                "-o",
                "trajectory.json",
            ]
        )
        assert args.research_command == "morph"
        assert args.source == 0
        assert args.target == 1
        assert args.method == "spherical"
        assert args.steps == 20
        assert args.normalize is True
        assert args.output == "trajectory.json"

    def test_tokenizer_instrument_histogram(self):
        """Test tokenizer instrument histogram command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "instrument",
                "histogram",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--bins",
                "30",
                "--width",
                "100",
                "--quick",
            ]
        )
        assert args.instrument_command == "histogram"
        assert args.bins == 30
        assert args.width == 100
        assert args.quick is True

    def test_tokenizer_instrument_oov(self):
        """Test tokenizer instrument oov command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "instrument",
                "oov",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--vocab-size",
                "30000",
                "--show-rare",
                "--max-freq",
                "10",
                "--top-k",
                "50",
            ]
        )
        assert args.instrument_command == "oov"
        assert args.vocab_size == 30000
        assert args.show_rare is True
        assert args.max_freq == 10
        assert args.top_k == 50

    def test_tokenizer_instrument_waste(self):
        """Test tokenizer instrument waste command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "instrument",
                "waste",
                "-t",
                "gpt2",
                "-f",
                "corpus.txt",
                "--max-length",
                "2048",
            ]
        )
        assert args.instrument_command == "waste"
        assert args.max_length == 2048

    def test_tokenizer_instrument_vocab_diff(self):
        """Test tokenizer instrument vocab-diff command."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "instrument",
                "vocab-diff",
                "-t1",
                "gpt2",
                "-t2",
                "llama",
                "-f",
                "corpus.txt",
                "--examples",
                "10",
                "--cost",
            ]
        )
        assert args.instrument_command == "vocab-diff"
        assert args.examples == 10
        assert args.cost is True

    def test_data_lengths_build(self):
        """Test data lengths build command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "lengths",
                "build",
                "-d",
                "train.jsonl",
                "-t",
                "gpt2",
                "-o",
                "lengths.jsonl",
            ]
        )
        assert args.command == "data"
        assert args.data_command == "lengths"
        assert args.lengths_command == "build"
        assert args.dataset == "train.jsonl"
        assert args.tokenizer == "gpt2"
        assert args.output == "lengths.jsonl"

    def test_data_lengths_stats(self):
        """Test data lengths stats command."""
        parser = app()
        args = parser.parse_args(["data", "lengths", "stats", "-c", "lengths.jsonl"])
        assert args.lengths_command == "stats"
        assert args.cache == "lengths.jsonl"

    def test_data_batchplan_build(self):
        """Test data batchplan build command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batchplan",
                "build",
                "-l",
                "lengths.jsonl",
                "-e",
                "3",
                "-b",
                "8192",
                "--bucket-edges",
                "64,128,256",
                "--overflow-max",
                "1024",
                "-p",
                "-s",
                "99",
                "--dataset-hash",
                "abc123",
                "-o",
                "plan/",
            ]
        )
        assert args.batchplan_command == "build"
        assert args.lengths == "lengths.jsonl"
        assert args.epochs == 3
        assert args.token_budget == 8192
        assert args.bucket_edges == "64,128,256"
        assert args.overflow_max == 1024
        assert args.predictable is True
        assert args.seed == 99
        assert args.dataset_hash == "abc123"
        assert args.output == "plan/"

    def test_data_batchplan_info(self):
        """Test data batchplan info command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batchplan",
                "info",
                "-p",
                "plan/",
                "-n",
                "5",
                "-r",
                "0",
                "-w",
                "4",
            ]
        )
        assert args.batchplan_command == "info"
        assert args.plan == "plan/"
        assert args.show_batches == 5
        assert args.rank == 0
        assert args.world_size == 4

    def test_data_batchplan_verify(self):
        """Test data batchplan verify command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batchplan",
                "verify",
                "-p",
                "plan/",
                "-l",
                "lengths.jsonl",
            ]
        )
        assert args.batchplan_command == "verify"
        assert args.plan == "plan/"
        assert args.lengths == "lengths.jsonl"

    def test_data_batchplan_shard(self):
        """Test data batchplan shard command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batchplan",
                "shard",
                "-p",
                "plan/",
                "-w",
                "8",
                "-o",
                "shards/",
            ]
        )
        assert args.batchplan_command == "shard"
        assert args.plan == "plan/"
        assert args.world_size == 8
        assert args.output == "shards/"

    def test_data_batching_analyze(self):
        """Test data batching analyze command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batching",
                "analyze",
                "-c",
                "lengths.jsonl",
                "--bucket-edges",
                "100,200,300",
                "--overflow-max",
                "4096",
                "-o",
                "report.json",
            ]
        )
        assert args.batching_command == "analyze"
        assert args.cache == "lengths.jsonl"
        assert args.bucket_edges == "100,200,300"
        assert args.overflow_max == 4096
        assert args.output == "report.json"

    def test_data_batching_histogram(self):
        """Test data batching histogram command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batching",
                "histogram",
                "-c",
                "lengths.jsonl",
                "--bins",
                "20",
                "--width",
                "60",
            ]
        )
        assert args.batching_command == "histogram"
        assert args.bins == 20
        assert args.width == 60

    def test_data_batching_suggest(self):
        """Test data batching suggest command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batching",
                "suggest",
                "-c",
                "lengths.jsonl",
                "-n",
                "6",
                "-g",
                "balance",
                "--max-length",
                "4096",
            ]
        )
        assert args.batching_command == "suggest"
        assert args.num_buckets == 6
        assert args.goal == "balance"
        assert args.max_length == 4096

    def test_data_batch_generate(self):
        """Test data batch generate command."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batch",
                "generate",
                "-p",
                "plan/",
                "-d",
                "train.jsonl",
                "-t",
                "gpt2",
                "-o",
                "batches/",
            ]
        )
        assert args.batch_command == "generate"
        assert args.plan == "plan/"
        assert args.dataset == "train.jsonl"
        assert args.tokenizer == "gpt2"
        assert args.output == "batches/"

    def test_gym_run(self):
        """Test gym run command."""
        parser = app()
        args = parser.parse_args(
            [
                "gym",
                "run",
                "-t",
                "gpt2",
                "--mock",
                "--num-episodes",
                "5",
                "--host",
                "example.com",
                "--port",
                "9000",
                "--output",
                "buffer.json",
            ]
        )
        assert args.command == "gym"
        assert args.gym_command == "run"
        assert args.tokenizer == "gpt2"
        assert args.mock is True
        assert args.num_episodes == 5
        assert args.host == "example.com"
        assert args.port == 9000
        assert args.output == "buffer.json"

    def test_gym_info(self):
        """Test gym info command."""
        parser = app()
        args = parser.parse_args(["gym", "info"])
        assert args.gym_command == "info"

    def test_bench_command(self):
        """Test bench command."""
        parser = app()
        args = parser.parse_args(["bench"])
        assert args.command == "bench"

    def test_bench_with_options(self):
        """Test bench with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "bench",
                "-d",
                "train.jsonl",
                "-t",
                "gpt2",
                "--bucket-edges",
                "128,256,512",
                "--token-budget",
                "8192",
            ]
        )
        assert args.dataset == "train.jsonl"
        assert args.tokenizer == "gpt2"
        assert args.bucket_edges == "128,256,512"
        assert args.token_budget == 8192

    def test_introspect_analyze(self):
        """Test introspect analyze command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "analyze",
                "-m",
                "model",
                "-p",
                "prompt",
                "--track",
                "token1",
                "--track",
                "token2",
                "--layer-strategy",
                "all",
            ]
        )
        assert args.command == "introspect"
        assert args.introspect_command == "analyze"
        assert args.model == "model"
        assert args.prompt == "prompt"
        # track uses action="append" so it's a list
        assert args.track == ["token1", "token2"]
        assert args.layer_strategy == "all"

    def test_introspect_compare(self):
        """Test introspect compare command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "compare",
                "-m1",
                "model1",
                "-m2",
                "model2",
                "-p",
                "prompt",
                "--track",
                "token",
            ]
        )
        assert args.introspect_command == "compare"
        assert args.model1 == "model1"
        assert args.model2 == "model2"

    def test_introspect_hooks(self):
        """Test introspect hooks command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "hooks",
                "-m",
                "model",
                "-p",
                "prompt",
                "--layers",
                "0,4,8",
                "--capture-attention",
            ]
        )
        assert args.introspect_command == "hooks"
        assert args.layers == "0,4,8"
        assert args.capture_attention is True

    def test_introspect_ablate(self):
        """Test introspect ablate command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "ablate",
                "-m",
                "model",
                "-p",
                "prompt",
                "-c",
                "function_call",
                "--component",
                "mlp",
                "--layers",
                "5,8,10",
            ]
        )
        assert args.introspect_command == "ablate"
        assert args.criterion == "function_call"
        assert args.component == "mlp"
        assert args.layers == "5,8,10"

    def test_introspect_weight_diff(self):
        """Test introspect weight-diff command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "weight-diff",
                "-b",
                "base",
                "-f",
                "finetuned",
                "-o",
                "diff.json",
            ]
        )
        assert args.introspect_command == "weight-diff"
        assert args.base == "base"
        assert args.finetuned == "finetuned"
        assert args.output == "diff.json"

    def test_introspect_activation_diff(self):
        """Test introspect activation-diff command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "activation-diff",
                "-b",
                "base",
                "-f",
                "finetuned",
                "-p",
                "prompt1,prompt2",
            ]
        )
        assert args.introspect_command == "activation-diff"

    def test_introspect_layer(self):
        """Test introspect layer command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "layer",
                "-m",
                "model",
                "-p",
                "prompt1|prompt2",
                "--layers",
                "5,10",
            ]
        )
        assert args.introspect_command == "layer"
        assert args.prompts == "prompt1|prompt2"
        assert args.layers == "5,10"

    def test_introspect_format_sensitivity(self):
        """Test introspect format-sensitivity command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "format-sensitivity",
                "-m",
                "model",
                "-p",
                "prompt1|prompt2",
            ]
        )
        assert args.introspect_command == "format-sensitivity"
        assert args.prompts == "prompt1|prompt2"

    def test_introspect_generate(self):
        """Test introspect generate command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "generate",
                "-m",
                "model",
                "-p",
                "prompt1|prompt2",
                "--max-tokens",
                "100",
                "--temperature",
                "0.8",
            ]
        )
        assert args.introspect_command == "generate"
        assert args.prompts == "prompt1|prompt2"
        assert args.max_tokens == 100
        assert args.temperature == 0.8

    def test_introspect_metacognitive(self):
        """Test introspect metacognitive command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "metacognitive",
                "-m",
                "model",
                "-p",
                "prompt",
            ]
        )
        assert args.introspect_command == "metacognitive"

    def test_introspect_steer(self):
        """Test introspect steer command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "steer",
                "-m",
                "model",
                "--extract",
                "--positive",
                "good",
                "--negative",
                "bad",
                "-o",
                "direction.npz",
            ]
        )
        assert args.introspect_command == "steer"
        assert args.extract is True
        assert args.positive == "good"
        assert args.negative == "bad"
        assert args.output == "direction.npz"

    def test_introspect_arithmetic(self):
        """Test introspect arithmetic command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "arithmetic",
                "-m",
                "model",
                "--quick",
            ]
        )
        assert args.introspect_command == "arithmetic"
        assert args.quick is True

    def test_introspect_uncertainty(self):
        """Test introspect uncertainty command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "uncertainty",
                "-m",
                "model",
                "-p",
                "prompt",
            ]
        )
        assert args.introspect_command == "uncertainty"

    def test_introspect_probe(self):
        """Test introspect probe command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "probe",
                "-m",
                "model",
                "--class-a",
                "prompt1|prompt2",
                "--class-b",
                "prompt3|prompt4",
            ]
        )
        assert args.introspect_command == "probe"
        assert args.class_a == "prompt1|prompt2"
        assert args.class_b == "prompt3|prompt4"

    def test_introspect_neurons(self):
        """Test introspect neurons command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "neurons",
                "-m",
                "model",
                "-p",
                "prompt1|prompt2",
            ]
        )
        assert args.introspect_command == "neurons"
        assert args.prompts == "prompt1|prompt2"

    def test_introspect_activation_cluster(self):
        """Test introspect cluster command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "cluster",
                "-m",
                "model",
                "--class-a",
                "prompt1|prompt2",
                "--class-b",
                "prompt3|prompt4",
            ]
        )
        assert args.introspect_command == "cluster"
        assert args.class_a == "prompt1|prompt2"
        assert args.class_b == "prompt3|prompt4"

    def test_introspect_memory(self):
        """Test introspect memory command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "memory",
                "-m",
                "model",
                "-f",
                "multiplication",
            ]
        )
        assert args.introspect_command == "memory"
        assert args.facts == "multiplication"

    def test_introspect_memory_inject(self):
        """Test introspect memory-inject command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "memory-inject",
                "-m",
                "model",
                "-f",
                "multiplication",
                "-q",
                "7*8=",
            ]
        )
        assert args.introspect_command == "memory-inject"
        assert args.facts == "multiplication"
        assert args.query == "7*8="

    def test_introspect_directions(self):
        """Test introspect directions command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "directions",
                "dir1.npz",
                "dir2.npz",
            ]
        )
        assert args.introspect_command == "directions"
        assert args.files == ["dir1.npz", "dir2.npz"]

    def test_introspect_operand_directions(self):
        """Test introspect operand-directions command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "operand-directions",
                "-m",
                "model",
            ]
        )
        assert args.introspect_command == "operand-directions"

    def test_introspect_embedding(self):
        """Test introspect embedding command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "embedding",
                "-m",
                "model",
                "--operation",
                "mult",
            ]
        )
        assert args.introspect_command == "embedding"
        assert args.operation == "mult"

    def test_introspect_commutativity(self):
        """Test introspect commutativity command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "commutativity",
                "-m",
                "model",
            ]
        )
        assert args.introspect_command == "commutativity"

    def test_introspect_patch(self):
        """Test introspect patch command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "patch",
                "-m",
                "model",
                "-s",
                "7*8=",
                "-t",
                "7+8=",
            ]
        )
        assert args.introspect_command == "patch"
        assert args.source == "7*8="
        assert args.target == "7+8="

    def test_introspect_early_layers(self):
        """Test introspect early-layers command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "early-layers",
                "-m",
                "model",
            ]
        )
        assert args.introspect_command == "early-layers"

    def test_introspect_circuit_capture(self):
        """Test introspect circuit capture command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "capture",
                "-m",
                "model",
                "-p",
                "prompt1|prompt2",
                "-l",
                "10",
                "-o",
                "circuit.npz",
            ]
        )
        assert args.introspect_command == "circuit"
        assert args.circuit_command == "capture"
        assert args.prompts == "prompt1|prompt2"
        assert args.layer == 10
        assert args.save == "circuit.npz"

    def test_introspect_circuit_invoke(self):
        """Test introspect circuit invoke command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "invoke",
                "-c",
                "circuit.npz",
                "--operands",
                "5,6",
            ]
        )
        assert args.circuit_command == "invoke"
        assert args.circuit == "circuit.npz"
        assert args.operands == "5,6"

    def test_introspect_circuit_decode(self):
        """Test introspect circuit decode command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "decode",
                "-m",
                "model",
                "-p",
                "prompt",
                "-i",
                "activations.npz",
            ]
        )
        assert args.circuit_command == "decode"
        assert args.prompt == "prompt"
        assert args.inject == "activations.npz"

    def test_introspect_circuit_test(self):
        """Test introspect circuit test command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "test",
                "-m",
                "model",
                "-c",
                "circuit.json",
            ]
        )
        assert args.circuit_command == "test"

    def test_introspect_circuit_compare(self):
        """Test introspect circuit compare command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "compare",
                "-c",
                "circuit1.npz",
                "circuit2.npz",
            ]
        )
        assert args.circuit_command == "compare"
        assert args.circuits == ["circuit1.npz", "circuit2.npz"]

    def test_introspect_circuit_view(self):
        """Test introspect circuit view command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "view",
                "-c",
                "circuit.npz",
            ]
        )
        assert args.circuit_command == "view"
        assert args.circuit == "circuit.npz"

    def test_introspect_virtual_expert(self):
        """Test introspect virtual-expert command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "virtual-expert",
                "-m",
                "model",
                "-p",
                "prompt",
            ]
        )
        assert args.introspect_command == "virtual-expert"

    def test_introspect_moe_expert(self):
        """Test introspect moe-expert command."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "moe-expert",
                "-m",
                "model",
                "-p",
                "prompt",
            ]
        )
        assert args.introspect_command == "moe-expert"


class TestMainFunction:
    """Tests for the main() entry point function."""

    @patch("chuk_lazarus.cli.main.app")
    def test_main_no_command_prints_help(self, mock_app):
        """Test main() prints help when no command is provided."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = Mock(command=None)
        mock_app.return_value = mock_parser

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        mock_parser.print_help.assert_called_once()

    @patch("chuk_lazarus.cli.main.app")
    def test_main_with_func_calls_it(self, mock_app):
        """Test main() calls func when it exists."""
        mock_func = Mock()
        mock_args = Mock(command="train", func=mock_func)
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_app.return_value = mock_parser

        main()

        mock_func.assert_called_once_with(mock_args)

    @patch("chuk_lazarus.cli.main.app")
    def test_main_train_without_type_shows_help(self, mock_app):
        """Test main() shows train help when train_type is None."""
        mock_args = Mock(command="train", train_type=None)
        delattr(mock_args, "func")
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_app.return_value = mock_parser

        main()

        # Should call parse_args again with train --help
        assert mock_parser.parse_args.call_count == 2
        mock_parser.parse_args.assert_any_call(["train", "--help"])

    @patch("chuk_lazarus.cli.main.app")
    def test_main_tokenizer_without_command_shows_help(self, mock_app):
        """Test main() shows tokenizer help when tok_command is None."""
        mock_args = Mock(command="tokenizer", tok_command=None)
        delattr(mock_args, "func")
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_app.return_value = mock_parser

        main()

        assert mock_parser.parse_args.call_count == 2
        mock_parser.parse_args.assert_any_call(["tokenizer", "--help"])

    @patch("chuk_lazarus.cli.main.app")
    def test_main_gym_without_command_shows_help(self, mock_app):
        """Test main() shows gym help when gym_command is None."""
        mock_args = Mock(command="gym", gym_command=None)
        delattr(mock_args, "func")
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_app.return_value = mock_parser

        main()

        assert mock_parser.parse_args.call_count == 2
        mock_parser.parse_args.assert_any_call(["gym", "--help"])

    @patch("chuk_lazarus.cli.main.app")
    def test_main_introspect_without_command_shows_help(self, mock_app):
        """Test main() shows introspect help when introspect_command is None."""
        mock_args = Mock(command="introspect", introspect_command=None)
        delattr(mock_args, "func")
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_app.return_value = mock_parser

        main()

        assert mock_parser.parse_args.call_count == 2
        mock_parser.parse_args.assert_any_call(["introspect", "--help"])


class TestCommandFunctionMapping:
    """Tests to verify that commands have their func set correctly.

    Note: The CLI now uses lambdas to wrap async handlers, so we verify
    that func is callable rather than checking exact function equality.
    """

    def test_train_sft_has_func(self):
        """Test train sft sets a callable func."""
        parser = app()
        args = parser.parse_args(["train", "sft", "--model", "m", "--data", "d"])

        assert hasattr(args, "func")
        assert callable(args.func)

    def test_train_dpo_has_func(self):
        """Test train dpo sets a callable func."""
        parser = app()
        args = parser.parse_args(["train", "dpo", "--model", "m", "--data", "d"])

        assert hasattr(args, "func")
        assert callable(args.func)

    def test_generate_has_func(self):
        """Test generate sets a callable func."""
        parser = app()
        args = parser.parse_args(["generate", "--type", "math"])

        assert hasattr(args, "func")
        assert callable(args.func)

    def test_infer_has_func(self):
        """Test infer sets a callable func."""
        parser = app()
        args = parser.parse_args(["infer", "--model", "m", "--prompt", "p"])

        assert hasattr(args, "func")
        assert callable(args.func)

    def test_tokenizer_encode_has_func(self):
        """Test tokenizer encode sets the correct func."""
        parser = app()
        args = parser.parse_args(["tokenizer", "encode", "-t", "gpt2", "--text", "hi"])
        from chuk_lazarus.cli.commands.tokenizer import tokenizer_encode

        assert hasattr(args, "func")
        assert args.func == tokenizer_encode

    def test_tokenizer_decode_has_func(self):
        """Test tokenizer decode sets the correct func."""
        parser = app()
        args = parser.parse_args(["tokenizer", "decode", "-t", "gpt2", "--ids", "1"])
        from chuk_lazarus.cli.commands.tokenizer import tokenizer_decode

        assert hasattr(args, "func")
        assert args.func == tokenizer_decode

    def test_gym_run_has_func(self):
        """Test gym run sets the correct func."""
        parser = app()
        args = parser.parse_args(["gym", "run", "-t", "gpt2"])
        from chuk_lazarus.cli.commands.gym import gym_run

        assert hasattr(args, "func")
        assert args.func == gym_run

    def test_gym_info_has_func(self):
        """Test gym info sets the correct func."""
        parser = app()
        args = parser.parse_args(["gym", "info"])
        from chuk_lazarus.cli.commands.gym import gym_info

        assert hasattr(args, "func")
        assert args.func == gym_info

    def test_bench_has_func(self):
        """Test bench sets the correct func."""
        parser = app()
        args = parser.parse_args(["bench"])
        from chuk_lazarus.cli.commands.gym import bench_pipeline

        assert hasattr(args, "func")
        assert args.func == bench_pipeline

    def test_data_lengths_build_has_func(self):
        """Test data lengths build sets the correct func."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "lengths",
                "build",
                "-d",
                "d",
                "-t",
                "t",
                "-o",
                "o",
            ]
        )
        from chuk_lazarus.cli.commands.data import data_lengths_build

        assert hasattr(args, "func")
        assert args.func == data_lengths_build

    def test_introspect_analyze_has_func(self):
        """Test introspect analyze sets the correct func."""
        parser = app()
        args = parser.parse_args(["introspect", "analyze", "-m", "m", "-p", "p"])
        from chuk_lazarus.cli.commands.introspect import introspect_analyze

        assert hasattr(args, "func")
        assert args.func == introspect_analyze

    def test_introspect_steer_has_func(self):
        """Test introspect steer sets the correct func."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "steer",
                "-m",
                "m",
                "--extract",
                "--positive",
                "p",
                "--negative",
                "n",
                "-o",
                "o",
            ]
        )
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        assert hasattr(args, "func")
        assert args.func == introspect_steer

    def test_introspect_circuit_capture_has_func(self):
        """Test introspect circuit capture sets the correct func."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "circuit",
                "capture",
                "-m",
                "m",
                "-p",
                "p",
                "-l",
                "10",
                "-o",
                "c.npz",
            ]
        )
        from chuk_lazarus.cli.commands.introspect import introspect_circuit_capture

        assert hasattr(args, "func")
        assert args.func == introspect_circuit_capture


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_parser_with_invalid_train_type(self):
        """Test parser rejects invalid train type."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "invalid"])

    def test_parser_with_invalid_generate_type(self):
        """Test parser rejects invalid generate type."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--type", "invalid"])

    def test_parser_with_invalid_pack_mode(self):
        """Test parser rejects invalid pack mode."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "train",
                    "sft",
                    "--model",
                    "m",
                    "--data",
                    "d",
                    "--pack-mode",
                    "invalid",
                ]
            )

    def test_parser_with_invalid_doctor_format(self):
        """Test parser rejects invalid doctor format."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "tokenizer",
                    "doctor",
                    "-t",
                    "gpt2",
                    "--format",
                    "invalid",
                ]
            )

    def test_parser_with_invalid_init_method(self):
        """Test parser rejects invalid init method."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "tokenizer",
                    "research",
                    "soft-tokens",
                    "--init-method",
                    "invalid",
                ]
            )

    def test_parser_with_invalid_morph_method(self):
        """Test parser rejects invalid morph method."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "tokenizer",
                    "research",
                    "morph",
                    "-f",
                    "emb.json",
                    "-s",
                    "0",
                    "-t",
                    "1",
                    "-m",
                    "invalid",
                ]
            )

    def test_parser_with_invalid_batching_goal(self):
        """Test parser rejects invalid batching goal."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "data",
                    "batching",
                    "suggest",
                    "-c",
                    "cache",
                    "-g",
                    "invalid",
                ]
            )

    def test_missing_required_model(self):
        """Test parser requires model argument."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "sft", "--data", "d"])

    def test_missing_required_data(self):
        """Test parser requires data argument."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "sft", "--model", "m"])

    def test_missing_required_tokenizer(self):
        """Test parser requires tokenizer argument."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["tokenizer", "encode", "--text", "hello"])

    def test_missing_required_ids(self):
        """Test parser requires ids argument for decode."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["tokenizer", "decode", "-t", "gpt2"])


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_train_sft_integration(self):
        """Test complete train sft flow - argument parsing only."""
        parser = app()
        args = parser.parse_args(
            [
                "train",
                "sft",
                "--model",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--data",
                "train.jsonl",
                "--epochs",
                "5",
                "--batch-size",
                "8",
            ]
        )

        # Verify the arguments are parsed correctly
        assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert args.data == "train.jsonl"
        assert args.epochs == 5
        assert args.batch_size == 8
        assert hasattr(args, "func")

    def test_tokenizer_encode_integration(self):
        """Test complete tokenizer encode flow - argument parsing only."""
        parser = app()
        args = parser.parse_args(
            [
                "tokenizer",
                "encode",
                "-t",
                "gpt2",
                "--text",
                "Hello world",
                "--special-tokens",
            ]
        )

        # Verify the arguments are parsed correctly
        assert args.tokenizer == "gpt2"
        assert args.text == "Hello world"
        assert args.special_tokens is True
        assert hasattr(args, "func")

    def test_gym_run_integration(self):
        """Test complete gym run flow - argument parsing only."""
        parser = app()
        args = parser.parse_args(
            [
                "gym",
                "run",
                "-t",
                "gpt2",
                "--mock",
                "--num-episodes",
                "10",
                "--output",
                "buffer.json",
            ]
        )

        # Verify the arguments are parsed correctly
        assert args.tokenizer == "gpt2"
        assert args.mock is True
        assert args.num_episodes == 10
        assert args.output == "buffer.json"
        assert hasattr(args, "func")

    def test_data_batchplan_build_integration(self):
        """Test complete data batchplan build flow - argument parsing only."""
        parser = app()
        args = parser.parse_args(
            [
                "data",
                "batchplan",
                "build",
                "-l",
                "lengths.jsonl",
                "-e",
                "3",
                "-b",
                "4096",
                "-o",
                "plan/",
            ]
        )

        # Verify the arguments are parsed correctly
        assert args.lengths == "lengths.jsonl"
        assert args.epochs == 3
        assert args.token_budget == 4096
        assert args.output == "plan/"
        assert hasattr(args, "func")

    def test_introspect_analyze_integration(self):
        """Test complete introspect analyze flow - argument parsing only."""
        parser = app()
        args = parser.parse_args(
            [
                "introspect",
                "analyze",
                "-m",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "-p",
                "The capital of France is",
                "--track",
                "Paris",
            ]
        )

        # Verify the arguments are parsed correctly
        assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert args.prompt == "The capital of France is"
        assert args.track == ["Paris"]
        assert hasattr(args, "func")
