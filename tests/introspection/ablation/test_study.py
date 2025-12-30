"""Tests for ablation study module."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.ablation.adapter import ModelAdapter
from chuk_lazarus.introspection.ablation.config import AblationConfig, ComponentType
from chuk_lazarus.introspection.ablation.models import AblationResult, LayerSweepResult
from chuk_lazarus.introspection.ablation.study import AblationStudy


# Mock classes
class MockAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size)


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size)


class MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockBackbone(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]


class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._call_count = 0

    def __call__(self, input_ids):
        self._call_count += 1
        batch_size, seq_len = input_ids.shape
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        # Make EOS likely after a few tokens
        if self._call_count > 2:
            logits = logits.at[:, -1, 2].add(100.0)
        return MockModelOutput(logits)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.unk_token_id = 0

    def encode(self, text, **kwargs):
        return [[1, 2, 3]]

    def decode(self, ids, **kwargs):
        return "generated text output"

    def convert_tokens_to_ids(self, token):
        return 0


class MockConfig:
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size


class TestAblationStudy:
    """Tests for AblationStudy class."""

    def test_init(self):
        """Test initialization with adapter."""
        model = MockModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        adapter = ModelAdapter(model, tokenizer, config)
        study = AblationStudy(adapter)
        assert study.adapter is adapter

    def test_detect_family_gemma(self):
        """Test family detection for gemma models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "gemma", "architectures": ["GemmaForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "gemma"

    def test_detect_family_qwen(self):
        """Test family detection for qwen models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "qwen3"

    def test_detect_family_gpt_oss(self):
        """Test family detection for gpt_oss models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "gpt_oss"

    def test_detect_family_llama(self):
        """Test family detection for llama models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "llama"

    def test_detect_family_llama4(self):
        """Test family detection for llama4 models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "llama4", "architectures": ["Llama4ForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "llama4"

    def test_detect_family_mamba(self):
        """Test family detection for mamba models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "mamba", "architectures": ["MambaForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "mamba"

    def test_detect_family_jamba(self):
        """Test family detection for jamba models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "jamba", "architectures": ["JambaForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "jamba"

    def test_detect_family_granite(self):
        """Test family detection for granite models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "granite", "architectures": ["GraniteForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "granite"

    def test_detect_family_starcoder(self):
        """Test family detection for starcoder models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "starcoder2", "architectures": ["Starcoder2ForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "starcoder2"

    def test_detect_family_default(self):
        """Test family detection defaults to llama."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"model_type": "unknown", "architectures": ["UnknownForCausalLM"]})
            )
            family = AblationStudy._detect_family(tmpdir)
            assert family == "llama"

    def test_is_coherent_true(self):
        """Test coherence check for normal text."""
        assert AblationStudy._is_coherent("This is a normal response.") is True

    def test_is_coherent_escape_spam(self):
        """Test coherence check for escape sequence spam."""
        text = "<escape>" * 10
        assert AblationStudy._is_coherent(text) is False

    def test_is_coherent_newline_spam(self):
        """Test coherence check for newline spam."""
        text = "\n" * 25 + "short"
        assert AblationStudy._is_coherent(text) is False

    def test_is_coherent_low_diversity(self):
        """Test coherence check for repetitive text."""
        text = "aaaaaaaa" * 20
        assert AblationStudy._is_coherent(text) is False


class TestAblationStudyWithMockAdapter:
    """Tests for AblationStudy with mock adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = MockModel(num_layers=4, hidden_size=64)
        self.tokenizer = MockTokenizer()
        self.config = MockConfig(64)
        self.adapter = ModelAdapter(self.model, self.tokenizer, self.config)
        self.study = AblationStudy(self.adapter)

    def test_ablate_and_generate_no_layers(self):
        """Test ablation with no layers (baseline)."""
        output = self.study.ablate_and_generate("test prompt", layers=[])
        assert isinstance(output, str)

    def test_ablate_and_generate_single_layer_mlp(self):
        """Test ablation of single MLP layer."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[0], component=ComponentType.MLP
        )
        assert isinstance(output, str)

    def test_ablate_and_generate_single_layer_attention(self):
        """Test ablation of single attention layer."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[1], component=ComponentType.ATTENTION
        )
        assert isinstance(output, str)

    def test_ablate_and_generate_both_components(self):
        """Test ablation of both MLP and attention."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[0], component=ComponentType.BOTH
        )
        assert isinstance(output, str)

    def test_ablate_and_generate_multiple_layers(self):
        """Test ablation of multiple layers."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[0, 1, 2], component=ComponentType.MLP
        )
        assert isinstance(output, str)

    def test_ablate_and_generate_with_config(self):
        """Test ablation with custom config."""
        config = AblationConfig(max_new_tokens=10, temperature=0.5)
        output = self.study.ablate_and_generate("test prompt", layers=[0], config=config)
        assert isinstance(output, str)

    def test_ablate_and_generate_mlp_down(self):
        """Test ablation of MLP down projection specifically."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[0], component=ComponentType.MLP_DOWN
        )
        assert isinstance(output, str)

    def test_ablate_and_generate_attn_o(self):
        """Test ablation of attention output projection specifically."""
        output = self.study.ablate_and_generate(
            "test prompt", layers=[0], component=ComponentType.ATTN_O
        )
        assert isinstance(output, str)

    def test_run_layer_sweep(self):
        """Test running a layer sweep."""

        def criterion(text):
            return "generated" in text.lower()

        result = self.study.run_layer_sweep(
            prompt="test prompt",
            criterion=criterion,
            layers=[0, 1],
            component=ComponentType.MLP,
            task_name="test_task",
        )

        assert isinstance(result, LayerSweepResult)
        assert result.task_name == "test_task"
        assert len(result.results) == 2

    def test_run_layer_sweep_all_layers(self):
        """Test layer sweep with default all layers."""

        def criterion(text):
            return True

        result = self.study.run_layer_sweep(
            prompt="test prompt",
            criterion=criterion,
            component=ComponentType.MLP,
        )

        assert len(result.results) == 4  # num_layers

    def test_run_layer_sweep_with_config(self):
        """Test layer sweep with custom config."""

        def criterion(text):
            return True

        config = AblationConfig(max_new_tokens=5)
        result = self.study.run_layer_sweep(
            prompt="test prompt",
            criterion=criterion,
            layers=[0],
            config=config,
        )

        assert isinstance(result, LayerSweepResult)

    def test_run_multi_task_sweep(self):
        """Test multi-task sweep."""
        tasks = [
            ("task1", "prompt1", lambda x: True),
            ("task2", "prompt2", lambda x: False),
        ]

        results = self.study.run_multi_task_sweep(
            tasks=tasks,
            layers=[0],
            component=ComponentType.MLP,
        )

        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results

    def test_print_sweep_summary(self, capsys):
        """Test printing sweep summary."""
        results = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="b",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
                output_coherent=True,
            ),
            AblationResult(
                layer=1,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
                output_coherent=True,
            ),
        ]
        sweep = LayerSweepResult(
            task_name="test",
            criterion_name="criterion",
            results=results,
        )

        self.study.print_sweep_summary(sweep)
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "YES ***" in captured.out  # Changed layer

    def test_print_multi_task_matrix(self, capsys):
        """Test printing multi-task matrix."""
        results1 = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="b",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
            ),
        ]
        results2 = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
            ),
        ]
        task_results = {
            "task1": LayerSweepResult(
                task_name="task1",
                criterion_name="c1",
                results=results1,
            ),
            "task2": LayerSweepResult(
                task_name="task2",
                criterion_name="c2",
                results=results2,
            ),
        }

        self.study.print_multi_task_matrix(task_results)
        captured = capsys.readouterr()
        assert "CAUSALITY MATRIX" in captured.out

    def test_print_multi_task_matrix_empty(self, capsys):
        """Test printing multi-task matrix with empty results."""
        self.study.print_multi_task_matrix({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_save_results(self):
        """Test saving results to JSON."""
        results = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="original output text",
                ablated_output="ablated output text",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
                output_coherent=True,
            ),
        ]
        task_results = {
            "test_task": LayerSweepResult(
                task_name="test_task",
                criterion_name="test_criterion",
                results=results,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            self.study.save_results(task_results, output_path)

            assert output_path.exists()
            with open(output_path) as f:
                saved_data = json.load(f)

            assert "test_task" in saved_data
            assert saved_data["test_task"]["task_name"] == "test_task"
            assert saved_data["test_task"]["criterion_name"] == "test_criterion"
            assert len(saved_data["test_task"]["results"]) == 1


class TestAblationStudyLoadModel:
    """Tests for AblationStudy._load_model method."""

    def test_load_model_unsupported_family(self):
        """Test that unsupported family raises error."""
        with pytest.raises(ValueError, match="Unsupported model family"):
            AblationStudy._load_model("/nonexistent/path", "unsupported_family")
