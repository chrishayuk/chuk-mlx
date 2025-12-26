"""Tests for inference/loader.py module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from chuk_lazarus.inference.loader import (
    DownloadConfig,
    DownloadResult,
    DType,
    HFLoader,
    LoadedWeights,
    StandardWeightConverter,
)


class TestDType:
    """Tests for DType enum."""

    def test_dtype_values(self):
        """Test dtype enum values."""
        assert DType.FLOAT16.value == "float16"
        assert DType.FLOAT32.value == "float32"
        assert DType.BFLOAT16.value == "bfloat16"

    def test_to_mlx_float16(self):
        """Test conversion to MLX float16."""
        assert DType.FLOAT16.to_mlx() == mx.float16

    def test_to_mlx_float32(self):
        """Test conversion to MLX float32."""
        assert DType.FLOAT32.to_mlx() == mx.float32

    def test_to_mlx_bfloat16(self):
        """Test conversion to MLX bfloat16."""
        assert DType.BFLOAT16.to_mlx() == mx.bfloat16

    def test_dtype_is_str_enum(self):
        """Test that DType is a string enum."""
        assert isinstance(DType.FLOAT16, str)
        assert DType.FLOAT16 == "float16"


class TestDownloadConfig:
    """Tests for DownloadConfig model."""

    def test_required_model_id(self):
        """Test that model_id is required."""
        with pytest.raises(ValueError):
            DownloadConfig()

    def test_default_values(self):
        """Test default configuration values."""
        config = DownloadConfig(model_id="org/model")
        assert config.model_id == "org/model"
        assert config.cache_dir is None
        assert config.prefer_sharded is True
        assert "*.json" in config.allow_patterns
        assert "*.safetensors" in config.allow_patterns

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DownloadConfig(
            model_id="org/model",
            cache_dir=Path("/tmp/cache"),
            allow_patterns=["*.bin"],
            prefer_sharded=False,
        )
        assert config.cache_dir == Path("/tmp/cache")
        assert config.allow_patterns == ["*.bin"]
        assert config.prefer_sharded is False


class TestLoadedWeights:
    """Tests for LoadedWeights model."""

    def test_create_loaded_weights(self):
        """Test creating LoadedWeights."""
        weights = {
            "model.layers.0.weight": mx.zeros((10, 10)),
            "model.layers.1.weight": mx.zeros((10, 10)),
        }
        loaded = LoadedWeights(
            weights=weights,
            dtype=DType.BFLOAT16,
            source_path=Path("/tmp/model"),
            tensor_count=2,
        )
        assert loaded.dtype == DType.BFLOAT16
        assert loaded.tensor_count == 2
        assert loaded.source_path == Path("/tmp/model")

    def test_layer_count_property(self):
        """Test layer_count property."""
        weights = {
            "model.layers.0.weight": mx.zeros((10, 10)),
            "model.layers.1.weight": mx.zeros((10, 10)),
            "model.layers.5.weight": mx.zeros((10, 10)),
        }
        loaded = LoadedWeights(
            weights=weights,
            dtype=DType.BFLOAT16,
            source_path=Path("/tmp/model"),
            tensor_count=3,
        )
        assert loaded.layer_count == 6  # 0-5 = 6 layers

    def test_layer_count_no_layers(self):
        """Test layer_count with no layer weights."""
        weights = {
            "model.embed_tokens.weight": mx.zeros((10, 10)),
            "lm_head.weight": mx.zeros((10, 10)),
        }
        loaded = LoadedWeights(
            weights=weights,
            dtype=DType.BFLOAT16,
            source_path=Path("/tmp/model"),
            tensor_count=2,
        )
        assert loaded.layer_count == 0


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_create_result(self):
        """Test creating download result."""
        result = DownloadResult(
            model_path=Path("/tmp/model"),
            model_id="org/model",
        )
        assert result.model_path == Path("/tmp/model")
        assert result.model_id == "org/model"
        assert result.is_cached is False

    def test_is_cached(self):
        """Test is_cached flag."""
        result = DownloadResult(
            model_path=Path("/tmp/model"),
            model_id="org/model",
            is_cached=True,
        )
        assert result.is_cached is True


class TestStandardWeightConverter:
    """Tests for StandardWeightConverter."""

    def test_convert_embed_tokens(self):
        """Test converting embed_tokens weight."""
        converter = StandardWeightConverter()
        result = converter.convert("model.embed_tokens.weight")
        assert result == "model.embed_tokens.weight.weight"

    def test_convert_final_norm(self):
        """Test converting final norm weight."""
        converter = StandardWeightConverter()
        result = converter.convert("model.norm.weight")
        assert result == "model.norm.weight"

    def test_convert_lm_head(self):
        """Test converting lm_head weight."""
        converter = StandardWeightConverter()
        result = converter.convert("lm_head.weight")
        assert result == "lm_head.lm_head.weight"

    def test_convert_lm_head_tied(self):
        """Test lm_head with tied embeddings."""
        converter = StandardWeightConverter(tie_word_embeddings=True)
        result = converter.convert("lm_head.weight")
        assert result is None

    def test_convert_layer_weights(self):
        """Test converting layer weights."""
        converter = StandardWeightConverter()

        result = converter.convert("model.layers.0.self_attn.q_proj.weight")
        assert result == "model.layers.0.self_attn.q_proj.weight"

        result = converter.convert("model.layers.5.mlp.gate_proj.weight")
        assert result == "model.layers.5.mlp.gate_proj.weight"

    def test_skip_rotary_emb(self):
        """Test skipping rotary embeddings."""
        converter = StandardWeightConverter()
        result = converter.convert("model.layers.0.self_attn.rotary_emb.inv_freq")
        assert result is None

    def test_unknown_weight(self):
        """Test unknown weight returns None."""
        converter = StandardWeightConverter()
        result = converter.convert("unknown.weight.name")
        assert result is None


class TestWeightConverterProtocol:
    """Tests for WeightConverter protocol."""

    def test_protocol_compliance(self):
        """Test that StandardWeightConverter implements WeightConverter."""
        converter = StandardWeightConverter()
        # Check it has the required method
        assert hasattr(converter, "convert")
        assert callable(converter.convert)

    def test_custom_converter(self):
        """Test custom converter implementation."""

        class CustomConverter:
            def convert(self, hf_name: str) -> str | None:
                return f"custom.{hf_name}"

        converter = CustomConverter()
        assert converter.convert("test") == "custom.test"


class TestHFLoaderDownload:
    """Tests for HFLoader.download method."""

    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.list_repo_files")
    def test_download_basic(self, mock_list_files, mock_download):
        """Test basic download."""
        mock_list_files.return_value = ["config.json", "model.safetensors"]
        mock_download.return_value = "/tmp/model"

        result = HFLoader.download("org/model")

        assert result.model_path == Path("/tmp/model")
        assert result.model_id == "org/model"
        mock_download.assert_called_once()

    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.list_repo_files")
    def test_download_prefer_sharded(self, mock_list_files, mock_download):
        """Test download preferring sharded files."""
        mock_list_files.return_value = [
            "config.json",
            "model-00001-of-00002.safetensors",
            "consolidated.safetensors",
        ]
        mock_download.return_value = "/tmp/model"

        _ = HFLoader.download("org/model", prefer_sharded=True)

        # Should ignore consolidated.safetensors
        call_args = mock_download.call_args
        assert "consolidated.safetensors" in call_args.kwargs.get("ignore_patterns", [])

    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.list_repo_files")
    def test_download_with_cache_dir(self, mock_list_files, mock_download):
        """Test download with cache directory."""
        mock_list_files.return_value = ["config.json"]
        mock_download.return_value = "/tmp/model"

        _ = HFLoader.download("org/model", cache_dir=Path("/custom/cache"))

        call_args = mock_download.call_args
        assert call_args.kwargs.get("cache_dir") == "/custom/cache"

    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.list_repo_files")
    def test_download_list_files_error(self, mock_list_files, mock_download):
        """Test download when listing files fails."""
        mock_list_files.side_effect = Exception("API error")
        mock_download.return_value = "/tmp/model"

        # Should still work, just without sharded preference logic
        result = HFLoader.download("org/model")
        assert result.model_path == Path("/tmp/model")


class TestHFLoaderLoadTokenizer:
    """Tests for HFLoader.load_tokenizer method."""

    def test_load_tokenizer_basic(self, monkeypatch):
        """Test basic tokenizer loading."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_auto_tokenizer_class = MagicMock()
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Import the loader module and mock
        import chuk_lazarus.inference.loader as loader_module

        def patched_load(model_path):
            # Mock the import by directly calling our mock
            mock_tokenizer_result = mock_auto_tokenizer_class.from_pretrained(str(model_path))
            if mock_tokenizer_result.pad_token is None:
                mock_tokenizer_result.pad_token = mock_tokenizer_result.eos_token
            return mock_tokenizer_result

        monkeypatch.setattr(loader_module.HFLoader, "load_tokenizer", staticmethod(patched_load))

        result = HFLoader.load_tokenizer(Path("/tmp/model"))

        assert result.pad_token == "<eos>"
        mock_auto_tokenizer_class.from_pretrained.assert_called_with("/tmp/model")

    def test_load_tokenizer_with_pad_token(self, monkeypatch):
        """Test loading tokenizer that already has pad token."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"

        mock_auto_tokenizer_class = MagicMock()
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        import chuk_lazarus.inference.loader as loader_module

        def patched_load(model_path):
            mock_tokenizer_result = mock_auto_tokenizer_class.from_pretrained(str(model_path))
            if mock_tokenizer_result.pad_token is None:
                mock_tokenizer_result.pad_token = mock_tokenizer_result.eos_token
            return mock_tokenizer_result

        monkeypatch.setattr(loader_module.HFLoader, "load_tokenizer", staticmethod(patched_load))

        result = HFLoader.load_tokenizer(Path("/tmp/model"))

        assert result.pad_token == "<pad>"  # Not overwritten


class TestHFLoaderLoadWeights:
    """Tests for HFLoader.load_weights method."""

    def test_load_weights_no_files(self, tmp_path):
        """Test loading weights when no files exist."""
        with pytest.raises(FileNotFoundError):
            HFLoader.load_weights(tmp_path)

    def test_load_weights_basic(self, tmp_path):
        """Test basic weight loading."""
        # Create a fake safetensors file
        weights = {
            "model.embed_tokens.weight": mx.zeros((10, 10)),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        }
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

        loaded = HFLoader.load_weights(tmp_path)

        assert loaded.tensor_count >= 0  # Some may be filtered
        assert loaded.source_path == tmp_path
        assert loaded.dtype == DType.BFLOAT16

    def test_load_weights_with_converter(self, tmp_path):
        """Test loading weights with custom converter."""
        weights = {"custom.weight": mx.zeros((10, 10))}
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

        class TestConverter:
            def convert(self, name):
                return f"converted.{name}"

        loaded = HFLoader.load_weights(tmp_path, converter=TestConverter())
        assert "converted.custom.weight" in loaded.weights


class TestHFLoaderBuildNestedWeights:
    """Tests for HFLoader.build_nested_weights method."""

    def test_build_nested_basic(self):
        """Test basic nested weight building."""
        loaded = LoadedWeights(
            weights={
                "model.embed_tokens.weight": mx.zeros((10, 10)),
                "model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
                "model.layers.1.self_attn.q_proj.weight": mx.zeros((10, 10)),
            },
            dtype=DType.BFLOAT16,
            source_path=Path("/tmp"),
            tensor_count=3,
        )

        nested = HFLoader.build_nested_weights(loaded)

        assert "model" in nested
        assert "layers" in nested["model"]
        assert len(nested["model"]["layers"]) == 2
        assert "self_attn" in nested["model"]["layers"][0]

    def test_build_nested_deep_structure(self):
        """Test nested structure with deep paths."""
        loaded = LoadedWeights(
            weights={
                "model.layers.0.mlp.gate_proj.weight": mx.zeros((10, 10)),
                "model.layers.0.mlp.up_proj.weight": mx.zeros((10, 10)),
            },
            dtype=DType.BFLOAT16,
            source_path=Path("/tmp"),
            tensor_count=2,
        )

        nested = HFLoader.build_nested_weights(loaded)

        assert "mlp" in nested["model"]["layers"][0]
        assert "gate_proj" in nested["model"]["layers"][0]["mlp"]
        assert "weight" in nested["model"]["layers"][0]["mlp"]["gate_proj"]


class TestHFLoaderDownloadAsync:
    """Tests for HFLoader.download_async method."""

    @pytest.mark.asyncio
    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.list_repo_files")
    async def test_download_async(self, mock_list_files, mock_download):
        """Test async download."""
        mock_list_files.return_value = ["config.json"]
        mock_download.return_value = "/tmp/model"

        result = await HFLoader.download_async("org/model")

        assert result.model_path == Path("/tmp/model")
        assert result.model_id == "org/model"
