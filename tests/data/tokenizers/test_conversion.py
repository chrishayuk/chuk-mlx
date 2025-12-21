"""Tests for conversion module."""

import json

from chuk_lazarus.data.tokenizers.conversion import (
    ConversionResult,
    ExportFormat,
    TokenizerConfig,
    TokenizerFormat,
    TokenMapping,
    VocabExport,
    create_huggingface_tokenizer_json,
    create_token_mappings,
    export_vocab_csv,
    export_vocab_json,
    export_vocab_tsv,
    export_vocabulary,
    extract_config_from_tokenizer,
    import_vocab_csv,
    import_vocab_json,
    import_vocab_tsv,
    import_vocabulary,
    load_vocabulary_file,
    save_huggingface_format,
    save_vocabulary_file,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"hello": 0, "world": 1, "<unk>": 2}
        self.pad_token_id = None
        self.unk_token_id = 2
        self.bos_token_id = None
        self.eos_token_id = None
        self.model_max_length = 512

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, 2) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestTokenizerFormatEnum:
    """Tests for TokenizerFormat enum."""

    def test_values(self):
        assert TokenizerFormat.HUGGINGFACE == "huggingface"
        assert TokenizerFormat.SENTENCEPIECE == "sentencepiece"
        assert TokenizerFormat.TIKTOKEN == "tiktoken"
        assert TokenizerFormat.CUSTOM_JSON == "custom_json"


class TestExportFormatEnum:
    """Tests for ExportFormat enum."""

    def test_values(self):
        assert ExportFormat.JSON == "json"
        assert ExportFormat.TSV == "tsv"
        assert ExportFormat.CSV == "csv"


class TestTokenizerConfigModel:
    """Tests for TokenizerConfig Pydantic model."""

    def test_default_values(self):
        config = TokenizerConfig(vocab_size=1000)
        assert config.vocab_size == 1000
        assert config.model_max_length == 512
        assert config.tokenizer_class == "PreTrainedTokenizer"

    def test_with_special_tokens(self):
        config = TokenizerConfig(
            vocab_size=1000,
            pad_token_id=0,
            unk_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
        )
        assert config.pad_token_id == 0


class TestVocabExportModel:
    """Tests for VocabExport Pydantic model."""

    def test_basic(self):
        export = VocabExport(
            format=ExportFormat.JSON,
            vocab_size=100,
            content='{"a": 0}',
        )
        assert export.format == ExportFormat.JSON


class TestConversionResultModel:
    """Tests for ConversionResult Pydantic model."""

    def test_success(self):
        result = ConversionResult(
            success=True,
            source_format=TokenizerFormat.CUSTOM_JSON,
            target_format=TokenizerFormat.HUGGINGFACE,
            vocab_size=1000,
            output_path="/path/to/output",
        )
        assert result.success

    def test_failure(self):
        result = ConversionResult(
            success=False,
            source_format=TokenizerFormat.CUSTOM_JSON,
            target_format=TokenizerFormat.HUGGINGFACE,
            vocab_size=0,
            errors=["Failed to convert"],
        )
        assert not result.success


class TestTokenMappingModel:
    """Tests for TokenMapping Pydantic model."""

    def test_basic(self):
        mapping = TokenMapping(
            token_id=0,
            token_str="hello",
            byte_fallback=b"hello",
        )
        assert mapping.token_id == 0


class TestExtractConfigFromTokenizer:
    """Tests for extract_config_from_tokenizer function."""

    def test_basic_extraction(self):
        tokenizer = MockTokenizer()
        config = extract_config_from_tokenizer(tokenizer)
        assert config.vocab_size == 3
        assert config.unk_token_id == 2
        assert config.model_max_length == 512


class TestExportVocabJson:
    """Tests for export_vocab_json function."""

    def test_basic_export(self):
        vocab = {"hello": 0, "world": 1}
        export = export_vocab_json(vocab)
        assert export.format == ExportFormat.JSON
        assert export.vocab_size == 2
        parsed = json.loads(export.content)
        assert parsed["hello"] == 0

    def test_pretty_format(self):
        vocab = {"a": 0}
        export = export_vocab_json(vocab, pretty=True)
        assert "\n" in export.content

    def test_compact_format(self):
        vocab = {"a": 0, "b": 1}
        export = export_vocab_json(vocab, pretty=False)
        assert "\n" not in export.content


class TestExportVocabTsv:
    """Tests for export_vocab_tsv function."""

    def test_with_header(self):
        vocab = {"hello": 0, "world": 1}
        export = export_vocab_tsv(vocab, include_header=True)
        assert export.format == ExportFormat.TSV
        lines = export.content.split("\n")
        assert "token\tid" in lines[0]

    def test_without_header(self):
        vocab = {"hello": 0}
        export = export_vocab_tsv(vocab, include_header=False)
        lines = export.content.split("\n")
        assert "token" not in lines[0]

    def test_escaping(self):
        vocab = {"hello\tworld": 0}  # Tab in token
        export = export_vocab_tsv(vocab)
        assert "\\t" in export.content


class TestExportVocabCsv:
    """Tests for export_vocab_csv function."""

    def test_with_header(self):
        vocab = {"hello": 0, "world": 1}
        export = export_vocab_csv(vocab, include_header=True)
        assert export.format == ExportFormat.CSV
        assert "token,id" in export.content

    def test_escaping_comma(self):
        vocab = {"hello,world": 0}
        export = export_vocab_csv(vocab)
        assert '"hello,world"' in export.content

    def test_escaping_quotes(self):
        vocab = {'hello"world': 0}
        export = export_vocab_csv(vocab)
        assert '""' in export.content  # Escaped quotes


class TestExportVocabulary:
    """Tests for export_vocabulary function."""

    def test_json_format(self):
        vocab = {"a": 0}
        export = export_vocabulary(vocab, format=ExportFormat.JSON)
        assert export.format == ExportFormat.JSON

    def test_tsv_format(self):
        vocab = {"a": 0}
        export = export_vocabulary(vocab, format=ExportFormat.TSV)
        assert export.format == ExportFormat.TSV

    def test_csv_format(self):
        vocab = {"a": 0}
        export = export_vocabulary(vocab, format=ExportFormat.CSV)
        assert export.format == ExportFormat.CSV


class TestImportVocabJson:
    """Tests for import_vocab_json function."""

    def test_basic_import(self):
        content = '{"hello": 0, "world": 1}'
        vocab = import_vocab_json(content)
        assert vocab["hello"] == 0
        assert vocab["world"] == 1


class TestImportVocabTsv:
    """Tests for import_vocab_tsv function."""

    def test_with_header(self):
        content = "token\tid\nhello\t0\nworld\t1"
        vocab = import_vocab_tsv(content, has_header=True)
        assert vocab["hello"] == 0
        assert vocab["world"] == 1

    def test_without_header(self):
        content = "hello\t0\nworld\t1"
        vocab = import_vocab_tsv(content, has_header=False)
        assert vocab["hello"] == 0

    def test_unescaping(self):
        content = "token\tid\nhello\\tworld\t0"
        vocab = import_vocab_tsv(content)
        assert "hello\tworld" in vocab


class TestImportVocabCsv:
    """Tests for import_vocab_csv function."""

    def test_with_header(self):
        content = "token,id\nhello,0\nworld,1"
        vocab = import_vocab_csv(content, has_header=True)
        assert vocab["hello"] == 0

    def test_quoted_values(self):
        content = 'token,id\n"hello,world",0'
        vocab = import_vocab_csv(content)
        assert "hello,world" in vocab


class TestImportVocabulary:
    """Tests for import_vocabulary function."""

    def test_json(self):
        content = '{"a": 0}'
        vocab = import_vocabulary(content, format=ExportFormat.JSON)
        assert vocab["a"] == 0

    def test_tsv(self):
        content = "token\tid\na\t0"
        vocab = import_vocabulary(content, format=ExportFormat.TSV)
        assert vocab["a"] == 0

    def test_csv(self):
        content = "token,id\na,0"
        vocab = import_vocabulary(content, format=ExportFormat.CSV)
        assert vocab["a"] == 0


class TestSaveVocabularyFile:
    """Tests for save_vocabulary_file function."""

    def test_save_json(self, tmp_path):
        vocab = {"hello": 0, "world": 1}
        path = tmp_path / "vocab.json"
        export = save_vocabulary_file(vocab, path)
        assert export.format == ExportFormat.JSON
        assert path.exists()

    def test_save_tsv(self, tmp_path):
        vocab = {"hello": 0}
        path = tmp_path / "vocab.tsv"
        export = save_vocabulary_file(vocab, path)
        assert export.format == ExportFormat.TSV

    def test_explicit_format(self, tmp_path):
        vocab = {"hello": 0}
        path = tmp_path / "vocab.txt"
        export = save_vocabulary_file(vocab, path, format=ExportFormat.JSON)
        assert export.format == ExportFormat.JSON


class TestLoadVocabularyFile:
    """Tests for load_vocabulary_file function."""

    def test_load_json(self, tmp_path):
        vocab = {"hello": 0, "world": 1}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(vocab))
        loaded = load_vocabulary_file(path)
        assert loaded == vocab

    def test_load_tsv(self, tmp_path):
        path = tmp_path / "vocab.tsv"
        path.write_text("token\tid\nhello\t0\nworld\t1")
        loaded = load_vocabulary_file(path)
        assert loaded["hello"] == 0


class TestCreateHuggingfaceTokenizerJson:
    """Tests for create_huggingface_tokenizer_json function."""

    def test_basic_structure(self):
        vocab = {"hello": 0, "world": 1}
        hf_json = create_huggingface_tokenizer_json(vocab)
        assert "version" in hf_json
        assert "model" in hf_json
        assert hf_json["model"]["type"] == "WordLevel"
        assert hf_json["model"]["vocab"] == vocab

    def test_with_config(self):
        vocab = {"<pad>": 0, "<unk>": 1, "hello": 2}
        config = TokenizerConfig(
            vocab_size=3,
            pad_token_id=0,
            unk_token_id=1,
        )
        hf_json = create_huggingface_tokenizer_json(vocab, config)
        assert len(hf_json["added_tokens"]) >= 1


class TestSaveHuggingfaceFormat:
    """Tests for save_huggingface_format function."""

    def test_creates_files(self, tmp_path):
        vocab = {"hello": 0, "world": 1}
        result = save_huggingface_format(vocab, tmp_path)
        assert result.success
        assert (tmp_path / "tokenizer.json").exists()
        assert (tmp_path / "vocab.json").exists()
        assert (tmp_path / "tokenizer_config.json").exists()

    def test_with_config(self, tmp_path):
        vocab = {"<pad>": 0, "hello": 1}
        config = TokenizerConfig(vocab_size=2, pad_token_id=0)
        result = save_huggingface_format(vocab, tmp_path, config)
        assert result.success


class TestCreateTokenMappings:
    """Tests for create_token_mappings function."""

    def test_basic_mappings(self):
        vocab = {"hello": 0, "world": 1}
        mappings = create_token_mappings(vocab)
        assert len(mappings) == 2
        assert mappings[0].token_id == 0
        assert mappings[0].token_str == "hello"
        assert mappings[0].byte_fallback == b"hello"

    def test_sorted_by_id(self):
        vocab = {"world": 1, "hello": 0}  # Unsorted
        mappings = create_token_mappings(vocab)
        assert mappings[0].token_id == 0  # Should be sorted
        assert mappings[1].token_id == 1
