"""Tokenizer conversion and serialization utilities with Pydantic models."""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class TokenizerFormat(str, Enum):
    """Supported tokenizer formats."""

    HUGGINGFACE = "huggingface"
    SENTENCEPIECE = "sentencepiece"
    TIKTOKEN = "tiktoken"
    CUSTOM_JSON = "custom_json"


class ExportFormat(str, Enum):
    """Formats for exporting vocabulary."""

    JSON = "json"
    TSV = "tsv"
    CSV = "csv"


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_vocab(self) -> dict[str, int]: ...


class TokenizerConfig(BaseModel):
    """Configuration for a tokenizer."""

    vocab_size: int = Field(ge=0, description="Size of vocabulary")
    pad_token_id: int | None = Field(default=None, description="PAD token ID")
    unk_token_id: int | None = Field(default=None, description="UNK token ID")
    bos_token_id: int | None = Field(default=None, description="BOS token ID")
    eos_token_id: int | None = Field(default=None, description="EOS token ID")
    model_max_length: int = Field(default=512, description="Maximum sequence length")
    tokenizer_class: str = Field(default="PreTrainedTokenizer", description="Tokenizer class name")
    additional_special_tokens: list[str] = Field(
        default_factory=list, description="Additional special tokens"
    )


class VocabExport(BaseModel):
    """Exported vocabulary data."""

    format: ExportFormat = Field(description="Export format used")
    vocab_size: int = Field(ge=0, description="Number of tokens exported")
    content: str = Field(description="Exported content as string")


class ConversionResult(BaseModel):
    """Result of a tokenizer conversion."""

    success: bool = Field(description="Whether conversion succeeded")
    source_format: TokenizerFormat = Field(description="Original format")
    target_format: TokenizerFormat = Field(description="Target format")
    vocab_size: int = Field(ge=0, description="Vocabulary size")
    output_path: str | None = Field(default=None, description="Path to output files")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")


class TokenMapping(BaseModel):
    """Mapping between token representations."""

    token_id: int = Field(description="Token ID")
    token_str: str = Field(description="Token string")
    byte_fallback: bytes | None = Field(default=None, description="Byte fallback for unknown chars")


def extract_config_from_tokenizer(
    tokenizer: TokenizerProtocol,
) -> TokenizerConfig:
    """
    Extract configuration from a tokenizer instance.

    Args:
        tokenizer: Tokenizer to extract config from

    Returns:
        TokenizerConfig with extracted settings
    """
    vocab = tokenizer.get_vocab()

    additional_special = []
    if hasattr(tokenizer, "additional_special_tokens"):
        tokens = getattr(tokenizer, "additional_special_tokens", [])
        if tokens:
            additional_special = list(tokens)

    model_max_length = 512
    if hasattr(tokenizer, "model_max_length"):
        length = getattr(tokenizer, "model_max_length", 512)
        if isinstance(length, int) and length < 1_000_000:
            model_max_length = length

    return TokenizerConfig(
        vocab_size=len(vocab),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        unk_token_id=getattr(tokenizer, "unk_token_id", None),
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        model_max_length=model_max_length,
        tokenizer_class=type(tokenizer).__name__,
        additional_special_tokens=additional_special,
    )


def export_vocab_json(
    vocab: dict[str, int],
    pretty: bool = True,
) -> VocabExport:
    """
    Export vocabulary as JSON.

    Args:
        vocab: Token to ID mapping
        pretty: Whether to format with indentation

    Returns:
        VocabExport with JSON content
    """
    indent = 2 if pretty else None
    content = json.dumps(vocab, ensure_ascii=False, indent=indent)

    return VocabExport(
        format=ExportFormat.JSON,
        vocab_size=len(vocab),
        content=content,
    )


def export_vocab_tsv(
    vocab: dict[str, int],
    include_header: bool = True,
) -> VocabExport:
    """
    Export vocabulary as TSV.

    Args:
        vocab: Token to ID mapping
        include_header: Whether to include column headers

    Returns:
        VocabExport with TSV content
    """
    lines = []
    if include_header:
        lines.append("token\tid")

    # Sort by ID for consistent output
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    for token, token_id in sorted_items:
        # Escape tabs in tokens
        escaped = token.replace("\t", "\\t").replace("\n", "\\n")
        lines.append(f"{escaped}\t{token_id}")

    return VocabExport(
        format=ExportFormat.TSV,
        vocab_size=len(vocab),
        content="\n".join(lines),
    )


def export_vocab_csv(
    vocab: dict[str, int],
    include_header: bool = True,
) -> VocabExport:
    """
    Export vocabulary as CSV.

    Args:
        vocab: Token to ID mapping
        include_header: Whether to include column headers

    Returns:
        VocabExport with CSV content
    """
    lines = []
    if include_header:
        lines.append("token,id")

    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    for token, token_id in sorted_items:
        # Escape quotes and commas
        escaped = token.replace('"', '""')
        if "," in escaped or '"' in escaped or "\n" in escaped:
            escaped = f'"{escaped}"'
        lines.append(f"{escaped},{token_id}")

    return VocabExport(
        format=ExportFormat.CSV,
        vocab_size=len(vocab),
        content="\n".join(lines),
    )


def export_vocabulary(
    vocab: dict[str, int],
    format: ExportFormat = ExportFormat.JSON,
    **kwargs: Any,
) -> VocabExport:
    """
    Export vocabulary in specified format.

    Args:
        vocab: Token to ID mapping
        format: Export format to use
        **kwargs: Additional format-specific options

    Returns:
        VocabExport with formatted content
    """
    if format == ExportFormat.JSON:
        return export_vocab_json(vocab, **kwargs)
    elif format == ExportFormat.TSV:
        return export_vocab_tsv(vocab, **kwargs)
    elif format == ExportFormat.CSV:
        return export_vocab_csv(vocab, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def import_vocab_json(content: str) -> dict[str, int]:
    """
    Import vocabulary from JSON content.

    Args:
        content: JSON string

    Returns:
        Token to ID mapping
    """
    return json.loads(content)


def import_vocab_tsv(
    content: str,
    has_header: bool = True,
) -> dict[str, int]:
    """
    Import vocabulary from TSV content.

    Args:
        content: TSV string
        has_header: Whether first line is header

    Returns:
        Token to ID mapping
    """
    lines = content.strip().split("\n")
    if has_header and lines:
        lines = lines[1:]

    vocab = {}
    for line in lines:
        if "\t" in line:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                token = parts[0].replace("\\t", "\t").replace("\\n", "\n")
                try:
                    vocab[token] = int(parts[1])
                except ValueError:
                    continue

    return vocab


def import_vocab_csv(
    content: str,
    has_header: bool = True,
) -> dict[str, int]:
    """
    Import vocabulary from CSV content.

    Args:
        content: CSV string
        has_header: Whether first line is header

    Returns:
        Token to ID mapping
    """
    import csv
    from io import StringIO

    reader = csv.reader(StringIO(content))
    if has_header:
        next(reader, None)

    vocab = {}
    for row in reader:
        if len(row) >= 2:
            try:
                vocab[row[0]] = int(row[1])
            except ValueError:
                continue

    return vocab


def import_vocabulary(
    content: str,
    format: ExportFormat = ExportFormat.JSON,
    **kwargs: Any,
) -> dict[str, int]:
    """
    Import vocabulary from string content.

    Args:
        content: File content as string
        format: Format of the content
        **kwargs: Additional format-specific options

    Returns:
        Token to ID mapping
    """
    if format == ExportFormat.JSON:
        return import_vocab_json(content)
    elif format == ExportFormat.TSV:
        return import_vocab_tsv(content, **kwargs)
    elif format == ExportFormat.CSV:
        return import_vocab_csv(content, **kwargs)
    else:
        raise ValueError(f"Unsupported import format: {format}")


def save_vocabulary_file(
    vocab: dict[str, int],
    path: str | Path,
    format: ExportFormat | None = None,
) -> VocabExport:
    """
    Save vocabulary to a file.

    Args:
        vocab: Token to ID mapping
        path: Output file path
        format: Export format (inferred from extension if None)

    Returns:
        VocabExport with save details
    """
    path = Path(path)

    if format is None:
        ext = path.suffix.lower()
        format_map = {
            ".json": ExportFormat.JSON,
            ".tsv": ExportFormat.TSV,
            ".csv": ExportFormat.CSV,
        }
        format = format_map.get(ext, ExportFormat.JSON)

    export = export_vocabulary(vocab, format)
    path.write_text(export.content, encoding="utf-8")

    return export


def load_vocabulary_file(
    path: str | Path,
    format: ExportFormat | None = None,
) -> dict[str, int]:
    """
    Load vocabulary from a file.

    Args:
        path: Input file path
        format: Import format (inferred from extension if None)

    Returns:
        Token to ID mapping
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")

    if format is None:
        ext = path.suffix.lower()
        format_map = {
            ".json": ExportFormat.JSON,
            ".tsv": ExportFormat.TSV,
            ".csv": ExportFormat.CSV,
        }
        format = format_map.get(ext, ExportFormat.JSON)

    return import_vocabulary(content, format)


def create_huggingface_tokenizer_json(
    vocab: dict[str, int],
    config: TokenizerConfig | None = None,
) -> dict[str, Any]:
    """
    Create a HuggingFace tokenizer.json structure.

    Args:
        vocab: Token to ID mapping
        config: Optional tokenizer configuration

    Returns:
        Dictionary matching HuggingFace tokenizer.json schema
    """
    if config is None:
        config = TokenizerConfig(vocab_size=len(vocab))

    # Build special tokens list
    special_tokens = []
    special_mapping = {
        "pad_token_id": ("<pad>", config.pad_token_id),
        "unk_token_id": ("<unk>", config.unk_token_id),
        "bos_token_id": ("<s>", config.bos_token_id),
        "eos_token_id": ("</s>", config.eos_token_id),
    }

    added_tokens = []
    for attr_name, (default_content, token_id) in special_mapping.items():
        if token_id is not None:
            # Find actual token content
            content = default_content
            for tok, tid in vocab.items():
                if tid == token_id:
                    content = tok
                    break

            added_tokens.append(
                {
                    "id": token_id,
                    "content": content,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
            )
            special_tokens.append({"id": token_id, "content": content})

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }


def save_huggingface_format(
    vocab: dict[str, int],
    output_dir: str | Path,
    config: TokenizerConfig | None = None,
) -> ConversionResult:
    """
    Save vocabulary in HuggingFace format.

    Args:
        vocab: Token to ID mapping
        output_dir: Output directory path
        config: Optional tokenizer configuration

    Returns:
        ConversionResult with save details
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []

    try:
        # Save tokenizer.json
        tokenizer_json = create_huggingface_tokenizer_json(vocab, config)
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer_path.write_text(
            json.dumps(tokenizer_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Save vocab.json
        vocab_path = output_dir / "vocab.json"
        vocab_path.write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Save tokenizer_config.json
        if config is None:
            config = TokenizerConfig(vocab_size=len(vocab))

        config_dict = {
            "tokenizer_class": config.tokenizer_class,
            "vocab_size": config.vocab_size,
            "model_max_length": config.model_max_length,
            "pad_token": "<pad>" if config.pad_token_id is not None else None,
            "unk_token": "<unk>" if config.unk_token_id is not None else None,
            "bos_token": "<s>" if config.bos_token_id is not None else None,
            "eos_token": "</s>" if config.eos_token_id is not None else None,
        }
        config_path = output_dir / "tokenizer_config.json"
        config_path.write_text(
            json.dumps(config_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    except Exception as e:
        errors.append(str(e))

    return ConversionResult(
        success=len(errors) == 0,
        source_format=TokenizerFormat.CUSTOM_JSON,
        target_format=TokenizerFormat.HUGGINGFACE,
        vocab_size=len(vocab),
        output_path=str(output_dir),
        errors=errors,
    )


def create_token_mappings(
    vocab: dict[str, int],
) -> list[TokenMapping]:
    """
    Create detailed token mappings with byte fallbacks.

    Args:
        vocab: Token to ID mapping

    Returns:
        List of TokenMapping for each token
    """
    mappings = []

    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        try:
            byte_fallback = token.encode("utf-8")
        except UnicodeEncodeError:
            byte_fallback = None

        mappings.append(
            TokenMapping(
                token_id=token_id,
                token_str=token,
                byte_fallback=byte_fallback,
            )
        )

    return mappings
