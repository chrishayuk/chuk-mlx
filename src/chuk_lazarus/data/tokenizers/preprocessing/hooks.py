"""
Pre/post tokenization hooks.

Composable transforms that run before encoding and after decoding:
- Numeric normalization
- Structure token injection
- Whitespace normalization
- Custom transforms
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
    NumericConfig,
    NumericEncoding,
    normalize_numbers,
    restore_numbers,
)
from chuk_lazarus.data.tokenizers.preprocessing.structure import (
    StructureConfig,
    StructureEncoding,
    inject_structure_tokens,
    restore_structures,
)


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


class TransformResult(BaseModel):
    """Result of a pre-tokenization transform."""

    text: str = Field(description="Transformed text")
    metadata: dict = Field(default_factory=dict, description="Transform metadata")


class PreTokenizeHook(ABC):
    """Abstract base class for pre-tokenization hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for identification."""
        ...

    @abstractmethod
    def transform(self, text: str) -> TransformResult:
        """
        Transform text before tokenization.

        Args:
            text: Input text

        Returns:
            TransformResult with transformed text and metadata
        """
        ...

    @abstractmethod
    def inverse(self, text: str, metadata: dict) -> str:
        """
        Inverse transform for decoding.

        Args:
            text: Decoded text to restore
            metadata: Metadata from transform

        Returns:
            Restored text
        """
        ...


class PostDecodeHook(ABC):
    """Abstract base class for post-decode hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for identification."""
        ...

    @abstractmethod
    def transform(self, text: str) -> str:
        """
        Transform text after decoding.

        Args:
            text: Decoded text

        Returns:
            Transformed text
        """
        ...


class NumericNormalizationHook(PreTokenizeHook):
    """Hook for numeric normalization."""

    def __init__(self, config: NumericConfig | None = None):
        self.config = config or NumericConfig()

    @property
    def name(self) -> str:
        return "numeric_normalization"

    def transform(self, text: str) -> TransformResult:
        encoding: NumericEncoding = normalize_numbers(text, self.config)
        return TransformResult(
            text=encoding.encoded_text,
            metadata={
                "mapping": encoding.mapping,
                "span_count": len(encoding.spans),
            },
        )

    def inverse(self, text: str, metadata: dict) -> str:
        mapping = metadata.get("mapping", {})
        return restore_numbers(text, mapping)


class StructureInjectionHook(PreTokenizeHook):
    """Hook for structure token injection."""

    def __init__(self, config: StructureConfig | None = None):
        self.config = config or StructureConfig()

    @property
    def name(self) -> str:
        return "structure_injection"

    def transform(self, text: str) -> TransformResult:
        encoding: StructureEncoding = inject_structure_tokens(text, self.config)
        return TransformResult(
            text=encoding.encoded_text,
            metadata={
                "mapping": encoding.mapping,
                "span_count": len(encoding.spans),
            },
        )

    def inverse(self, text: str, metadata: dict) -> str:
        mapping = metadata.get("mapping", {})
        return restore_structures(text, mapping)


class WhitespaceNormalizationHook(PreTokenizeHook):
    """Hook for whitespace normalization."""

    def __init__(
        self,
        collapse_spaces: bool = True,
        normalize_newlines: bool = True,
        strip: bool = True,
    ):
        self.collapse_spaces = collapse_spaces
        self.normalize_newlines = normalize_newlines
        self.strip = strip

    @property
    def name(self) -> str:
        return "whitespace_normalization"

    def transform(self, text: str) -> TransformResult:
        result = text

        if self.normalize_newlines:
            result = result.replace("\r\n", "\n").replace("\r", "\n")

        if self.collapse_spaces:
            import re

            result = re.sub(r"[ \t]+", " ", result)

        if self.strip:
            result = result.strip()

        return TransformResult(
            text=result,
            metadata={"original_length": len(text), "normalized_length": len(result)},
        )

    def inverse(self, text: str, metadata: dict) -> str:
        # Whitespace normalization is not reversible
        return text


class LowercaseHook(PreTokenizeHook):
    """Hook for case normalization."""

    def __init__(self, preserve_acronyms: bool = False):
        self.preserve_acronyms = preserve_acronyms

    @property
    def name(self) -> str:
        return "lowercase"

    def transform(self, text: str) -> TransformResult:
        if self.preserve_acronyms:
            # Keep all-caps words unchanged
            words = text.split()
            result = []
            for word in words:
                if word.isupper() and len(word) > 1:
                    result.append(word)
                else:
                    result.append(word.lower())
            return TransformResult(text=" ".join(result), metadata={})
        return TransformResult(text=text.lower(), metadata={})

    def inverse(self, text: str, metadata: dict) -> str:
        # Case normalization is not reversible
        return text


class ASCIIFoldingHook(PreTokenizeHook):
    """Hook for folding Unicode to ASCII equivalents."""

    def __init__(self):
        import unicodedata

        self._normalize = unicodedata.normalize

    @property
    def name(self) -> str:
        return "ascii_folding"

    def transform(self, text: str) -> TransformResult:
        # NFD decomposition then filter to ASCII
        normalized = self._normalize("NFD", text)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return TransformResult(
            text=ascii_text,
            metadata={"original_length": len(text), "ascii_length": len(ascii_text)},
        )

    def inverse(self, text: str, metadata: dict) -> str:
        # ASCII folding is not reversible
        return text


class CustomHook(PreTokenizeHook):
    """Hook for custom transform functions."""

    def __init__(
        self,
        name: str,
        transform_fn: Callable,
        inverse_fn: Callable | None = None,
    ):
        self._name = name
        self._transform_fn = transform_fn
        self._inverse_fn = inverse_fn or (lambda t, m: t)

    @property
    def name(self) -> str:
        return self._name

    def transform(self, text: str) -> TransformResult:
        result = self._transform_fn(text)
        if isinstance(result, tuple):
            return TransformResult(text=result[0], metadata=result[1])
        return TransformResult(text=result, metadata={})

    def inverse(self, text: str, metadata: dict) -> str:
        return self._inverse_fn(text, metadata)


class HookPipeline:
    """Pipeline of pre/post tokenization hooks."""

    def __init__(self, hooks: list[PreTokenizeHook] | None = None):
        self.hooks: list[PreTokenizeHook] = hooks or []
        self._metadata_stack: list[dict] = []

    def add_hook(self, hook: PreTokenizeHook) -> "HookPipeline":
        """Add a hook to the pipeline."""
        self.hooks.append(hook)
        return self

    def pre_tokenize(self, text: str) -> str:
        """
        Apply all pre-tokenization hooks in order.

        Args:
            text: Input text

        Returns:
            Transformed text
        """
        self._metadata_stack = []
        result = text

        for hook in self.hooks:
            transform_result = hook.transform(result)
            result = transform_result.text
            self._metadata_stack.append({"hook": hook.name, "metadata": transform_result.metadata})

        return result

    def post_decode(self, text: str) -> str:
        """
        Apply inverse transforms in reverse order.

        Args:
            text: Decoded text

        Returns:
            Restored text
        """
        result = text

        # Apply inverses in reverse order
        for hook_data in reversed(self._metadata_stack):
            hook_name = hook_data["hook"]
            metadata = hook_data["metadata"]

            # Find the hook
            for hook in self.hooks:
                if hook.name == hook_name:
                    result = hook.inverse(result, metadata)
                    break

        return result

    def get_metadata(self) -> list[dict]:
        """Get metadata from last pre_tokenize call."""
        return self._metadata_stack.copy()


class HookedTokenizer:
    """Tokenizer wrapper that applies hooks."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        pipeline: HookPipeline,
    ):
        self.tokenizer = tokenizer
        self.pipeline = pipeline

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode with pre-tokenization hooks."""
        transformed = self.pipeline.pre_tokenize(text)
        return self.tokenizer.encode(transformed, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int]) -> str:
        """Decode with post-decode hooks."""
        decoded = self.tokenizer.decode(token_ids)
        return self.pipeline.post_decode(decoded)

    @property
    def vocab_size(self) -> int:
        """Get vocab size from underlying tokenizer."""
        if hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer.vocab_size
        return 0


def create_standard_pipeline(
    numeric: bool = True,
    structure: bool = True,
    whitespace: bool = False,
    numeric_config: NumericConfig | None = None,
    structure_config: StructureConfig | None = None,
) -> HookPipeline:
    """
    Create a standard preprocessing pipeline.

    Args:
        numeric: Enable numeric normalization
        structure: Enable structure token injection
        whitespace: Enable whitespace normalization
        numeric_config: Config for numeric normalization
        structure_config: Config for structure injection

    Returns:
        Configured HookPipeline
    """
    pipeline = HookPipeline()

    if whitespace:
        pipeline.add_hook(WhitespaceNormalizationHook())

    if numeric:
        pipeline.add_hook(NumericNormalizationHook(numeric_config))

    if structure:
        pipeline.add_hook(StructureInjectionHook(structure_config))

    return pipeline


def create_math_pipeline(
    numeric_config: NumericConfig | None = None,
) -> HookPipeline:
    """
    Create a pipeline optimized for math/reasoning.

    Args:
        numeric_config: Config for numeric normalization

    Returns:
        Configured HookPipeline for math
    """
    from chuk_lazarus.data.tokenizers.preprocessing.structure import (
        create_math_aware_config,
    )

    config = numeric_config or NumericConfig(use_placeholder=True)

    pipeline = HookPipeline()
    pipeline.add_hook(NumericNormalizationHook(config))
    pipeline.add_hook(StructureInjectionHook(create_math_aware_config()))

    return pipeline


def create_tool_pipeline(
    structure_config: StructureConfig | None = None,
) -> HookPipeline:
    """
    Create a pipeline optimized for tool/agent traces.

    Args:
        structure_config: Config for structure injection

    Returns:
        Configured HookPipeline for tool use
    """
    from chuk_lazarus.data.tokenizers.preprocessing.structure import (
        create_tool_aware_config,
    )

    config = structure_config or create_tool_aware_config()

    pipeline = HookPipeline()
    pipeline.add_hook(NumericNormalizationHook())
    pipeline.add_hook(StructureInjectionHook(config))

    return pipeline
