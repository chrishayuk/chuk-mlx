"""Tests for tokenization hooks."""

from chuk_lazarus.data.tokenizers.preprocessing.hooks import (
    CustomHook,
    HookedTokenizer,
    HookPipeline,
    LowercaseHook,
    NumericNormalizationHook,
    StructureInjectionHook,
    WhitespaceNormalizationHook,
    create_math_pipeline,
    create_standard_pipeline,
    create_tool_pipeline,
)
from chuk_lazarus.data.tokenizers.preprocessing.numeric import NumericConfig


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {"hello": 1, "world": 2, "<NUM_0>": 100}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab.get(w, 0) for w in text.lower().split()]

    def decode(self, token_ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class TestNumericNormalizationHook:
    """Tests for NumericNormalizationHook."""

    def test_hook_name(self):
        hook = NumericNormalizationHook()
        assert hook.name == "numeric_normalization"

    def test_transform(self):
        hook = NumericNormalizationHook()
        result = hook.transform("Value is 42")
        assert "<NUM_0>" in result.text
        assert "mapping" in result.metadata

    def test_inverse(self):
        hook = NumericNormalizationHook()
        result = hook.transform("Value is 42")
        restored = hook.inverse(result.text, result.metadata)
        assert restored == "Value is 42"

    def test_custom_config(self):
        config = NumericConfig(use_placeholder=False)
        hook = NumericNormalizationHook(config)
        result = hook.transform("Value is 42")
        # Without placeholder, number is normalized but not replaced
        assert result.text is not None


class TestStructureInjectionHook:
    """Tests for StructureInjectionHook."""

    def test_hook_name(self):
        hook = StructureInjectionHook()
        assert hook.name == "structure_injection"

    def test_transform(self):
        hook = StructureInjectionHook()
        result = hook.transform("Email: test@example.com")
        assert "<EMAIL_0>" in result.text

    def test_inverse(self):
        hook = StructureInjectionHook()
        text = "Email: test@example.com"
        result = hook.transform(text)
        restored = hook.inverse(result.text, result.metadata)
        assert restored == text


class TestWhitespaceNormalizationHook:
    """Tests for WhitespaceNormalizationHook."""

    def test_hook_name(self):
        hook = WhitespaceNormalizationHook()
        assert hook.name == "whitespace_normalization"

    def test_collapse_spaces(self):
        hook = WhitespaceNormalizationHook(collapse_spaces=True)
        result = hook.transform("hello    world")
        assert result.text == "hello world"

    def test_normalize_newlines(self):
        hook = WhitespaceNormalizationHook(normalize_newlines=True)
        result = hook.transform("hello\r\nworld")
        assert result.text == "hello\nworld"

    def test_strip(self):
        hook = WhitespaceNormalizationHook(strip=True)
        result = hook.transform("  hello  ")
        assert result.text == "hello"


class TestLowercaseHook:
    """Tests for LowercaseHook."""

    def test_hook_name(self):
        hook = LowercaseHook()
        assert hook.name == "lowercase"

    def test_lowercase(self):
        hook = LowercaseHook()
        result = hook.transform("Hello WORLD")
        assert result.text == "hello world"

    def test_preserve_acronyms(self):
        hook = LowercaseHook(preserve_acronyms=True)
        result = hook.transform("Hello NASA world")
        assert "NASA" in result.text
        assert "hello" in result.text


class TestCustomHook:
    """Tests for CustomHook."""

    def test_custom_transform(self):
        hook = CustomHook(
            name="custom",
            transform_fn=lambda t: t.upper(),
        )
        result = hook.transform("hello")
        assert result.text == "HELLO"

    def test_custom_inverse(self):
        hook = CustomHook(
            name="custom",
            transform_fn=lambda t: (t.upper(), {"original": t}),
            inverse_fn=lambda t, m: m.get("original", t),
        )
        result = hook.transform("hello")
        restored = hook.inverse(result.text, result.metadata)
        assert restored == "hello"


class TestHookPipeline:
    """Tests for HookPipeline."""

    def test_empty_pipeline(self):
        pipeline = HookPipeline()
        result = pipeline.pre_tokenize("hello")
        assert result == "hello"

    def test_add_hook(self):
        pipeline = HookPipeline()
        pipeline.add_hook(WhitespaceNormalizationHook())
        assert len(pipeline.hooks) == 1

    def test_chain_add_hook(self):
        pipeline = HookPipeline()
        pipeline.add_hook(WhitespaceNormalizationHook()).add_hook(LowercaseHook())
        assert len(pipeline.hooks) == 2

    def test_pre_tokenize(self):
        pipeline = HookPipeline(
            [
                WhitespaceNormalizationHook(),
                LowercaseHook(),
            ]
        )
        result = pipeline.pre_tokenize("  Hello  World  ")
        assert result == "hello world"

    def test_post_decode(self):
        pipeline = HookPipeline(
            [
                NumericNormalizationHook(),
            ]
        )
        # Pre-tokenize to set up metadata
        text = "Value is 42"
        encoded = pipeline.pre_tokenize(text)
        # Simulate decode that returns encoded text
        restored = pipeline.post_decode(encoded)
        assert restored == text

    def test_get_metadata(self):
        pipeline = HookPipeline(
            [
                NumericNormalizationHook(),
            ]
        )
        pipeline.pre_tokenize("Value is 42")
        metadata = pipeline.get_metadata()
        assert len(metadata) == 1
        assert metadata[0]["hook"] == "numeric_normalization"


class TestHookedTokenizer:
    """Tests for HookedTokenizer."""

    def test_encode(self):
        tokenizer = MockTokenizer()
        pipeline = HookPipeline([WhitespaceNormalizationHook()])
        hooked = HookedTokenizer(tokenizer, pipeline)
        tokens = hooked.encode("  hello  world  ")
        assert len(tokens) == 2

    def test_decode(self):
        tokenizer = MockTokenizer()
        pipeline = HookPipeline([NumericNormalizationHook()])
        hooked = HookedTokenizer(tokenizer, pipeline)

        # Encode with number
        tokens = hooked.encode("hello 42")
        # Decode should work
        decoded = hooked.decode(tokens)
        assert decoded is not None

    def test_vocab_size(self):
        tokenizer = MockTokenizer()
        pipeline = HookPipeline()
        hooked = HookedTokenizer(tokenizer, pipeline)
        assert hooked.vocab_size == tokenizer.vocab_size


class TestPipelineFactories:
    """Tests for pipeline factory functions."""

    def test_create_standard_pipeline(self):
        pipeline = create_standard_pipeline()
        assert len(pipeline.hooks) > 0

    def test_create_standard_pipeline_options(self):
        pipeline = create_standard_pipeline(
            numeric=True,
            structure=False,
            whitespace=True,
        )
        hook_names = [h.name for h in pipeline.hooks]
        assert "numeric_normalization" in hook_names
        assert "whitespace_normalization" in hook_names

    def test_create_math_pipeline(self):
        pipeline = create_math_pipeline()
        hook_names = [h.name for h in pipeline.hooks]
        assert "numeric_normalization" in hook_names

    def test_create_tool_pipeline(self):
        pipeline = create_tool_pipeline()
        hook_names = [h.name for h in pipeline.hooks]
        assert "structure_injection" in hook_names
