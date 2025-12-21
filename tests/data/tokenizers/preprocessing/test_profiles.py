"""Tests for tokenizer profiles."""

import pytest

from chuk_lazarus.data.tokenizers.preprocessing.profiles import (
    ProfiledTokenizer,
    ProfileManager,
    ProfileMode,
    TokenizerProfile,
    create_default_manager,
    create_inference_profile,
    create_math_profile,
    create_tool_profile,
    create_training_profile,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {"hello": 1, "world": 2, "value": 3, "is": 4}
        self._vocab_size = 100

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [self.vocab.get(w, 0) for w in text.lower().split()]
        if add_special_tokens:
            tokens = [1] + tokens + [2]  # BOS, EOS
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size


class TestTokenizerProfile:
    """Tests for TokenizerProfile model."""

    def test_default_values(self):
        profile = TokenizerProfile(name="test")
        assert profile.mode == ProfileMode.INFERENCE
        assert profile.stochastic is False
        assert profile.byte_fallback is True

    def test_custom_values(self):
        profile = TokenizerProfile(
            name="custom",
            mode=ProfileMode.TRAINING,
            stochastic=True,
            normalize_numbers=True,
        )
        assert profile.mode == ProfileMode.TRAINING
        assert profile.stochastic is True


class TestProfiledTokenizer:
    """Tests for ProfiledTokenizer."""

    def test_basic_encode(self):
        tokenizer = MockTokenizer()
        profile = create_inference_profile()
        profiled = ProfiledTokenizer(tokenizer, profile)
        tokens = profiled.encode("hello world")
        assert len(tokens) > 0

    def test_basic_decode(self):
        tokenizer = MockTokenizer()
        profile = create_inference_profile()
        profiled = ProfiledTokenizer(tokenizer, profile)
        tokens = profiled.encode("hello world")
        decoded = profiled.decode(tokens)
        assert "hello" in decoded or "<unk>" in decoded

    def test_set_profile(self):
        tokenizer = MockTokenizer()
        profile1 = create_inference_profile()
        profile2 = create_training_profile()
        profiled = ProfiledTokenizer(tokenizer, profile1)
        assert profiled.profile.name == "inference"
        profiled.set_profile(profile2)
        assert profiled.profile.name == "training"

    def test_truncation(self):
        tokenizer = MockTokenizer()
        profile = TokenizerProfile(
            name="truncate",
            max_length=3,
            truncation=True,
        )
        profiled = ProfiledTokenizer(tokenizer, profile)
        tokens = profiled.encode("hello world value is")
        assert len(tokens) <= 3

    def test_add_special_tokens_override(self):
        tokenizer = MockTokenizer()
        profile = TokenizerProfile(name="test", add_special_tokens=True)
        profiled = ProfiledTokenizer(tokenizer, profile)

        tokens_with = profiled.encode("hello", add_special_tokens=True)
        tokens_without = profiled.encode("hello", add_special_tokens=False)
        assert len(tokens_with) >= len(tokens_without)

    def test_vocab_size(self):
        tokenizer = MockTokenizer()
        profile = create_inference_profile()
        profiled = ProfiledTokenizer(tokenizer, profile)
        assert profiled.vocab_size == 100


class TestProfileFactories:
    """Tests for profile factory functions."""

    def test_create_training_profile(self):
        profile = create_training_profile()
        assert profile.name == "training"
        assert profile.mode == ProfileMode.TRAINING
        assert profile.truncation is True

    def test_create_inference_profile(self):
        profile = create_inference_profile()
        assert profile.name == "inference"
        assert profile.mode == ProfileMode.INFERENCE
        assert profile.stochastic is False

    def test_create_math_profile(self):
        profile = create_math_profile()
        assert profile.name == "math"
        assert profile.normalize_numbers is True

    def test_create_tool_profile(self):
        profile = create_tool_profile()
        assert profile.name == "tool"
        assert profile.inject_structures is True

    def test_custom_name(self):
        profile = create_training_profile(name="my_training")
        assert profile.name == "my_training"

    def test_custom_max_length(self):
        profile = create_training_profile(max_length=1024)
        assert profile.max_length == 1024


class TestProfileManager:
    """Tests for ProfileManager."""

    def test_register(self):
        manager = ProfileManager()
        profile = create_inference_profile()
        manager.register(profile)
        assert manager.get("inference") is not None

    def test_get(self):
        manager = ProfileManager()
        profile = create_inference_profile()
        manager.register(profile)
        retrieved = manager.get("inference")
        assert retrieved is not None
        assert retrieved.name == "inference"

    def test_get_missing(self):
        manager = ProfileManager()
        assert manager.get("nonexistent") is None

    def test_set_active(self):
        manager = ProfileManager()
        manager.register(create_inference_profile())
        manager.register(create_training_profile())
        manager.set_active("training")
        assert manager.active is not None
        assert manager.active.name == "training"

    def test_set_active_missing(self):
        manager = ProfileManager()
        with pytest.raises(ValueError):
            manager.set_active("nonexistent")

    def test_list_profiles(self):
        manager = ProfileManager()
        manager.register(create_inference_profile())
        manager.register(create_training_profile())
        profiles = manager.list_profiles()
        assert "inference" in profiles
        assert "training" in profiles

    def test_create_tokenizer(self):
        manager = ProfileManager()
        manager.register(create_inference_profile())
        manager.set_active("inference")

        tokenizer = MockTokenizer()
        profiled = manager.create_tokenizer(tokenizer)
        assert profiled.profile.name == "inference"

    def test_create_tokenizer_with_name(self):
        manager = ProfileManager()
        manager.register(create_inference_profile())
        manager.register(create_training_profile())

        tokenizer = MockTokenizer()
        profiled = manager.create_tokenizer(tokenizer, "training")
        assert profiled.profile.name == "training"

    def test_create_tokenizer_no_profile(self):
        manager = ProfileManager()
        tokenizer = MockTokenizer()
        with pytest.raises(ValueError):
            manager.create_tokenizer(tokenizer)


class TestCreateDefaultManager:
    """Tests for create_default_manager function."""

    def test_has_standard_profiles(self):
        manager = create_default_manager()
        profiles = manager.list_profiles()
        assert "training" in profiles
        assert "inference" in profiles
        assert "math" in profiles
        assert "tool" in profiles

    def test_default_active(self):
        manager = create_default_manager()
        assert manager.active is not None
        assert manager.active.name == "inference"
