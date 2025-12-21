"""Tests for special_tokens module."""

from chuk_lazarus.data.tokenizers.special_tokens import (
    SpecialTokenConfig,
    SpecialTokenCount,
    SpecialTokenType,
    add_bos_token,
    add_eos_token,
    add_special_tokens,
    count_special_tokens,
    ensure_special_tokens,
    find_eos_positions,
    get_special_token_ids,
    get_special_token_mask,
    split_on_eos,
    strip_padding,
    strip_special_tokens,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3


class MockTokenizerWithExtras:
    """Mock tokenizer with additional special tokens."""

    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.additional_special_tokens_ids = [4, 5]


class TestSpecialTokenTypeEnum:
    """Tests for SpecialTokenType enum."""

    def test_all_values(self):
        assert SpecialTokenType.PAD == "pad"
        assert SpecialTokenType.UNK == "unk"
        assert SpecialTokenType.BOS == "bos"
        assert SpecialTokenType.EOS == "eos"
        assert SpecialTokenType.SEP == "sep"
        assert SpecialTokenType.CLS == "cls"
        assert SpecialTokenType.MASK == "mask"


class TestSpecialTokenCountModel:
    """Tests for SpecialTokenCount Pydantic model."""

    def test_default_values(self):
        count = SpecialTokenCount()
        assert count.pad == 0
        assert count.unk == 0
        assert count.total == 0

    def test_custom_values(self):
        count = SpecialTokenCount(pad=5, unk=2, bos=1, eos=1, other_special=3, total=12)
        assert count.pad == 5
        assert count.total == 12


class TestSpecialTokenConfigModel:
    """Tests for SpecialTokenConfig Pydantic model."""

    def test_default_values(self):
        config = SpecialTokenConfig()
        assert config.pad_token_id is None
        assert config.additional_special_ids == set()

    def test_with_ids(self):
        config = SpecialTokenConfig(
            pad_token_id=0,
            unk_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
        )
        assert config.pad_token_id == 0

    def test_from_tokenizer(self):
        tokenizer = MockTokenizer()
        config = SpecialTokenConfig.from_tokenizer(tokenizer)
        assert config.pad_token_id == 0
        assert config.eos_token_id == 3

    def test_from_tokenizer_with_extras(self):
        tokenizer = MockTokenizerWithExtras()
        config = SpecialTokenConfig.from_tokenizer(tokenizer)
        assert 4 in config.additional_special_ids
        assert 5 in config.additional_special_ids

    def test_all_special_ids(self):
        config = SpecialTokenConfig(
            pad_token_id=0,
            unk_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            additional_special_ids={4, 5},
        )
        all_ids = config.all_special_ids()
        assert all_ids == {0, 1, 2, 3, 4, 5}


class TestGetSpecialTokenIds:
    """Tests for get_special_token_ids function."""

    def test_basic(self):
        tokenizer = MockTokenizer()
        ids = get_special_token_ids(tokenizer)
        assert 0 in ids  # pad
        assert 1 in ids  # unk
        assert 2 in ids  # bos
        assert 3 in ids  # eos


class TestGetSpecialTokenMask:
    """Tests for get_special_token_mask function."""

    def test_with_special_tokens(self):
        tokenizer = MockTokenizer()
        token_ids = [2, 10, 11, 12, 3]  # bos, regular, regular, regular, eos
        mask = get_special_token_mask(token_ids, tokenizer)
        assert mask == [True, False, False, False, True]

    def test_no_special_tokens(self):
        tokenizer = MockTokenizer()
        token_ids = [10, 11, 12]
        mask = get_special_token_mask(token_ids, tokenizer)
        assert mask == [False, False, False]


class TestStripSpecialTokens:
    """Tests for strip_special_tokens function."""

    def test_strip_special(self):
        tokenizer = MockTokenizer()
        token_ids = [2, 10, 11, 3]  # bos, regular, regular, eos
        stripped = strip_special_tokens(token_ids, tokenizer)
        assert stripped == [10, 11]

    def test_no_special(self):
        tokenizer = MockTokenizer()
        token_ids = [10, 11, 12]
        stripped = strip_special_tokens(token_ids, tokenizer)
        assert stripped == [10, 11, 12]


class TestStripPadding:
    """Tests for strip_padding function."""

    def test_strip_right_padding(self):
        token_ids = [1, 2, 3, 0, 0, 0]
        stripped = strip_padding(token_ids, pad_token_id=0, from_left=False)
        assert stripped == [1, 2, 3]

    def test_strip_left_padding(self):
        token_ids = [0, 0, 1, 2, 3]
        stripped = strip_padding(token_ids, pad_token_id=0, from_left=True)
        assert stripped == [1, 2, 3]

    def test_no_padding(self):
        token_ids = [1, 2, 3]
        stripped = strip_padding(token_ids, pad_token_id=0)
        assert stripped == [1, 2, 3]

    def test_all_padding(self):
        token_ids = [0, 0, 0]
        stripped = strip_padding(token_ids, pad_token_id=0)
        assert stripped == []

    def test_empty_list(self):
        stripped = strip_padding([], pad_token_id=0)
        assert stripped == []


class TestAddBosToken:
    """Tests for add_bos_token function."""

    def test_add_bos(self):
        token_ids = [10, 11, 12]
        result = add_bos_token(token_ids, bos_token_id=2)
        assert result == [2, 10, 11, 12]

    def test_already_has_bos(self):
        token_ids = [2, 10, 11]
        result = add_bos_token(token_ids, bos_token_id=2)
        assert result == [2, 10, 11]

    def test_empty_list(self):
        result = add_bos_token([], bos_token_id=2)
        assert result == [2]


class TestAddEosToken:
    """Tests for add_eos_token function."""

    def test_add_eos(self):
        token_ids = [10, 11, 12]
        result = add_eos_token(token_ids, eos_token_id=3)
        assert result == [10, 11, 12, 3]

    def test_already_has_eos(self):
        token_ids = [10, 11, 3]
        result = add_eos_token(token_ids, eos_token_id=3)
        assert result == [10, 11, 3]

    def test_empty_list(self):
        result = add_eos_token([], eos_token_id=3)
        assert result == [3]


class TestAddSpecialTokens:
    """Tests for add_special_tokens function."""

    def test_add_both(self):
        token_ids = [10, 11]
        result = add_special_tokens(token_ids, bos_token_id=2, eos_token_id=3)
        assert result == [2, 10, 11, 3]

    def test_add_only_bos(self):
        token_ids = [10, 11]
        result = add_special_tokens(token_ids, bos_token_id=2)
        assert result == [2, 10, 11]

    def test_add_only_eos(self):
        token_ids = [10, 11]
        result = add_special_tokens(token_ids, eos_token_id=3)
        assert result == [10, 11, 3]


class TestEnsureSpecialTokens:
    """Tests for ensure_special_tokens function."""

    def test_ensure_both(self):
        tokenizer = MockTokenizer()
        token_ids = [10, 11]
        result = ensure_special_tokens(token_ids, tokenizer)
        assert result[0] == 2  # bos
        assert result[-1] == 3  # eos


class TestFindEosPositions:
    """Tests for find_eos_positions function."""

    def test_multiple_eos(self):
        token_ids = [1, 2, 3, 1, 3, 1]  # 3 is eos
        positions = find_eos_positions(token_ids, eos_token_id=3)
        assert positions == [2, 4]

    def test_no_eos(self):
        token_ids = [1, 2, 4, 5]
        positions = find_eos_positions(token_ids, eos_token_id=3)
        assert positions == []


class TestSplitOnEos:
    """Tests for split_on_eos function."""

    def test_split_basic(self):
        token_ids = [1, 2, 3, 4, 5, 3, 6, 7]  # 3 is eos
        segments = split_on_eos(token_ids, eos_token_id=3, keep_eos=True)
        assert len(segments) == 3
        assert segments[0] == [1, 2, 3]
        assert segments[1] == [4, 5, 3]
        assert segments[2] == [6, 7]

    def test_split_without_eos(self):
        token_ids = [1, 2, 3, 4, 5, 3]
        segments = split_on_eos(token_ids, eos_token_id=3, keep_eos=False)
        assert segments[0] == [1, 2]
        assert segments[1] == [4, 5]

    def test_no_eos_present(self):
        token_ids = [1, 2, 4, 5]
        segments = split_on_eos(token_ids, eos_token_id=3)
        assert segments == [[1, 2, 4, 5]]

    def test_empty_list(self):
        segments = split_on_eos([], eos_token_id=3)
        assert segments == []


class TestCountSpecialTokens:
    """Tests for count_special_tokens function."""

    def test_count_all_types(self):
        tokenizer = MockTokenizer()
        token_ids = [0, 0, 1, 2, 3, 3, 10, 11]  # 2 pad, 1 unk, 1 bos, 2 eos
        count = count_special_tokens(token_ids, tokenizer)
        assert count.pad == 2
        assert count.unk == 1
        assert count.bos == 1
        assert count.eos == 2
        assert count.total == 6

    def test_no_special(self):
        tokenizer = MockTokenizer()
        token_ids = [10, 11, 12]
        count = count_special_tokens(token_ids, tokenizer)
        assert count.pad == 0
        assert count.total == 0
