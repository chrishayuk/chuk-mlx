"""Tests for token_debug module."""

from chuk_lazarus.data.tokenizers.token_debug import (
    TokenComparison,
    TokenInfo,
    UnknownTokenAnalysis,
    analyze_unknown_tokens,
    compare_tokenizations,
    find_token_by_string,
    format_token_table,
    get_similar_tokens,
    get_token_info,
    get_tokens_info,
    highlight_tokens,
    token_to_bytes,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {"hello": 0, "world": 1, "<unk>": 2, "the": 3}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._vocab.get("<unk>", 2)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestTokenInfoModel:
    """Tests for TokenInfo Pydantic model."""

    def test_valid_token_info(self):
        info = TokenInfo(
            token_id=42,
            token_str="hello",
            byte_repr="68 65 6c 6c 6f",
            char_count=5,
            byte_count=5,
        )
        assert info.token_id == 42
        assert info.token_str == "hello"
        assert info.byte_count == 5


class TestTokenComparisonModel:
    """Tests for TokenComparison Pydantic model."""

    def test_valid_comparison(self):
        comp = TokenComparison(
            text="hello world",
            tokenizer1_ids=[0, 1],
            tokenizer2_ids=[0, 1, 2],
            tokenizer1_tokens=["hello", "world"],
            tokenizer2_tokens=["hel", "lo", "world"],
            tokenizer1_count=2,
            tokenizer2_count=3,
        )
        assert comp.tokenizer1_count == 2
        assert comp.tokenizer2_count == 3


class TestUnknownTokenAnalysisModel:
    """Tests for UnknownTokenAnalysis Pydantic model."""

    def test_valid_analysis(self):
        analysis = UnknownTokenAnalysis(
            text="hello unknown",
            unknown_count=1,
            total_count=2,
            unknown_ratio=0.5,
            unknown_positions=[1],
            unknown_segments=["unknown"],
        )
        assert analysis.unknown_ratio == 0.5


class TestGetTokenInfo:
    """Tests for get_token_info function."""

    def test_basic_info(self):
        tokenizer = MockTokenizer()
        info = get_token_info(0, tokenizer)  # "hello"
        assert info.token_id == 0
        assert info.token_str == "hello"
        assert info.char_count == 5
        assert info.byte_count == 5
        assert "68" in info.byte_repr  # 'h' in hex

    def test_decode_error_handling(self):
        # Create tokenizer with ID that might fail
        tokenizer = MockTokenizer({})
        info = get_token_info(999, tokenizer)
        assert info.token_id == 999
        # Should handle the error gracefully


class TestGetTokensInfo:
    """Tests for get_tokens_info function."""

    def test_multiple_tokens(self):
        tokenizer = MockTokenizer()
        infos = get_tokens_info([0, 1], tokenizer)
        assert len(infos) == 2
        assert infos[0].token_str == "hello"
        assert infos[1].token_str == "world"

    def test_empty_list(self):
        tokenizer = MockTokenizer()
        infos = get_tokens_info([], tokenizer)
        assert infos == []


class TestCompareTokenizations:
    """Tests for compare_tokenizations function."""

    def test_same_tokenizer(self):
        tokenizer = MockTokenizer()
        comparison = compare_tokenizations("hello world", tokenizer, tokenizer)
        assert comparison.tokenizer1_count == comparison.tokenizer2_count
        assert comparison.tokenizer1_ids == comparison.tokenizer2_ids

    def test_different_tokenizers(self):
        tokenizer1 = MockTokenizer({"hello": 0, "world": 1, "<unk>": 2})
        tokenizer2 = MockTokenizer({"hel": 0, "lo": 1, "world": 2, "<unk>": 3})
        comparison = compare_tokenizations("hello world", tokenizer1, tokenizer2)
        assert comparison.text == "hello world"
        assert comparison.tokenizer1_count == 2


class TestAnalyzeUnknownTokens:
    """Tests for analyze_unknown_tokens function."""

    def test_no_unknowns(self):
        tokenizer = MockTokenizer()
        analysis = analyze_unknown_tokens("hello world", tokenizer)
        assert analysis.unknown_count == 0
        assert analysis.unknown_ratio == 0.0

    def test_with_unknowns(self):
        tokenizer = MockTokenizer()
        analysis = analyze_unknown_tokens("hello xyz", tokenizer)
        assert analysis.unknown_count == 1
        assert analysis.total_count == 2
        assert analysis.unknown_ratio == 0.5

    def test_custom_unk_id(self):
        tokenizer = MockTokenizer()
        analysis = analyze_unknown_tokens("hello xyz", tokenizer, unk_token_id=2)
        assert analysis.unknown_count == 1


class TestHighlightTokens:
    """Tests for highlight_tokens function."""

    def test_default_separator(self):
        tokenizer = MockTokenizer()
        highlighted = highlight_tokens("hello world", tokenizer)
        assert "|" in highlighted

    def test_custom_separator(self):
        tokenizer = MockTokenizer()
        highlighted = highlight_tokens("hello world", tokenizer, separator=" | ")
        assert " | " in highlighted

    def test_with_special_tokens(self):
        tokenizer = MockTokenizer()
        highlighted = highlight_tokens("hello world", tokenizer, add_special_tokens=True)
        assert "hello" in highlighted


class TestTokenToBytes:
    """Tests for token_to_bytes function."""

    def test_ascii_token(self):
        tokenizer = MockTokenizer()
        bytes_result = token_to_bytes(0, tokenizer)  # "hello"
        assert bytes_result == b"hello"

    def test_unicode_token(self):
        tokenizer = MockTokenizer({"日本語": 0})
        bytes_result = token_to_bytes(0, tokenizer)
        assert bytes_result == "日本語".encode()


class TestFormatTokenTable:
    """Tests for format_token_table function."""

    def test_basic_table(self):
        tokenizer = MockTokenizer()
        table = format_token_table([0, 1], tokenizer)
        assert "Index" in table
        assert "ID" in table
        assert "Token" in table
        assert "hello" in table
        assert "world" in table

    def test_empty_list(self):
        tokenizer = MockTokenizer()
        table = format_token_table([], tokenizer)
        assert "Index" in table  # Header still present


class TestFindTokenByString:
    """Tests for find_token_by_string function."""

    def test_find_matching_tokens(self):
        tokenizer = MockTokenizer({"hello": 0, "help": 1, "world": 2})
        matches = find_token_by_string("hel", tokenizer)
        assert 0 in matches  # hello
        assert 1 in matches  # help
        assert 2 not in matches  # world

    def test_no_matches(self):
        tokenizer = MockTokenizer({"hello": 0, "world": 1})
        matches = find_token_by_string("xyz", tokenizer)
        assert len(matches) == 0


class TestGetSimilarTokens:
    """Tests for get_similar_tokens function."""

    def test_find_similar(self):
        tokenizer = MockTokenizer({"hello": 0, "help": 1, "helm": 2, "world": 3})
        similar = get_similar_tokens(0, tokenizer, max_results=5)  # Similar to "hello"
        # Should find tokens starting with "he"
        assert any(t.token_str == "help" for t in similar)
        assert any(t.token_str == "helm" for t in similar)

    def test_no_similar(self):
        tokenizer = MockTokenizer({"a": 0, "b": 1})
        similar = get_similar_tokens(0, tokenizer)  # "a"
        # Might not find any similar (depends on prefix matching)
        assert isinstance(similar, list)

    def test_max_results(self):
        vocab = {f"test{i}": i for i in range(20)}
        tokenizer = MockTokenizer(vocab)
        similar = get_similar_tokens(0, tokenizer, max_results=5)
        assert len(similar) <= 5

    def test_empty_token(self):
        tokenizer = MockTokenizer({"": 0, "a": 1})
        similar = get_similar_tokens(0, tokenizer)
        assert similar == []
