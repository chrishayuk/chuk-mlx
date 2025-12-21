"""Tests for vocab_manager module."""

from collections import Counter

from chuk_lazarus.data.tokenizers.vocab_manager import (
    ConflictResolution,
    SortOrder,
    VocabularyDiff,
    VocabularyIssues,
    VocabularyStats,
    create_id_to_token,
    extend_vocabulary,
    filter_vocabulary,
    get_vocabulary_diff,
    get_vocabulary_stats,
    merge_vocabularies,
    renumber_vocabulary,
    shrink_vocabulary,
    validate_vocabulary,
)


class TestConflictResolutionEnum:
    """Tests for ConflictResolution enum."""

    def test_values(self):
        assert ConflictResolution.FIRST == "first"
        assert ConflictResolution.SECOND == "second"
        assert ConflictResolution.RENUMBER == "renumber"


class TestSortOrderEnum:
    """Tests for SortOrder enum."""

    def test_values(self):
        assert SortOrder.BY_ID == "id"
        assert SortOrder.ALPHABETICAL == "alpha"


class TestVocabularyStatsModel:
    """Tests for VocabularyStats Pydantic model."""

    def test_valid_stats(self):
        stats = VocabularyStats(
            size=1000,
            min_id=0,
            max_id=999,
            id_range=1000,
            avg_token_length=4.5,
            max_token_length=20,
            min_token_length=1,
        )
        assert stats.size == 1000
        assert stats.avg_token_length == 4.5


class TestVocabularyIssuesModel:
    """Tests for VocabularyIssues Pydantic model."""

    def test_no_issues(self):
        issues = VocabularyIssues()
        assert not issues.has_issues()
        assert issues.duplicate_ids == []
        assert issues.missing_ids == []
        assert issues.negative_ids == []

    def test_has_issues(self):
        issues = VocabularyIssues(duplicate_ids=[{"id": 1, "tokens": ["a", "b"]}])
        assert issues.has_issues()


class TestVocabularyDiffModel:
    """Tests for VocabularyDiff Pydantic model."""

    def test_valid_diff(self):
        diff = VocabularyDiff(
            only_in_first={"token1": 0},
            only_in_second={"token2": 1},
            in_both_count=5,
        )
        assert diff.in_both_count == 5


class TestMergeVocabularies:
    """Tests for merge_vocabularies function."""

    def test_merge_first_priority(self):
        vocab1 = {"a": 0, "b": 1}
        vocab2 = {"c": 2, "d": 3}
        merged = merge_vocabularies(vocab1, vocab2, ConflictResolution.FIRST)
        assert "a" in merged
        assert "c" in merged
        assert len(merged) == 4

    def test_merge_second_priority(self):
        vocab1 = {"a": 0, "b": 1}
        vocab2 = {"b": 10, "c": 2}
        merged = merge_vocabularies(vocab1, vocab2, ConflictResolution.SECOND)
        assert merged["b"] == 10  # Second vocab's ID

    def test_merge_renumber(self):
        vocab1 = {"a": 0, "b": 1}
        vocab2 = {"c": 0, "d": 1}  # Same IDs
        merged = merge_vocabularies(vocab1, vocab2, ConflictResolution.RENUMBER)
        assert len(merged) == 4
        # All IDs should be unique
        assert len(set(merged.values())) == 4

    def test_merge_empty_first(self):
        vocab1 = {}
        vocab2 = {"a": 0, "b": 1}
        merged = merge_vocabularies(vocab1, vocab2)
        assert merged == {"a": 0, "b": 1}

    def test_merge_empty_second(self):
        vocab1 = {"a": 0, "b": 1}
        vocab2 = {}
        merged = merge_vocabularies(vocab1, vocab2)
        assert merged == {"a": 0, "b": 1}


class TestFilterVocabulary:
    """Tests for filter_vocabulary function."""

    def test_filter_by_min_freq(self):
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        counts = Counter({"a": 10, "b": 5, "c": 1, "d": 1})
        filtered = filter_vocabulary(vocab, counts, min_freq=5)
        assert "a" in filtered
        assert "b" in filtered
        assert "c" not in filtered
        assert "d" not in filtered

    def test_filter_by_max_freq(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        counts = Counter({"a": 100, "b": 50, "c": 10})
        filtered = filter_vocabulary(vocab, counts, min_freq=1, max_freq=60)
        assert "a" not in filtered
        assert "b" in filtered
        assert "c" in filtered

    def test_keep_special(self):
        vocab = {"<pad>": 0, "a": 1, "b": 2}
        counts = Counter({"a": 10, "b": 1})
        filtered = filter_vocabulary(vocab, counts, min_freq=5, keep_special={"<pad>"})
        assert "<pad>" in filtered
        assert "a" in filtered
        assert "b" not in filtered


class TestExtendVocabulary:
    """Tests for extend_vocabulary function."""

    def test_extend_basic(self):
        vocab = {"a": 0, "b": 1}
        extended = extend_vocabulary(vocab, ["c", "d"])
        assert "c" in extended
        assert "d" in extended
        assert extended["c"] == 2
        assert extended["d"] == 3

    def test_extend_with_start_id(self):
        vocab = {"a": 0, "b": 1}
        extended = extend_vocabulary(vocab, ["c", "d"], start_id=100)
        assert extended["c"] == 100
        assert extended["d"] == 101

    def test_extend_no_duplicates(self):
        vocab = {"a": 0, "b": 1}
        extended = extend_vocabulary(vocab, ["b", "c"])  # b already exists
        assert extended["b"] == 1  # Original ID preserved
        assert "c" in extended

    def test_extend_empty_vocab(self):
        vocab = {}
        extended = extend_vocabulary(vocab, ["a", "b"])
        assert extended["a"] == 0
        assert extended["b"] == 1


class TestShrinkVocabulary:
    """Tests for shrink_vocabulary function."""

    def test_shrink_basic(self):
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        shrunk = shrink_vocabulary(vocab, {"b", "d"})
        assert "a" in shrunk
        assert "c" in shrunk
        assert "b" not in shrunk
        assert "d" not in shrunk

    def test_shrink_renumber(self):
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        shrunk = shrink_vocabulary(vocab, {"b"}, renumber=True)
        # IDs should be contiguous after renumbering
        ids = sorted(shrunk.values())
        assert ids == list(range(len(ids)))

    def test_shrink_no_renumber(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        shrunk = shrink_vocabulary(vocab, {"b"}, renumber=False)
        assert shrunk["a"] == 0
        assert shrunk["c"] == 2  # Original ID preserved


class TestGetVocabularyDiff:
    """Tests for get_vocabulary_diff function."""

    def test_diff_basic(self):
        vocab1 = {"a": 0, "b": 1, "c": 2}
        vocab2 = {"b": 1, "c": 2, "d": 3}
        diff = get_vocabulary_diff(vocab1, vocab2)
        assert diff.only_in_first == {"a": 0}
        assert diff.only_in_second == {"d": 3}
        assert diff.in_both_count == 2

    def test_diff_identical(self):
        vocab = {"a": 0, "b": 1}
        diff = get_vocabulary_diff(vocab, vocab)
        assert diff.only_in_first == {}
        assert diff.only_in_second == {}
        assert diff.in_both_count == 2

    def test_diff_disjoint(self):
        vocab1 = {"a": 0, "b": 1}
        vocab2 = {"c": 2, "d": 3}
        diff = get_vocabulary_diff(vocab1, vocab2)
        assert diff.in_both_count == 0


class TestRenumberVocabulary:
    """Tests for renumber_vocabulary function."""

    def test_renumber_by_id(self):
        vocab = {"a": 5, "b": 10, "c": 3}
        renumbered = renumber_vocabulary(vocab, start_id=0, sort_by=SortOrder.BY_ID)
        # Should maintain order by original ID: c(3), a(5), b(10)
        assert renumbered["c"] == 0
        assert renumbered["a"] == 1
        assert renumbered["b"] == 2

    def test_renumber_alphabetical(self):
        vocab = {"c": 0, "a": 1, "b": 2}
        renumbered = renumber_vocabulary(vocab, start_id=0, sort_by=SortOrder.ALPHABETICAL)
        assert renumbered["a"] == 0
        assert renumbered["b"] == 1
        assert renumbered["c"] == 2

    def test_renumber_custom_start(self):
        vocab = {"a": 0, "b": 1}
        renumbered = renumber_vocabulary(vocab, start_id=100)
        assert renumbered["a"] == 100
        assert renumbered["b"] == 101


class TestCreateIdToToken:
    """Tests for create_id_to_token function."""

    def test_basic(self):
        vocab = {"hello": 0, "world": 1}
        id_to_token = create_id_to_token(vocab)
        assert id_to_token[0] == "hello"
        assert id_to_token[1] == "world"

    def test_empty_vocab(self):
        id_to_token = create_id_to_token({})
        assert id_to_token == {}


class TestValidateVocabulary:
    """Tests for validate_vocabulary function."""

    def test_valid_vocab(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        issues = validate_vocabulary(vocab)
        assert not issues.has_issues()

    def test_duplicate_ids(self):
        vocab = {"a": 0, "b": 0, "c": 1}  # a and b have same ID
        issues = validate_vocabulary(vocab)
        assert issues.has_issues()
        assert len(issues.duplicate_ids) == 1
        assert issues.duplicate_ids[0]["id"] == 0

    def test_negative_ids(self):
        vocab = {"a": -1, "b": 0, "c": 1}
        issues = validate_vocabulary(vocab)
        assert issues.has_issues()
        assert len(issues.negative_ids) == 1

    def test_missing_ids(self):
        vocab = {"a": 0, "b": 2, "c": 4}  # Missing 1 and 3
        issues = validate_vocabulary(vocab)
        assert issues.has_issues()
        assert 1 in issues.missing_ids
        assert 3 in issues.missing_ids

    def test_empty_vocab(self):
        issues = validate_vocabulary({})
        assert not issues.has_issues()


class TestGetVocabularyStats:
    """Tests for get_vocabulary_stats function."""

    def test_basic_stats(self):
        vocab = {"a": 0, "bb": 1, "ccc": 2}
        stats = get_vocabulary_stats(vocab)
        assert stats.size == 3
        assert stats.min_id == 0
        assert stats.max_id == 2
        assert stats.id_range == 3
        assert stats.min_token_length == 1
        assert stats.max_token_length == 3
        assert stats.avg_token_length == 2.0

    def test_empty_vocab(self):
        stats = get_vocabulary_stats({})
        assert stats.size == 0
        assert stats.min_id == 0
        assert stats.max_id == 0
        assert stats.avg_token_length == 0.0
