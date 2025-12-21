"""Tests for length_buckets module."""

from chuk_lazarus.data.tokenizers.curriculum.length_buckets import (
    CurriculumSchedule,
    LengthBucket,
    LengthBucketConfig,
    create_length_buckets,
    get_curriculum_schedule,
    sort_by_length,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"<unk>": 0, "a": 1, "b": 2, "c": 3}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1] * len(text.split())

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestLengthBucketConfigModel:
    """Tests for LengthBucketConfig model."""

    def test_default_values(self):
        config = LengthBucketConfig()
        assert config.num_buckets == 5
        assert config.min_length == 1
        assert config.max_length is None
        assert config.log_scale is False

    def test_custom_values(self):
        config = LengthBucketConfig(
            num_buckets=10,
            min_length=5,
            max_length=100,
            log_scale=True,
        )
        assert config.num_buckets == 10
        assert config.log_scale is True


class TestLengthBucketModel:
    """Tests for LengthBucket model."""

    def test_valid_bucket(self):
        bucket = LengthBucket(
            bucket_id=0,
            min_tokens=1,
            max_tokens=10,
            sample_count=50,
            sample_indices=[0, 1, 2, 3, 4],
            avg_length=5.5,
        )
        assert bucket.bucket_id == 0
        assert bucket.sample_count == 50
        assert len(bucket.sample_indices) == 5

    def test_empty_bucket(self):
        bucket = LengthBucket(
            bucket_id=1,
            min_tokens=10,
            max_tokens=20,
            sample_count=0,
            sample_indices=[],
            avg_length=0.0,
        )
        assert bucket.sample_count == 0


class TestCurriculumScheduleModel:
    """Tests for CurriculumSchedule model."""

    def test_valid_schedule(self):
        buckets = [
            LengthBucket(
                bucket_id=0,
                min_tokens=1,
                max_tokens=10,
                sample_count=20,
                sample_indices=[],
                avg_length=5.0,
            ),
            LengthBucket(
                bucket_id=1,
                min_tokens=10,
                max_tokens=20,
                sample_count=30,
                sample_indices=[],
                avg_length=15.0,
            ),
        ]
        schedule = CurriculumSchedule(
            buckets=buckets,
            total_samples=50,
            schedule_order=[0, 1],
            warmup_samples=20,
            ramp_samples=10,
        )
        assert len(schedule.buckets) == 2
        assert schedule.warmup_samples == 20


class TestCreateLengthBuckets:
    """Tests for create_length_buckets function."""

    def test_basic_bucketing(self):
        tokenizer = MockTokenizer()
        texts = [
            "a",
            "a b",
            "a b c",
            "a b c d",
            "a b c d e",
        ]
        buckets = create_length_buckets(texts, tokenizer)
        assert len(buckets) > 0
        assert all(isinstance(b, LengthBucket) for b in buckets)

    def test_custom_num_buckets(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e", "a b c d e f"]
        config = LengthBucketConfig(num_buckets=3)
        buckets = create_length_buckets(texts, tokenizer, config)
        # Should have exactly 3 buckets
        assert len(buckets) == 3

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        buckets = create_length_buckets(texts, tokenizer)
        assert len(buckets) == 0

    def test_log_scale_buckets(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c d", "a b c d e f g h"]
        config = LengthBucketConfig(log_scale=True, num_buckets=3)
        buckets = create_length_buckets(texts, tokenizer, config)
        assert len(buckets) > 0

    def test_min_tokens_in_bucket(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c d e"]
        buckets = create_length_buckets(texts, tokenizer)
        # Check min_tokens is set correctly
        for bucket in buckets:
            assert bucket.min_tokens >= 0

    def test_max_length_cap(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c d e f g h i j"]
        config = LengthBucketConfig(max_length=5)
        buckets = create_length_buckets(texts, tokenizer, config)
        assert len(buckets) > 0

    def test_sample_indices_correct(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b c", "a b c d e"]
        buckets = create_length_buckets(texts, tokenizer)
        # All indices should be valid
        all_indices = []
        for bucket in buckets:
            all_indices.extend(bucket.sample_indices)
        assert all(0 <= i < len(texts) for i in all_indices)

    def test_all_texts_assigned(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d"]
        buckets = create_length_buckets(texts, tokenizer)
        total_assigned = sum(b.sample_count for b in buckets)
        # Should assign all texts
        assert total_assigned == len(texts)

    def test_avg_length_calculation(self):
        tokenizer = MockTokenizer()
        texts = ["a b", "a b c", "a b c d"]
        config = LengthBucketConfig(num_buckets=1)
        buckets = create_length_buckets(texts, tokenizer, config)
        if buckets and buckets[0].sample_count > 0:
            assert buckets[0].avg_length > 0


class TestSortByLength:
    """Tests for sort_by_length function."""

    def test_basic_sorting(self):
        tokenizer = MockTokenizer()
        texts = ["a b c", "a", "a b"]
        sorted_items = sort_by_length(texts, tokenizer)
        # Returns list of (index, text, length)
        assert len(sorted_items) == 3
        # Check sorted by length
        lengths = [item[2] for item in sorted_items]
        assert lengths == sorted(lengths)

    def test_descending_sort(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b c", "a b"]
        sorted_items = sort_by_length(texts, tokenizer, reverse=True)
        lengths = [item[2] for item in sorted_items]
        assert lengths == sorted(lengths, reverse=True)

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        sorted_items = sort_by_length(texts, tokenizer)
        assert len(sorted_items) == 0

    def test_preserves_content(self):
        tokenizer = MockTokenizer()
        texts = ["a b c", "a", "a b"]
        sorted_items = sort_by_length(texts, tokenizer)
        sorted_texts = [item[1] for item in sorted_items]
        assert set(sorted_texts) == set(texts)

    def test_single_text(self):
        tokenizer = MockTokenizer()
        texts = ["a b c"]
        sorted_items = sort_by_length(texts, tokenizer)
        assert len(sorted_items) == 1
        assert sorted_items[0][2] == 3  # 3 tokens

    def test_returns_tuples(self):
        tokenizer = MockTokenizer()
        texts = ["a b c", "a"]
        sorted_items = sort_by_length(texts, tokenizer)
        for item in sorted_items:
            assert len(item) == 3  # (index, text, length)
            assert isinstance(item[0], int)  # index
            assert isinstance(item[1], str)  # text
            assert isinstance(item[2], int)  # length


class TestGetCurriculumSchedule:
    """Tests for get_curriculum_schedule function."""

    def test_basic_schedule(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        schedule = get_curriculum_schedule(texts, tokenizer)
        assert isinstance(schedule, CurriculumSchedule)
        assert schedule.total_samples == 5

    def test_schedule_order(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d"]
        schedule = get_curriculum_schedule(texts, tokenizer)
        # Schedule should order buckets from shortest to longest
        assert len(schedule.schedule_order) > 0

    def test_warmup_samples(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        schedule = get_curriculum_schedule(texts, tokenizer)
        assert schedule.warmup_samples >= 0

    def test_ramp_samples(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        schedule = get_curriculum_schedule(texts, tokenizer)
        assert schedule.ramp_samples >= 0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        schedule = get_curriculum_schedule(texts, tokenizer)
        assert schedule.total_samples == 0

    def test_config_parameter(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d"]
        config = LengthBucketConfig(num_buckets=2)
        schedule = get_curriculum_schedule(texts, tokenizer, config=config)
        assert len(schedule.buckets) == 2

    def test_custom_warmup_ratio(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        schedule = get_curriculum_schedule(texts, tokenizer, warmup_ratio=0.5)
        # Warmup should be about 50% of samples
        assert schedule.warmup_samples >= 2

    def test_custom_ramp_ratio(self):
        tokenizer = MockTokenizer()
        texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        schedule = get_curriculum_schedule(texts, tokenizer, ramp_ratio=0.5)
        # Ramp should be about 50% of samples
        assert schedule.ramp_samples >= 2
