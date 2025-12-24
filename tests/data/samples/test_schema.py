"""Tests for canonical sample schema."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from chuk_lazarus.data.samples import (
    DatasetFingerprint,
    DatasetSource,
    PreferenceSample,
    Sample,
    SampleMeta,
    SampleType,
    compute_dataset_fingerprint,
)
from chuk_lazarus.data.samples.schema import (
    DifficultyLevel,
    load_samples,
    save_samples,
)


def raises_validation_error_matching(pattern: str):
    """Context manager that checks for ValidationError with message matching pattern."""
    return pytest.raises(ValidationError, match=pattern)


# =============================================================================
# SampleMeta Tests
# =============================================================================


class TestSampleMeta:
    """Tests for SampleMeta model."""

    def test_create_minimal(self):
        """Test creating metadata with required fields only."""
        meta = SampleMeta(sample_id="s001", dataset_id="train")
        assert meta.sample_id == "s001"
        assert meta.dataset_id == "train"
        assert meta.source == DatasetSource.LOCAL  # default

    def test_create_with_all_fields(self):
        """Test creating metadata with all fields."""
        meta = SampleMeta(
            sample_id="s001",
            dataset_id="train",
            episode_id="ep_123",
            source=DatasetSource.GYM,
            difficulty=DifficultyLevel.HARD,
            difficulty_score=0.75,
            reward=1.0,
            success=True,
            original_index=42,
            split="train",
        )
        assert meta.episode_id == "ep_123"
        assert meta.source == DatasetSource.GYM
        assert meta.difficulty == DifficultyLevel.HARD
        assert meta.difficulty_score == 0.75
        assert meta.reward == 1.0
        assert meta.success is True
        assert meta.original_index == 42
        assert meta.split == "train"

    def test_create_factory(self):
        """Test create factory method."""
        meta = SampleMeta.create(
            sample_id="s002",
            dataset_id="eval",
            difficulty=DifficultyLevel.EASY,
        )
        assert meta.sample_id == "s002"
        assert meta.difficulty == DifficultyLevel.EASY

    def test_immutable(self):
        """Test that metadata is immutable (frozen)."""
        meta = SampleMeta(sample_id="s001", dataset_id="train")
        with pytest.raises(ValidationError):
            meta.sample_id = "changed"

    def test_difficulty_score_bounds(self):
        """Test difficulty_score validation (0.0-1.0)."""
        # Valid scores
        SampleMeta(sample_id="s1", dataset_id="d1", difficulty_score=0.0)
        SampleMeta(sample_id="s1", dataset_id="d1", difficulty_score=0.5)
        SampleMeta(sample_id="s1", dataset_id="d1", difficulty_score=1.0)

        # Invalid scores
        with pytest.raises(ValidationError):
            SampleMeta(sample_id="s1", dataset_id="d1", difficulty_score=-0.1)
        with pytest.raises(ValidationError):
            SampleMeta(sample_id="s1", dataset_id="d1", difficulty_score=1.1)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            SampleMeta(
                sample_id="s001",
                dataset_id="train",
                unknown_field="value",
            )


# =============================================================================
# Sample Tests
# =============================================================================


class TestSample:
    """Tests for Sample model."""

    def test_create_minimal(self):
        """Test creating sample with required fields."""
        sample = Sample(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        assert sample.length == 5
        assert sample.num_loss_tokens == 3
        assert sample.sample_type == SampleType.SFT

    def test_create_with_segment_ids(self):
        """Test creating sample with segment IDs (packing)."""
        sample = Sample(
            input_ids=[1, 2, 3, 4, 5, 6],
            loss_mask=[1, 1, 1, 1, 1, 1],
            segment_ids=[0, 0, 0, 1, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        assert sample.num_segments == 2

    def test_list_to_tuple_conversion(self):
        """Test that lists are converted to tuples."""
        sample = Sample(
            input_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        assert isinstance(sample.input_ids, tuple)
        assert isinstance(sample.loss_mask, tuple)

    def test_immutable(self):
        """Test that samples are immutable (frozen)."""
        sample = Sample(
            input_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        with pytest.raises(ValidationError):
            sample.input_ids = (4, 5, 6)

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise validation error."""
        with raises_validation_error_matching("input_ids length"):
            Sample(
                input_ids=[1, 2, 3, 4, 5],
                loss_mask=[0, 1, 1],  # Wrong length
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            )

    def test_segment_ids_length_mismatch_raises(self):
        """Test that segment_ids length mismatch raises validation error."""
        with raises_validation_error_matching("segment_ids length"):
            Sample(
                input_ids=[1, 2, 3, 4, 5],
                loss_mask=[0, 0, 1, 1, 1],
                segment_ids=[0, 0, 1],  # Wrong length
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            )

    def test_invalid_loss_mask_values(self):
        """Test that loss_mask must contain only 0 or 1."""
        with raises_validation_error_matching("loss_mask must contain"):
            Sample(
                input_ids=[1, 2, 3],
                loss_mask=[0, 2, 1],  # Invalid value
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            )

    def test_segment_ids_non_monotonic_raises(self):
        """Test that non-monotonic segment_ids raise validation error."""
        with raises_validation_error_matching("monotonically non-decreasing"):
            Sample(
                input_ids=[1, 2, 3, 4, 5],
                loss_mask=[1, 1, 1, 1, 1],
                segment_ids=[0, 1, 0, 1, 2],  # Not monotonic
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            )

    def test_valid_segment_ids_patterns(self):
        """Test valid segment ID patterns."""
        # Single segment
        Sample(
            input_ids=[1, 2, 3],
            loss_mask=[1, 1, 1],
            segment_ids=[0, 0, 0],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )

        # Multiple segments
        Sample(
            input_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            loss_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            segment_ids=[0, 0, 0, 1, 1, 2, 2, 2, 2],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )

        # Same ID repeated (valid)
        Sample(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[1, 1, 1, 1, 1],
            segment_ids=[0, 0, 1, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )

    def test_properties(self):
        """Test sample properties."""
        sample = Sample(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            segment_ids=[0, 0, 0, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        assert sample.length == 5
        assert sample.num_loss_tokens == 3
        assert sample.num_segments == 2

    def test_num_segments_no_segment_ids(self):
        """Test num_segments when segment_ids is None."""
        sample = Sample(
            input_ids=[1, 2, 3],
            loss_mask=[1, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        assert sample.num_segments == 1

    def test_to_lists(self):
        """Test conversion to dict with lists."""
        sample = Sample(
            input_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            meta=SampleMeta(sample_id="s001", dataset_id="train"),
        )
        data = sample.to_lists()

        assert data["input_ids"] == [1, 2, 3]
        assert data["loss_mask"] == [0, 1, 1]
        assert data["segment_ids"] is None
        assert data["meta"]["sample_id"] == "s001"
        assert data["sample_type"] == "sft"

    def test_from_lists(self):
        """Test creation from dict with lists."""
        data = {
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "meta": {"sample_id": "s001", "dataset_id": "train"},
            "sample_type": "sft",
        }
        sample = Sample.from_lists(data)

        assert sample.input_ids == (1, 2, 3)
        assert sample.loss_mask == (0, 1, 1)
        assert sample.meta.sample_id == "s001"

    def test_roundtrip_serialization(self):
        """Test that to_lists -> from_lists preserves data."""
        original = Sample(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 0, 1, 1, 1],
            segment_ids=[0, 0, 0, 1, 1],
            meta=SampleMeta(
                sample_id="s001",
                dataset_id="train",
                difficulty=DifficultyLevel.MEDIUM,
            ),
            sample_type=SampleType.RL,
        )

        data = original.to_lists()
        restored = Sample.from_lists(data)

        assert restored.input_ids == original.input_ids
        assert restored.loss_mask == original.loss_mask
        assert restored.segment_ids == original.segment_ids
        assert restored.meta.sample_id == original.meta.sample_id
        assert restored.sample_type == original.sample_type


# =============================================================================
# PreferenceSample Tests
# =============================================================================


class TestPreferenceSample:
    """Tests for PreferenceSample model."""

    def test_create_basic(self):
        """Test creating a preference sample."""
        sample = PreferenceSample(
            chosen_input_ids=[1, 2, 3, 4, 5],
            chosen_loss_mask=[0, 0, 1, 1, 1],
            rejected_input_ids=[1, 2, 3, 6, 7, 8],
            rejected_loss_mask=[0, 0, 1, 1, 1, 1],
            prompt_length=2,
            meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
        )
        assert sample.chosen_length == 5
        assert sample.rejected_length == 6
        assert sample.max_length == 6
        assert sample.prompt_length == 2

    def test_length_mismatch_chosen(self):
        """Test validation of chosen sequence length consistency."""
        with raises_validation_error_matching("chosen_input_ids length"):
            PreferenceSample(
                chosen_input_ids=[1, 2, 3, 4, 5],
                chosen_loss_mask=[0, 1, 1],  # Wrong length
                rejected_input_ids=[1, 2, 3],
                rejected_loss_mask=[0, 1, 1],
                prompt_length=1,
                meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
            )

    def test_length_mismatch_rejected(self):
        """Test validation of rejected sequence length consistency."""
        with raises_validation_error_matching("rejected_input_ids length"):
            PreferenceSample(
                chosen_input_ids=[1, 2, 3],
                chosen_loss_mask=[0, 1, 1],
                rejected_input_ids=[1, 2, 3, 4, 5],
                rejected_loss_mask=[0, 1, 1],  # Wrong length
                prompt_length=1,
                meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
            )

    def test_prompt_length_exceeds_chosen(self):
        """Test that prompt_length cannot exceed chosen sequence."""
        with raises_validation_error_matching("prompt_length.*chosen_input_ids"):
            PreferenceSample(
                chosen_input_ids=[1, 2, 3],
                chosen_loss_mask=[0, 1, 1],
                rejected_input_ids=[1, 2, 3, 4, 5],
                rejected_loss_mask=[0, 1, 1, 1, 1],
                prompt_length=10,  # Too long
                meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
            )

    def test_prompt_length_exceeds_rejected(self):
        """Test that prompt_length cannot exceed rejected sequence."""
        with raises_validation_error_matching("prompt_length.*rejected_input_ids"):
            PreferenceSample(
                chosen_input_ids=[1, 2, 3, 4, 5, 6, 7],
                chosen_loss_mask=[0, 0, 0, 1, 1, 1, 1],
                rejected_input_ids=[1, 2, 3],
                rejected_loss_mask=[0, 1, 1],
                prompt_length=5,  # Exceeds rejected length
                meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
            )

    def test_to_samples(self):
        """Test conversion to separate Sample objects."""
        pref = PreferenceSample(
            chosen_input_ids=[1, 2, 3, 4, 5],
            chosen_loss_mask=[0, 0, 1, 1, 1],
            rejected_input_ids=[1, 2, 3, 6, 7],
            rejected_loss_mask=[0, 0, 1, 1, 1],
            prompt_length=2,
            meta=SampleMeta(sample_id="p001", dataset_id="prefs"),
        )

        chosen, rejected = pref.to_samples()

        assert isinstance(chosen, Sample)
        assert isinstance(rejected, Sample)
        assert chosen.input_ids == (1, 2, 3, 4, 5)
        assert rejected.input_ids == (1, 2, 3, 6, 7)
        assert chosen.sample_type == SampleType.DPO
        assert rejected.sample_type == SampleType.DPO
        assert "rejected" in rejected.meta.sample_id


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_sample_type_values(self):
        """Test SampleType enum values."""
        assert SampleType.SFT.value == "sft"
        assert SampleType.DPO.value == "dpo"
        assert SampleType.PRETRAIN.value == "pretrain"
        assert SampleType.RL.value == "rl"
        assert SampleType.GRPO.value == "grpo"
        assert SampleType.PPO.value == "ppo"

    def test_dataset_source_values(self):
        """Test DatasetSource enum values."""
        assert DatasetSource.LOCAL.value == "local"
        assert DatasetSource.HUGGINGFACE.value == "huggingface"
        assert DatasetSource.GYM.value == "gym"
        assert DatasetSource.SYNTHETIC.value == "synthetic"
        assert DatasetSource.REMOTE.value == "remote"

    def test_difficulty_level_values(self):
        """Test DifficultyLevel enum values."""
        assert DifficultyLevel.TRIVIAL.value == "trivial"
        assert DifficultyLevel.EASY.value == "easy"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.HARD.value == "hard"
        assert DifficultyLevel.EXPERT.value == "expert"


# =============================================================================
# DatasetFingerprint Tests
# =============================================================================


class TestDatasetFingerprint:
    """Tests for DatasetFingerprint model."""

    def test_create_fingerprint(self):
        """Test creating a dataset fingerprint."""
        fp = DatasetFingerprint(
            fingerprint="abc123def456",
            full_hash="abc123def456" * 4,
            content_hash="content123",
            tokenizer_hash="tokenizer456",
            num_samples=1000,
            source_path="/path/to/data.jsonl",
        )
        assert fp.fingerprint == "abc123def456"
        assert fp.num_samples == 1000

    def test_matches(self):
        """Test fingerprint matching."""
        fp1 = DatasetFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            content_hash="c1",
            tokenizer_hash="t1",
            num_samples=100,
        )
        fp2 = DatasetFingerprint(
            fingerprint="abc",
            full_hash="abc123",
            content_hash="c1",
            tokenizer_hash="t1",
            num_samples=100,
        )
        assert fp1.matches(fp2)

    def test_matches_content(self):
        """Test content-only matching."""
        fp1 = DatasetFingerprint(
            fingerprint="a",
            full_hash="a1",
            content_hash="same",
            tokenizer_hash="t1",
            num_samples=100,
        )
        fp2 = DatasetFingerprint(
            fingerprint="b",
            full_hash="b1",
            content_hash="same",
            tokenizer_hash="t2",  # Different tokenizer
            num_samples=100,
        )
        assert fp1.matches_content(fp2)
        assert not fp1.matches(fp2)


class TestComputeDatasetFingerprint:
    """Tests for compute_dataset_fingerprint function."""

    def test_compute_fingerprint(self):
        """Test computing fingerprint from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prompt": "Hello", "response": "Hi"}\n')
            f.write('{"prompt": "How are you?", "response": "Fine"}\n')
            path = f.name

        try:
            fp = compute_dataset_fingerprint(path, "tokenizer_abc123")
            assert len(fp.fingerprint) == 16
            assert len(fp.full_hash) == 64
            assert fp.num_samples == 2
        finally:
            Path(path).unlink()

    def test_fingerprint_deterministic(self):
        """Test that fingerprint is deterministic."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "sample"}\n')
            path = f.name

        try:
            fp1 = compute_dataset_fingerprint(path, "tok123")
            fp2 = compute_dataset_fingerprint(path, "tok123")
            assert fp1.fingerprint == fp2.fingerprint
            assert fp1.full_hash == fp2.full_hash
        finally:
            Path(path).unlink()

    def test_different_tokenizer_different_fingerprint(self):
        """Test that different tokenizer produces different fingerprint."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "sample"}\n')
            path = f.name

        try:
            fp1 = compute_dataset_fingerprint(path, "tokenizer_a")
            fp2 = compute_dataset_fingerprint(path, "tokenizer_b")
            assert fp1.fingerprint != fp2.fingerprint
            assert fp1.content_hash == fp2.content_hash  # Content is same
        finally:
            Path(path).unlink()

    def test_sample_limit(self):
        """Test sample limit parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(100):
                f.write(f'{{"id": {i}}}\n')
            path = f.name

        try:
            fp_full = compute_dataset_fingerprint(path, "tok", sample_limit=None)
            fp_limited = compute_dataset_fingerprint(path, "tok", sample_limit=10)
            assert fp_full.num_samples == 100
            assert fp_limited.num_samples == 10
            assert fp_full.fingerprint != fp_limited.fingerprint
        finally:
            Path(path).unlink()


# =============================================================================
# I/O Tests
# =============================================================================


class TestSampleIO:
    """Tests for sample I/O functions."""

    def test_save_load_samples(self):
        """Test save and load samples roundtrip."""
        samples = [
            Sample(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            ),
            Sample(
                input_ids=[4, 5, 6, 7],
                loss_mask=[0, 0, 1, 1],
                meta=SampleMeta(sample_id="s002", dataset_id="train"),
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_samples(samples, path)
            loaded = load_samples(path)

            assert len(loaded) == 2
            assert loaded[0].input_ids == (1, 2, 3)
            assert loaded[1].input_ids == (4, 5, 6, 7)
            assert loaded[0].meta.sample_id == "s001"
            assert loaded[1].meta.sample_id == "s002"
        finally:
            Path(path).unlink()

    def test_save_samples_with_segments(self):
        """Test saving samples with segment IDs."""
        samples = [
            Sample(
                input_ids=[1, 2, 3, 4, 5, 6],
                loss_mask=[1, 1, 1, 1, 1, 1],
                segment_ids=[0, 0, 0, 1, 1, 1],
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_samples(samples, path)
            loaded = load_samples(path)

            assert loaded[0].segment_ids == (0, 0, 0, 1, 1, 1)
        finally:
            Path(path).unlink()

    def test_save_creates_valid_jsonl(self):
        """Test that saved file is valid JSONL."""
        samples = [
            Sample(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            save_samples(samples, path)

            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    assert "input_ids" in data
                    assert "loss_mask" in data
                    assert "meta" in data
        finally:
            Path(path).unlink()


class TestAsyncIO:
    """Tests for async sample I/O."""

    def test_save_load_async(self):
        """Test async save and load."""
        import asyncio

        from chuk_lazarus.data.samples.schema import (
            load_samples_async,
            save_samples_async,
        )

        samples = [
            Sample(
                input_ids=[1, 2, 3],
                loss_mask=[0, 1, 1],
                meta=SampleMeta(sample_id="s001", dataset_id="train"),
            ),
        ]

        async def run_async():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                await save_samples_async(samples, path)
                loaded = await load_samples_async(path)
                return loaded
            finally:
                Path(path).unlink()

        loaded = asyncio.run(run_async())
        assert len(loaded) == 1
        assert loaded[0].input_ids == (1, 2, 3)
