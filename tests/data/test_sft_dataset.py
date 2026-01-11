"""Tests for SFT dataset."""

import json
import tempfile
from unittest.mock import MagicMock


class TestSFTSample:
    """Tests for SFTSample."""

    def test_create_sample(self):
        """Test creating SFT sample."""
        from chuk_lazarus.data.sft_dataset import SFTSample

        sample = SFTSample(prompt="Hello", response="Hi there!")

        assert sample.prompt == "Hello"
        assert sample.response == "Hi there!"
        assert sample.metadata is None

    def test_create_sample_with_metadata(self):
        """Test creating SFT sample with metadata."""
        from chuk_lazarus.data.sft_dataset import SFTSample

        sample = SFTSample(
            prompt="Hello",
            response="Hi there!",
            metadata={"source": "test"},
        )

        assert sample.metadata == {"source": "test"}


class TestSFTDataset:
    """Tests for SFTDataset."""

    def test_import(self):
        """Test SFT dataset can be imported."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        assert SFTDataset is not None

    def test_load_simple_format(self):
        """Test loading simple prompt/response format."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is the capital of France?", "response": "Paris"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)

            assert len(dataset) == 2
            assert dataset.samples[0].prompt == "What is 2+2?"
            assert dataset.samples[0].response == "4"

    def test_load_messages_format(self):
        """Test loading chat messages format."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)

            assert len(dataset) == 1
            assert "Hello" in dataset.samples[0].prompt
            assert dataset.samples[0].response == "Hi there!"

    def test_load_alternative_keys(self):
        """Test loading with alternative keys (input/output/completion)."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [
            {"input": "Question", "output": "Answer"},
            {"input": "Another", "completion": "Response"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)

            assert len(dataset) == 2
            assert dataset.samples[0].prompt == "Question"
            assert dataset.samples[0].response == "Answer"

    def test_getitem(self):
        """Test __getitem__ method."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [{"prompt": "Test", "response": "Response"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)
            item = dataset[0]

            assert "input_ids" in item
            assert "labels" in item
            assert "loss_mask" in item
            assert "prompt_length" in item

    def test_tokenize_with_mask(self):
        """Test tokenization with prompt masking."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [{"prompt": "Q", "response": "A"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            f.flush()

            tokenizer = MagicMock()
            # Prompt encodes to [1, 2], full encodes to [1, 2, 3, 4]
            tokenizer.encode = MagicMock(side_effect=lambda x: [1, 2] if x == "Q" else [1, 2, 3, 4])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512, mask_prompt=True)
            item = dataset[0]

            # First 2 tokens should be masked (0.0), rest should be 1.0
            assert item["loss_mask"][:2] == [0.0, 0.0]

    def test_tokenize_without_mask(self):
        """Test tokenization without prompt masking."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [{"prompt": "Q", "response": "A"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512, mask_prompt=False)
            item = dataset[0]

            # All tokens should have loss mask = 1.0
            assert all(m == 1.0 for m in item["loss_mask"])

    def test_truncation(self):
        """Test sequence truncation."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [{"prompt": "Q", "response": "A"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data[0]) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=list(range(100)))  # 100 tokens
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=50, mask_prompt=False)
            item = dataset[0]

            # Should be truncated to max_length
            assert len(item["input_ids"]) == 50

    def test_get_batch(self):
        """Test get_batch method."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [
            {"prompt": "Q1", "response": "A1"},
            {"prompt": "Q2", "response": "A2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(side_effect=lambda x: [1, 2, 3] if "1" in x else [4, 5])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)
            batch = dataset.get_batch([0, 1], pad_token_id=0)

            assert "input_ids" in batch
            assert "labels" in batch
            assert "loss_mask" in batch
            assert "attention_mask" in batch

            # Batch should have 2 samples
            assert batch["input_ids"].shape[0] == 2

    def test_iter_batches(self):
        """Test iter_batches method."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        data = [{"prompt": f"Q{i}", "response": f"A{i}"} for i in range(5)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()

            tokenizer = MagicMock()
            tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            tokenizer.eos_token_id = 0

            dataset = SFTDataset(f.name, tokenizer, max_length=512)

            batches = list(dataset.iter_batches(batch_size=2, shuffle=False))

            # 5 samples, batch_size=2 -> 3 batches
            assert len(batches) == 3


class TestPreferenceDataset:
    """Tests for PreferenceDataset."""

    def test_import(self):
        """Test preference dataset can be imported."""
        from chuk_lazarus.data.preference_dataset import PreferenceDataset

        assert PreferenceDataset is not None


class TestClassificationDataset:
    """Tests for ClassificationDataset."""

    def test_import(self):
        """Test classification dataset can be imported."""
        from chuk_lazarus.data.classification_dataset import ClassificationDataset

        assert ClassificationDataset is not None


class TestBaseDatasetImport:
    """Tests for BaseDataset import."""

    def test_import(self):
        """Test base dataset can be imported."""
        from chuk_lazarus.data.base_dataset import BaseDataset

        assert BaseDataset is not None
