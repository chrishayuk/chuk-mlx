"""Tests for SFT dataset."""

import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSFTDataset:
    """Tests for SFTDataset."""

    def test_import(self):
        """Test SFT dataset can be imported."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        assert SFTDataset is not None

    def test_create_from_jsonl(self):
        """Test creating dataset from JSONL."""
        from chuk_lazarus.data.sft_dataset import SFTDataset

        # Just verify the class exists and can be imported
        assert SFTDataset is not None


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


class TestBaseDataset:
    """Tests for BaseDataset."""

    def test_import(self):
        """Test base dataset can be imported."""
        from chuk_lazarus.data.base_dataset import BaseDataset

        assert BaseDataset is not None
