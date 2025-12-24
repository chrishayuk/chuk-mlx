"""Tests for async length cache."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.data.batching import LengthCache, LengthEntry


class TestLengthEntry:
    """Tests for LengthEntry model."""

    def test_create(self):
        """Test creating entry."""
        entry = LengthEntry(
            sample_id="sample_001",
            tokenizer_hash="abc123",
            length=256,
        )
        assert entry.sample_id == "sample_001"
        assert entry.tokenizer_hash == "abc123"
        assert entry.length == 256

    def test_to_json(self):
        """Test JSON serialization."""
        entry = LengthEntry(
            sample_id="sample_001",
            tokenizer_hash="abc123",
            length=256,
        )
        json_str = entry.to_json()
        assert "sample_001" in json_str
        assert "abc123" in json_str
        assert "256" in json_str

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"sample_id": "s1", "tokenizer_hash": "h1", "length": 100}'
        entry = LengthEntry.from_json(json_str)
        assert entry.sample_id == "s1"
        assert entry.tokenizer_hash == "h1"
        assert entry.length == 100

    def test_roundtrip(self):
        """Test JSON roundtrip."""
        original = LengthEntry(
            sample_id="test_sample",
            tokenizer_hash="tok_hash_123",
            length=512,
        )
        json_str = original.to_json()
        restored = LengthEntry.from_json(json_str)
        assert restored == original


class TestLengthCache:
    """Tests for async LengthCache."""

    def test_create_empty(self):
        """Test creating empty cache."""
        cache = LengthCache.create("/tmp/test.jsonl", "tok_abc")
        assert len(cache) == 0
        assert cache.tokenizer_hash == "tok_abc"

    def test_build_cache(self):
        """Test building cache with async context manager."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                async with LengthCache.create(path, "tok_123") as cache:
                    await cache.add("s1", 100)
                    await cache.add("s2", 200)
                    await cache.add("s3", 150)

                assert len(cache) == 3
                assert cache.get("s1") == 100
                assert cache.get("s2") == 200
                assert cache.get("s3") == 150

                # Verify file was written
                assert Path(path).exists()
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_load_cache(self):
        """Test loading cache from file."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                # Build cache
                async with LengthCache.create(path, "tok_abc") as cache:
                    await cache.add("sample_1", 128)
                    await cache.add("sample_2", 256)
                    await cache.add("sample_3", 512)

                # Load cache
                loaded = await LengthCache.load(path)
                assert len(loaded) == 3
                assert loaded.tokenizer_hash == "tok_abc"
                assert loaded.get("sample_1") == 128
                assert loaded.get("sample_2") == 256
                assert loaded.get("sample_3") == 512
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_load_with_hash_validation(self):
        """Test loading with tokenizer hash validation."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                async with LengthCache.create(path, "tok_abc") as cache:
                    await cache.add("s1", 100)

                # Load with correct hash - should succeed
                loaded = await LengthCache.load(path, expected_tokenizer_hash="tok_abc")
                assert len(loaded) == 1

                # Load with wrong hash - should fail
                with pytest.raises(ValueError, match="Tokenizer hash mismatch"):
                    await LengthCache.load(path, expected_tokenizer_hash="tok_xyz")
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_load_or_create_existing(self):
        """Test load_or_create with existing cache."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                async with LengthCache.create(path, "tok_abc") as cache:
                    await cache.add("s1", 100)

                loaded = await LengthCache.load_or_create(path, "tok_abc")
                assert len(loaded) == 1
                assert loaded.get("s1") == 100
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_load_or_create_missing(self):
        """Test load_or_create with missing cache."""

        async def run():
            path = "/tmp/nonexistent_cache_12345.jsonl"
            cache = await LengthCache.load_or_create(path, "tok_abc")
            assert len(cache) == 0
            assert cache.tokenizer_hash == "tok_abc"

        asyncio.run(run())

    def test_load_or_create_invalid_hash(self):
        """Test load_or_create with invalid hash creates new."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                async with LengthCache.create(path, "tok_old") as cache:
                    await cache.add("s1", 100)

                # Different hash - should create new empty cache
                loaded = await LengthCache.load_or_create(path, "tok_new")
                assert len(loaded) == 0
                assert loaded.tokenizer_hash == "tok_new"
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_contains(self):
        """Test __contains__ method."""
        cache = LengthCache.create("/tmp/test.jsonl", "tok")
        cache._entries = {"s1": 100, "s2": 200}

        assert "s1" in cache
        assert "s2" in cache
        assert "s3" not in cache

    def test_get_all(self):
        """Test get_all method."""
        cache = LengthCache.create("/tmp/test.jsonl", "tok")
        cache._entries = {"s1": 100, "s2": 200}

        all_entries = cache.get_all()
        assert all_entries == {"s1": 100, "s2": 200}
        # Should return a copy
        all_entries["s3"] = 300
        assert "s3" not in cache

    def test_items(self):
        """Test items iteration."""
        cache = LengthCache.create("/tmp/test.jsonl", "tok")
        cache._entries = {"s1": 100, "s2": 200}

        items = list(cache.items())
        assert ("s1", 100) in items
        assert ("s2", 200) in items

    def test_stream(self):
        """Test streaming entries."""

        async def run():
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = f.name

            try:
                async with LengthCache.create(path, "tok_123") as cache:
                    for i in range(100):
                        await cache.add(f"sample_{i}", i * 10)

                # Stream without loading all into memory
                entries = []
                async for entry in LengthCache.stream(path):
                    entries.append(entry)

                assert len(entries) == 100
                assert entries[0].sample_id == "sample_0"
                assert entries[0].length == 0
                assert entries[50].sample_id == "sample_50"
                assert entries[50].length == 500
            finally:
                Path(path).unlink()

        asyncio.run(run())

    def test_validate(self):
        """Test cache validation."""

        async def run():
            cache = LengthCache.create("/tmp/test.jsonl", "tok")
            cache._entries = {"s1": 100, "s2": 200, "s3": 300}

            # All present
            missing, extra = await cache.validate({"s1", "s2", "s3"})
            assert missing == set()
            assert extra == set()

            # Some missing
            missing, extra = await cache.validate({"s1", "s2", "s3", "s4", "s5"})
            assert missing == {"s4", "s5"}
            assert extra == set()

            # Some extra
            missing, extra = await cache.validate({"s1", "s2"})
            assert missing == set()
            assert extra == {"s3"}

        asyncio.run(run())

    def test_is_complete(self):
        """Test is_complete check."""
        cache = LengthCache.create("/tmp/test.jsonl", "tok")
        cache._entries = {"s1": 100, "s2": 200, "s3": 300}

        assert cache.is_complete({"s1", "s2"})
        assert cache.is_complete({"s1", "s2", "s3"})
        assert not cache.is_complete({"s1", "s2", "s3", "s4"})

    def test_add_without_context_raises(self):
        """Test that add() without context manager raises."""

        async def run():
            cache = LengthCache.create("/tmp/test.jsonl", "tok")
            with pytest.raises(RuntimeError, match="not opened"):
                await cache.add("s1", 100)

        asyncio.run(run())
