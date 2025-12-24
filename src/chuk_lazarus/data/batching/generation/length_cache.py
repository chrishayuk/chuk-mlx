"""
Async-native length cache for pre-computed sequence lengths.

The length cache stores sequence lengths keyed by sample ID and tokenizer hash.
This enables fast bucket assignment without re-tokenizing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
from pydantic import BaseModel, ConfigDict, Field


class LengthEntry(BaseModel):
    """
    A single length cache entry.

    Maps sample_id + tokenizer_hash to sequence length.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(description="Unique sample identifier")
    tokenizer_hash: str = Field(description="Tokenizer fingerprint hash")
    length: int = Field(ge=0, description="Sequence length in tokens")

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(self.model_dump())

    @classmethod
    def from_json(cls, line: str) -> LengthEntry:
        """Deserialize from JSON line."""
        return cls(**json.loads(line))


class LengthCache:
    """
    Async-native cache for sequence lengths.

    Stores lengths in JSONL format for streaming reads.
    Supports incremental building and validation.

    Usage:
        # Build cache
        async with LengthCache.create(path, tokenizer_hash) as cache:
            async for sample in dataset:
                await cache.add(sample.id, len(tokenizer.encode(sample.text)))

        # Load cache
        cache = await LengthCache.load(path)
        length = cache.get(sample_id)
    """

    def __init__(
        self,
        path: Path,
        tokenizer_hash: str,
        entries: dict[str, int] | None = None,
    ):
        self._path = path
        self._tokenizer_hash = tokenizer_hash
        self._entries: dict[str, int] = entries or {}
        self._file = None
        self._dirty = False

    @property
    def path(self) -> Path:
        """Cache file path."""
        return self._path

    @property
    def tokenizer_hash(self) -> str:
        """Tokenizer fingerprint this cache was built with."""
        return self._tokenizer_hash

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    def __contains__(self, sample_id: str) -> bool:
        """Check if sample is in cache."""
        return sample_id in self._entries

    def get(self, sample_id: str) -> int | None:
        """Get cached length for sample."""
        return self._entries.get(sample_id)

    def get_all(self) -> dict[str, int]:
        """Get all cached lengths."""
        return dict(self._entries)

    def items(self):
        """Iterate over (sample_id, length) pairs."""
        return self._entries.items()

    # =========================================================================
    # Async Context Manager (for building)
    # =========================================================================

    @classmethod
    def create(cls, path: str | Path, tokenizer_hash: str) -> LengthCache:
        """
        Create a new cache for building.

        Use as async context manager:
            async with LengthCache.create(path, hash) as cache:
                await cache.add(...)
        """
        return cls(Path(path), tokenizer_hash)

    async def __aenter__(self) -> LengthCache:
        """Open cache file for writing."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = await aiofiles.open(self._path, "w")
        # Write header with tokenizer hash
        header = {"tokenizer_hash": self._tokenizer_hash, "version": 1}
        await self._file.write(json.dumps(header) + "\n")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close cache file."""
        if self._file:
            await self._file.close()
            self._file = None

    async def add(self, sample_id: str, length: int) -> None:
        """
        Add a length entry to the cache.

        Must be called within async context manager.
        """
        if self._file is None:
            raise RuntimeError("Cache not opened for writing. Use 'async with' context.")

        entry = LengthEntry(
            sample_id=sample_id,
            tokenizer_hash=self._tokenizer_hash,
            length=length,
        )
        await self._file.write(entry.to_json() + "\n")
        self._entries[sample_id] = length

    # =========================================================================
    # Async Loading
    # =========================================================================

    @classmethod
    async def load(
        cls,
        path: str | Path,
        expected_tokenizer_hash: str | None = None,
    ) -> LengthCache:
        """
        Load cache from file.

        Args:
            path: Path to cache file
            expected_tokenizer_hash: If provided, validate cache matches this hash

        Returns:
            Loaded LengthCache

        Raises:
            ValueError: If tokenizer hash doesn't match
            FileNotFoundError: If cache file doesn't exist
        """
        path = Path(path)
        entries: dict[str, int] = {}
        tokenizer_hash: str = ""

        async with aiofiles.open(path) as f:
            # First line is header
            header_line = await f.readline()
            header = json.loads(header_line)
            tokenizer_hash = header["tokenizer_hash"]

            if expected_tokenizer_hash and tokenizer_hash != expected_tokenizer_hash:
                raise ValueError(
                    f"Tokenizer hash mismatch: cache has {tokenizer_hash}, "
                    f"expected {expected_tokenizer_hash}"
                )

            # Rest are entries
            async for line in f:
                if line.strip():
                    entry = LengthEntry.from_json(line)
                    entries[entry.sample_id] = entry.length

        return cls(path, tokenizer_hash, entries)

    @classmethod
    async def load_or_create(
        cls,
        path: str | Path,
        tokenizer_hash: str,
    ) -> LengthCache:
        """
        Load cache if exists and valid, otherwise create empty.

        Args:
            path: Path to cache file
            tokenizer_hash: Expected tokenizer hash

        Returns:
            LengthCache (loaded or empty)
        """
        path = Path(path)
        if path.exists():
            try:
                return await cls.load(path, expected_tokenizer_hash=tokenizer_hash)
            except (ValueError, json.JSONDecodeError):
                # Invalid cache, will rebuild
                pass
        return cls(path, tokenizer_hash)

    # =========================================================================
    # Async Iteration
    # =========================================================================

    @classmethod
    async def stream(cls, path: str | Path) -> AsyncIterator[LengthEntry]:
        """
        Stream entries from cache without loading all into memory.

        Useful for very large caches.
        """
        path = Path(path)
        async with aiofiles.open(path) as f:
            # Skip header
            await f.readline()
            async for line in f:
                if line.strip():
                    yield LengthEntry.from_json(line)

    # =========================================================================
    # Validation
    # =========================================================================

    async def validate(self, sample_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Validate cache against expected sample IDs.

        Args:
            sample_ids: Expected sample IDs

        Returns:
            Tuple of (missing_ids, extra_ids)
        """
        cached_ids = set(self._entries.keys())
        missing = sample_ids - cached_ids
        extra = cached_ids - sample_ids
        return missing, extra

    def is_complete(self, sample_ids: set[str]) -> bool:
        """Check if cache contains all expected sample IDs."""
        return sample_ids <= set(self._entries.keys())
