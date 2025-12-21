"""
Tokenizer Profile API.

Switch between training and inference behavior without changing tokenizers:
- Training: stochastic tokenization, aggressive normalization
- Inference: deterministic, byte-safe, minimal transforms
"""

import random
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from chuk_lazarus.data.tokenizers.preprocessing.hooks import (
    HookedTokenizer,
    HookPipeline,
    create_standard_pipeline,
)
from chuk_lazarus.data.tokenizers.preprocessing.numeric import NumericConfig
from chuk_lazarus.data.tokenizers.preprocessing.structure import StructureConfig


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...


class ProfileMode(str, Enum):
    """Tokenizer profile modes."""

    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class TokenizerProfile(BaseModel):
    """Configuration for tokenizer behavior."""

    name: str = Field(description="Profile name")
    mode: ProfileMode = Field(default=ProfileMode.INFERENCE, description="Profile mode")

    # Stochastic tokenization (for Unigram LM)
    stochastic: bool = Field(default=False, description="Enable stochastic tokenization sampling")
    stochastic_alpha: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Sampling temperature for stochastic"
    )

    # Normalization
    normalize_numbers: bool = Field(default=False, description="Normalize numbers to placeholders")
    inject_structures: bool = Field(default=False, description="Inject structure tokens")
    normalize_whitespace: bool = Field(default=False, description="Normalize whitespace")

    # Safety
    byte_fallback: bool = Field(default=True, description="Enable byte fallback for unknown chars")
    max_length: int | None = Field(default=None, description="Maximum sequence length")
    truncation: bool = Field(default=False, description="Enable truncation")

    # Configs
    numeric_config: NumericConfig = Field(
        default_factory=NumericConfig, description="Numeric normalization config"
    )
    structure_config: StructureConfig = Field(
        default_factory=StructureConfig, description="Structure injection config"
    )

    # Special tokens
    add_special_tokens: bool = Field(default=True, description="Add special tokens during encoding")


class ProfiledTokenizer:
    """Tokenizer that applies profile-based behavior."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        profile: TokenizerProfile,
    ):
        self.base_tokenizer = tokenizer
        self.profile = profile
        self._pipeline: HookPipeline | None = None
        self._hooked_tokenizer: HookedTokenizer | None = None
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Set up the hook pipeline based on profile."""
        if self.profile.normalize_numbers or self.profile.inject_structures:
            self._pipeline = create_standard_pipeline(
                numeric=self.profile.normalize_numbers,
                structure=self.profile.inject_structures,
                whitespace=self.profile.normalize_whitespace,
                numeric_config=self.profile.numeric_config,
                structure_config=self.profile.structure_config,
            )
            self._hooked_tokenizer = HookedTokenizer(self.base_tokenizer, self._pipeline)
        else:
            self._pipeline = None
            self._hooked_tokenizer = None

    def set_profile(self, profile: TokenizerProfile) -> None:
        """Switch to a different profile."""
        self.profile = profile
        self._setup_pipeline()

    def encode(
        self,
        text: str,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """
        Encode text according to profile.

        Args:
            text: Input text
            add_special_tokens: Override profile setting

        Returns:
            Token IDs
        """
        add_special = (
            add_special_tokens
            if add_special_tokens is not None
            else self.profile.add_special_tokens
        )

        # Use hooked tokenizer if pipeline is set up
        if self._hooked_tokenizer is not None:
            token_ids = self._hooked_tokenizer.encode(text, add_special_tokens=add_special)
        else:
            token_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special)

        # Apply stochastic sampling if enabled
        if self.profile.stochastic and self.profile.mode == ProfileMode.TRAINING:
            token_ids = self._apply_stochastic(token_ids)

        # Apply truncation if enabled
        if self.profile.truncation and self.profile.max_length is not None:
            token_ids = token_ids[: self.profile.max_length]

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode tokens according to profile.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text
        """
        if self._hooked_tokenizer is not None:
            return self._hooked_tokenizer.decode(token_ids)
        return self.base_tokenizer.decode(token_ids)

    def _apply_stochastic(self, token_ids: list[int]) -> list[int]:
        """
        Apply stochastic perturbation to tokenization.

        This simulates the effect of sampling from multiple valid
        segmentations in Unigram LM tokenizers.

        For BPE tokenizers, we can approximate this by:
        - Occasionally merging adjacent tokens
        - Occasionally splitting tokens

        Args:
            token_ids: Original token IDs

        Returns:
            Perturbed token IDs
        """
        # Simple approximation: randomly drop merge operations
        # This is a placeholder - real implementation would need
        # access to tokenizer internals
        if random.random() > self.profile.stochastic_alpha:
            return token_ids

        # For now, just return original
        # TODO: Implement proper stochastic sampling for Unigram
        return token_ids

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.base_tokenizer.vocab_size

    def get_pipeline_metadata(self) -> list[dict]:
        """Get metadata from last encode operation."""
        if self._pipeline is not None:
            return self._pipeline.get_metadata()
        return []


def create_training_profile(
    name: str = "training",
    normalize_numbers: bool = True,
    inject_structures: bool = True,
    stochastic: bool = False,
    max_length: int | None = 2048,
) -> TokenizerProfile:
    """
    Create a profile optimized for training.

    Args:
        name: Profile name
        normalize_numbers: Enable numeric normalization
        inject_structures: Enable structure injection
        stochastic: Enable stochastic tokenization
        max_length: Maximum sequence length

    Returns:
        Training-optimized TokenizerProfile
    """
    return TokenizerProfile(
        name=name,
        mode=ProfileMode.TRAINING,
        stochastic=stochastic,
        normalize_numbers=normalize_numbers,
        inject_structures=inject_structures,
        normalize_whitespace=True,
        byte_fallback=True,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )


def create_inference_profile(
    name: str = "inference",
    byte_fallback: bool = True,
    max_length: int | None = None,
) -> TokenizerProfile:
    """
    Create a profile optimized for inference.

    Args:
        name: Profile name
        byte_fallback: Enable byte fallback
        max_length: Maximum sequence length

    Returns:
        Inference-optimized TokenizerProfile
    """
    return TokenizerProfile(
        name=name,
        mode=ProfileMode.INFERENCE,
        stochastic=False,
        normalize_numbers=False,
        inject_structures=False,
        normalize_whitespace=False,
        byte_fallback=byte_fallback,
        max_length=max_length,
        truncation=max_length is not None,
        add_special_tokens=True,
    )


def create_math_profile(
    name: str = "math",
    max_length: int | None = 2048,
) -> TokenizerProfile:
    """
    Create a profile optimized for math/reasoning.

    Args:
        name: Profile name
        max_length: Maximum sequence length

    Returns:
        Math-optimized TokenizerProfile
    """
    return TokenizerProfile(
        name=name,
        mode=ProfileMode.TRAINING,
        stochastic=False,
        normalize_numbers=True,
        inject_structures=True,
        normalize_whitespace=False,
        byte_fallback=True,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        numeric_config=NumericConfig(
            use_placeholder=True,
            detect_scientific=True,
            detect_fractions=True,
        ),
    )


def create_tool_profile(
    name: str = "tool",
    max_length: int | None = 4096,
) -> TokenizerProfile:
    """
    Create a profile optimized for tool/agent traces.

    Args:
        name: Profile name
        max_length: Maximum sequence length

    Returns:
        Tool-optimized TokenizerProfile
    """
    from chuk_lazarus.data.tokenizers.preprocessing.structure import (
        create_tool_aware_config,
    )

    return TokenizerProfile(
        name=name,
        mode=ProfileMode.TRAINING,
        stochastic=False,
        normalize_numbers=True,
        inject_structures=True,
        normalize_whitespace=False,
        byte_fallback=True,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        structure_config=create_tool_aware_config(),
    )


class ProfileManager:
    """Manage multiple tokenizer profiles."""

    def __init__(self):
        self._profiles: dict[str, TokenizerProfile] = {}
        self._active_profile: str | None = None

    def register(self, profile: TokenizerProfile) -> None:
        """Register a profile."""
        self._profiles[profile.name] = profile

    def get(self, name: str) -> TokenizerProfile | None:
        """Get a profile by name."""
        return self._profiles.get(name)

    def set_active(self, name: str) -> None:
        """Set the active profile."""
        if name not in self._profiles:
            raise ValueError(f"Profile '{name}' not found")
        self._active_profile = name

    @property
    def active(self) -> TokenizerProfile | None:
        """Get the active profile."""
        if self._active_profile is None:
            return None
        return self._profiles.get(self._active_profile)

    def list_profiles(self) -> list[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())

    def create_tokenizer(
        self,
        base_tokenizer: TokenizerProtocol,
        profile_name: str | None = None,
    ) -> ProfiledTokenizer:
        """
        Create a profiled tokenizer.

        Args:
            base_tokenizer: Base tokenizer to wrap
            profile_name: Profile to use (or active profile)

        Returns:
            ProfiledTokenizer with the specified profile
        """
        name = profile_name or self._active_profile
        if name is None:
            raise ValueError("No profile specified and no active profile set")

        profile = self._profiles.get(name)
        if profile is None:
            raise ValueError(f"Profile '{name}' not found")

        return ProfiledTokenizer(base_tokenizer, profile)


def create_default_manager() -> ProfileManager:
    """
    Create a profile manager with standard profiles.

    Returns:
        ProfileManager with training, inference, math, and tool profiles
    """
    manager = ProfileManager()
    manager.register(create_training_profile())
    manager.register(create_inference_profile())
    manager.register(create_math_profile())
    manager.register(create_tool_profile())
    manager.set_active("inference")
    return manager
