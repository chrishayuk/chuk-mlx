"""
Mamba configuration.

Configuration for Mamba SSM models.
"""

from __future__ import annotations

from pydantic import Field

from ...core.config import ModelConfig, SSMConfig


class MambaConfig(ModelConfig):
    """
    Configuration for Mamba models.

    Mamba uses SSM (State Space Model) blocks instead of attention.
    This provides O(n) complexity and constant memory during inference.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layer: Number of Mamba blocks
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension

    Example:
        >>> config = MambaConfig(
        ...     vocab_size=50280,
        ...     d_model=768,
        ...     n_layer=24,
        ...     d_state=16,
        ...     d_conv=4,
        ...     expand=2,
        ... )
    """

    model_type: str = "mamba"

    # Mamba-specific parameters
    d_state: int = Field(default=16, description="SSM state dimension")
    d_conv: int = Field(default=4, description="Convolution kernel size")
    expand: int = Field(default=2, description="Expansion factor")

    # Override hidden_size as d_model for clarity
    @property
    def d_model(self) -> int:
        return self.hidden_size

    # Alias for num_hidden_layers
    @property
    def n_layer(self) -> int:
        return self.num_hidden_layers

    def get_ssm_config(self) -> SSMConfig:
        """Get SSM configuration for blocks."""
        return SSMConfig(
            hidden_size=self.hidden_size,
            state_size=self.d_state,
            conv_kernel_size=self.d_conv,
            expand_factor=self.expand,
        )

    @classmethod
    def mamba_130m(cls) -> MambaConfig:
        """Create Mamba 130M configuration."""
        return cls(
            vocab_size=50280,
            hidden_size=768,
            num_hidden_layers=24,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    @classmethod
    def mamba_370m(cls) -> MambaConfig:
        """Create Mamba 370M configuration."""
        return cls(
            vocab_size=50280,
            hidden_size=1024,
            num_hidden_layers=48,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    @classmethod
    def mamba_790m(cls) -> MambaConfig:
        """Create Mamba 790M configuration."""
        return cls(
            vocab_size=50280,
            hidden_size=1536,
            num_hidden_layers=48,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    @classmethod
    def mamba_1_4b(cls) -> MambaConfig:
        """Create Mamba 1.4B configuration."""
        return cls(
            vocab_size=50280,
            hidden_size=2048,
            num_hidden_layers=48,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    @classmethod
    def mamba_2_8b(cls) -> MambaConfig:
        """Create Mamba 2.8B configuration."""
        return cls(
            vocab_size=50280,
            hidden_size=2560,
            num_hidden_layers=64,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    @classmethod
    def tiny(cls) -> MambaConfig:
        """Create tiny Mamba for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            d_state=8,
            d_conv=4,
            expand=2,
        )
