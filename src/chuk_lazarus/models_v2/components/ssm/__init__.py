"""
State Space Model (SSM) components.

Provides:
- SelectiveSSM: Core selective scan operation (Mamba)
- Mamba: Full Mamba layer with conv + SSM
- MambaBlock: Mamba with residual and normalization

Reference:
- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2: https://arxiv.org/abs/2405.21060
"""

from .mamba import Mamba, MambaBlock, create_mamba, create_mamba_block
from .selective_ssm import SelectiveSSM, selective_scan, selective_scan_step

__all__ = [
    "SelectiveSSM",
    "selective_scan",
    "selective_scan_step",
    "Mamba",
    "MambaBlock",
    "create_mamba",
    "create_mamba_block",
]
