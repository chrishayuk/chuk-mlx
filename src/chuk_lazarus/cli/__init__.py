"""
CHUK-MLX Command Line Interface.

Provides unified CLI for:
- Training (SFT, DPO, GRPO, PPO)
- Model loading and inference
- Data generation
"""

from .main import app as app
from .main import main as main
