"""
Tool Calling Dynamics Experiment

Investigates how GPT-OSS internally represents and generates tool calls.
"""

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RESULTS_DIR = EXPERIMENT_DIR / "results"
