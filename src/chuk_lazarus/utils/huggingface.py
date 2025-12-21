"""HuggingFace Hub utilities."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def load_from_hub(path_or_hf_repo: str, allow_patterns: list[str] = None) -> Path:
    """
    Load model from HuggingFace hub if not found locally.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo ID
        allow_patterns: File patterns to download (default: model files)

    Returns:
        Path to the model directory
    """
    model_path = Path(path_or_hf_repo)

    if model_path.exists():
        return model_path

    if allow_patterns is None:
        allow_patterns = [
            "*.json",
            "*.safetensors",
            "*.bin",
            "tokenizer.model",
            "*.tiktoken",
        ]

    try:
        logger.info(f"Downloading from HuggingFace hub: {path_or_hf_repo}")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=allow_patterns,
            )
        )
        logger.info(f"Downloaded to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to download from HuggingFace hub: {e}")
        raise

    return model_path
