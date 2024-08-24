import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_from_hub(path_or_hf_repo):
    """Load model from Hugging Face hub if not found locally."""
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        try:
            logger.info(f"Attempting to download model from Hugging Face hub: {path_or_hf_repo}")
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "tokenizer.model"],
                )
            )
        except Exception as e:
            logger.error(f"Error downloading model from Hugging Face hub: {e}")
            raise

    return model_path