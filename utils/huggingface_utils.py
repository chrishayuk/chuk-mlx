from pathlib import Path
from huggingface_hub import snapshot_download


def load_from_hub(path_or_hf_repo):
    # get the model path
    model_path = Path(path_or_hf_repo)

    # if the path is not a local path, then load from the hub
    if not model_path.exists():
        # download the model from huggingface and set the path
        model_path = Path(
            # download the model from huggingface
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    # return the model path
    return model_path
