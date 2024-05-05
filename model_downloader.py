import argparse
from utils.huggingface_utils import load_from_hub

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="HuggingFace Model Downloader")

    # look for the model
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )

    # parse arguments
    args = parser.parse_args()

    # load the model from huggingface
    print(f"Loading Model: {args.model}")
    load_from_hub(args.model)
    print(f"Model Loaded: {args.model}")

