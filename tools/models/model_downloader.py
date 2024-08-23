import argparse
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
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

