import argparse
import logging
from utils.model_loader import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # look for the model
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )

    # specify action
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--show-layers",
        action="store_true",
        help="Show layers of the model.",
    )

    action_group.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration table of the model. (default)",
    )

    # parse arguments
    args = parser.parse_args()

    # load the model
    logger.info(f"Loading Model: {args.model}")
    model_config = load_model(args.model)

    # perform action
    if args.show_layers:
        # print the layers
        model_config.print_layers()
    else:
        # print the config
        model_config.print_config()
