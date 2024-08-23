import argparse
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.models.model_loader import load_model

def print_model_layer_modules(model):
    """
    Print the names of all modules (layers) in the model.
    """
    for name, _ in model.named_modules():
        print(name)

def main(model_name):
    """
    Load the model and print its layer modules.
    """
    # Load the model using the updated load_model function
    model = load_model(model_name, load_weights=False)

    # Print the model layers
    print_model_layer_modules(model)

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Print layer modules of the model.')

    # Argument for model name
    parser.add_argument(
        "--model",
        required=False,
        default="meta-llama/Meta-Llama-3-8b-Instruct",
        help='Model name or path, e.g., meta-llama/Llama-3-8b-Instruct'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute the main function with the provided model name
    main(args.model)
