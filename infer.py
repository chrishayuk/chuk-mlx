import argparse
import mlx.core as mx
from models.model_config import ModelConfig
from models.mlx_model_weights_loader import load_weight_files
from utils.tokenizer_loader import load_tokenizer
from utils.huggingface_utils import load_from_hub

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # look for the model
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Argument for tokenizer; if not specified, use the model path
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="The name or path for the tokenizer; if not specified, use the model path."
    )

    # Argument for tokenizer; if not specified, use the model path
    parser.add_argument(
        "--prompt",
        default="Who is Ada Lovelace?",
        help="The prompt"
    )

    # parse arguments
    args = parser.parse_args()

    # load the model from huggingface
    print(f"Loading Model: {args.model}")
    model_path = load_from_hub(args.model)

    # load config
    print(f"Loading Model Config")
    model_config = ModelConfig.load(model_path)

    # loading weights
    print(f"Loading Model Weights")
    weights = load_weight_files(model_path)

    # load the tokenizer
    print(f"Loading Tokenizer")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = load_tokenizer(tokenizer_path)

    # Encode the prompt
    encoded_input = tokenizer.encode(args.prompt, add_special_tokens=True)

    # load the prompt into the array
    prompt = mx.array(encoded_input)
    print(prompt)
