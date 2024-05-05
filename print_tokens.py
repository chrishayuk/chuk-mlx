import argparse
from utils.tokenizer_loader import load_tokenizer
from utils.token_display_utility import TokenDisplayUtility

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="MLX Inference")

    # look for the model
    parser.add_argument(
        "--model",
        default=None,
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Argument for tokenizer; if not specified, use the model path
    parser.add_argument(
        "--tokenizer",
        default="mistralai/Mistral-7B-Instruct-v0.2",
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

    # load the tokenizer
    print(f"Loading Tokenizer")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = load_tokenizer(tokenizer_path)

    # display the tokens
    tokenizer_utility = TokenDisplayUtility(tokenizer)
    tokenizer_utility.display_tokens_from_prompt(args.prompt)

