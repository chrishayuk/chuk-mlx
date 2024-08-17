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

    # Argument for prompt
    parser.add_argument(
        "--prompt",
        default=None,
        help="The prompt"
    )

    # New argument to control special tokens
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Don't add special tokens when tokenizing the prompt"
    )

    # parse arguments
    args = parser.parse_args()

    # load the tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = load_tokenizer(tokenizer_path)

    # display the tokens
    tokenizer_utility = TokenDisplayUtility(tokenizer)

    # check if a prompt is provided
    if args.prompt:
        # display the prompt tokens, respecting the --no_special_tokens flag
        tokenizer_utility.display_tokens_from_prompt(args.prompt, add_special_tokens=not args.skip_special_tokens)
    else:
        # display the full vocabulary
        tokenizer_utility.display_full_vocabulary(chunk_size=500, pause_between_chunks=True)