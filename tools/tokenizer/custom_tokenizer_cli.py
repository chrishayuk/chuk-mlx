import argparse
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.tokenizers.custom_tokenizer import CustomTokenizer

def get_tokenizer_config(tokenizer_name):
    """Construct the path to the tokenizer's vocab file based on its name."""

    # get the configuration from the model configuration folder
    vocab_file = f"model_configuration/{tokenizer_name}/tokenizer.json"

    # check if existsa
    if os.path.exists(vocab_file):
        # return the config
        return vocab_file
    else:
        # throw an error
        raise ValueError(f"Tokenizer configuration file '{vocab_file}' not found.")

def get_default_save_path(tokenizer_name):
    """Construct the default save path for the vocabulary."""
    return f"output/tokenizers/{tokenizer_name}"

def main():
    # setup the cli parser
    parser = argparse.ArgumentParser(description="Custom Tokenizer CLI")
    
    # setup the sub parsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common argument for all commands: the tokenizer name
    parser.add_argument("--tokenizer-name", type=str, default="lazyfox", help="Name of the tokenizer (e.g., lazyfox). Defaults to 'lazyfox'.")
    
    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text into token IDs")
    encode_parser.add_argument("text", type=str, help="Text to encode")
    encode_parser.add_argument("--add-special-tokens", action="store_true", help="Add special tokens like <bos> and <eos>")
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs back to text")
    decode_parser.add_argument("token_ids", type=int, nargs='+', help="Token IDs to decode")
    decode_parser.add_argument("--skip-special-tokens", action="store_true", help="Skip special tokens like <bos> and <eos>")
    
    # Pad command
    pad_parser = subparsers.add_parser("pad", help="Pad a sequence of token IDs")
    pad_parser.add_argument("sequences", type=int, nargs='+', help="Tokenized sequence to pad")
    pad_parser.add_argument("--max-length", type=int, required=True, help="Maximum length for padding")
    pad_parser.add_argument("--pad-to-multiple-of", type=int, help="Pad to a multiple of this value")
    pad_parser.add_argument("--return-attention-mask", action="store_true", help="Return the attention mask")

    # Save vocabulary command
    save_parser = subparsers.add_parser("save-vocab", help="Save the vocabulary to a file")
    save_parser.add_argument("--output-file", type=str, required=False, help="Path to save the vocabulary file")
    
     # Load vocabulary command
    load_parser = subparsers.add_parser("load-vocab", help="Load the vocabulary from a file")
    load_parser.add_argument("--input-file", type=str, help="Path to load the vocabulary file from. Defaults to 'model_configuration/{tokenizer_name}/vocab.json'")

    # parse
    args = parser.parse_args()

    # Resolve the vocab file path based on the tokenizer name
    vocab_file = get_tokenizer_config(args.tokenizer_name)

    if args.command == "encode":
        tokenizer = CustomTokenizer(vocab_file=vocab_file)
        encoded = tokenizer.encode(args.text, add_special_tokens=args.add_special_tokens)
        print("Encoded IDs:", encoded)

    elif args.command == "decode":
        tokenizer = CustomTokenizer(vocab_file=vocab_file)
        decoded = tokenizer.decode(args.token_ids, skip_special_tokens=args.skip_special_tokens)
        print("Decoded Text:", decoded)

    elif args.command == "pad":
        tokenizer = CustomTokenizer(vocab_file=vocab_file)
        # Directly pass the list of integers to pad
        if args.return_attention_mask:
            padded_sequence, attention_mask = tokenizer.pad(
                args.sequences, max_length=args.max_length, 
                pad_to_multiple_of=args.pad_to_multiple_of, 
                return_attention_mask=True
            )
            print("Padded Sequence:", padded_sequence)
            print("Attention Mask:", attention_mask)
        else:
            padded_sequence = tokenizer.pad(
                args.sequences, max_length=args.max_length, 
                pad_to_multiple_of=args.pad_to_multiple_of
            )
            print("Padded Sequence:", padded_sequence)

    elif args.command == "save-vocab":
        tokenizer = CustomTokenizer(vocab_file=vocab_file)
        output_file = args.output_file or os.path.join(get_default_save_path(args.tokenizer_name), "tokenizer.json")
        tokenizer.save_vocabulary(os.path.dirname(output_file))
        print(f"Vocabulary saved to {output_file}")

    elif args.command == "load-vocab":
        # Use the provided input file or fall back to the default path
        input_file = args.input_file or os.path.join(get_tokenizer_config(args.tokenizer_name))
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The vocabulary file '{input_file}' was not found.")
        
        tokenizer = CustomTokenizer(vocab_file=input_file)
        print(f"Loaded vocabulary from {input_file}")
        print("Vocabulary:", tokenizer.get_vocab())

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
