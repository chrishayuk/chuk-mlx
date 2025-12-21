import json
import os


def load_vocabulary(vocab_file):
    """Load vocabulary and special tokens from a JSON file."""
    if not vocab_file or not os.path.exists(vocab_file):
        raise ValueError("A valid vocab_file path must be provided")

    # open the vocab
    with open(vocab_file) as f:
        # load the json
        vocab_data = json.load(f)

    # get the vocab, special tokens and added tokens
    vocab = vocab_data.get("vocab", {})
    special_tokens = vocab_data.get("special_tokens", {})
    added_tokens = vocab_data.get("added_tokens", [])

    # return the vocab
    return vocab, special_tokens, added_tokens


def save_vocabulary(vocab, special_tokens, added_tokens, save_directory, version="1.0"):
    """Save the vocabulary, special tokens, and added tokens to a JSON file."""

    # check the save directory exists
    if not os.path.exists(save_directory):
        # make the directory if not
        os.makedirs(save_directory)

    # load the vocab file
    vocab_file = os.path.join(save_directory, "tokenizer.json")

    #
    vocab_data = {
        "version": version,
        "vocab": vocab,
        "special_tokens": special_tokens,
        "added_tokens": added_tokens,
    }

    with open(vocab_file, "w") as f:
        json.dump(vocab_data, f, indent=2)

    return vocab_file
