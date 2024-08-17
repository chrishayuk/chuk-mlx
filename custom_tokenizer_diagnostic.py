import json
import os
import shutil
from chuk_tokenizers.custom_tokenizer import CustomTokenizer

def check_special_token_ids(tokenizer):
    """Check that special token IDs are correctly set."""
    try:
        assert tokenizer.pad_token_id == 0, f"Expected pad_token_id to be 0, but got {tokenizer.pad_token_id}"
        assert tokenizer.unk_token_id == 1, f"Expected unk_token_id to be 1, but got {tokenizer.unk_token_id}"
        assert tokenizer.bos_token_id == 2, f"Expected bos_token_id to be 2, but got {tokenizer.bos_token_id}"
        assert tokenizer.eos_token_id == 3, f"Expected eos_token_id to be 3, but got {tokenizer.eos_token_id}"
        print("Special token IDs are correctly set.")
    except AssertionError as e:
        print(f"Error in check_special_token_ids: {e}")
        raise

def check_vocab_loading(tokenizer):
    """Check that the vocabulary is correctly loaded and consistent."""
    try:
        expected_vocab = {
            "the": 4,
            "quick": 5,
            "brown": 6,
            "fox": 7,
            "jumps": 8,
            "over": 9,
            "lazy": 10,
            "dog": 11
        }
        actual_vocab = {token: id for token, id in tokenizer.get_vocab().items() if token not in tokenizer.special_tokens}
        assert actual_vocab == expected_vocab, f"Vocabulary does not match. Got {actual_vocab}"
        print("Vocabulary is correctly loaded.")
    except AssertionError as e:
        print(f"Error in check_vocab_loading: {e}")
        raise

def check_token_to_id_conversion(tokenizer):
    """Check that tokens are correctly converted to IDs."""
    try:
        token_ids = tokenizer.convert_tokens_to_ids(["the", "quick", "brown", "fox"])
        assert token_ids == [4, 5, 6, 7], f"Token to ID conversion failed. Got {token_ids}"

        special_token_ids = tokenizer.convert_tokens_to_ids(["<bos>", "the", "quick", "<eos>"])
        assert special_token_ids == [2, 4, 5, 3], f"Special token to ID conversion failed. Got {special_token_ids}"

        print("Token to ID conversion works correctly.")
    except AssertionError as e:
        print(f"Error in check_token_to_id_conversion: {e}")
        raise

def check_id_to_token_conversion(tokenizer):
    """Check that IDs are correctly converted back to tokens."""
    try:
        tokens = tokenizer.convert_ids_to_tokens([4, 5, 6, 7])
        assert tokens == ["the", "quick", "brown", "fox"], f"ID to token conversion failed. Got {tokens}"

        special_tokens = tokenizer.convert_ids_to_tokens([2, 4, 5, 3])
        assert special_tokens == ["<bos>", "the", "quick", "<eos>"], f"Special ID to token conversion failed. Got {special_tokens}"

        print("ID to token conversion works correctly.")
    except AssertionError as e:
        print(f"Error in check_id_to_token_conversion: {e}")
        raise

def check_encoding(tokenizer):
    """Check that text is correctly encoded into token IDs."""
    try:
        text = "the quick brown fox"
        encoded = tokenizer.encode(text)
        expected_encoded = [2, 4, 5, 6, 7, 3]
        assert encoded == expected_encoded, f"Encoding failed. Got {encoded}"

        print("Text encoding works correctly.")
    except AssertionError as e:
        print(f"Error in check_encoding: {e}")
        raise

def check_padding(tokenizer):
    """Check that sequences are correctly padded."""
    try:
        text = ["the quick brown", "fox"]
        encoded = [tokenizer.encode(t, add_special_tokens=False) for t in text]

        # Flatten the list of lists before padding
        padded_sequences = [tokenizer.pad(seq, max_length=7, padding=True) for seq in encoded]
        
        expected_padded = [
            [4, 5, 6, 3, 0, 0, 0],
            [7, 3, 0, 0, 0, 0, 0]
        ]

        assert padded_sequences == expected_padded, f"Padding failed. Got {padded_sequences}"

        print("Sequence padding works correctly.")
    except AssertionError as e:
        print(f"Error in check_padding: {e}")
        raise


def check_decoding(tokenizer):
    """Check that token IDs are correctly decoded back into text."""
    try:
        token_ids = [2, 4, 5, 6, 7, 3]
        decoded = tokenizer.decode(token_ids)
        expected_decoded = "<bos> the quick brown fox <eos>"
        assert decoded == expected_decoded, f"Decoding failed. Got {decoded}"

        decoded_no_specials = tokenizer.decode(token_ids, skip_special_tokens=True)
        expected_decoded_no_specials = "the quick brown fox"
        assert decoded_no_specials == expected_decoded_no_specials, f"Decoding without special tokens failed. Got {decoded_no_specials}"

        print("Token decoding works correctly.")
    except AssertionError as e:
        print(f"Error in check_decoding: {e}")
        raise

def check_save_and_load(tokenizer):
    """Check that the vocabulary can be saved and loaded correctly."""
    save_dir = os.path.join(os.getcwd(), "test_vocab")
    try:
        os.makedirs(save_dir, exist_ok=True)
        vocab_file = tokenizer.save_vocabulary(save_dir)
        
        print(f"Vocabulary file saved at: {vocab_file}")  # Debugging line
        
        assert os.path.isfile(vocab_file), f"Expected a file but got a directory: {vocab_file}"

        with open(vocab_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["version"] == "1.0", "Incorrect version in saved vocab file."
        assert loaded_data["vocab"] == tokenizer.get_vocab(), "Vocab does not match after saving and loading."
        assert loaded_data["special_tokens"] == tokenizer.special_tokens, "Special tokens do not match after saving and loading."

        print("Vocabulary save and load works correctly.")
    except AssertionError as e:
        print(f"Error in check_save_and_load: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in check_save_and_load: {e}")
        raise
    finally:
        # Clean up the saved vocab directory
        shutil.rmtree(save_dir)


def run_all_diagnostics():
    """Run all diagnostic checks on the CustomTokenizer."""
    try:
        # Initialize the tokenizer
        tokenizer = CustomTokenizer(vocab_file="model_configuration/lazyfox/tokenizer.json")

        # Run all diagnostics
        check_special_token_ids(tokenizer)
        check_vocab_loading(tokenizer)
        check_token_to_id_conversion(tokenizer)
        check_id_to_token_conversion(tokenizer)
        check_encoding(tokenizer)
        check_padding(tokenizer)
        check_decoding(tokenizer)
        check_save_and_load(tokenizer)

        # Final outcome
        print("All diagnostic checks passed successfully.")
    except Exception as e:
        print(f"An error occurred during diagnostics: {e}")

if __name__ == "__main__":
    # Run diagnostics
    run_all_diagnostics()
