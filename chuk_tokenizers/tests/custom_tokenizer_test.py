import pytest
import json
import os
from chuk_tokenizers.lazyfox_tokenizer import CustomTokenizer

@pytest.fixture
def tokenizer():
    # set the vocab file
    vocab_file = "model_configuration/lazyfox/tokenizer.json"

    # return the tokenizer
    return CustomTokenizer(vocab_file=vocab_file)

def test_vocab_loading(tokenizer):
    """Test if the vocabulary contains the correct tokens."""
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

    # Remove special tokens from the vocabulary for comparison
    actual_vocab = {token: id for token, id in tokenizer.get_vocab().items() if token not in ['<pad>', '<unk>', '<bos>', '<eos>']}
    assert actual_vocab == expected_vocab

    # Check that special tokens are correctly loaded
    expected_special_tokens = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    for token, token_id in expected_special_tokens.items():
        assert tokenizer.special_tokens[token] == token_id

    # Check reverse mapping
    expected_ids_to_tokens = {v: k for k, v in {**expected_special_tokens, **expected_vocab}.items()}
    assert tokenizer.ids_to_tokens == expected_ids_to_tokens

def test_special_token_ids(tokenizer):
    """Test if special tokens have the correct IDs."""
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3

def test_token_to_id_conversion(tokenizer):
    """Test conversion from tokens to IDs."""
    token_ids = tokenizer.convert_tokens_to_ids(["the", "quick", "brown", "fox"])
    assert token_ids == [4, 5, 6, 7]

    # Test with special tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(["<bos>", "the", "quick", "<eos>"])
    assert special_token_ids == [2, 4, 5, 3]

def test_id_to_token_conversion(tokenizer):
    """Test conversion from IDs to tokens."""
    tokens = tokenizer.convert_ids_to_tokens([4, 5, 6, 7])
    assert tokens == ["the", "quick", "brown", "fox"]

    # Test with special token IDs
    special_tokens = tokenizer.convert_ids_to_tokens([2, 4, 5, 3])
    assert special_tokens == ["<bos>", "the", "quick", "<eos>"]

def test_encode(tokenizer):
    """Test encoding of text to token IDs with special tokens."""
    text = "the quick brown fox"
    encoded = tokenizer.encode(text)
    expected_encoded = [2, 4, 5, 6, 7, 3]  # [<bos>, 'the', 'quick', 'brown', 'fox', <eos>]
    assert encoded == expected_encoded

def test_padding(tokenizer):
    """Test padding of sequences."""
    text = ["the quick brown", "fox"]
    encoded = [tokenizer.encode(t, add_special_tokens=False) for t in text]
    padded = tokenizer.pad(encoded, max_length=7, padding=True)['input_ids']
    
    expected_padded = [
        [4, 5, 6, 3, 0, 0, 0],  # ['the', 'quick', 'brown', <eos>, <pad>, <pad>, <pad>]
        [7, 3, 0, 0, 0, 0, 0]   # ['fox', <eos>, <pad>, <pad>, <pad>, <pad>, <pad>]
    ]
    assert padded == expected_padded

def test_decode(tokenizer):
    """Test decoding from token IDs to text."""
    token_ids = [2, 4, 5, 6, 7, 3]  # [<bos>, 'the', 'quick', 'brown', 'fox', <eos>]

    # Test including special tokens (default behavior)
    decoded = tokenizer.decode(token_ids)
    expected_decoded = "<bos> the quick brown fox <eos>"
    assert decoded == expected_decoded

    # Test skipping special tokens
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    expected_decoded = "the quick brown fox"
    assert decoded == expected_decoded

def test_save_and_load_vocabulary(tmp_path, tokenizer):
    """Test saving and loading the vocabulary with the Hugging Face structure."""
    vocab_file = tokenizer.save_vocabulary(tmp_path)[0]
    assert os.path.exists(vocab_file)

    with open(vocab_file, 'r') as f:
        loaded_data = json.load(f)

    # Ensure that the saved data matches the tokenizer's configuration
    assert loaded_data["version"] == "1.0"
    assert loaded_data["vocab"] == tokenizer.get_vocab()
    assert loaded_data["special_tokens"] == tokenizer.special_tokens
    assert loaded_data["added_tokens"] == []

    # Re-initialize a tokenizer with the saved vocab and check consistency
    reloaded_tokenizer = CustomTokenizer(vocab_file=vocab_file)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reloaded_tokenizer.special_tokens == tokenizer.special_tokens
