import pytest
import json
import os
from chuk_tokenizers.custom_tokenizer import CustomTokenizer

@pytest.fixture
def tokenizer():
    # Set the vocab file
    vocab_file = "model_configuration/lazyfox/tokenizer.json"

    # Return the tokenizer
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
    sequence = [4, 5, 6, 7]
    padded = tokenizer.pad(sequence, max_length=7, padding=True)
    
    expected_padded = [4, 5, 6, 7, 3, 0, 0]  # ['the', 'quick', 'brown', 'fox', <eos>, <pad>, <pad>]
    assert padded == expected_padded

def test_no_padding_needed(tokenizer):
    """Test sequences that do not require padding."""
    sequence = [2, 4, 5, 6, 7, 3]
    padded = tokenizer.pad(sequence, max_length=6, padding=True)
    
    expected_padded = [2, 4, 5, 6, 7, 3]  # [<bos>, 'the', 'quick', 'brown', 'fox', <eos>]
    assert padded == expected_padded

def test_truncation(tokenizer):
    """Test truncation when sequences exceed max_length."""
    sequence = [4, 5, 6, 7, 8]  # Input sequence
    max_length = 5

    # The expected behavior is to truncate and add <eos> at the end
    padded = tokenizer.pad(sequence, max_length=max_length, padding=True)

    # Since the sequence is longer than max_length, it should be truncated to 4 tokens + <eos>
    expected_padded = [4, 5, 6, 7, 3]  # [<the>, <quick>, <brown>, <fox>, <eos>]
    assert padded == expected_padded

def test_pad_to_multiple_of(tokenizer):
    """Test padding to a multiple of a specific value."""
    sequence = [4, 5, 6]
    padded = tokenizer.pad(sequence, max_length=6, pad_to_multiple_of=7, padding=True)
    
    expected_padded = [4, 5, 6, 3, 0, 0, 0]  # ['the', 'quick', 'brown', <eos>, <pad>, <pad>, <pad>]
    assert padded == expected_padded

def test_attention_mask(tokenizer):
    """Test attention mask generation during padding."""
    sequences = [[4, 5, 6], [7]]
    padded, attention_mask = tokenizer.pad(sequences[0], max_length=7, padding=True, return_attention_mask=True)
    
    expected_padded = [4, 5, 6, 3, 0, 0, 0]  # ['the', 'quick', 'brown', <eos>, <pad>, <pad>, <pad>]
    expected_attention_mask = [1, 1, 1, 1, 0, 0, 0]  # 'the', 'quick', 'brown', <eos> are attended to; <pad> are not
    assert padded == expected_padded
    assert attention_mask == expected_attention_mask

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
    vocab_file = tokenizer.save_vocabulary(tmp_path)
    assert os.path.isfile(vocab_file)  # Ensure the path points to a file

    with open(vocab_file, 'r') as f:
        loaded_data = json.load(f)

    assert loaded_data["version"] == "1.0"
    assert loaded_data["vocab"] == tokenizer.get_vocab()
    assert loaded_data["special_tokens"] == tokenizer.special_tokens
    assert loaded_data["added_tokens"] == []

    reloaded_tokenizer = CustomTokenizer(vocab_file=vocab_file)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reloaded_tokenizer.special_tokens == tokenizer.special_tokens

