from transformers import PreTrainedTokenizer
from chuk_lazarus.data.tokenizers.vocab_utils import load_vocabulary, save_vocabulary

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        # Load vocabulary and special tokens using the utility function
        self.vocab, self.special_tokens, self.added_tokens = load_vocabulary(vocab_file)

        # Merge special tokens into the vocabulary
        self.vocab.update(self.special_tokens)

        # Create reverse mapping from IDs to tokens
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}

        # Call the base class initializer first
        super().__init__(**kwargs)

        # Now assign special token IDs based on vocab dictionary
        self.pad_token_id = self.special_tokens.get('<pad>')
        self.unk_token_id = self.special_tokens.get('<unk>')
        self.bos_token_id = self.special_tokens.get('<bos>')
        self.eos_token_id = self.special_tokens.get('<eos>')

        # Ensure all special tokens are set correctly
        if None in (self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id):
            raise ValueError("Special token IDs are not correctly set.")

    def get_vocab(self):
        """Return the full vocabulary including special tokens."""
        return self.vocab

    def tokenize(self, text):
        """Split by whitespace to tokenize."""
        return text.split()

    def _convert_token_to_id(self, token):
        """Convert token to ID, falling back to <unk> if not found."""
        return self.vocab.get(token, self.vocab['<unk>'])
    
    def _convert_id_to_token(self, index):
        """Convert ID to token, falling back to <unk> if not found."""
        return self.ids_to_tokens.get(index, '<unk>')

    def convert_tokens_to_ids(self, tokens):
        """Convert a single token or list of tokens to IDs."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Convert a single ID or list of IDs to tokens."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [
            self._convert_id_to_token(id_)
            for id_ in ids
            if not (skip_special_tokens and id_ in self.special_tokens.values())
        ]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Add special tokens to a sequence of token IDs."""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return (
                [self.bos_token_id] + token_ids_0 + [self.eos_token_id] +
                token_ids_1 + [self.eos_token_id]
            )

    def encode(self, text, text_pair=None, add_special_tokens=True, max_length=None, padding=False, truncation=False, return_tensors=None):
        """Tokenize and convert text to input IDs."""
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(input_ids, self.tokenize(text_pair) if text_pair else None)

        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        if padding:
            input_ids = self.pad({'input_ids': input_ids}, padding=padding, max_length=max_length)['input_ids']

        return input_ids

    def pad(self, sequence, padding=True, max_length=None, pad_to_multiple_of=None, return_attention_mask=False):
        # Ensure the sequence is a list of integers
        if not isinstance(sequence, list) or not all(isinstance(i, int) for i in sequence):
            raise ValueError("Input must be a list of integers.")
        
        # Determine the maximum length for padding
        if max_length is None:
            max_length = len(sequence)

        # Adjust max_length to the nearest multiple of pad_to_multiple_of, if specified
        if pad_to_multiple_of is not None:
            max_length = (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        # Truncate the sequence if it exceeds max_length, add EOS token
        if len(sequence) >= max_length:
            sequence_with_eos = sequence[:max_length - 1] + [self.eos_token_id]
        else:
            if sequence[-1] != self.eos_token_id and len(sequence) < max_length:
                sequence_with_eos = sequence + [self.eos_token_id]
            else:
                sequence_with_eos = sequence

        # Calculate the padding length needed
        padding_length = max_length - len(sequence_with_eos)

        if padding and padding_length > 0:
            padded_sequence = sequence_with_eos + [self.pad_token_id] * padding_length
        else:
            padded_sequence = sequence_with_eos

        # Handle attention mask if requested
        if return_attention_mask:
            attention_mask = [1] * len(sequence_with_eos) + [0] * padding_length
            return padded_sequence, attention_mask

        return padded_sequence




    def save_vocabulary(self, save_directory):
        """Save the vocabulary to the specified directory."""
        return save_vocabulary(self.vocab, self.special_tokens, self.added_tokens, save_directory)
