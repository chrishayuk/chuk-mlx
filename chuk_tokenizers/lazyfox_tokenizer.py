from transformers import PreTrainedTokenizer
import json
import os

from transformers import PreTrainedTokenizer
import json
import os

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        # Ensure a vocab_file is provided and exists
        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError("A valid vocab_file path must be provided")

        # Load vocabulary from the provided JSON file
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)

        # Extract vocab and special tokens
        self.vocab = vocab_data.get('vocab', {})
        self.special_tokens = vocab_data.get('special_tokens', {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        })

        # Ensure all special tokens are in the vocabulary
        for token, id in self.special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = id

        # Create reverse mapping from IDs to tokens
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}

        # Call the base class initializer first
        super().__init__(**kwargs)

        # Now assign special token IDs based on vocab dictionary
        self.pad_token_id = self.vocab.get('<pad>')
        self.unk_token_id = self.vocab.get('<unk>')
        self.bos_token_id = self.vocab.get('<bos>')
        self.eos_token_id = self.vocab.get('<eos>')

        # Ensure all special tokens are set correctly
        if None in (self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id):
            raise ValueError("Special token IDs are not correctly set.")

    def get_vocab(self):
        # Return the full vocabulary including special tokens
        return self.vocab

    def tokenize(self, text):
        # Split by whitespace to tokenize
        return text.split()

    def _convert_token_to_id(self, token):
        # Convert token to ID, falling back to <unk> if not found
        return self.vocab.get(token, self.vocab['<unk>'])
    
    def _convert_id_to_token(self, index):
        # Convert ID to token, falling back to <unk> if not found
        return self.ids_to_tokens.get(index, '<unk>')

    def convert_tokens_to_ids(self, tokens):
        # Convert a single token or list of tokens to IDs
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # Convert a single ID or list of IDs to tokens
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [
            self._convert_id_to_token(id)
            for id in ids
            if not (skip_special_tokens and id in self.special_tokens.values())
        ]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # Add special tokens to a sequence of token IDs
        if token_ids_1 is None:
            return [self.vocab['<bos>']] + token_ids_0 + [self.vocab['<eos>']]
        else:
            return [self.vocab['<bos>']] + token_ids_0 + [self.vocab['<eos>']] + token_ids_1 + [self.vocab['<eos>']]

    def encode(self, text, text_pair=None, add_special_tokens=True, max_length=None, padding=False, truncation=False, return_tensors=None):
        # Tokenize and convert text to input IDs
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(input_ids)

        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        if padding:
            input_ids = self.pad({'input_ids': input_ids}, padding=padding, max_length=max_length)['input_ids']

        return input_ids

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_attention_mask=None):
        # Pad sequences to a maximum length
        if isinstance(encoded_inputs, (list, tuple)):
            encoded_inputs = {'input_ids': encoded_inputs}

        input_ids = encoded_inputs['input_ids']

        # Ensure input_ids is a list of lists
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        # Determine the maximum length for padding
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        padded_inputs = []
        for ids in input_ids:
            # Calculate the padding length needed
            padding_length = max_length - len(ids) - 1  # Subtract 1 to account for the EOS token

            if padding_length > 0:
                # Append the EOS token, then pad
                padded_sequence = ids + [self.vocab['<eos>']] + [self.vocab['<pad>']] * padding_length
            else:
                # If no padding is needed, just add the EOS token
                padded_sequence = ids[:max_length - 1] + [self.vocab['<eos>']]

            padded_inputs.append(padded_sequence)

        encoded_inputs['input_ids'] = padded_inputs

        # Handle attention mask if requested
        if return_attention_mask:
            attention_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in padded_inputs]
            encoded_inputs['attention_mask'] = attention_mask

        return encoded_inputs

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        vocab_data = {
            "version": "1.0",
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "added_tokens": []  # Ensure added_tokens is included, even if empty
        }
        with open(vocab_file, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        return (vocab_file,)
