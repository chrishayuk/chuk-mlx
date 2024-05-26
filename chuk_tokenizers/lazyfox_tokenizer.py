from transformers import PreTrainedTokenizer
import json
import os

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        # Set pad, unk, bos, and eos tokens
        pad_token = '<pad>'
        unk_token = '<unk>'
        bos_token = '<bos>'
        eos_token = '<eos>'

        # Initialize vocabulary
        local_vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']

        # Combine special tokens with the local vocabulary
        combined_vocab = [pad_token, unk_token, bos_token, eos_token] + local_vocab
        
        # Create a dictionary with token content
        self.vocab = {token: i for i, token in enumerate(combined_vocab)}

        # Set kwargs for pad, unk, bos, and eos tokens
        kwargs['pad_token'] = pad_token
        kwargs['unk_token'] = unk_token
        kwargs['bos_token'] = bos_token
        kwargs['eos_token'] = eos_token

        # Assign special token IDs based on vocab dictionary
        kwargs['pad_token_id'] = self.vocab[pad_token]
        kwargs['unk_token_id'] = self.vocab[unk_token]
        kwargs['bos_token_id'] = self.vocab[bos_token]
        kwargs['eos_token_id'] = self.vocab[eos_token]

        # Call the base class initializer
        super().__init__(**kwargs)

        # Set special token IDs as attributes
        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]
        self.bos_token_id = self.vocab[bos_token]
        self.eos_token_id = self.vocab[eos_token]

    def get_vocab(self):
        return self.vocab

    def tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get('<unk>'))
    
    def _convert_id_to_token(self, index):
        if isinstance(index, list):
            return [self._convert_id_to_token(idx) for idx in index]
        else:
            return {idx: tok for tok, idx in self.vocab.items()}.get(index, '<unk>')

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]  # Ensure ids are always handled as a list
        return [self._convert_id_to_token(id) for id in ids if not (skip_special_tokens and id in [self.vocab['<pad>'], self.vocab['<unk>'], self.vocab['<bos>'], self.vocab['<eos>']])]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            # Ensure tokens are always handled as a list
            tokens = [tokens]
        
        # Convert the tokens to a list of ids
        return [self._convert_token_to_id(token) for token in tokens]

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            f.write(json.dumps(self.vocab, indent=2))
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        A sequence has the following format: [BOS_token] X [EOS_token]
        """
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids):
        """
        Creates a token type ID tensor from a sequence of token IDs.
        """
        return [0] * len(token_ids)
