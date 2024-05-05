import json
import os
from transformers import PreTrainedTokenizer

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        # Set pad and unk tokens
        pad_token = '<pad>'
        unk_token = '<unk>'

        # Initialize vocabulary
        local_vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
        # Combine pad and unk with the local vocabulary
        combined_vocab = [pad_token, unk_token] + local_vocab
        
        # Create a dictionary with token content
        self.vocab = {token: i for i, token in enumerate(combined_vocab)}

        # Set kwargs for pad and unk tokens
        kwargs['pad_token'] = pad_token
        kwargs['unk_token'] = unk_token

        # Assign special token IDs based on vocab dictionary
        kwargs['pad_token_id'] = self.vocab[pad_token]
        kwargs['unk_token_id'] = self.vocab[unk_token]

        # Call the base class initializer
        super().__init__(**kwargs)

    def get_vocab(self):
        return self.vocab

    def tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get('<unk>'))

    def _convert_id_to_token(self, index):
        # Reverse lookup by index
        return {idx: tok for tok, idx in self.vocab.items()}.get(index, '<unk>')

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]  # Ensure tokens are always handled as a list
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]  # Ensure ids are always handled as a list
        return [self._convert_id_to_token(id) for id in ids if not skip_special_tokens or id not in [self.vocab.get('<pad>'), self.vocab.get('<unk>')]]

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            f.write(json.dumps(self.vocab, indent=2))
        return (vocab_file,)
