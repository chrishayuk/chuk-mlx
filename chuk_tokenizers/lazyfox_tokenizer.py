from transformers import PreTrainedTokenizer
import json
import os

class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        # Define special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'

        # Initialize vocabulary
        local_vocab = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
        
        # Combine special tokens with the local vocabulary
        combined_vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + local_vocab
        
        # Create a dictionary with token content
        self.vocab = {token: i for i, token in enumerate(combined_vocab)}

        # Assign special token IDs based on vocab dictionary
        kwargs['pad_token'] = self.pad_token
        kwargs['unk_token'] = self.unk_token
        kwargs['bos_token'] = self.bos_token
        kwargs['eos_token'] = self.eos_token
        kwargs['pad_token_id'] = self.vocab[self.pad_token]
        kwargs['unk_token_id'] = self.vocab[self.unk_token]
        kwargs['bos_token_id'] = self.vocab[self.bos_token]
        kwargs['eos_token_id'] = self.vocab[self.eos_token]

        # Call the base class initializer
        super().__init__(**kwargs)

    def get_vocab(self):
        return self.vocab

    def tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        return {idx: tok for tok, idx in self.vocab.items()}.get(index, self.unk_token)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        return [self._convert_id_to_token(id) for id in ids if not (skip_special_tokens and id in [self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id])]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self._convert_token_to_id(token) for token in tokens]

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            f.write(json.dumps(self.vocab, indent=2))
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.bos_token_id] + token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(self, token_ids):
        return [0] * len(token_ids)
