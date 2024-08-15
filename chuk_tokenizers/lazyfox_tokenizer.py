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
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}

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
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(id) for id in ids 
                if not (skip_special_tokens and id in [self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id])]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def encode(self, text, text_pair=None, add_special_tokens=True, max_length=None, padding=False, truncation=False, return_tensors=None):
        if isinstance(text, str):
            tokens = self.tokenize(text)
        elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
            tokens = [t for sent in text for t in self.tokenize(sent)]
        else:
            raise ValueError("Text input must be a string or a list of strings")

        input_ids = self.convert_tokens_to_ids(tokens)

        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(input_ids)

        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        if padding:
            input_ids = self.pad({'input_ids': input_ids}, padding=padding, max_length=max_length)['input_ids']

        return input_ids

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_attention_mask=None):
        if isinstance(encoded_inputs, (list, tuple)):
            encoded_inputs = {'input_ids': encoded_inputs}

        input_ids = encoded_inputs['input_ids']

        if isinstance(input_ids[0], int):
            input_ids = [input_ids]

        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        padded_inputs = []
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_inputs.append(ids + [self.pad_token_id] * padding_length)

        encoded_inputs['input_ids'] = padded_inputs

        if return_attention_mask:
            attention_mask = [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids]
            encoded_inputs['attention_mask'] = attention_mask

        return encoded_inputs

    def save_vocabulary(self, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w') as f:
            f.write(json.dumps(self.vocab, indent=2))
        return (vocab_file,)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        text = ' '.join(filtered_tokens)
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text

    def clean_up_tokenization(self, text):
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        """
        text = text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
        text = text.replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m")
        text = text.replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        return text