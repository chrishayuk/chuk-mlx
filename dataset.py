import random

class Dataset:
    def __init__(self, input_sequences, tokenizer, add_special_tokens=True):
        self.input_sequences = input_sequences
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if index >= len(self.input_sequences) or index < 0:
            raise IndexError("Index out of bounds")
        input_sequence = self.input_sequences[index]
        input_indices = self.tokenizer.encode(input_sequence, add_special_tokens=self.add_special_tokens)
        return input_indices

    def get_batch(self, batch_size, start_index=0):
        """Retrieve a batch of data starting from a specific index."""
        end_index = min(len(self.input_sequences), start_index + batch_size)
        batch_sequences = self.input_sequences[start_index:end_index]
        batch_indices = [self.tokenizer.encode(seq, add_special_tokens=self.add_special_tokens) for seq in batch_sequences]
        return batch_indices

    def shuffle(self):
        """Shuffle the dataset sequences."""
        random.shuffle(self.input_sequences)
