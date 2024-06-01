import numpy as np

class SequenceUtility:
    def __init__(self, max_seq_length, padding_value=0, initial_pad_token=1):
        self.max_seq_length = max_seq_length
        self.padding_value = padding_value
        self.initial_pad_token = initial_pad_token

    def pad_sequence(self, sequence):
        """Pad the sequence to the maximum length with the specified padding value."""
        if not sequence:
            # Return a list of padding tokens if the sequence is empty
            return [self.initial_pad_token] + [self.padding_value] * (self.max_seq_length - 1)

        # Truncate the sequence if it is longer than max_seq_length
        padded_sequence = sequence[:self.max_seq_length]

        # Determine the padding length
        padding_length = self.max_seq_length - len(padded_sequence)

        if padding_length > 0:
            # Add the initial padding token followed by the regular padding tokens
            padded_sequence += [self.initial_pad_token] + [self.padding_value] * (padding_length - 1)

        # Ensure there are no None values in the padded sequence
        padded_sequence = [token if token is not None else self.padding_value for token in padded_sequence]

        return padded_sequence

    def batch_sequences(self, sequences):
        # Pad the sequences and filter out any None values
        padded_sequences = [self.pad_sequence(seq) for seq in sequences if seq]

        return padded_sequences
