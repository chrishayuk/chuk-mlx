import pandas
import numpy as np

class SequenceUtility:
    def __init__(self, max_seq_length, padding_value=0):
        self.max_seq_length = max_seq_length
        self.padding_value = padding_value

    def pad_sequence(self, sequence):
        """Pad the sequence to the maximum length with the specified padding value."""
        if not sequence:
            # Return a list of padding tokens if the sequence is empty
            return [self.padding_value] * self.max_seq_length

        # Truncate the sequence if it is longer than max_seq_length
        padded_sequence = sequence[:self.max_seq_length]

        # Set default padding value if not set
        pad_token = self.padding_value[0] if isinstance(self.padding_value, list) else self.padding_value or 0

        # Pad the sequence if it is shorter than max_seq_length
        padding_length = self.max_seq_length - len(padded_sequence)
        padded_sequence += [pad_token] * padding_length

        # Ensure there are no None values in the padded sequence
        padded_sequence = [token if token is not None else pad_token for token in padded_sequence]

        return padded_sequence

    def batch_sequences(self, sequences):
        # Pad the sequences and filter out any None values
        padded_sequences = [self.pad_sequence(seq) for seq in sequences if seq]

        return padded_sequences
