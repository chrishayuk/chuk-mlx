import pandas
import numpy as np

class SequenceUtility:
    def __init__(self, max_seq_length, padding_value=0):
        self.max_seq_length = max_seq_length
        self.padding_value = padding_value

    def pad_sequence(self, sequence):
        """Pad the sequence to the maximum length with the specified padding value."""
        # Truncate the sequence if it is longer than max_seq_length
        padded_sequence = sequence[:self.max_seq_length]
        
        # Check if padding_value is not set
        if self.padding_value is None:
            self.padding_value = 0  # Default padding value if none provided
        
        # Check if padding_value is a list
        if isinstance(self.padding_value, list):
            pad_token = self.padding_value[0]
        else:
            pad_token = self.padding_value
        
        # Pad the sequence if it is shorter than max_seq_length
        padding_length = self.max_seq_length - len(padded_sequence)
        padded_sequence += [pad_token] * padding_length
        
        return padded_sequence

    def batch_sequences(self, sequences):
        # Ensure all sequences have the same length
        sequences_padded = []
        for seq in sequences:
            seq_padded = self.pad_sequence(seq)
            sequences_padded.append(seq_padded)
    
        return sequences_padded

    def create_next_token_target_sequence(self, input_sequences, pad_token_id):
        """Create next token target sequences by shifting input sequences by one position to the right."""
        target_sequences = []
        for seq in input_sequences:
            # Shift right: remove the first element, append pad_token_id
            target_seq = seq[1:] + pad_token_id
            # Pad the rest to make sure all are the same length
            padded_target_seq = self.pad_sequence(target_seq)
            target_sequences.append(padded_target_seq)
        return np.array(target_sequences, dtype=int)
