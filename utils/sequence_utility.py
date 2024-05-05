import pandas
import numpy as np

class SequenceUtility:
    def __init__(self, max_seq_length, padding_value=0):
        self.max_seq_length = max_seq_length
        self.padding_value = padding_value

    def pad_sequence(self, sequence):
        """Pad the sequence to the maximum length with the specified padding value."""

        # truncate the sequence if it is longer than max_seq_length
        padded_sequence = sequence[:self.max_seq_length]

        # check if padding_value is not set or not an integer
        if self.padding_value is None or not isinstance(self.padding_value, int):
            self.padding_value = 0  # default padding value if none provided

        # pad the sequence if it is shorter than max_seq_length
        padding_length = self.max_seq_length - len(padded_sequence)
        padded_sequence += [self.padding_value] * padding_length

        # return the padded sequence
        return padded_sequence


    def batch_sequences(self, sequences):
        """Convert a list of sequences into a batch, applying padding as necessary."""
        return np.array([self.pad_sequence(seq) for seq in sequences], dtype=int)

    def create_next_token_target_sequence(self, input_sequences, pad_token_id):
        """Create next token target sequences by shifting input sequences by one position to the right."""
        target_sequences = []
        for seq in input_sequences:
            # Shift right: remove the first element, append pad_token_id
            target_seq = seq[1:] + [pad_token_id]
            # Pad the rest to make sure all are the same length
            padded_target_seq = self.pad_sequence(target_seq)
            target_sequences.append(padded_target_seq)
        return np.array(target_sequences, dtype=int)
    
    def visualize_sequences(self, sequences, tokenizer, max_columns=8):
        # Ensure max_columns does not exceed max_seq_length
        max_columns = min(max_columns, self.max_seq_length)

        # Fixed width for each column
        fixed_width = 10  

        # Prepare column headers
        columns = [f"T{i}" for i in range(max_columns - 1)] + ["LT", f"T{self.max_seq_length - 1}"]
        header = '|'.join(col.center(fixed_width) for col in columns)
        separator = '-' * len(header)

        # Print the headers
        print(header)
        print(separator)

        # Process each sequence in the batch
        for seq in sequences:
            # Decode tokens and gather token IDs
            tokens = [tokenizer.decode([token_id]) for token_id in seq[:max_columns - 1]]
            token_ids = [str(token_id) for token_id in seq[:max_columns - 1]]

            # Find the last non-pad token's index
            last_non_pad_index = next((i for i, token_id in enumerate(seq[:self.max_seq_length - 1]) if token_id != 0), None)
            if last_non_pad_index == 0:
                last_non_pad_index = self.max_seq_length-1

            # Find the last non-pad token's index correctly
            last_non_pad_index = max((index for index, token_id in enumerate(seq[:max_columns - 1]) if token_id != 0), default=0)
            last_token = tokens[last_non_pad_index]
            last_token_id = token_ids[last_non_pad_index]
            lt_display = f"T{last_non_pad_index}:({last_token_id})"

            # Handle the token at the maximum sequence length - 1
            max_seq_token_index = self.max_seq_length - 1
            max_seq_token = tokenizer.decode([seq[max_seq_token_index]])
            max_seq_token_id = str(seq[max_seq_token_index])
            t_max_display = f"({max_seq_token_id})"

            # Truncate tokens if necessary and prepare display lines
            token_line = ''
            id_line = ''
            for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
                truncated_token = (token[:fixed_width-2]+".." if len(token) > fixed_width-2 else token).center(fixed_width)
                formatted_id = f"({token_id})".center(fixed_width)
                token_line += truncated_token + '|'
                id_line += formatted_id + '|'

            # Add the last non-pad token and the final token displays
            token_line += last_token.center(fixed_width) + '|' + max_seq_token.center(fixed_width) + '|'
            id_line += lt_display.center(fixed_width) + '|' + t_max_display.center(fixed_width) + '|'

            # Print the tokens and their IDs, ensuring exact column alignment
            print(token_line[:-1])  # Trim the trailing '|'
            print(id_line[:-1])  # Trim the trailing '|'
            print(separator)
