import numpy as np
from core.batch.batch_analysis_summary import generate_batch_analysis_summary_table
from core.batch.batch_base import BatchBase
from core.batch.batch_generation_summary import generate_batch_generation_summary
from core.batch.bucketing import add_to_buckets

class FineTuneBatch(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Initialize the base class
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def pad_sequences(self, input_sequences, target_sequences, pad_token_id):
        """
        Pads both input and target sequences to the length of the longest sequence in the batch.
        Handles cases where sequences are empty or have different lengths.
        """

        # check for input and targets
        if not input_sequences and not target_sequences:
            # return
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        # Determine the maximum length across both input and target sequences
        max_input_length = max((len(seq) for seq in input_sequences), default=0)
        max_target_length = max((len(seq) for seq in target_sequences), default=0)
        
        # Set the max length to pad both input and target sequences
        max_length = max(max_input_length, max_target_length)
        
        def pad(seq, max_len):
            padded_seq = seq + [pad_token_id] * (max_len - len(seq))
            return padded_seq
        
        # Handle cases where input_sequences or target_sequences might be empty
        if input_sequences:
            padded_input_sequences = [pad(seq, max_length) for seq in input_sequences]
        else:
            padded_input_sequences = np.array([], dtype=np.int32)
        
        if target_sequences:
            padded_target_sequences = [pad(seq, max_length) for seq in target_sequences]
        else:
            padded_target_sequences = np.array([], dtype=np.int32)
        
        # Convert to numpy arrays and ensure shapes are consistent
        padded_input_sequences = np.array(padded_input_sequences, dtype=np.int32)
        padded_target_sequences = np.array(padded_target_sequences, dtype=np.int32)

        # return the final padded sequences        
        return padded_input_sequences, padded_target_sequences


    def tokenize_and_batch(self, input_files):
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)
        
        # Initialize buckets for storing sequences
        buckets = {}
        
        # Process each tokenized sequence
        for input_tokens, target_tokens in tokenized_dataset:
            # Pad both input and target sequences to the same length
            input_tokens_padded, target_tokens_padded = self.pad_sequences(
                [input_tokens], [target_tokens], self.tokenizer.pad_token_id
            )

            # Add the padded input and target sequences to the appropriate buckets
            add_to_buckets(buckets, input_tokens_padded[0], target_tokens_padded[0])
        
        # Create batches from the filled buckets and process them
        self.create_batches(buckets)






