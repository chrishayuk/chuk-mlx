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

    def pad_sequences(self, input_sequences, target_sequences, attention_masks, pad_token_id):
        """
        Pads input, target sequences, and attention masks to the length of the longest sequence in the batch.
        Handles cases where sequences are empty or have different lengths.
        """

        # Check for input and targets
        if not input_sequences and not target_sequences:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32)
            )
        
        # Determine the maximum length across input, target sequences, and attention masks
        max_input_length = max((len(seq) for seq in input_sequences), default=0)
        max_target_length = max((len(seq) for seq in target_sequences), default=0)
        max_attention_length = max((len(seq) for seq in attention_masks), default=0)
        
        # Set the max length to pad input, target sequences, and attention masks
        max_length = max(max_input_length, max_target_length, max_attention_length)
        
        def pad(seq, max_len, pad_value):
            return seq + [pad_value] * (max_len - len(seq))
        
        # Handle cases where input_sequences, target_sequences, or attention_masks might be empty
        if input_sequences:
            padded_input_sequences = [pad(seq, max_length, pad_token_id) for seq in input_sequences]
        else:
            padded_input_sequences = np.array([], dtype=np.int32)
        
        if target_sequences:
            padded_target_sequences = [pad(seq, max_length, pad_token_id) for seq in target_sequences]
        else:
            padded_target_sequences = np.array([], dtype=np.int32)

        if attention_masks:
            padded_attention_masks = [pad(seq, max_length, 0) for seq in attention_masks]  # Attention mask is padded with 0
        else:
            padded_attention_masks = np.array([], dtype=np.int32)
        
        # Convert to numpy arrays and ensure shapes are consistent
        padded_input_sequences = np.array(padded_input_sequences, dtype=np.int32)
        padded_target_sequences = np.array(padded_target_sequences, dtype=np.int32)
        padded_attention_masks = np.array(padded_attention_masks, dtype=np.int32)

        # Return the final padded sequences
        return padded_input_sequences, padded_target_sequences, padded_attention_masks


    def tokenize_and_batch(self, input_files):
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)
        
        # Initialize buckets for storing sequences
        buckets = {}
        
        # Process each tokenized sequence
        for idx, (input_tokens, target_tokens, attention_mask) in enumerate(tokenized_dataset):
            # Pad input, target sequences, and attention masks to the same length
            input_tokens_padded, target_tokens_padded, attention_mask_padded = self.pad_sequences(
                [input_tokens], [target_tokens], [attention_mask], self.tokenizer.pad_token_id
            )

            # Add the padded input, target sequences, and attention mask to the appropriate buckets
            add_to_buckets(buckets, input_tokens_padded[0], target_tokens_padded[0], attention_mask_padded[0])
        
        # Create batches from the filled buckets and process them
        self.create_batches(buckets)







