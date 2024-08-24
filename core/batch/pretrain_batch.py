import numpy as np
from core.batch.batch_base import BatchBase
from core.batch.text_utils import get_line_text
from core.batch.bucketing import add_to_buckets

class PretrainBatchGenerator(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Initialize the base class
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        # get the text
        text = get_line_text(line)

        # encode the line as tokens
        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)

        # return the tokens
        return tokens

    def save_batch(self, batch_data, file_path):
        # Use the base class's method to process the input sequences
        input_tensor = self.process_batch_data(batch_data)
        
        if input_tensor is None:
            return None
        
        # Create target sequences specific to pretraining
        target_tensor, lengths = self.create_target_batch(input_tensor, self.tokenizer.pad_token_id, self.max_sequence_length)
        
        # Save both input and target tensors in a .npz file
        np.savez(file_path, input_tensor=input_tensor, target_tensor=target_tensor)
        
        # return the input tensor
        return input_tensor

    def create_target_batch(self, input_batch, pad_token_id, max_seq_length=None):
        target_indices = []

        for seq in input_batch:
            # Shift input sequence by one to create the target sequence
            target_seq = seq[1:].tolist() + [pad_token_id]

            # Ensure the target sequence is padded to match the input sequence length
            target_seq += [pad_token_id] * (len(seq) - len(target_seq))

            target_indices.append(target_seq)

        # Convert the target sequences to a numpy array
        target_tensor = np.array(target_indices, dtype=np.int32)

        # Return the target batch tensor
        return target_tensor, np.array([len(seq) for seq in target_indices], dtype=np.int32)


    def tokenize_and_batch(self, input_files):
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)
        
        # Initialize buckets for storing sequences
        buckets = {}
        
        # Process each tokenized sequence
        for input_tokens in tokenized_dataset:
            # Generate target tokens for the current input tokens
            target_tokens, _ = self.create_target_batch(np.array([input_tokens], dtype=np.int32), self.tokenizer.pad_token_id, self.max_sequence_length)
            target_tokens = target_tokens[0]
            
            # Add the input and target sequences to the appropriate buckets
            add_to_buckets(buckets, input_tokens, target_tokens)
        
        # Create batches from the filled buckets and process them
        self.create_batches(buckets)