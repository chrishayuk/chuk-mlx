import numpy as np
from batch_generation.batch_base import BatchBase
from batch_generation.text_utils import get_line_text
from batch_generation.bucketing import add_to_buckets

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

    def create_target_batch(self, input_batch, pad_token_id, max_seq_length):
        target_indices = []

        # loop through the batch
        for seq in input_batch:
            # Shift input sequence by one to create the target sequence
            target_seq = seq[1:].tolist() + [pad_token_id]
            
            # Pad or truncate the target sequence to match max_seq_length
            if len(target_seq) < max_seq_length:
                target_seq += [pad_token_id] * (max_seq_length - len(target_seq))
            else:
                target_seq = target_seq[:max_seq_length]
            
            # add the target
            target_indices.append(target_seq)
        
        # return the target batch
        return np.array(target_indices, dtype=np.int32), np.array([len(seq) for seq in target_indices], dtype=np.int32)

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
