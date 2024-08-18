import os
import time
import numpy as np
from batch_generation.bucketing import add_to_buckets, get_batch_from_buckets, merge_small_buckets, split_large_buckets
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table

class BatchBase:
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Set the various parameters
        self.tokenizer = tokenizer
        self.output_directory = output_directory
        self.file_prefix = file_prefix
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.print_summaries = print_summaries
    
    def tokenize_and_batch(self, input_files):
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)

        # Add tokenized sequences to buckets
        buckets = {}

        # Loop through the dataset
        for tokens in tokenized_dataset:
            # In the base case, use the same tokens for both input and target
            add_to_buckets(buckets, tokens, tokens)

        # Create batches from the buckets and process them
        self.create_batches(buckets)


    def tokenize_dataset(self, input_files):
        # Empty dataset
        tokenized_dataset = []

        # Loop through each file and tokenize lines
        for input_file in input_files:
            with open(input_file, 'r') as file:
                for line in file:
                    # tokenize the line
                    tokens = self.tokenize_line(line)

                    # check we have tokens
                    if tokens:
                        # add the tokens
                        tokenized_dataset.append(tokens)

        # Return the tokenized dataset
        return tokenized_dataset

    def create_batches(self, buckets):
        # batch index is zero
        batch_idx = 0

        # Process batches from buckets until no more batches can be formed
        batch = get_batch_from_buckets(buckets, self.batch_size)

        # loop through the batches
        while batch is not None:
            # get the batch filename
            file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}')

            # process the batch
            self.process_batch(batch_idx, batch, file_path)
            batch_idx += 1

            # get the batch from the buckets
            batch = get_batch_from_buckets(buckets, self.batch_size)

        # Handle any remaining sequences in the buckets
        for bucket in buckets.values():
            while bucket:
                # get the file
                file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}')
                
                # get the batch
                batch = bucket[:self.batch_size]

                # process the batch
                self.process_batch(batch_idx, batch, file_path)

                # set the bucket
                bucket = bucket[self.batch_size:]

                # move to next back
                batch_idx += 1

    def process_batch(self, batch_idx, batch_data, file_path):
        # Start the batch timer
        batch_start_time = time.time()

        # Process the batch data (encode and pad)
        input_tensor = self.save_batch(batch_data, file_path)

        # Capture batch end time
        batch_end_time = time.time()

        # Generate and print summaries if requested
        summary_table = generate_batch_analysis_summary_table(input_tensor, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, input_tensor, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

        if self.print_summaries:
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

    def save_batch(self, batch_data, file_path):
        # Use the base class's method to process the input sequences
        input_tensor = self.process_batch_data(batch_data)
        
        if input_tensor is None:
            return None
        
        # Save both input and target tensors in a .npz file
        np.savez(file_path, input_tensor=input_tensor)

        # return the input tensor
        return input_tensor
    
    def process_batch_data(self, batch_data):
        # Initialize processed batch
        processed_batch = []

        # Loop through the sequences in the batch
        for seq in batch_data:
            # Flatten the sequence (handle cases where items may be tuples or lists)
            flat_seq = [item if isinstance(item, int) else item[0] for item in seq]

            # Truncate or pad sequences as necessary
            if len(flat_seq) >= self.max_sequence_length:
                flat_seq = flat_seq[:self.max_sequence_length - 1] + [self.tokenizer.eos_token_id]
            else:
                if flat_seq[-1] != self.tokenizer.eos_token_id:
                    flat_seq.append(self.tokenizer.eos_token_id)
                padding_needed = self.max_sequence_length - len(flat_seq)
                flat_seq += [self.tokenizer.pad_token_id] * padding_needed
            
            # Add the processed sequence to the batch
            processed_batch.append(flat_seq)
        
        # Convert to numpy array
        input_tensor = np.array(processed_batch, dtype=np.int32)

        # Return the processed tensor
        return input_tensor

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

