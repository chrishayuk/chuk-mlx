import os
import time
import numpy as np
from core.batch.bucketing import add_to_buckets, get_batch_from_buckets, merge_small_buckets, split_large_buckets
from core.batch.batch_generation_summary import generate_batch_generation_summary
from core.batch.batch_analysis_summary import generate_batch_analysis_summary_table
from core.batch.tokenization_utils import batch_tokenize_and_pad, tokenize_and_pad

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
        while True:
            # Get the next batch from the buckets
            batch = get_batch_from_buckets(buckets, self.batch_size)

            if batch is None or len(batch) == 0:
                # No more batches can be formed, exit the loop
                break

            # get the batch filename
            file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npz')

            # process the batch
            self.process_batch(batch_idx, batch, file_path)
            batch_idx += 1

            # Check if all buckets are empty
            if all(len(bucket) == 0 for bucket in buckets.values()):
                break

        # Handle any remaining sequences in the buckets (unlikely with the above logic)
        for bucket_key in list(buckets.keys()):
            while buckets[bucket_key]:
                # get the file path
                file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npz')

                # get the batch
                batch = buckets[bucket_key][:self.batch_size]

                # process the batch
                self.process_batch(batch_idx, batch, file_path)

                # remove processed sequences from the bucket
                buckets[bucket_key] = buckets[bucket_key][self.batch_size:]

                # Check if the bucket is empty and remove it
                if not buckets[bucket_key]:
                    del buckets[bucket_key]

                # move to the next batch index
                batch_idx += 1

                # Exit the loop if no buckets remain
                if not buckets:
                    break


    def process_batch(self, batch_idx, batch_data, file_path):
        # Start the batch timer
        batch_start_time = time.time()

        # Save the batch
        result = self.save_batch(batch_data, file_path)

        # Capture batch end time
        batch_end_time = time.time()

        if isinstance(result, tuple):
            input_tensor, target_tensor = result
        else:
            input_tensor = result

        # Generate and print summaries if requested
        summary_table = generate_batch_analysis_summary_table(input_tensor, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, input_tensor, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

        if self.print_summaries:
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

    def save_batch(self, batch_data, file_path):
        def pad_sequences(sequences, pad_value, max_length=None):
            if not max_length:
                max_length = max(len(seq) for seq in sequences)
            return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]

        # Check if batch_data contains tuples (input, target) or just inputs
        if isinstance(batch_data[0], tuple):
            inputs = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]

            # Pad sequences
            inputs_padded = pad_sequences(inputs, self.tokenizer.pad_token_id)
            targets_padded = pad_sequences(targets, self.tokenizer.pad_token_id)

            # Convert to numpy arrays
            inputs_array = np.array(inputs_padded, dtype=np.int32)
            targets_array = np.array(targets_padded, dtype=np.int32)

            # Save the inputs and targets to a .npz file
            np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array)

            # Return both input and target tensors
            return inputs_array, targets_array
        else:
            # Assume batch_data contains only inputs
            inputs_padded = pad_sequences(batch_data, self.tokenizer.pad_token_id)

            # Convert to numpy array
            inputs_array = np.array(inputs_padded, dtype=np.int32)

            # Save only the input tensor to a .npz file
            np.savez(file_path, input_tensor=inputs_array)

            # Return only the input tensor
            return inputs_array
    
    def process_batch_data(self, batch_data):
        # Use the batch_tokenize_and_pad function to process the batch
        processed_batch = batch_tokenize_and_pad(batch_data, self.tokenizer, self.max_sequence_length)
        
        # Convert to numpy array
        input_tensor = np.array(processed_batch, dtype=np.int32)

        # Return the processed tensor
        return input_tensor

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")
