import time
import numpy as np
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table
from batch_generation.batch_base import BatchBase
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.bucketing import add_to_buckets

class FineTuneBatch(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Initialize the base class
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def save_batch(self, batch_data, file_path):
        inputs = [item[0] for item in batch_data]
        targets = [item[1] for item in batch_data]

        # Check if any of the inputs or targets are empty
        if not inputs or not targets:
            print(f"Skipping empty batch: {file_path}")
            return None, None

        # Convert to numpy arrays
        inputs_array = np.array(inputs, dtype=np.int32)
        targets_array = np.array(targets, dtype=np.int32)

        # Save the inputs and targets to a .npz file
        np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array)

        return inputs_array, targets_array

    def tokenize_and_batch(self, input_files):
        # Tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)
        
        # Initialize buckets for storing sequences
        buckets = {}
        
        # Process each tokenized sequence
        for input_tokens, target_tokens in tokenized_dataset:
            # Add the input and target sequences to the appropriate buckets
            add_to_buckets(buckets, input_tokens, target_tokens)
        
        # Create batches from the filled buckets and process them
        self.create_batches(buckets)

    def tokenize_dataset(self, input_files):
        tokenized_dataset = []
        
        for input_file in input_files:
            try:
                with open(input_file, 'r') as file:
                    for line in file:
                        # tokenize the line using the subclass method
                        input_tokens, target_tokens = self.tokenize_line(line)
                        
                        if input_tokens is not None and target_tokens is not None:
                            tokenized_dataset.append((input_tokens, target_tokens))
            except (OSError, IOError) as e:
                print(f"Error reading file {input_file}: {e}")
        
        return tokenized_dataset
    
    def process_batch(self, batch_idx, batch_data, file_path):
        # Start the batch timer
        batch_start_time = time.time()

        # Save the batch
        input_tensor, target_tensor = self.save_batch(batch_data, file_path)

        # Capture batch end time
        batch_end_time = time.time()

        # Generate and print summaries if requested
        summary_table = generate_batch_analysis_summary_table(input_tensor, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, input_tensor, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

        if self.print_summaries:
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

