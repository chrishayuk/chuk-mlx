import os
import time
import numpy as np
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.sequence_utility import SequenceUtility
from utils.tokenizer_loader import load_tokenizer
from .batch_base import BatchBase

class FineTuneBatch(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        self.tokenizer = load_tokenizer(tokenizer)
        self.output_directory = output_directory
        self.file_prefix = file_prefix
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.print_summaries = print_summaries

    def tokenize_line(self, line):
        # Implement the logic to extract input and target sequences from the line
        # This method should be overridden by subclasses for specific formats
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def get_batch_file_path(self, batch_idx):
        # set the concatenated file path
        concatenated_file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npz')

        # return the file path
        return concatenated_file_path

    def process_batch(self, batch_idx, batch_data, file_path):
        # start the batch timer
        batch_start_time = time.time()

        # save the batch
        inputs_array, targets_array = self.save_batch(batch_data, file_path)

        # capture batch end time
        batch_end_time = time.time()

        # calculate the batch generation time
        summary_table = generate_batch_analysis_summary_table(inputs_array, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, inputs_array, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

        if self.print_summaries:
            # print out the batch summary
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

    def save_batch(self, batch_data, file_path):
        # Method to save batch data to file
        # Implementation should be provided in the subclass
        raise NotImplementedError("Subclasses must implement the save_batch method.")

    def tokenize_and_batch(self, input_files):
        # create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # initialize batch variables
        batch_idx = 0
        current_batch = []
        total_batches = 0
        
        for input_file in input_files:
            try:
                with open(input_file, 'r') as file:
                    for line in file:
                        # tokenize the line using the subclass method
                        input_tokens, target_tokens = self.tokenize_line(line)
                        
                        if input_tokens is not None and target_tokens is not None:
                            # add the tokens to the current batch
                            current_batch.append((input_tokens, target_tokens))
                        
                            # check if the current batch is full
                            if len(current_batch) == self.batch_size:
                                # construct the file path for the batch
                                file_path = self.get_batch_file_path(batch_idx)

                                # process the batch
                                self.process_batch(batch_idx, current_batch, file_path)
                                
                                # reset the current batch
                                current_batch = []
                                batch_idx += 1
                                total_batches += 1
            except (OSError, IOError) as e:
                print(f"Error reading file {input_file}: {e}")
        
        # process any remaining samples in the current batch
        if current_batch:
            file_path = self.get_batch_file_path(batch_idx)
            self.process_batch(batch_idx, current_batch, file_path)
            total_batches += 1

        print(f"Total batches processed: {total_batches}")

# Example usage:
# tokenizer = load_tokenizer('tokenizer_name')
# fine_tune_batch = LLaMAFineTuneBatch(tokenizer, 'output_dir', 'file_prefix', 128, 32, True)
# fine_tune_batch.tokenize_and_batch(['input1.txt', 'input2.txt'])
