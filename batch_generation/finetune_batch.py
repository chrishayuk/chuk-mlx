import os
import time
import random
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.sequence_utility import SequenceUtility
from utils.tokenizer_loader import load_tokenizer
from batch_generation.batch_base import BatchBase
from batch_generation.bucketing import add_to_buckets, split_large_buckets, merge_small_buckets, get_batch_from_buckets

class FineTuneBatch(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        self.tokenizer = load_tokenizer(tokenizer)
        self.output_directory = output_directory
        self.file_prefix = file_prefix
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.print_summaries = print_summaries
        self.buckets = {}

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
        
        for input_file in input_files:
            try:
                with open(input_file, 'r') as file:
                    for line in file:
                        # tokenize the line using the subclass method
                        input_tokens, target_tokens = self.tokenize_line(line)
                        
                        if input_tokens is not None and target_tokens is not None:
                            # add the tokens to the appropriate bucket
                            add_to_buckets(self.buckets, input_tokens, target_tokens)
            except (OSError, IOError) as e:
                print(f"Error reading file {input_file}: {e}")
        
        # split large buckets and merge small buckets
        split_buckets = split_large_buckets(self.buckets, self.batch_size)
        merged_buckets = merge_small_buckets(split_buckets, self.batch_size)
        
        # create a list to store the batches
        batches = []
        
        # process buckets and create batches
        batch_idx = 0
        total_batches = 0
        while True:
            batch_data = get_batch_from_buckets(merged_buckets, self.batch_size)
            if batch_data is None:
                break
            
            file_path = self.get_batch_file_path(batch_idx)
            batches.append((batch_idx, batch_data, file_path))
            batch_idx += 1
            total_batches += 1
        
        # shuffle the batches
        random.shuffle(batches)
        
        # process and save the shuffled batches
        for batch_idx, batch_data, file_path in batches:
            self.process_batch(batch_idx, batch_data, file_path)
        
        print(f"Total batches processed: {total_batches}")