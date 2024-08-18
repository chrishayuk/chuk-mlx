import os
import time
import numpy as np
from batch_generation.bucketing import add_to_buckets, get_batch_from_buckets, merge_small_buckets, split_large_buckets
from utils.tokenizer_loader import load_tokenizer
from batch_generation.sequence_utility import SequenceUtility
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table

class BatchBase:
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # set the various parameters
        self.tokenizer = tokenizer
        self.output_directory = output_directory
        self.file_prefix = file_prefix
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.print_summaries = print_summaries

    def save_batch(self, batch_data, file_path):
        # calculate the maximum sequence length within the batch
        min_seq_length = min(len(seq) for seq in batch_data)
        max_seq_length = max(len(seq) for seq in batch_data)

        # check if the batch is missized
        if min_seq_length != max_seq_length:
            print("Missized batch detected!")
            print(f"Min sequence length: {min_seq_length}")
            print(f"Max sequence length: {max_seq_length}")
            print(f"Batch data: {batch_data}")
            return None

        # create sequence utility with the batch-specific maximum sequence length
        seq_util = SequenceUtility(max_seq_length=max_seq_length, padding_value=self.tokenizer.pad_token_id)

        # pad the batch
        padded_batch = seq_util.batch_sequences(batch_data)

        # Remove sequences with None values from the padded batch
        valid_padded_batch = [seq for seq in padded_batch if None not in seq]

        # Check if the padded batch is empty
        if not valid_padded_batch:
            print("Skipping empty padded batch.")
            return None

        # load the padded batch
        try:
            batch_data = np.array(valid_padded_batch, dtype=np.int32)
        except TypeError as e:
            print(f"TypeError during np.array conversion: {e}")
            print(f"padded_batch: {valid_padded_batch}")
            return None

        # save the batch
        np.save(file_path, batch_data)
        return batch_data
    
    def create_batches(self, tokenized_dataset, output_directory, file_prefix, max_sequence_length, batch_size, tokenizer, print_summaries):
        # create the output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # start at the beginning for the batch
        batch_idx = 0
        total_batches = 0

        # get the batch from the buckets
        batch_data = get_batch_from_buckets(tokenized_dataset, batch_size)

        while batch_data is not None:
            # get the file path
            file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')

            # process the batch
            self.process_batch(batch_idx, batch_data, file_path)

            batch_idx += 1
            total_batches += 1

            # check if another batch can be formed
            batch_data = get_batch_from_buckets(tokenized_dataset, batch_size)

        # process any remaining batches in the buckets
        for bucket in tokenized_dataset.values():
            while bucket:
                # get the file path for the remaining batch
                file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')

                # process the remaining batch
                batch_data = bucket[:batch_size]
                self.process_batch(batch_idx, batch_data, file_path)

                bucket = bucket[batch_size:]
                batch_idx += 1
                total_batches += 1

    def tokenize_dataset(self, input_files):
        # empty dataset
        tokenized_dataset = []

        # loop through each file
        for input_file in input_files:
            # open the file
            with open(input_file, 'r') as file:
                for line in file:
                    # tokenize the line
                    tokens = self.tokenize_line(line)
                    
                    # if we have tokens, add them
                    if tokens:
                        tokenized_dataset.append(tokens)

        # return the dataset
        return tokenized_dataset

    def process_batch(self, batch_idx, batch_data, file_path):
        # start the batch timer
        batch_start_time = time.time()

        # save the concatenated batch
        concatenated_batch = self.save_batch(batch_data, file_path)

        # If the concatenated batch is None, skip the rest of the processing
        if concatenated_batch is None:
            return

        # capture batch end time
        batch_end_time = time.time()

        # calculate the batch generation time
        summary_table = generate_batch_analysis_summary_table(concatenated_batch, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, concatenated_batch, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

        if self.print_summaries:
            # print out the batch summary
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

    def tokenize_and_batch(self, input_files):
        # tokenize the dataset
        tokenized_dataset = self.tokenize_dataset(input_files)

        # add tokenized sequences to buckets
        buckets = {}
        for tokens in tokenized_dataset:
            add_to_buckets(buckets, tokens)

        # split large buckets
        #split_buckets = split_large_buckets(buckets, self.batch_size)

        # merge small buckets
        #merged_buckets = merge_small_buckets(split_buckets, self.batch_size)

        # create batches from the merged buckets
        self.create_batches(buckets, self.output_directory, self.file_prefix, self.max_sequence_length,
                       self.batch_size, self.tokenizer, self.print_summaries)

    def tokenize_line(self, line):
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def get_batch_file_path(self, batch_idx):
        raise NotImplementedError("Subclasses must implement the get_batch_file_path method.")
