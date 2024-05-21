import os
import time
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from batches.sequence_utility import SequenceUtility
from .batch_generation_summary import generate_batch_generation_summary
from .batch_analysis_summary import generate_batch_analysis_summary_table

class BatchBase:
    def __init__(self, tokenizer_name, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.output_directory = output_directory
        self.file_prefix = file_prefix
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.print_summaries = print_summaries

    def save_batch(self, batch_data, file_path):
        # get sequence utility
        seq_util = SequenceUtility(max_seq_length=self.max_sequence_length, padding_value=self.tokenizer.pad_token_id)

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
        # create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # start at the beginning for the batch
        batch_idx = 0
        current_batch = []
        total_batches = 0

        for input_file in input_files:
            with open(input_file, 'r') as file:
                for line in file:
                    # tokenize the line
                    tokens = self.tokenize_line(line)

                    # Debug statement to check tokens
                    if tokens is None:
                        print("tokenize_and_batch found a None value in tokens:", line)
                    elif not tokens:
                        print("tokenize_and_batch found an empty token list:", line)
                    else:
                        # add the tokens to the batch
                        current_batch.append(tokens)

                    # check if the current batch is full
                    if len(current_batch) == self.batch_size:
                        # get the file path
                        file_path = self.get_batch_file_path(batch_idx)

                        # process the batch
                        self.process_batch(batch_idx, current_batch, file_path)

                        # next batch
                        current_batch = []
                        batch_idx += 1
                        total_batches += 1

        # check if there are any remaining samples in the current batch
        if current_batch:
            # get the file path for the last batch
            file_path = self.get_batch_file_path(batch_idx)

            # process the last batch
            self.process_batch(batch_idx, current_batch, file_path)
            total_batches += 1

    def tokenize_line(self, line):
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def get_batch_file_path(self, batch_idx):
        raise NotImplementedError("Subclasses must implement the get_batch_file_path method.")
