import os
import time
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility
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
        
        # load the padded batch
        batch_data = np.array(padded_batch, dtype=np.int32)
        
        # save the batch
        np.save(file_path, batch_data)
        return batch_data

    def process_batch(self, batch_idx, batch_data, file_path):
        # start the batch timer
        batch_start_time = time.time()

        # save the batch
        batch_data = self.save_batch(batch_data, file_path, self.max_sequence_length, self.pad_token_id)

        # capture batch end time
        batch_end_time = time.time()

        # calculate the batch generation time
        summary_table = generate_batch_analysis_summary_table(batch_data, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)

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