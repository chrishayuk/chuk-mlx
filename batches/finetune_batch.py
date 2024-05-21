import os
import time
from batches.batch_analysis_summary import generate_batch_analysis_summary_table
from batches.batch_generation_summary import generate_batch_generation_summary
from .batch_base import BatchBase

class FineTuneBatch(BatchBase):
    def tokenize_line(self, line):
        # Implement the logic to extract input and target sequences from the line
        # This method should be overridden by subclasses for specific formats
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")

    def get_batch_file_path(self, batch_idx):
        # set the concatenated file path
        concatenated_file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npy')

        # return the file path
        return concatenated_file_path

    def process_batch(self, batch_idx, batch_data, file_path):
        # start the batch timer
        batch_start_time = time.time()

        # save the concatenated batch
        concatenated_batch = self.save_batch(batch_data, file_path)

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