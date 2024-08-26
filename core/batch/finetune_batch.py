import numpy as np
from core.batch.batch_analysis_summary import generate_batch_analysis_summary_table
from core.batch.batch_base import BatchBase
from core.batch.batch_generation_summary import generate_batch_generation_summary
from core.batch.bucketing import add_to_buckets

class FineTuneBatch(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Initialize the base class
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")







