import numpy as np

from chuk_lazarus.data.preprocessing.batch_base import BatchBase


class FineTuneBatch(BatchBase):
    def __init__(
        self,
        tokenizer,
        output_directory,
        file_prefix,
        max_sequence_length,
        batch_size,
        print_summaries,
        dtype=np.int32,
    ):
        # Initialize the base class with the dtype parameter
        super().__init__(
            tokenizer,
            output_directory,
            file_prefix,
            max_sequence_length,
            batch_size,
            print_summaries,
            dtype,
        )

    def tokenize_line(self, line):
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement the tokenize_line method.")
