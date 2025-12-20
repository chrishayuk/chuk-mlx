import numpy as np
from chuk_lazarus.data.preprocessing.batch_base import BatchBase
from chuk_lazarus.data.preprocessing.text_utils import get_line_text
from chuk_lazarus.data.preprocessing.bucketing import add_to_buckets

class PretrainBatchGenerator(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries, dtype=np.int32):
        # Initialize the base class with the dtype parameter
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries, dtype)

    def tokenize_line(self, line):
        # get the line to tokenize
        text = get_line_text(line)

        # tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)
        
        # calculate the target and attention mask
        target_tokens, attention_mask = self.create_target_and_mask(tokens)

        # return the tokens, target and attention
        return tokens, target_tokens, attention_mask

    def create_target_and_mask(self, tokens):
        """
        Generates target tokens by shifting the input tokens and creates an attention mask.
        """

        # calculate the target tokens by doing a shift, and adding a pad token
        target_tokens = tokens[1:] + [self.tokenizer.pad_token_id]

        # calculate the attention mask by just 1's it
        attention_mask = [1] * len(tokens)

        # return target tokens and attention mask
        return target_tokens, attention_mask

    def save_batch(self, batch_data, file_path=None):
        """
        Preprocesses batch data to generate targets and attention masks if necessary, then calls the base class save_batch.
        
        Args:
            batch_data (list): List of input sequences or tuples of (input, target, attention_mask).
            file_path (str): File path to save the batch.
        
        Returns:
            tuple: Numpy arrays of inputs, targets, and attention masks.
        """
        if not isinstance(batch_data[0], tuple):
            # Preprocess batch_data to generate targets and attention masks
            batch_data = [
                (seq, seq[1:] + [self.tokenizer.pad_token_id], [1] * len(seq))
                for seq in batch_data
            ]

        # save the batch using the dtype from the parent class
        return super().save_batch(batch_data, file_path)
