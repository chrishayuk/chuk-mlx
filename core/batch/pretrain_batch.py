import numpy as np
from core.batch.batch_base import BatchBase
from core.batch.text_utils import get_line_text
from core.batch.bucketing import add_to_buckets

class PretrainBatchGenerator(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

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
        if not batch_data:
            raise ValueError("Batch data is empty or not properly structured.")
        
        # check if we already have inputs, targets and attention masks in batch
        if isinstance(batch_data[0], tuple) and len(batch_data[0]) == 3:
            # get the inputs, targets and attention masks from the batch
            inputs = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]
            attention_masks = [item[2] for item in batch_data]
        else:
            # set the input as the batch
            inputs = batch_data

            # calculate target and attention masks from inputs
            targets, attention_masks = zip(*[self.create_target_and_mask(seq) for seq in inputs])
        
        # pad the sequences
        inputs_padded, targets_padded, attention_masks_padded = self.pad_sequences(
            inputs, targets, attention_masks, self.tokenizer.pad_token_id
        )

        # load the inputs, targets and attention mask as a numpy array
        inputs_array = np.array(inputs_padded, dtype=np.int32)
        targets_array = np.array(targets_padded, dtype=np.int32)
        attention_masks_array = np.array(attention_masks_padded, dtype=np.int32)

        # check if we have a path to save
        if file_path:
            # save
            np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array, attention_mask_tensor=attention_masks_array)

        # return inputs, targets and attention masks
        return inputs_array, targets_array, attention_masks_array
