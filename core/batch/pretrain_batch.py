import numpy as np
from core.batch.batch_base import BatchBase
from core.batch.text_utils import get_line_text
from core.batch.bucketing import add_to_buckets

class PretrainBatchGenerator(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        text = get_line_text(line)
        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)
        attention_mask = [1] * len(tokens)
        target_tokens = tokens[1:] + [self.tokenizer.pad_token_id]
        return tokens, target_tokens, attention_mask


    def save_batch(self, batch_data, file_path=None):
        if not batch_data:
            raise ValueError("Batch data is empty or not properly structured.")
        
        # Handle batch_data that is either a list of inputs or a list of (input, target, attention_mask) tuples
        if isinstance(batch_data[0], tuple) and len(batch_data[0]) == 3:
            inputs = [item[0] for item in batch_data]
            targets = [item[1] for item in batch_data]
            attention_masks = [item[2] for item in batch_data]
        else:
            inputs = batch_data
            targets = [seq[1:] + [self.tokenizer.pad_token_id] for seq in inputs]
            attention_masks = [[1] * len(seq) for seq in inputs]
        
        max_length = max(len(seq) for seq in inputs)
        
        # Use the complex pad_sequences method from the base class
        inputs_padded, targets_padded, attention_masks_padded = self.pad_sequences(
            inputs, targets, attention_masks, self.tokenizer.pad_token_id
        )

        inputs_array = np.array(inputs_padded, dtype=np.int32)
        targets_array = np.array(targets_padded, dtype=np.int32)
        attention_masks_array = np.array(attention_masks_padded, dtype=np.int32)

        if file_path:
            np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array, attention_mask_tensor=attention_masks_array)

        return inputs_array, targets_array, attention_masks_array


