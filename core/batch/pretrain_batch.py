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

    def tokenize_dataset(self, input_files):
        tokenized_dataset = []
        for input_file in input_files:
            with open(input_file, 'r') as file:
                for line in file:
                    tokenized_line = self.tokenize_line(line)
                    if len(tokenized_line) == 3:
                        tokenized_dataset.append(tokenized_line)
        return tokenized_dataset

    def pad_sequences(self, sequences, pad_value, max_length):
        padded_sequences = []
        for seq in sequences:
            seq = seq.tolist() if isinstance(seq, np.ndarray) else seq
            padded_seq = seq + [pad_value] * (max_length - len(seq))
            padded_sequences.append(padded_seq[:max_length])  # Ensure no sequence exceeds max_length
        return padded_sequences

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
        
        inputs_padded = self.pad_sequences(inputs, self.tokenizer.pad_token_id, max_length)
        targets_padded = self.pad_sequences(targets, self.tokenizer.pad_token_id, max_length)
        attention_masks_padded = self.pad_sequences(attention_masks, 0, max_length)

        inputs_array = np.array(inputs_padded, dtype=np.int32)
        targets_array = np.array(targets_padded, dtype=np.int32)
        attention_masks_array = np.array(attention_masks_padded, dtype=np.int32)

        if file_path:
            np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array, attention_mask_tensor=attention_masks_array)

        return inputs_array, targets_array, attention_masks_array


    def create_target_batch(self, input_batch, pad_token_id):
        max_length_in_batch = len(input_batch[0])
        target_indices = [seq[1:] + [pad_token_id] for seq in input_batch]
        target_padded = self.pad_sequences(target_indices, pad_token_id, max_length_in_batch)

        target_tensor = np.array(target_padded, dtype=np.int32)
        return target_tensor, np.array([len(seq) for seq in target_padded], dtype=np.int32)

    def tokenize_and_batch(self, input_files):
        tokenized_dataset = self.tokenize_dataset(input_files)
        buckets = {}
        
        for input_tokens, target_tokens, attention_mask in tokenized_dataset:
            input_tokens_padded, target_tokens_padded, attention_mask_padded = self.save_batch(
                [(input_tokens, target_tokens, attention_mask)], file_path=None
            )
            add_to_buckets(buckets, input_tokens_padded[0], target_tokens_padded[0], attention_mask_padded[0])
            
        self.create_batches(buckets)
