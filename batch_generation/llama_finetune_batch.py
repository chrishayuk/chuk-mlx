import json
import os
import numpy as np
from .finetune_batch import FineTuneBatch
from batch_generation.sequence_utility import SequenceUtility

class LLaMAFineTuneBatch(FineTuneBatch):
    def tokenize_line(self, line):
        try:
            # Parse the JSON line
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None, None

        # Ensure we have a text column
        text = data.get('text', '')
        if not text:
            print("Text column is missing in the data.")
            return None, None

        # Ensure we have instruction tags
        inst_start = text.find('[INST]') + len('[INST]')
        inst_end = text.find('[/INST]')
        if inst_start == -1 + len('[INST]') or inst_end == -1:
            print("Instruction tags are missing or incorrect in the text.")
            return None, None

        # Extract the instruction and target from the LLaMA format
        instruction = text[inst_start:inst_end].strip()
        target = text[inst_end + len('[/INST]'):].strip()

        try:
            # Tokenize the instruction and target
            input_tokens = self.tokenizer.encode(instruction)
            target_tokens = self.tokenizer.encode(target)
        except Exception as e:
            print(f"Error tokenizing input or target: {e}")
            print(f"Instruction: {instruction}")
            print(f"Target: {target}")
            return None, None

        return input_tokens, target_tokens

    def save_batch(self, batch_data, file_path):
        inputs = [item[0] for item in batch_data]
        targets = [item[1] for item in batch_data]
        input_lengths = [len(input_seq) for input_seq in inputs]
        target_lengths = [len(target_seq) for target_seq in targets]

        # Check if any of the inputs or targets are empty
        if not inputs or not targets:
            print(f"Skipping empty batch: {file_path}")
            return None, None

        # Determine the maximum sequence length for the current batch
        max_seq_length = max(max(input_lengths), max(target_lengths)) + 1  # +1 for the EOS token

        # Get sequence utility for inputs and targets
        seq_util_inputs = SequenceUtility(max_seq_length=max_seq_length, padding_value=0, initial_pad_token=self.tokenizer.pad_token_id)
        seq_util_targets = SequenceUtility(max_seq_length=max_seq_length, padding_value=0, initial_pad_token=self.tokenizer.pad_token_id)

        # Pad the inputs
        padded_inputs = seq_util_inputs.batch_sequences(inputs)
        
        # Process targets to strip padding and existing EOS, then add a single EOS and pad
        eos_token_id = self.tokenizer.eos_token_id
        processed_targets = []
        for seq in targets:
            # Strip existing padding and EOS
            stripped_seq = [token for token in seq if token != eos_token_id and token != 0]
            # Append a single EOS token
            stripped_seq.append(eos_token_id)
            # Pad to the max sequence length
            if len(stripped_seq) < max_seq_length:
                padded_target = stripped_seq + [0] * (max_seq_length - len(stripped_seq))
            else:
                padded_target = stripped_seq[:max_seq_length]
            processed_targets.append(padded_target)
        
        # Convert the padded batches to numpy arrays
        inputs_array = np.array(padded_inputs, dtype=np.int32)
        targets_array = np.array(processed_targets, dtype=np.int32)
        input_lengths_array = np.array(input_lengths, dtype=np.int32)
        target_lengths_array = np.array(target_lengths, dtype=np.int32)

        # Ensure that the inputs and targets have the same shape
        assert inputs_array.shape == targets_array.shape, "Inconsistent shapes for inputs and targets"

        # Save the batch to a .npz file
        try:
            np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array, input_lengths=input_lengths_array, target_lengths=target_lengths_array)
        except Exception as e:
            print(f"Error saving batch: {file_path}")
            print(f"Error message: {str(e)}")
            return None, None
        
        return inputs_array, targets_array
