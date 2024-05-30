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
        target = text[inst_end + len('[/INST]'):].strip('</s>').strip()

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

        # Get sequence utility
        seq_util = SequenceUtility(max_seq_length=self.max_sequence_length, padding_value=self.tokenizer.pad_token_id)
        
        # Pad the inputs and targets
        padded_inputs = seq_util.batch_sequences(inputs)
        padded_targets = seq_util.batch_sequences(targets)
        
        # Convert the padded batches to numpy arrays
        inputs_array = np.array(padded_inputs, dtype=np.int32)
        targets_array = np.array(padded_targets, dtype=np.int32)
        input_lengths_array = np.array(input_lengths, dtype=np.int32)
        target_lengths_array = np.array(target_lengths, dtype=np.int32)

        # Save the batch to a .npz file
        np.savez(file_path, input_tensor=inputs_array, target_tensor=targets_array, input_lengths=input_lengths_array, target_lengths=target_lengths_array)
        
        return inputs_array, targets_array
