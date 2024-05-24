import json
from .finetune_batch import FineTuneBatch

class LLaMAFineTuneBatch(FineTuneBatch):
    def tokenize_line(self, line):
        try:
            # parse the JSON line
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []

        # ensure we have a text column
        text = data.get('text', '')
        if not text:
            print("Text column is missing in the data.")
            return []

        # ensure we have instruction tags
        inst_start = text.find('[INST]') + len('[INST]')
        inst_end = text.find('[/INST]')
        if inst_start == -1 or inst_end == -1:
            print("Instruction tags are missing or incorrect in the text.")
            return []

        # extract the instruction and target from the LLaMA format
        instruction = text[inst_start:inst_end].strip()
        target = text[inst_end + len('[/INST]'):].strip('</s>').strip()

        # tokenize the instruction and target
        input_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)

        # Concatenate input and target tokens, with a separator if needed
        sequence = input_tokens + [self.tokenizer.sep_token_id] + target_tokens

        # Check if the sequence is valid and within the maximum sequence length
        if len(sequence) > self.max_sequence_length:
            print(f"Skipping sequence due to exceeding max_sequence_length: {len(sequence)}")
            return []

        # Check if there is a mismatch in lengths
        if len(input_tokens) + len(target_tokens) + 1 != len(sequence):
            print(f"Mismatched lengths detected:")
            print(f"Instruction tokens length: {len(input_tokens)}")
            print(f"Target tokens length: {len(target_tokens)}")
            print(f"Concatenated sequence length: {len(sequence)}")

        return sequence