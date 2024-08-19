import json
import os
import numpy as np
from .finetune_batch import FineTuneBatch
from batch_generation.sequence_utility import SequenceUtility

class LLaMAFineTuneBatch(FineTuneBatch):
    def preprocess_text(self, text):
        # Encode the text as Unicode escape sequences using a custom function to force \u format
        return ''.join(f'\\u{ord(c):04x}' if ord(c) > 127 else c for c in text)

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

        # Preprocess the text to handle special characters using Unicode escape
        text = self.preprocess_text(text)

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

