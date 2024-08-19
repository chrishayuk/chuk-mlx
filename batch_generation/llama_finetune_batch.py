import json
import unicodedata
import re
from .finetune_batch import FineTuneBatch

class LLaMAFineTuneBatch(FineTuneBatch):
    def preprocess_text(self, text):
        # Normalize Unicode characters to NFKC form
        text = unicodedata.normalize('NFKC', text)

        # Remove or replace any unsupported characters (e.g., non-ASCII characters)
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Optionally replace with '[UNK]' or another placeholder

        # return the pre-processed text
        return text

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

        # Preprocess the text
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
            # Tokenize the instruction and target using the plain text
            input_tokens = self.tokenizer.encode(instruction)
            target_tokens = self.tokenizer.encode(target)
        except Exception as e:
            print(f"Error tokenizing input or target: {e}")
            print(f"Instruction: {instruction}")
            print(f"Target: {target}")
            return None, None

        return input_tokens, target_tokens
