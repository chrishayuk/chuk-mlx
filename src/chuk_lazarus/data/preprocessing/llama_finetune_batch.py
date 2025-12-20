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
            return None, None, None

        # Preprocess the text
        text = self.preprocess_text(text)

        # Handle cases without instruction tags
        inst_start = text.find('[INST]')
        inst_end = text.find('[/INST]')

        if inst_start == -1 or inst_end == -1:
            # Add special tokens for input
            input_tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Set target to be empty with only EOS token
            target_tokens = [self.tokenizer.eos_token_id]
        else:
            # Extract the instruction and target from the LLaMA format
            inst_start += len('[INST]')
            instruction = text[inst_start:inst_end].strip()
            target = text[inst_end + len('[/INST]'):].strip()

            try:
                # Tokenize the instruction with special tokens
                input_tokens = self.tokenizer.encode(instruction, add_special_tokens=True)
                
                # Tokenize target without special tokens and then manually add end-of-sequence token
                target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
                
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None and (not target_tokens or target_tokens[-1] != eos_token_id):
                    target_tokens.append(eos_token_id)
            except Exception as e:
                print(f"Error tokenizing input or target: {e}")
                print(f"Instruction: {instruction}")
                print(f"Target: {target}")
                return None, None

        # Generate the attention mask: 1 for non-pad tokens, 0 for pad tokens
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        attention_mask = [1 if token != pad_token_id else 0 for token in input_tokens]

        return input_tokens, target_tokens, attention_mask