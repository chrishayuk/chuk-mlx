import json
from .finetune_batch import FineTuneBatch

class LLaMAFineTuneBatch(FineTuneBatch):
    def tokenize_line(self, line):
        try:
            # parse the JSON line
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        
        # ensure we have a text column
        text = data.get('text', '')
        if not text:
            print("Text column is missing in the data.")
            return None
        
        # ensure we have instruction tags
        inst_start = text.find('[INST]') + len('[INST]')
        inst_end = text.find('[/INST]')
        if inst_start == -1 or inst_end == -1:
            print("Instruction tags are missing or incorrect in the text.")
            return None
        
        # extract the instruction and target from the LLaMA format
        instruction = text[inst_start:inst_end].strip()
        target = text[inst_end + len('[/INST]'):].strip('</s>').strip()

        # tokenize the instruction and target
        input_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)

        # Concatenate input and target tokens, with a separator if needed
        sequence = input_tokens + [self.tokenizer.sep_token_id] + target_tokens

        # return the sequence
        #return sequence

        # return the input and target tokens
        return input_tokens, target_tokens