import json
from .finetune_batch import FineTuneBatch

class LLaMAFineTuneBatch(FineTuneBatch):
    def tokenize_line(self, line):
        # parse the JSON line
        data = json.loads(line)

        # extract the text column
        text = data['text']

        # extract the instruction and target from the LLaMA format
        inst_start = text.find('[INST]') + len('[INST]')
        inst_end = text.find('[/INST]')
        instruction = text[inst_start:inst_end]
        target = text[inst_end + len('[/INST]'):].strip('</s>')

        # tokenize the instruction and target
        input_tokens = self.tokenizer.encode(instruction, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)

        # return the input and target tokens
        return input_tokens, target_tokens