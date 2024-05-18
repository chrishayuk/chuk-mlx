from tabulate import tabulate

class TokenDisplayUtility:
    def __init__(self, tokenizer):
        # set the tokenizer
        self.tokenizer = tokenizer

    def display_tokens_from_prompt(self, prompt):
        # encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # display the tokens
        self.display_tokens(input_ids)

    def display_tokens_from_ids(self, input_ids):
        # display the tokens
        self.display_tokens(input_ids)

    def display_tokens(self, input_ids):
        # create a table of prompts
        table_data = [
            [i, token_id, self.tokenizer.decode([token_id])]
            for i, token_id in enumerate(input_ids)
        ]
        
        # print the table
        print(tabulate(table_data, headers=["Index", "Token ID", "Decoded Token"], tablefmt="grid"))

    def display_full_vocabulary(self):
        # get the full vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # get the token IDs
        token_ids = list(vocab.values())
        
        # create a table of the vocabulary
        table_data = [
            [i, token_id, self.tokenizer.decode([token_id])]
            for i, token_id in enumerate(token_ids)
        ]
        
        # print the table
        print(tabulate(table_data, headers=["Index", "Token ID", "Decoded Token"], tablefmt="grid"))