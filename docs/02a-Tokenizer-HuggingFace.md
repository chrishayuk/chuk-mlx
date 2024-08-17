# Print Tokens Utility
The print tokens utility allows you to print tokens or full vocabulary for either a hugginface tokenizer or a custom tokenizer.

## Hugging Face Tokenizers
The following shows you how to view the tokens for a given string for a huggingface tokenizer.

```bash
python print_tokens.py --tokenizer ibm-granite/granite-3b-code-instruct --prompt "Who is Ada Lovelace?" --no_special_tokens
```

and if you want to see the whole vocabulary

```bash
python print_tokens.py --tokenizer ibm-granite/granite-3b-code-instruct
```

## Custom Tokenizers
You can also use the print tokens utility on custom tokenizers.

If you wish to tokenize a prompt for a custom tokenizer, you can view the prompt tokens using the print_tokens utility.

```bash
python print_tokens.py --tokenizer lazyfox --prompt "the quick brown fox"  --no_special_tokens
```

And if you want to see the whole vocabulary

```bash
python print_tokens.py --tokenizer lazyfox
```
