# Print Tokens Utility
The print tokens utility allows you to print tokens or full vocabulary for either a huggingface tokenizer or a custom tokenizer.

## Hugging Face Tokenizers
The following shows you how to view the tokens for a given string for a huggingface tokenizer.

```bash
python tools/tokenizer/print_tokens.py --tokenizer HuggingFaceTB/SmolLM-135M-Instruct --prompt "Who is Ada Lovelace?" --skip-special-tokens
```

and if you want to see the whole vocabulary

```bash
python tools/tokenizer/print_tokens.py --tokenizer HuggingFaceTB/SmolLM-135M-Instruct
```

## Custom Tokenizers
You can also use the print tokens utility on custom tokenizers.

### prompt tokenization
If you wish to tokenize a prompt for a custom tokenizer, you can view the prompt tokens using the print_tokens utility.

```bash
python tools/tokenizer/print_tokens.py --tokenizer lazyfox --prompt "the quick brown fox"  --skip-special-tokens
```

or

```bash
python tools/tokenizer/print_tokens.py --tokenizer math --prompt "1 + 1 = 2"  --skip-special-tokens
```

### full vocabulary
And if you want to see the whole vocabulary

```bash
python tools/tokenizer/print_tokens.py --tokenizer lazyfox
```

or

```bash
python tools/tokenizer/print_tokens.py --tokenizer math
```

