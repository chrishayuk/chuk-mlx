# Tokenizer
The following are a set of diagnostic tools that help you work with the tokenizers

## huggingface tokenizer
The following shows how you can use the tokenizer diagnostic tools to check tokenization for huggingface tokenizers

### print tokens for a prompt
The following shows you how to view the tokens for a given string for a huggingface tokenizer.

```bash
python print_tokens.py --tokenizer ibm-granite/granite-3b-code-instruct --prompt "Who is Ada Lovelace?"
```

### view the full vocabularly

and if you want to see the whole vocabulary

```bash
python print_tokens.py --tokenizer ibm-granite/granite-3b-code-instruct --no_special_tokens
```

## custom tokenizer
The chuk framework has full support for custom tokenizers.

### print tokens for a prompt
If you wish to tokenize a prompt for a custom tokenizer, you can view the prompt tokens using the print_tokens utility.

```bash
python print_tokens.py --tokenizer lazyfox --prompt "the quick brown fox"  --no_special_tokens
```

### view the full vocabularly
And if you want to see the whole vocabulary

```bash
python print_tokens.py --tokenizer lazyfox
```

### tokenizer diagnostics
If you wish to check the tokenizer is working correctly, you can run tokenizer diagnostics

```bash
python custom_tokenizer_diagnostics.py --tokenizer lazyfox
```

