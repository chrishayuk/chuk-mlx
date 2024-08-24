# cli
the cli allows you test various aspects of encoding, decoding of the tokenizer through the cli

## encoding
to encode a string, you can use the following command, and this will default to the lazyfox tokenizer

```bash
python tools/tokenizer/custom_tokenizer_cli.py encode "the quick brown fox"
```

alternatively you can pass in the tokenizer name

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox encode "the quick brown fox" 
```

if you wish to include special tokens (such as BOS, EOS) in the encoding then use the following

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox encode "the quick brown fox" --add-special-tokens
```

## decoding a sequence
to encode some tokens, you can use the following command, and this will default to the lazyfox tokenizer

```bash
python tools/tokenizer/custom_tokenizer_cli.py decode 2 4 5 6 7 3
```

if you wish to skip special tokens

```bash
python tools/tokenizer/custom_tokenizer_cli.py decode 2 4 5 6 7 3 --skip-special-tokens
```

and specify a tokenizer

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox decode 2 4 5 6 7 3 --skip-special-tokens
```

## testing padding
we can test that padding works correctly for the tokenizer using the following

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox pad 4 5 6 7 3 --max-length 10
```

and with attention mask

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox pad 4 5 6 7 3 --max-length 10 --return-attention-mask
```

## saving the vocabulary
if you wish to save the vocabulary of the tokenizer to json, you can do this via the following command

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox save-vocab
```

## load the vocabulary
if you wish to load the vocabulary of the tokenizer to json, you can do this via the following command

```bash
python tools/tokenizer/custom_tokenizer_cli.py --tokenizer-name lazyfox load-vocab
```
