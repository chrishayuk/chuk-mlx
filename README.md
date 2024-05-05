## Model Downloader
This is a simple utility that allows you to download a model from huggingface hug, and stick it the cache

```bash
python model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.2
```

## Model Viewer
This is a simple utility that allows you to view a models config or layers.

```bash
python model_viewer.py --model meta-llama/Meta-Llama-3-8B-Instruct --show-config
```

or

```bash
python model_viewer.py --model ibm/granite-7b-base --show-layers
```

## Print Layer Modules
This is a simple utility that allows you to print the modules of a layer.

```bash
python print_layer_modules.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

or

```bash
python print_layer_modules.py --model ibm/granite-7b-base
```
    
## Print Tokens
This is a simple utility that allows you to print tokens for the given tokenizer

```bash
python print_tokens.py --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --prompt "Who is Ada Lovelace?"
```

or

```bash
python print_tokens.py --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --prompt "Who is Ada Lovelace?"
```

or

```bash
python print_tokens.py --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --prompt "the quick brown fox jumps over the lazy dog"
```

or

```bash
python print_tokens.py --tokenizer ibm/granite-7b-base --prompt "the quick brown fox jumps over the lazy dog"
```

or

```bash
python print_tokens.py --tokenizer lazyfox_tokenizer --prompt "the quick brown fox jumps over the lazy dog"
```

## Infer

```bash
python infer.py --model mistralai/Mistral-7B-Instruct-v0.2
```

```bash
python infer.py --model ibm/granite-7b-base
```

```bash
python infer.py --model meta-llama/Meta-Llama-3-8B-Instruct
```